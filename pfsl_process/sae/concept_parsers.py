"""Offline normalization of frozen biological concept snapshots."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from pfsl_process.sae.external_snapshots import (
    SnapshotError,
    validate_external_snapshots,
)
from pfsl_process.sae.feature_atlas_io import (
    AtlasIOError,
    _reject_symlink_components,
    file_metadata,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
    write_jsonl_atomic,
)


class ConceptParseError(ValueError):
    """Raised when a frozen source or normalized index violates its contract."""


@dataclass(frozen=True)
class ConceptRecord:
    database: Literal["go", "reactome", "interpro", "corum"]
    concept_id: str
    concept_name: str
    concept_type: str
    protein_accessions: tuple[str, ...]
    original_accessions: tuple[str, ...]
    parent_ids: tuple[str, ...]
    evidence_tier: str
    source_release: str
    source_sha256: str
    membership_tracks: tuple[tuple[str, tuple[str, ...]], ...]
    redundancy_cluster_id: str | None


_CANONICAL_ACCESSION = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){2})\Z"
)
_ISOFORM = re.compile(r"(?P<base>.+)-[1-9][0-9]*\Z")
_PARTIAL = re.compile(r"(?P<base>.+)-PRO_[0-9]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GO_ID = re.compile(r"GO:[0-9]{7}\Z")
_INTERPRO_ID = re.compile(r"IPR[0-9]{6}\Z")
_PRIMARY_GO_CODES = frozenset(
    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP",
     "HGI", "HEP", "TAS", "IC"}
)
_OUTPUT_FILES = ("concept_index.jsonl", "parse_audit.json", "manifest.json")
_AUDIT_SCHEMAS = {
    "go": frozenset({
        "accepted_rows", "alias_resolutions", "concepts", "duplicate_rows",
        "filtered_not_rows", "input_rows", "lost_invalid_accession_rows",
        "lost_obsolete_unresolved_rows", "lost_unknown_term_rows",
        "lost_unmapped_accession_rows", "replacement_rows",
        "secondary_evidence_rows",
    }),
    "reactome": frozenset({
        "accepted_rows", "concepts", "duplicate_rows",
        "filtered_non_human_rows", "input_rows", "lost_invalid_accession_rows",
        "lost_unknown_pathway_rows", "lost_unmapped_accession_rows",
    }),
    "interpro": frozenset({
        "accepted_rows", "concepts", "duplicate_rows",
        "filtered_unsupported_type_rows", "input_rows",
        "lost_invalid_accession_rows", "lost_unknown_entry_rows",
        "lost_unmapped_accession_rows", "unsupported_entry_definitions",
    }),
    "corum": frozenset({
        "accepted_human_complex_rows", "accepted_member_tokens", "concepts",
        "duplicate_complex_rows", "duplicate_member_tokens",
        "filtered_non_human_complex_rows", "filtered_out_of_universe_members",
        "input_complex_rows", "input_member_tokens", "isoform_collapses",
        "lost_invalid_accession_members", "partial_protein_collapses",
        "redundancy_components",
    }),
}


def canonical_uniprot_accession(value: str) -> str:
    """Return a canonical accession after only documented suffix collapses."""

    if type(value) is not str or not value or value != value.strip():
        raise ValueError("UniProt accession must be one non-empty stripped string")
    match = _PARTIAL.fullmatch(value)
    if match is None:
        match = _ISOFORM.fullmatch(value)
    candidate = match.group("base") if match is not None else value
    if _CANONICAL_ACCESSION.fullmatch(candidate) is None:
        raise ValueError(f"Invalid UniProt accession: {value!r}")
    return candidate


def _validate_common(
    universe: frozenset[str], release: str, source_sha256: str
) -> None:
    if type(universe) is not frozenset:
        raise TypeError("universe must be a frozenset")
    for accession in universe:
        if canonical_uniprot_accession(accession) != accession:
            raise ConceptParseError(f"Universe accession is not canonical: {accession!r}")
    if type(release) is not str or not release.strip():
        raise ConceptParseError("source release is missing")
    if type(source_sha256) is not str or _SHA256.fullmatch(source_sha256) is None:
        raise ConceptParseError("source SHA-256 is invalid")


def _open_text(path: Path):
    if not isinstance(path, Path) or not path.is_file():
        raise ConceptParseError(f"Source file does not exist: {path}")
    try:
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", newline="")
        return path.open("rt", encoding="utf-8", newline="")
    except OSError as error:
        raise ConceptParseError(f"Cannot open source file: {path}") from error


def _require_acyclic(parents: dict[str, set[str]], label: str) -> None:
    states: dict[str, int] = {}

    def visit(node: str) -> None:
        state = states.get(node, 0)
        if state == 1:
            raise ConceptParseError(f"{label} parent cycle contains {node}")
        if state == 2:
            return
        states[node] = 1
        for parent in sorted(parents.get(node, ())):
            visit(parent)
        states[node] = 2

    for node in sorted(parents):
        visit(node)


def _ancestor_closure(node: str, parents: dict[str, set[str]]) -> tuple[str, ...]:
    found: set[str] = set()
    stack = list(parents.get(node, ()))
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        stack.extend(parents.get(current, ()))
    return tuple(sorted(found))


def _record(
    database: Literal["go", "reactome", "interpro", "corum"],
    concept_id: str,
    concept_name: str,
    concept_type: str,
    proteins: set[str],
    originals: set[str],
    parents: set[str] | tuple[str, ...],
    evidence_tier: str,
    release: str,
    checksum: str,
    membership_tracks: dict[str, set[str]] | None = None,
    redundancy_cluster_id: str | None = None,
) -> ConceptRecord:
    if not concept_name.strip():
        raise ConceptParseError(f"Missing concept name for {concept_id}")
    return ConceptRecord(
        database=database,
        concept_id=concept_id,
        concept_name=concept_name.strip(),
        concept_type=concept_type,
        protein_accessions=tuple(sorted(proteins)),
        original_accessions=tuple(sorted(originals)),
        parent_ids=tuple(sorted(parents)),
        evidence_tier=evidence_tier,
        source_release=release,
        source_sha256=checksum,
        membership_tracks=tuple(
            (track, tuple(sorted(values)))
            for track, values in sorted(
                (membership_tracks or {"default": proteins}).items()
            )
        ),
        redundancy_cluster_id=redundancy_cluster_id,
    )


def _parse_obo(
    path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, str], dict[str, set[str]], int]:
    try:
        with _open_text(path) as handle:
            text = handle.read()
    except (OSError, UnicodeError) as error:
        raise ConceptParseError(f"Cannot read OBO: {path}") from error
    if "format-version:" not in text.split("[Term]", 1)[0]:
        raise ConceptParseError("Malformed OBO header")
    terms: dict[str, dict[str, object]] = {}
    aliases: dict[str, str] = {}
    for raw_block in re.split(r"(?m)^\[Term\]\s*$", text)[1:]:
        block = re.split(r"(?m)^\[[^\]]+\]\s*$", raw_block, maxsplit=1)[0]
        fields: dict[str, list[str]] = defaultdict(list)
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("!") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()].append(value.strip())
        ids = fields.get("id", [])
        if len(ids) != 1 or _GO_ID.fullmatch(ids[0]) is None:
            raise ConceptParseError("OBO term has malformed or duplicate id")
        term_id = ids[0]
        if term_id in terms:
            raise ConceptParseError(f"Duplicate OBO term: {term_id}")
        names = fields.get("name", [])
        if len(names) != 1 or not names[0]:
            raise ConceptParseError(f"Missing OBO term name: {term_id}")
        namespaces = fields.get("namespace", [])
        if len(namespaces) != 1 or not namespaces[0]:
            raise ConceptParseError(f"Missing OBO namespace: {term_id}")
        obsolete = fields.get("is_obsolete", []) == ["true"]
        replaced = tuple(value.split()[0] for value in fields.get("replaced_by", []))
        safe: set[str] = set()
        for value in fields.get("is_a", []):
            safe.add(value.split()[0])
        for value in fields.get("relationship", []):
            parts = value.split()
            if len(parts) >= 2 and parts[0] == "part_of":
                safe.add(parts[1])
        terms[term_id] = {
            "name": names[0], "namespace": namespaces[0], "obsolete": obsolete,
            "replaced_by": replaced, "parents": safe,
        }
        for alt_id in fields.get("alt_id", []):
            if (_GO_ID.fullmatch(alt_id) is None or alt_id in aliases
                    or alt_id == term_id):
                raise ConceptParseError(f"Invalid duplicate OBO alt_id: {alt_id}")
            aliases[alt_id] = term_id
    conflict = sorted(set(aliases) & set(terms))
    if conflict:
        raise ConceptParseError(f"OBO alt_id conflicts with canonical id: {conflict[0]}")

    alias_resolutions = 0

    def canonical_go_id(value: str) -> str:
        nonlocal alias_resolutions
        if value in aliases:
            alias_resolutions += 1
            return aliases[value]
        return value

    parents: dict[str, set[str]] = {}
    for term_id, term in terms.items():
        safe = {canonical_go_id(value) for value in term["parents"]}  # type: ignore[union-attr]
        missing = sorted(parent for parent in safe if parent not in terms)
        if missing:
            raise ConceptParseError(f"Unknown GO parent {missing[0]} for {term_id}")
        if any(bool(terms[parent]["obsolete"]) for parent in safe):
            raise ConceptParseError(f"GO safe parent is obsolete for {term_id}")
        parents[term_id] = safe
        replacements = tuple(
            canonical_go_id(value) for value in term["replaced_by"]  # type: ignore[union-attr]
        )
        if bool(term["obsolete"]):
            if len(replacements) > 1:
                raise ConceptParseError(f"Ambiguous obsolete replacement for {term_id}")
            if replacements:
                replacement_id = replacements[0]
                if replacement_id not in terms:
                    raise ConceptParseError(f"Unknown obsolete replacement for {term_id}")
                if bool(terms[replacement_id]["obsolete"]):
                    raise ConceptParseError(f"Obsolete replacement chain for {term_id}")
        term["replaced_by"] = replacements
    _require_acyclic(parents, "GO")
    return terms, aliases, parents, alias_resolutions


def parse_go(
    gaf_path: Path,
    obo_path: Path,
    universe: frozenset[str],
    release: str,
    source_sha256: str,
) -> tuple[dict[str, ConceptRecord], dict[str, int]]:
    _validate_common(universe, release, source_sha256)
    terms, aliases, parents, metadata_alias_resolutions = _parse_obo(obo_path)
    secondary: dict[str, set[str]] = defaultdict(set)
    originals: dict[str, set[str]] = defaultdict(set)
    primary: dict[str, set[str]] = defaultdict(set)
    audit = {
        "accepted_rows": 0, "alias_resolutions": metadata_alias_resolutions,
        "concepts": 0, "duplicate_rows": 0, "filtered_not_rows": 0,
        "input_rows": 0, "lost_invalid_accession_rows": 0,
        "lost_obsolete_unresolved_rows": 0, "lost_unknown_term_rows": 0,
        "lost_unmapped_accession_rows": 0, "replacement_rows": 0,
        "secondary_evidence_rows": 0,
    }
    seen: set[tuple[str, str, str]] = set()
    version_seen = False
    try:
        with _open_text(gaf_path) as handle:
            for line_number, raw in enumerate(handle, 1):
                if raw.startswith("!"):
                    if raw.strip().lower() == "!gaf-version: 2.2":
                        version_seen = True
                    continue
                if not version_seen:
                    raise ConceptParseError("GAF 2.2 header is missing")
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) != 17:
                    raise ConceptParseError(f"Malformed GAF row {line_number}: expected 17 columns")
                audit["input_rows"] += 1
                if "NOT" in {item.strip() for item in fields[3].split("|")}:
                    audit["filtered_not_rows"] += 1
                    continue
                raw_accession, term_id, evidence = fields[1].strip(), fields[4].strip(), fields[6].strip()
                try:
                    accession = canonical_uniprot_accession(raw_accession)
                except ValueError:
                    audit["lost_invalid_accession_rows"] += 1
                    continue
                if accession not in universe:
                    audit["lost_unmapped_accession_rows"] += 1
                    continue
                if term_id in aliases:
                    term_id = aliases[term_id]
                    audit["alias_resolutions"] += 1
                term = terms.get(term_id)
                if term is None:
                    audit["lost_unknown_term_rows"] += 1
                    continue
                if bool(term["obsolete"]):
                    replacements = tuple(term["replaced_by"])  # type: ignore[arg-type]
                    if len(replacements) != 1:
                        audit["lost_obsolete_unresolved_rows"] += 1
                        continue
                    term_id = replacements[0]
                    term = terms[term_id]
                    audit["replacement_rows"] += 1
                identity = (accession, term_id, evidence)
                destinations = (term_id, *_ancestor_closure(term_id, parents))
                for destination in destinations:
                    originals[destination].add(raw_accession)
                if identity in seen:
                    audit["duplicate_rows"] += 1
                    continue
                seen.add(identity)
                audit["accepted_rows"] += 1
                is_primary = evidence in _PRIMARY_GO_CODES
                if not is_primary:
                    audit["secondary_evidence_rows"] += 1
                for destination in destinations:
                    secondary[destination].add(accession)
                    if is_primary:
                        primary[destination].add(accession)
            if not version_seen:
                raise ConceptParseError("GAF 2.2 header is missing")
    except (OSError, UnicodeError, EOFError) as error:
        raise ConceptParseError(f"Cannot read GAF: {gaf_path}") from error
    records: dict[str, ConceptRecord] = {}
    for term_id in sorted(secondary):
        term = terms[term_id]
        records[term_id] = _record(
            "go", term_id, str(term["name"]), str(term["namespace"]),
            primary[term_id], originals[term_id], parents.get(term_id, set()),
            "primary" if primary[term_id] else "secondary", release, source_sha256,
            {"primary": primary[term_id], "secondary": secondary[term_id]},
        )
    audit["concepts"] = len(records)
    return records, audit


def _reactome_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        with _open_text(path) as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip() or raw.startswith("#"):
                    continue
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) < 6:
                    raise ConceptParseError(f"Malformed Reactome mapping row {line_number}: {path}")
                rows.append(fields)
    except (OSError, UnicodeError) as error:
        raise ConceptParseError(f"Cannot read Reactome mapping: {path}") from error
    return rows


def parse_reactome(
    raw_dir: Path,
    universe: frozenset[str],
    release: str,
    source_sha256: str,
) -> tuple[dict[str, ConceptRecord], dict[str, int]]:
    _validate_common(universe, release, source_sha256)
    if not isinstance(raw_dir, Path) or not raw_dir.is_dir():
        raise ConceptParseError(f"Reactome raw directory does not exist: {raw_dir}")
    metadata: dict[str, str] = {}
    human_ids: set[str] = set()
    with _open_text(raw_dir / "ReactomePathways.txt") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ConceptParseError(f"Malformed Reactome pathway row {line_number}")
            pathway_id, name, species = (field.strip() for field in fields)
            if not pathway_id or not name:
                raise ConceptParseError(f"Missing Reactome pathway name at row {line_number}")
            if species.casefold() in {"homo sapiens", "human"}:
                if pathway_id in metadata and metadata[pathway_id] != name:
                    raise ConceptParseError(f"Conflicting Reactome name: {pathway_id}")
                metadata[pathway_id] = name
                human_ids.add(pathway_id)
    parents: dict[str, set[str]] = defaultdict(set)
    with _open_text(raw_dir / "ReactomePathwaysRelation.txt") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ConceptParseError(f"Malformed Reactome relation row {line_number}")
            parent, child = fields
            if parent in human_ids and child in human_ids:
                parents[child].add(parent)
    _require_acyclic(parents, "Reactome")
    audit = {
        "accepted_rows": 0, "concepts": 0, "duplicate_rows": 0,
        "filtered_non_human_rows": 0, "input_rows": 0,
        "lost_invalid_accession_rows": 0, "lost_unknown_pathway_rows": 0,
        "lost_unmapped_accession_rows": 0,
    }
    track_data: dict[str, dict[str, set[str]]] = {}
    track_originals: dict[str, dict[str, set[str]]] = {}
    for track, filename in (
        ("leaf", "UniProt2Reactome.txt"),
        ("all", "UniProt2Reactome_All_Levels.txt"),
    ):
        mapped: dict[str, set[str]] = defaultdict(set)
        original: dict[str, set[str]] = defaultdict(set)
        seen: set[tuple[str, str]] = set()
        for fields in _reactome_rows(raw_dir / filename):
            audit["input_rows"] += 1
            raw_accession, pathway_id = fields[0].strip(), fields[1].strip()
            species = fields[5].strip().casefold()
            if species not in {"homo sapiens", "human"}:
                audit["filtered_non_human_rows"] += 1
                continue
            if pathway_id not in metadata:
                audit["lost_unknown_pathway_rows"] += 1
                continue
            try:
                accession = canonical_uniprot_accession(raw_accession)
            except ValueError:
                audit["lost_invalid_accession_rows"] += 1
                continue
            if accession not in universe:
                audit["lost_unmapped_accession_rows"] += 1
                continue
            identity = (accession, pathway_id)
            original[pathway_id].add(raw_accession)
            if identity in seen:
                audit["duplicate_rows"] += 1
                continue
            seen.add(identity)
            audit["accepted_rows"] += 1
            mapped[pathway_id].add(accession)
        track_data[track] = mapped
        track_originals[track] = original
    records: dict[str, ConceptRecord] = {}
    concept_ids = sorted(set(track_data["leaf"]) | set(track_data["all"]))
    for pathway_id in concept_ids:
        leaf = track_data["leaf"].get(pathway_id, set())
        all_level = track_data["all"].get(pathway_id, set())
        records[pathway_id] = _record(
            "reactome", pathway_id, metadata[pathway_id], "pathway",
            leaf if leaf else all_level,
            track_originals["leaf"].get(pathway_id, set())
            | track_originals["all"].get(pathway_id, set()),
            {parent for parent in parents.get(pathway_id, ()) if parent in concept_ids},
            "official", release, source_sha256,
            {"all": all_level, "leaf": leaf},
        )
    audit["concepts"] = len(records)
    return records, audit


def parse_interpro(
    protein2ipr_path: Path,
    entry_list_path: Path,
    universe: frozenset[str],
    release: str,
    source_sha256: str,
) -> tuple[dict[str, ConceptRecord], dict[str, int]]:
    _validate_common(universe, release, source_sha256)
    supported = {
        "Family": "family",
        "Domain": "domain",
        "Repeat": "repeat",
        "Active_site": "site",
        "Binding_site": "site",
        "Conserved_site": "site",
        "FAMILY": "family",
        "DOMAIN": "domain",
        "REPEAT": "repeat",
        "SITE": "site",
    }
    entries: dict[str, tuple[str, str]] = {}
    unsupported: set[str] = set()
    with _open_text(entry_list_path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ConceptParseError("InterPro entry header is missing") from error
        if header[:3] != ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"]:
            raise ConceptParseError("Malformed InterPro entry header")
        for line_number, fields in enumerate(reader, 2):
            if not fields or not any(fields):
                continue
            if len(fields) < 3:
                raise ConceptParseError(f"Malformed InterPro entry row {line_number}")
            entry_id, entry_type, name = (field.strip() for field in fields[:3])
            if _INTERPRO_ID.fullmatch(entry_id) is None:
                raise ConceptParseError(f"Invalid InterPro id at row {line_number}")
            if not name:
                raise ConceptParseError(f"Missing InterPro entry name: {entry_id}")
            if entry_id in entries or entry_id in unsupported:
                raise ConceptParseError(f"Duplicate InterPro entry: {entry_id}")
            if entry_type not in supported:
                unsupported.add(entry_id)
            else:
                entries[entry_id] = (supported[entry_type], name)
    audit = {
        "accepted_rows": 0, "concepts": 0, "duplicate_rows": 0,
        "filtered_unsupported_type_rows": 0, "input_rows": 0,
        "lost_invalid_accession_rows": 0, "lost_unknown_entry_rows": 0,
        "lost_unmapped_accession_rows": 0,
        "unsupported_entry_definitions": len(unsupported),
    }
    proteins: dict[str, set[str]] = defaultdict(set)
    originals: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    with _open_text(protein2ipr_path) as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                raise ConceptParseError(f"Malformed protein2ipr row {line_number}")
            audit["input_rows"] += 1
            raw_accession, entry_id = fields[0].strip(), fields[1].strip()
            if entry_id in unsupported:
                audit["filtered_unsupported_type_rows"] += 1
                continue
            if entry_id not in entries:
                audit["lost_unknown_entry_rows"] += 1
                continue
            try:
                accession = canonical_uniprot_accession(raw_accession)
            except ValueError:
                audit["lost_invalid_accession_rows"] += 1
                continue
            if accession not in universe:
                audit["lost_unmapped_accession_rows"] += 1
                continue
            identity = (accession, entry_id)
            originals[entry_id].add(raw_accession)
            if identity in seen:
                audit["duplicate_rows"] += 1
                continue
            seen.add(identity)
            audit["accepted_rows"] += 1
            proteins[entry_id].add(accession)
    records = {
        entry_id: _record(
            "interpro", entry_id, entries[entry_id][1], entries[entry_id][0],
            proteins[entry_id], originals[entry_id], set(), "official", release,
            source_sha256,
        )
        for entry_id in sorted(proteins)
    }
    audit["concepts"] = len(records)
    return records, audit


def _corum_member_name(names: list[str]) -> str:
    candidates = [
        name for name in names
        if name.lower().endswith(".txt")
        and "complex" in Path(name).name.lower()
        and "readme" not in Path(name).name.lower()
    ]
    if len(candidates) != 1:
        raise ConceptParseError("CORUM archive must contain exactly one complex table")
    return candidates[0]


def _corum_columns(header: list[str]) -> dict[str, str]:
    aliases = {
        "complex_id": {"complexid", "complex_id"},
        "complex_name": {"complexname", "complex_name"},
        "organism": {"organism"},
        "subunits": {"subunits(uniprot ids)", "subunits_uniprot_id"},
    }
    columns: dict[str, str] = {}
    for semantic_name, spellings in aliases.items():
        matches = [
            name for name in header
            if type(name) is str and name.strip().casefold() in spellings
        ]
        if len(matches) != 1:
            raise ConceptParseError("Malformed CORUM table header")
        columns[semantic_name] = matches[0]
    return columns


def parse_corum(
    corum_path: Path,
    universe: frozenset[str],
    release: str,
    source_sha256: str,
) -> tuple[dict[str, ConceptRecord], dict[str, int]]:
    _validate_common(universe, release, source_sha256)
    audit = {
        "accepted_human_complex_rows": 0, "accepted_member_tokens": 0,
        "concepts": 0, "duplicate_complex_rows": 0,
        "duplicate_member_tokens": 0, "filtered_non_human_complex_rows": 0,
        "filtered_out_of_universe_members": 0, "input_complex_rows": 0,
        "input_member_tokens": 0, "isoform_collapses": 0,
        "lost_invalid_accession_members": 0, "partial_protein_collapses": 0,
        "redundancy_components": 0,
    }
    names: dict[str, str] = {}
    proteins: dict[str, set[str]] = defaultdict(set)
    full_members: dict[str, set[str]] = defaultdict(set)
    originals: dict[str, set[str]] = defaultdict(set)
    seen_members: dict[str, set[str]] = defaultdict(set)
    try:
        with ExitStack() as stack:
            if corum_path.suffix.casefold() == ".zip" or zipfile.is_zipfile(corum_path):
                archive = stack.enter_context(zipfile.ZipFile(corum_path))
                member = _corum_member_name(archive.namelist())
                raw = stack.enter_context(archive.open(member))
                text = stack.enter_context(
                    io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                )
            else:
                text = stack.enter_context(
                    corum_path.open("rt", encoding="utf-8-sig", newline="")
                )
            reader = csv.DictReader(text, delimiter="\t")
            columns = _corum_columns(reader.fieldnames or [])
            for line_number, row in enumerate(reader, 2):
                audit["input_complex_rows"] += 1
                organism = (row.get(columns["organism"]) or "").strip().casefold()
                if organism not in {"human", "homo sapiens"}:
                    audit["filtered_non_human_complex_rows"] += 1
                    continue
                raw_id = (row.get(columns["complex_id"]) or "").strip()
                name = (row.get(columns["complex_name"]) or "").strip()
                if not raw_id or not name:
                    raise ConceptParseError(f"Missing CORUM id or name at row {line_number}")
                concept_id = f"CORUM:{raw_id}"
                if concept_id in names:
                    if names[concept_id] != name:
                        raise ConceptParseError(f"Conflicting CORUM name: {concept_id}")
                    audit["duplicate_complex_rows"] += 1
                else:
                    audit["accepted_human_complex_rows"] += 1
                names[concept_id] = name
                raw_members = (row.get(columns["subunits"]) or "").strip()
                for raw_accession in (item.strip() for item in re.split(r"[;,]", raw_members)):
                    if not raw_accession:
                        continue
                    audit["input_member_tokens"] += 1
                    try:
                        accession = canonical_uniprot_accession(raw_accession)
                    except ValueError:
                        audit["lost_invalid_accession_members"] += 1
                        continue
                    audit["accepted_member_tokens"] += 1
                    if _PARTIAL.fullmatch(raw_accession):
                        audit["partial_protein_collapses"] += 1
                    elif _ISOFORM.fullmatch(raw_accession):
                        audit["isoform_collapses"] += 1
                    originals[concept_id].add(raw_accession)
                    if accession in seen_members[concept_id]:
                        audit["duplicate_member_tokens"] += 1
                    seen_members[concept_id].add(accession)
                    full_members[concept_id].add(accession)
                    if accession not in universe:
                        audit["filtered_out_of_universe_members"] += 1
                    else:
                        proteins[concept_id].add(accession)
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as error:
        raise ConceptParseError(f"Cannot read CORUM snapshot: {corum_path}") from error
    source_concept_ids = sorted(concept_id for concept_id in names if full_members[concept_id])
    concept_ids = sorted(concept_id for concept_id in source_concept_ids if proteins[concept_id])
    parent = {concept_id: concept_id for concept_id in source_concept_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for left_index, left in enumerate(source_concept_ids):
        for right in source_concept_ids[left_index + 1:]:
            intersection = len(full_members[left] & full_members[right])
            union_size = len(full_members[left] | full_members[right])
            if union_size and intersection / union_size >= 0.50:
                union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for concept_id in source_concept_ids:
        components[find(concept_id)].append(concept_id)
    cluster_ids: dict[str, str] = {}
    for members in sorted(components.values()):
        if len(members) < 2:
            continue
        representative = min(members, key=lambda value: (int(value.split(":", 1)[1])
                                                           if value.split(":", 1)[1].isdigit()
                                                           else value))
        cluster_id = f"CORUM:cluster:{representative.split(':', 1)[1]}"
        audit["redundancy_components"] += 1
        for concept_id in members:
            cluster_ids[concept_id] = cluster_id
    records = {
        concept_id: _record(
            "corum", concept_id, names[concept_id], "protein_complex",
            proteins[concept_id], originals[concept_id], set(),
            "official", release, source_sha256, None, cluster_ids.get(concept_id),
        )
        for concept_id in concept_ids
    }
    audit["concepts"] = len(records)
    return records, audit


def _database_provenance(
    manifest: dict[str, object], database: str
) -> tuple[str, str, dict[str, dict[str, object]]]:
    files = manifest.get("files")
    if type(files) is not list:
        raise ConceptParseError("Snapshot manifest files are invalid")
    selected = [row for row in files if type(row) is dict and row.get("database") == database]
    if not selected:
        raise ConceptParseError(f"Snapshot database is missing: {database}")
    by_name = {str(row["name"]): row for row in selected}
    releases_by_name = {name: row.get("release") for name, row in by_name.items()}
    if any(type(value) is not str or not value for value in releases_by_name.values()):
        raise ConceptParseError(f"Snapshot release is invalid: {database}")
    releases = set(releases_by_name.values())
    release = (
        str(next(iter(releases)))
        if len(releases) == 1
        else "files:" + json.dumps(
            releases_by_name, sort_keys=True, separators=(",", ":")
        )
    )
    checksums = sorted(str(row["sha256"]) for row in selected)
    checksum = checksums[0] if len(checksums) == 1 else hashlib.sha256(
        json.dumps(checksums, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return release, checksum, by_name


def _raw_path(snapshot_dir: Path, row: dict[str, object]) -> Path:
    relative = row.get("relative_path")
    if type(relative) is not str:
        raise ConceptParseError("Snapshot relative path is invalid")
    return snapshot_dir / relative


def _read_universe(units_path: Path) -> frozenset[str]:
    accessions: set[str] = set()
    try:
        for line_number, row in enumerate(strict_json_lines(units_path), 1):
            accession = row.get("accession")
            if type(accession) is not str:
                raise ConceptParseError(f"Unit row {line_number} has invalid accession")
            canonical = canonical_uniprot_accession(accession)
            if canonical != accession:
                raise ConceptParseError(f"Unit row {line_number} accession is not canonical")
            accessions.add(accession)
    except (AtlasIOError, ValueError) as error:
        if isinstance(error, ConceptParseError):
            raise
        raise ConceptParseError(f"Invalid unit corpus: {error}") from error
    if not accessions:
        raise ConceptParseError("Unit corpus accession universe is empty")
    return frozenset(accessions)


def _all_parsers(
    snapshot_dir: Path, manifest: dict[str, object], universe: frozenset[str]
) -> tuple[list[ConceptRecord], dict[str, dict[str, int]], dict[str, tuple[str, str]]]:
    provenance: dict[str, tuple[str, str]] = {}
    audits: dict[str, dict[str, int]] = {}
    all_records: list[ConceptRecord] = []
    go_release, go_sha, go = _database_provenance(manifest, "go")
    provenance["go"] = (go_release, go_sha)
    records, audits["go"] = parse_go(
        _raw_path(snapshot_dir, go["human_gaf"]),
        _raw_path(snapshot_dir, go["go_basic"]), universe, go_release, go_sha)
    all_records.extend(records.values())
    reactome_release, reactome_sha, reactome = _database_provenance(manifest, "reactome")
    provenance["reactome"] = (reactome_release, reactome_sha)
    records, audits["reactome"] = parse_reactome(
        _raw_path(snapshot_dir, reactome["pathways"]).parent,
        universe, reactome_release, reactome_sha)
    all_records.extend(records.values())
    interpro_release, interpro_sha, interpro = _database_provenance(manifest, "interpro")
    provenance["interpro"] = (interpro_release, interpro_sha)
    records, audits["interpro"] = parse_interpro(
        _raw_path(snapshot_dir, interpro["protein2ipr"]),
        _raw_path(snapshot_dir, interpro["entry_list"]), universe,
        interpro_release, interpro_sha)
    all_records.extend(records.values())
    corum_release, corum_sha, corum = _database_provenance(manifest, "corum")
    provenance["corum"] = (corum_release, corum_sha)
    records, audits["corum"] = parse_corum(
        _raw_path(snapshot_dir, corum["all_complexes"]), universe,
        corum_release, corum_sha)
    all_records.extend(records.values())
    all_records.sort(key=lambda record: (record.database, record.concept_id))
    identities = [(record.database, record.concept_id) for record in all_records]
    if len(identities) != len(set(identities)):
        raise ConceptParseError("Duplicate normalized concept identity")
    return all_records, audits, provenance


def _audit_total(audits: dict[str, dict[str, int]], universe_size: int) -> dict[str, int]:
    return {
        "collapse_count": sum(
            value for audit in audits.values() for key, value in audit.items()
            if key.endswith("_collapses")
        ),
        "concepts": sum(audit["concepts"] for audit in audits.values()),
        "duplicate_count": sum(
            value for audit in audits.values() for key, value in audit.items()
            if key.startswith("duplicate_")
        ),
        "explicit_loss_count": sum(
            value for audit in audits.values() for key, value in audit.items()
            if key.startswith("lost_")
        ),
        "filter_count": sum(
            value for audit in audits.values() for key, value in audit.items()
            if key.startswith("filtered_")
        ),
        "replacement_count": sum(
            value for audit in audits.values() for key, value in audit.items()
            if key.endswith("replacement_rows")
        ),
        "universe_accessions": universe_size,
    }


def _validate_audits(
    audits: dict[str, dict[str, int]], total: dict[str, object], universe_size: int
) -> None:
    if set(audits) != set(_AUDIT_SCHEMAS):
        raise ConceptParseError("Parse audit database schema is invalid")
    for database, schema in _AUDIT_SCHEMAS.items():
        audit = audits[database]
        if set(audit) != schema:
            raise ConceptParseError(f"Parse audit schema is invalid: {database}")
        if any(type(value) is not int or value < 0 for value in audit.values()):
            raise ConceptParseError(f"Parse audit counters are invalid: {database}")
    go = audits["go"]
    if go["input_rows"] != sum(go[key] for key in (
        "accepted_rows", "duplicate_rows", "filtered_not_rows",
        "lost_invalid_accession_rows", "lost_obsolete_unresolved_rows",
        "lost_unknown_term_rows", "lost_unmapped_accession_rows",
    )):
        raise ConceptParseError("GO audit terminal reconciliation failed")
    reactome = audits["reactome"]
    if reactome["input_rows"] != sum(reactome[key] for key in (
        "accepted_rows", "duplicate_rows", "filtered_non_human_rows",
        "lost_invalid_accession_rows", "lost_unknown_pathway_rows",
        "lost_unmapped_accession_rows",
    )):
        raise ConceptParseError("Reactome audit terminal reconciliation failed")
    interpro = audits["interpro"]
    if interpro["input_rows"] != sum(interpro[key] for key in (
        "accepted_rows", "duplicate_rows", "filtered_unsupported_type_rows",
        "lost_invalid_accession_rows", "lost_unknown_entry_rows",
        "lost_unmapped_accession_rows",
    )):
        raise ConceptParseError("InterPro audit terminal reconciliation failed")
    corum = audits["corum"]
    if corum["input_complex_rows"] != sum(corum[key] for key in (
        "accepted_human_complex_rows", "duplicate_complex_rows",
        "filtered_non_human_complex_rows",
    )):
        raise ConceptParseError("CORUM complex audit terminal reconciliation failed")
    if corum["input_member_tokens"] != (
        corum["accepted_member_tokens"] + corum["lost_invalid_accession_members"]
    ):
        raise ConceptParseError("CORUM member audit terminal reconciliation failed")
    expected_total = _audit_total(audits, universe_size)
    if total != expected_total:
        raise ConceptParseError("Parse audit total reconciliation failed")


def _publish_transaction(
    stage: Path,
    output_dir: Path,
    validate_committed: object,
) -> dict[str, object]:
    backup = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.concept-backup-", dir=output_dir.parent
    ))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            destination = output_dir / name
            if destination.exists():
                os.replace(destination, backup / name)
        for name in _OUTPUT_FILES:
            os.replace(stage / name, output_dir / name)
        validated = validate_committed()  # type: ignore[operator]
    except BaseException as error:
        rollback_errors: list[str] = []

        def recover_move(source: Path, destination: Path, label: str) -> None:
            try:
                os.replace(source, destination)
            except BaseException as rollback_error:
                # A rename can complete before an asynchronous exception reaches
                # Python. Filesystem state, not control-flow bookkeeping, is the
                # authoritative journal.
                if source.exists() or not destination.exists():
                    rollback_errors.append(f"{label}: {rollback_error}")

        for name in _OUTPUT_FILES:
            destination = output_dir / name
            staged = stage / name
            saved = backup / name
            had_saved_old = saved.exists()
            if had_saved_old:
                if destination.exists():
                    if staged.exists():
                        rollback_errors.append(
                            f"ambiguous new-output state for {name}"
                        )
                    else:
                        recover_move(
                            destination, staged, f"cannot recover new {name}"
                        )
                if saved.exists() and not destination.exists():
                    recover_move(saved, destination, f"cannot restore old {name}")
                if saved.exists() or not destination.exists():
                    rollback_errors.append(f"old-output restoration incomplete for {name}")
            else:
                # Either this was a fresh output, or the old->backup rename did
                # not occur. output+stage means the untouched old output and the
                # staged replacement; output without stage means a fresh rename
                # completed and must be moved back.
                if destination.exists() and not staged.exists():
                    recover_move(
                        destination, staged, f"cannot remove fresh output {name}"
                    )
                if destination.exists() and not staged.exists():
                    rollback_errors.append(f"fresh-output recovery incomplete for {name}")
        if rollback_errors:
            raise ConceptParseError(
                "Concept publication recovery required; "
                f"backup recovery path: {backup}; staging recovery path: {stage}; "
                + "; ".join(rollback_errors)
            ) from error
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise ConceptParseError(f"Concept publication failed: {error}") from error
    shutil.rmtree(stage, ignore_errors=True)
    try:
        shutil.rmtree(backup)
    except OSError as error:
        raise ConceptParseError(
            "Concept stage committed but backup retirement failed; "
            f"retained backup path: {backup}"
        ) from error
    return validated


def build_concept_index(
    snapshot_dir: Path,
    units_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    if not all(isinstance(path, Path) for path in (snapshot_dir, units_path, output_dir)):
        raise TypeError("snapshot_dir, units_path, and output_dir must be pathlib.Path values")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a built-in bool")
    try:
        _reject_symlink_components(output_dir)
    except AtlasIOError as error:
        raise ConceptParseError(str(error)) from error
    try:
        snapshot_manifest = validate_external_snapshots(snapshot_dir)
    except (SnapshotError, AtlasIOError) as error:
        raise ConceptParseError(f"Invalid external snapshot: {error}") from error
    universe = _read_universe(units_path)
    records, audits, provenance = _all_parsers(snapshot_dir, snapshot_manifest, universe)
    if not records:
        raise ConceptParseError("Normalized concept index is empty")
    existing = [output_dir / name for name in _OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise ConceptParseError("Concept outputs exist; use overwrite=True")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.concept-stage-", dir=output_dir.parent))
    try:
        index_path = stage / "concept_index.jsonl"
        audit_path = stage / "parse_audit.json"
        write_jsonl_atomic(index_path, (asdict(record) for record in records))
        total = _audit_total(audits, len(universe))
        _validate_audits(audits, total, len(universe))
        write_json_atomic(audit_path, {**audits, "total": total})
        manifest: dict[str, object] = {
            "databases": {
                database: {"release": release, "source_sha256": checksum}
                for database, (release, checksum) in sorted(provenance.items())
            },
            "files": {
                "concept_index": file_metadata(index_path),
                "parse_audit": file_metadata(audit_path),
            },
            "schema_version": 1,
            "snapshot_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
            "units_sha256": sha256_file(units_path),
        }
        # Staging paths are not provenance; publish stable output paths in the manifest.
        manifest["files"] = {
            "concept_index": {
                **manifest["files"]["concept_index"],  # type: ignore[index]
                "path": str((output_dir / "concept_index.jsonl").resolve()),
            },
            "parse_audit": {
                **manifest["files"]["parse_audit"],  # type: ignore[index]
                "path": str((output_dir / "parse_audit.json").resolve()),
            },
        }
        write_json_atomic(stage / "manifest.json", manifest)
    except (OSError, AtlasIOError, ConceptParseError) as error:
        shutil.rmtree(stage, ignore_errors=True)
        if isinstance(error, ConceptParseError):
            raise
        raise ConceptParseError(f"Cannot publish concept index: {error}") from error
    return _publish_transaction(
        stage, output_dir,
        lambda: validate_concept_index(snapshot_dir, units_path, output_dir),
    )


def validate_concept_index(
    snapshot_dir: Path, units_path: Path, output_dir: Path
) -> dict[str, object]:
    if not all(isinstance(path, Path) for path in (snapshot_dir, units_path, output_dir)):
        raise TypeError("snapshot_dir, units_path, and output_dir must be pathlib.Path values")
    try:
        _reject_symlink_components(output_dir)
    except AtlasIOError as error:
        raise ConceptParseError(str(error)) from error
    try:
        snapshot_manifest = validate_external_snapshots(snapshot_dir)
        manifest = strict_json_load(output_dir / "manifest.json")
        audit = strict_json_load(output_dir / "parse_audit.json")
    except (SnapshotError, AtlasIOError) as error:
        raise ConceptParseError(f"Invalid concept manifest or snapshot: {error}") from error
    if manifest.get("schema_version") != 1:
        raise ConceptParseError("Concept manifest schema_version must be 1")
    if manifest.get("snapshot_manifest_sha256") != sha256_file(snapshot_dir / "manifest.json"):
        raise ConceptParseError("Snapshot manifest changed")
    if manifest.get("units_sha256") != sha256_file(units_path):
        raise ConceptParseError("Unit corpus changed")
    files = manifest.get("files")
    if type(files) is not dict:
        raise ConceptParseError("Concept manifest files are invalid")
    for key, filename in (("concept_index", "concept_index.jsonl"),
                          ("parse_audit", "parse_audit.json")):
        metadata = files.get(key)
        path = output_dir / filename
        if type(metadata) is not dict or metadata.get("path") != str(path.resolve()):
            raise ConceptParseError(f"Invalid registered output metadata: {key}")
        if metadata.get("sha256") != sha256_file(path) or metadata.get("bytes") != path.stat().st_size:
            raise ConceptParseError(f"Registered concept output checksum changed: {key}")
    universe = _read_universe(units_path)
    databases = manifest.get("databases")
    if type(databases) is not dict or set(databases) != {"go", "reactome", "interpro", "corum"}:
        raise ConceptParseError("Concept database provenance is incomplete")
    for database in databases:
        release, checksum, _ = _database_provenance(snapshot_manifest, database)
        if databases[database] != {"release": release, "source_sha256": checksum}:
            raise ConceptParseError(f"Concept database provenance changed: {database}")
    identities: list[tuple[str, str]] = []
    parsed_rows: list[dict[str, object]] = []
    parent_references: list[tuple[str, str]] = []
    try:
        rows = strict_json_lines(output_dir / "concept_index.jsonl")
        for row_number, row in enumerate(rows, 1):
            required = {field.name for field in fields(ConceptRecord)}
            if set(row) != required:
                raise ConceptParseError(f"Concept row {row_number} schema is invalid")
            database, concept_id = row["database"], row["concept_id"]
            if type(database) is not str or database not in databases or type(concept_id) is not str:
                raise ConceptParseError(f"Concept row {row_number} identity is invalid")
            identity = (database, concept_id)
            identities.append(identity)
            parsed_rows.append(row)
            for field in ("concept_name", "concept_type", "evidence_tier",
                          "source_release", "source_sha256"):
                if type(row[field]) is not str or not row[field]:
                    raise ConceptParseError(f"Concept row {row_number} has invalid {field}")
            for field in ("protein_accessions", "original_accessions", "parent_ids"):
                values = row[field]
                if type(values) is not list or any(type(value) is not str for value in values):
                    raise ConceptParseError(f"Concept row {row_number} has invalid {field}")
                if values != sorted(set(values)):
                    raise ConceptParseError(f"Concept row {row_number} {field} is not sorted unique")
            if not set(row["protein_accessions"]).issubset(universe):
                raise ConceptParseError(f"Concept row {row_number} proteins violate universe")
            tracks = row["membership_tracks"]
            if type(tracks) is not list:
                raise ConceptParseError(f"Concept row {row_number} tracks are invalid")
            track_map: dict[str, list[str]] = {}
            for pair in tracks:
                if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                        or type(pair[1]) is not list
                        or any(type(value) is not str for value in pair[1])
                        or pair[1] != sorted(set(pair[1]))
                        or not set(pair[1]).issubset(universe)):
                    raise ConceptParseError(f"Concept row {row_number} tracks are invalid")
                track_map[pair[0]] = pair[1]
            if list(track_map) != sorted(track_map) or len(track_map) != len(tracks):
                raise ConceptParseError(f"Concept row {row_number} tracks are not sorted unique")
            redundancy = row["redundancy_cluster_id"]
            if database == "go":
                if (set(track_map) != {"primary", "secondary"}
                        or row["protein_accessions"] != track_map["primary"]
                        or not set(track_map["primary"]).issubset(track_map["secondary"])
                        or not track_map["secondary"] or redundancy is not None):
                    raise ConceptParseError(f"Concept row {row_number} GO track contract failed")
            elif database == "reactome":
                default = track_map.get("leaf") or track_map.get("all")
                if (set(track_map) != {"all", "leaf"} or not default
                        or row["protein_accessions"] != default
                        or redundancy is not None or "@" in concept_id):
                    raise ConceptParseError(f"Concept row {row_number} Reactome track contract failed")
            else:
                if (set(track_map) != {"default"}
                        or row["protein_accessions"] != track_map["default"]
                        or not track_map["default"]):
                    raise ConceptParseError(f"Concept row {row_number} default track contract failed")
                if database == "interpro" and redundancy is not None:
                    raise ConceptParseError(f"Concept row {row_number} InterPro redundancy is invalid")
                if database == "corum" and redundancy is not None and (
                    type(redundancy) is not str or not redundancy.startswith("CORUM:cluster:")
                ):
                    raise ConceptParseError(f"Concept row {row_number} CORUM redundancy is invalid")
            parent_references.extend((database, parent) for parent in row["parent_ids"])
            expected = databases[database]
            if (row["source_release"] != expected["release"] or
                    row["source_sha256"] != expected["source_sha256"]):
                raise ConceptParseError(f"Concept row {row_number} provenance mismatch")
    except AtlasIOError as error:
        raise ConceptParseError(f"Invalid concept index: {error}") from error
    if identities != sorted(set(identities)):
        raise ConceptParseError("Concept identities are not sorted and unique")
    if any(reference not in set(identities) for reference in parent_references):
        raise ConceptParseError("Concept index contains dangling parent reference")
    total = audit.get("total")
    parser_audits = {database: audit.get(database) for database in _AUDIT_SCHEMAS}
    if type(total) is not dict or any(type(value) is not dict for value in parser_audits.values()):
        raise ConceptParseError("Parse audit structure is invalid")
    if set(audit) != {*_AUDIT_SCHEMAS, "total"}:
        raise ConceptParseError("Parse audit top-level schema is invalid")
    _validate_audits(parser_audits, total, len(universe))  # type: ignore[arg-type]
    expected_records, expected_audits, _ = _all_parsers(
        snapshot_dir, snapshot_manifest, universe
    )
    expected_rows = [
        json.loads(json.dumps(asdict(record), sort_keys=True))
        for record in expected_records
    ]
    if parsed_rows != expected_rows:
        raise ConceptParseError("Concept index does not match recomputed normalization")
    if parser_audits != expected_audits:
        raise ConceptParseError("Parse audit does not match recomputed normalization")
    if total["concepts"] != len(identities):
        raise ConceptParseError("Parse audit concept total does not match index")
    return manifest
