"""Structured evidence validation and deterministic mechanism hypotheses."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from esmc_sae.calibrate import _write_json_create_only
from esmc_sae.manifest import file_sha256
from esmc_sae.mechanism_eval import evaluate_mechanism
from esmc_sae.pair_report import (
    _atomic_text_create_only,
    _verify_sha256_manifest,
)


_CLAIM_FIELDS = (
    "a_side_function",
    "b_side_function",
    "joint_vulnerability",
    "cell_context",
)
_EVIDENCE_KINDS = frozenset({"support", "counter", "context"})


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _feature_ids(value: object, label: str) -> list[int]:
    result: list[int] = []
    for item in _sequence(value, label):
        if type(item) is not int or not 0 <= item < 16384:
            raise ValueError(f"{label} contains an invalid feature")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate features")
    return result


def _evidence_ids(value: object, label: str, *, required: bool) -> list[str]:
    result = []
    for item in _sequence(value, label):
        result.append(_nonempty_text(item, label))
    if required and not result:
        raise ValueError(f"{label} requires at least one evidence ID")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate evidence IDs")
    return result


def _citation_is_traceable(citation: object) -> bool:
    if not isinstance(citation, Mapping):
        return False
    url = citation.get("url")
    pmid = citation.get("pmid")
    doi = citation.get("doi")
    return bool(
        (isinstance(url, str) and url.startswith(("https://", "http://")))
        or (isinstance(pmid, str) and pmid.strip())
        or (isinstance(doi, str) and doi.strip())
    )


def _validate_packet_structure(packet: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(packet, Mapping):
        raise ValueError("packet must be a mapping")
    if packet.get("schema_version") != 1:
        raise ValueError("packet schema_version must be 1")
    row_id = _nonempty_text(packet.get("row_id"), "row_id")
    gene_a = packet.get("gene_a")
    gene_b = packet.get("gene_b")
    if type(gene_a) is not int or type(gene_b) is not int:
        raise ValueError("packet gene IDs must be integers")
    model_features = _feature_ids(packet.get("model_features"), "model_features")
    if not model_features:
        raise ValueError("model_features must not be empty")
    model_feature_set = set(model_features)

    semantics = _sequence(packet.get("sae_semantics"), "sae_semantics")
    semantic_ids: list[int] = []
    for item in semantics:
        if not isinstance(item, Mapping):
            raise ValueError("sae_semantics item must be a mapping")
        feature_id = item.get("feature_id")
        if type(feature_id) is not int:
            raise ValueError("sae_semantics feature is invalid")
        semantic_ids.append(feature_id)
        _nonempty_text(item.get("description"), "sae_semantics description")
        source_url = _nonempty_text(
            item.get("source_url"), "sae_semantics source_url"
        )
        if not source_url.startswith(("https://", "http://")):
            raise ValueError("sae_semantics source_url must be traceable")
    if set(semantic_ids) != model_feature_set or len(semantic_ids) != len(
        model_features
    ):
        raise ValueError("sae_semantics must cover each model feature exactly")

    evidence_items = _sequence(packet.get("evidence"), "evidence")
    evidence_by_id: dict[str, Mapping[str, object]] = {}
    for item in evidence_items:
        if not isinstance(item, Mapping):
            raise ValueError("evidence item must be a mapping")
        evidence_id = _nonempty_text(item.get("evidence_id"), "evidence_id")
        if evidence_id in evidence_by_id:
            raise ValueError("evidence contains duplicate evidence IDs")
        if item.get("kind") not in _EVIDENCE_KINDS:
            raise ValueError("evidence kind is invalid")
        _nonempty_text(item.get("summary"), "evidence summary")
        if not _citation_is_traceable(item.get("citation")):
            raise ValueError("evidence citation lacks URL, PMID, or DOI")
        evidence_by_id[evidence_id] = item
    if not evidence_by_id:
        raise ValueError("evidence must not be empty")

    claim_features: set[int] = set()
    claims: dict[str, Mapping[str, object]] = {}
    for field in _CLAIM_FIELDS:
        claim = packet.get(field)
        if not isinstance(claim, Mapping):
            raise ValueError(f"{field} must be a mapping")
        _nonempty_text(claim.get("text"), f"{field} text")
        features = _feature_ids(claim.get("feature_ids"), f"{field} feature_ids")
        if not set(features) <= model_feature_set:
            raise ValueError(f"{field} references an unvalidated feature")
        claim_features.update(features)
        ids = _evidence_ids(
            claim.get("evidence_ids"),
            f"{field} evidence_ids",
            required=True,
        )
        if not set(ids) <= set(evidence_by_id):
            raise ValueError(f"{field} references unknown evidence")
        claims[field] = claim
    if claim_features != model_feature_set:
        raise ValueError("not every model feature is used by a mechanistic claim")

    counter = packet.get("counterevidence")
    if not isinstance(counter, Mapping):
        raise ValueError("counterevidence must be a mapping")
    _nonempty_text(counter.get("text"), "counterevidence text")
    counter_ids = _evidence_ids(
        counter.get("evidence_ids"),
        "counterevidence evidence_ids",
        required=False,
    )
    if not set(counter_ids) <= set(evidence_by_id):
        raise ValueError("counterevidence references unknown evidence")
    prediction = _nonempty_text(
        packet.get("testable_prediction"),
        "testable_prediction",
    )
    return {
        "row_id": row_id,
        "gene_a": gene_a,
        "gene_b": gene_b,
        "model_features": model_features,
        "evidence_by_id": evidence_by_id,
        "claims": claims,
        "counter": counter,
        "prediction": prediction,
    }


def _pair_features(pair_report: Mapping[str, object]) -> set[int]:
    mechanism_ids = set(
        _feature_ids(
            pair_report.get("mechanism_feature_ids"),
            "pair report mechanism_feature_ids",
        )
    )
    candidate_ids = set(
        _feature_ids(pair_report.get("feature_ids"), "pair report feature_ids")
    )
    if not mechanism_ids:
        raise ValueError("pair report has no mechanism features")
    if not mechanism_ids <= candidate_ids:
        raise ValueError("pair report mechanism features are not candidates")
    return mechanism_ids


def validate_evidence_packet(
    packet: Mapping[str, object],
    pair_report: Mapping[str, object],
) -> None:
    """Reject evidence that is not identity- and intervention-traceable."""

    parsed = _validate_packet_structure(packet)
    if not isinstance(pair_report, Mapping):
        raise ValueError("pair_report must be a mapping")
    if (
        pair_report.get("row_id") != parsed["row_id"]
        or pair_report.get("gene_a") != parsed["gene_a"]
        or pair_report.get("gene_b") != parsed["gene_b"]
    ):
        raise ValueError("packet row or gene identity does not match pair report")
    if (
        pair_report.get("prediction_type") != "TP"
        or pair_report.get("explanation_status") != "reliable"
        or pair_report.get("mechanism_eligible", True) is not True
    ):
        raise ValueError("pair report is not mechanism eligible")
    validated = _pair_features(pair_report)
    if set(parsed["model_features"]) != validated:
        raise ValueError("packet references a non-intervention-validated feature")


def _citation_text(citation: Mapping[str, object]) -> str:
    for key in ("url", "pmid", "doi"):
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key.upper()}:{value.strip()}"
    raise ValueError("citation is not traceable")


def render_mechanism(packet: Mapping[str, object]) -> str:
    """Render only validated structured fields into a deterministic report."""

    parsed = _validate_packet_structure(packet)
    features = ", ".join(
        f"feature {feature_id}" for feature_id in parsed["model_features"]
    )
    evidence_lines = []
    for item in packet["evidence"]:
        evidence_lines.append(
            f"- [{item['evidence_id']}] ({item['kind']}) {item['summary']} "
            f"({_citation_text(item['citation'])})"
        )
    final_paragraph = " ".join(
        [
            packet["a_side_function"]["text"],
            packet["b_side_function"]["text"],
            packet["joint_vulnerability"]["text"],
            f"Cell context: {packet['cell_context']['text']}",
            "Supporting evidence: "
            + "; ".join(item["summary"] for item in packet["evidence"]),
            f"Counterevidence/limitations: {packet['counterevidence']['text']}",
            f"Testable prediction: {parsed['prediction']}",
        ]
    )
    lines = [
        f"# Candidate mechanism hypothesis: {parsed['row_id']}",
        "",
        "## Identity and model dependency",
        "",
        f"- gene_a: {parsed['gene_a']}",
        f"- gene_b: {parsed['gene_b']}",
        f"- intervention-validated features: {features}",
        "- boundary: model-internal dependency supporting a candidate hypothesis",
        "",
        "## Structured claims",
        "",
        f"- A-side function: {packet['a_side_function']['text']}",
        f"- B-side function: {packet['b_side_function']['text']}",
        f"- Joint vulnerability: {packet['joint_vulnerability']['text']}",
        f"- Cell context: {packet['cell_context']['text']}",
        f"- Counterevidence/limitations: {packet['counterevidence']['text']}",
        f"- Testable prediction: {parsed['prediction']}",
        "",
        "## Traceable evidence",
        "",
        *evidence_lines,
        "",
        "## Final evidence-bounded paragraph",
        "",
        final_paragraph,
        "",
        (
            "This report is a candidate mechanism hypothesis grounded in a "
            "model-internal dependency; it is not proof of a biological mechanism."
        ),
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable or malformed") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"eligible manifest is invalid at line {line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError("eligible manifest row must be an object")
            records.append(value)
    return records


def _verify_upstream(
    project_root: Path,
    eligible_manifest: Path,
    pair_report_index: Path,
) -> None:
    code_root = project_root.parent
    task07 = project_root / "manifests" / "task07.sha256"
    task08 = project_root / "manifests" / "task08.sha256"
    if task07.is_file():
        _verify_sha256_manifest(
            task07,
            code_root=code_root,
            required_targets=(eligible_manifest,),
            label="Task 7",
        )
    if task08.is_file():
        _verify_sha256_manifest(
            task08,
            code_root=code_root,
            required_targets=(pair_report_index,),
            label="Task 8",
        )


def generate_mechanism_reports(
    eligible_manifest: Path,
    *,
    evidence_dir: Path,
    output_root: Path,
    pair_report_index: Path,
    reference_dir: Path | None = None,
) -> dict[str, int]:
    """Generate reports only for eligible pairs with complete evidence."""

    eligible_manifest = Path(eligible_manifest)
    evidence_dir = Path(evidence_dir)
    output_root = Path(output_root)
    pair_report_index = Path(pair_report_index)
    if output_root.exists():
        raise FileExistsError("mechanism output already exists; use a new root")
    project_root = eligible_manifest.resolve().parents[2]
    _verify_upstream(project_root, eligible_manifest, pair_report_index)
    records = [
        record
        for record in _read_jsonl(eligible_manifest)
        if record.get("prediction_type") == "TP"
        and record.get("explanation_status") == "reliable"
    ]
    index = _read_json(pair_report_index, "pair report index")
    report_entries = index.get("reports")
    if not isinstance(report_entries, Sequence):
        raise ValueError("pair report index lacks reports")
    by_row = {
        item["row_id"]: item
        for item in report_entries
        if isinstance(item, Mapping) and isinstance(item.get("row_id"), str)
    }
    if len(by_row) != len(report_entries):
        raise ValueError("pair report index has invalid or duplicate rows")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".tmp",
            dir=output_root.parent,
        )
    )
    status_rows = []
    generated = 0
    missing = 0
    direction_abstain = 0
    mechanism_eligible = 0
    try:
        for record in records:
            row_id = record.get("row_id")
            if not isinstance(row_id, str) or row_id not in by_row:
                raise ValueError("eligible row is absent from pair report index")
            pair_report = dict(by_row[row_id])
            candidate_features = {
                int(item["feature_id"])
                for item in _sequence(record.get("singles"), "eligible singles")
                if isinstance(item, Mapping) and type(item.get("feature_id")) is int
            }
            if candidate_features != set(pair_report.get("feature_ids", [])):
                raise ValueError(f"eligible feature mismatch: {row_id}")
            if pair_report.get("mechanism_eligible") is not True:
                direction_abstain += 1
                status_rows.append(
                    {"row_id": row_id, "status": "direction_abstain"}
                )
                continue
            mechanism_features = _pair_features(pair_report)
            if not mechanism_features <= candidate_features:
                raise ValueError(f"mechanism feature mismatch: {row_id}")
            mechanism_eligible += 1
            evidence_path = evidence_dir / f"{row_id}.json"
            if not evidence_path.is_file():
                missing += 1
                status_rows.append(
                    {"row_id": row_id, "status": "evidence_missing"}
                )
                continue
            packet = _read_json(evidence_path, f"evidence packet {row_id}")
            validate_evidence_packet(packet, pair_report)
            report = render_mechanism(packet)
            reference = None
            if reference_dir is not None:
                reference_path = Path(reference_dir) / f"{row_id}.json"
                if reference_path.is_file():
                    reference = _read_json(reference_path, f"reference {row_id}")
            evaluation = evaluate_mechanism(report, packet, reference)
            report_path = staging / f"{row_id}.md"
            evaluation_path = staging / f"{row_id}.evaluation.json"
            _atomic_text_create_only(report_path, report)
            _write_json_create_only(evaluation_path, evaluation)
            generated += 1
            status_rows.append(
                {
                    "row_id": row_id,
                    "status": "generated",
                    "report_sha256": file_sha256(report_path),
                    "evaluation_sha256": file_sha256(evaluation_path),
                }
            )
        counts = {
            "eligible": mechanism_eligible,
            "direction_abstain": direction_abstain,
            "generated": generated,
            "evidence_missing": missing,
            "errors": 0,
        }
        _write_json_create_only(
            staging / "status_index.json",
            {"schema_version": 1, **counts, "pairs": status_rows},
        )
        try:
            os.rename(staging, output_root)
        except FileExistsError as error:
            raise FileExistsError(
                "mechanism output already exists; use a new root"
            ) from error
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--pair-report-index", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args(argv)
    project_root = args.eligible_manifest.resolve().parents[2]
    counts = generate_mechanism_reports(
        args.eligible_manifest,
        evidence_dir=args.evidence_dir,
        output_root=(
            args.output_root
            or project_root / "experiment3" / "mechanism_reports"
        ),
        pair_report_index=(
            args.pair_report_index
            or project_root / "experiment2" / "pair_reports" / "report_index.json"
        ),
        reference_dir=args.reference_dir,
    )
    print(json.dumps(counts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
