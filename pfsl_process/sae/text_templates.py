"""Deterministic UniProt-to-text templates for SAE corpus construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


ASPECT_NAMES = (
    "Protein names",
    "Gene Names",
    "Activity regulation",
    "Developmental stage",
    "Induction",
    "Gene Ontology (biological process)",
    "Gene Ontology (cellular component)",
    "Gene Ontology (molecular function)",
    "Involvement in disease",
    "Subcellular location [CC]",
    "Domain [CC]",
    "Pathway",
    "Cofactor",
    "Catalytic activity",
    "EC number",
    "Caution",
    "Function [CC]",
    "Miscellaneous [CC]",
    "Polymorphism",
    "Post-translational modification",
    "Proteomes",
    "Sequence similarities",
    "Subunit structure",
    "Tissue specificity",
)

GENE_NAME_TEMPLATES = (
    "The ORF Name <> identifies the gene responsible for this protein.",
    "The Ordered Locus Name <> indicates the gene that codes for this protein.",
    "The nucleotide sequence <> corresponds to the synthesis of this protein.",
)


@dataclass(frozen=True)
class TextUnit:
    unit_type: str
    text: str


@dataclass(frozen=True)
class AspectText:
    index: int
    name: str
    text: str
    unit_type: str
    units: tuple[TextUnit, ...]


def _value(row: Mapping[str, str], column: str) -> str:
    value = row.get(column, "")
    if value is None:
        return ""
    return str(value).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_evidence(text: str) -> str:
    text = re.sub(r"(?<=[.!?])\s*\{[^{}]*\}\s*[.!?]", "", text)
    return _normalize(re.sub(r"\s*\{[^{}]*\}", "", text))


def _remove_trailing_evidence(text: str) -> str:
    return _normalize(
        re.sub(r"(?<=\.)\s+\{[^{}]*\}\s*[.!?]?\s*$", "", text)
    )


def _ensure_period(text: str) -> str:
    text = _normalize(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _deduplicate(units: Sequence[TextUnit]) -> tuple[TextUnit, ...]:
    seen: set[str] = set()
    result: list[TextUnit] = []
    for unit in units:
        text = _normalize(unit.text)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(TextUnit(unit.unit_type, text))
    return tuple(result)


def _sentences(text: str) -> list[str]:
    text = _ensure_period(_remove_trailing_evidence(text))
    if not text:
        return []
    protected = text.replace("e.g.", "e§g§").replace("i.e.", "i§e§")
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    return [piece.replace("§", ".").strip() for piece in pieces if piece.strip()]


def _split_labeled_blocks(raw: str, label: str) -> list[str]:
    if not raw:
        return []
    pattern = rf"(?:^|;\s*){re.escape(label)}:\s*"
    blocks = re.split(pattern, raw)
    return [block.strip().strip(";") for block in blocks if block.strip().strip(";")]


def _comment_units(raw: str, label: str, unit_type: str = "sentence") -> tuple[TextUnit, ...]:
    units: list[TextUnit] = []
    for block in _split_labeled_blocks(raw, label):
        units.extend(TextUnit(unit_type, sentence) for sentence in _sentences(block))
    return _deduplicate(units)


def _aspect(index: int, unit_type: str, units: Sequence[TextUnit]) -> AspectText:
    unique = _deduplicate(units)
    return AspectText(
        index=index,
        name=ASPECT_NAMES[index],
        text=" ".join(unit.text for unit in unique),
        unit_type=unit_type,
        units=unique,
    )


def _protein_name(raw: str) -> str:
    text = raw.strip()
    match = re.match(r"^(.*?)(?:\s*\([^)]*\))+(?:\s*\[[^]]*\])?$", text)
    if match:
        return match.group(1).strip()
    match = re.match(r"^(.*?)(?:\s*\[[^]]*\])$", text)
    return match.group(1).strip() if match else text


def stable_gene_sentence(accession: str, gene_names: str) -> str:
    """Render the primary gene name with stable template variation."""
    primary = gene_names.split()[0] if gene_names.split() else ""
    if not primary:
        return ""
    digest = hashlib.sha256(accession.encode()).hexdigest()
    template = GENE_NAME_TEMPLATES[int(digest, 16) % 3]
    return template.replace("<>", primary)


def _go_aspect(index: int, raw: str, category: str) -> AspectText:
    terms = [_normalize(term) for term in raw.split(";") if _normalize(term)]
    units = [
        TextUnit(
            "go_term",
            f"The GO term for this protein involving {category} incorporates {term}.",
        )
        for term in terms
    ]
    unique = _deduplicate(units)
    if not unique:
        return _aspect(index, "go_term", ())
    full = (
        f"The GO term for this protein involving {category} incorporates "
        f"{'; '.join(dict.fromkeys(terms))}."
    )
    return AspectText(index, ASPECT_NAMES[index], full, "go_term", unique)


def _disease_aspect(raw: str) -> AspectText:
    units: list[TextUnit] = []
    for block in _split_labeled_blocks(raw, "DISEASE"):
        disease = block.split("Note=")[0].split(" [")[0].strip().strip(";")
        if disease:
            units.append(
                TextUnit(
                    "disease",
                    f"{disease} is the disease tied to genetic variations in this protein.",
                )
            )
    return _aspect(8, "disease", units)


def _subcellular_aspect(raw: str) -> AspectText:
    units: list[TextUnit] = []
    location_text = raw
    if location_text.startswith("SUBCELLULAR LOCATION: "):
        location_text = location_text[len("SUBCELLULAR LOCATION: ") :]
    location_text = location_text.replace("; SUBCELLULAR LOCATION: ", " ")
    location_text = _remove_evidence(location_text)
    for sentence in _sentences(location_text):
        phrase = sentence.strip()
        if phrase.lower() == "secreted.":
            continue
        if any(
            token in phrase.lower()
            for token in ("located", "localizes", "found", "associate", "recruit", "detect")
        ):
            rendered = phrase
        else:
            rendered = f"The protein can be spotted at {phrase}"
        units.append(TextUnit("location", _ensure_period(rendered)))
    return _aspect(9, "location", units)


def _cofactor_aspect(raw: str) -> AspectText:
    text = raw[len("COFACTOR: ") :] if raw.startswith("COFACTOR: ") else raw
    units: list[TextUnit] = []
    for chunk in (part.strip() for part in text.split(";")):
        if chunk.startswith("Name="):
            units.append(
                TextUnit(
                    "cofactor",
                    f"Without {chunk[5:]} acting as a cofactor, this protein's function is compromised.",
                )
            )
        elif chunk.startswith("Note="):
            units.extend(
                TextUnit("cofactor", sentence) for sentence in _sentences(chunk[5:])
            )
    return _aspect(12, "cofactor", units)


def _catalytic_aspect(raw: str) -> AspectText:
    units: list[TextUnit] = []
    for block in _split_labeled_blocks(raw, "CATALYTIC ACTIVITY"):
        for chunk in (part.strip() for part in block.split(";")):
            if chunk.startswith("Reaction="):
                reaction = chunk[len("Reaction=") :]
                units.append(
                    TextUnit(
                        "reaction",
                        f"The reaction: {reaction} undergoes catalysis due to the presence of this protein.",
                    )
                )
    return _aspect(13, "reaction", units)


def _ec_aspect(raw: str, ec_descriptions: Mapping[str, str]) -> AspectText:
    units: list[TextUnit] = []
    for code in (_normalize(item) for item in raw.split(";") if _normalize(item)):
        key = code if code.startswith("EC ") else f"EC {code}"
        description = _ensure_period(ec_descriptions.get(key, ""))
        text = description or f"This protein has enzyme commission number {key}."
        units.append(TextUnit("ec_number", text))
    return _aspect(14, "ec_number", units)


def _proteome_aspect(raw: str) -> AspectText:
    if not raw:
        return _aspect(20, "proteome", ())
    name = raw.split(": ", 1)[1] if ": " in raw else raw
    if name == "Unplaced":
        return _aspect(20, "proteome", ())
    return _aspect(
        20,
        "proteome",
        [TextUnit("proteome", f"A proteome, containing {name}, is composed of this protein.")],
    )


def build_aspect_texts(
    row: Mapping[str, str],
    ec_descriptions: Mapping[str, str],
) -> tuple[AspectText, ...]:
    """Convert one UniProt row into the ordered 24-aspect representation."""
    accession = _value(row, "Entry")
    protein = _protein_name(_value(row, "Protein names"))
    protein_units = (
        [TextUnit("protein_name", f"This protein is identified as {protein}.")]
        if protein
        else []
    )
    gene_sentence = stable_gene_sentence(accession, _value(row, "Gene Names"))

    aspects = [
        _aspect(0, "protein_name", protein_units),
        _aspect(1, "gene_name", [TextUnit("gene_name", gene_sentence)] if gene_sentence else []),
        _aspect(2, "sentence", _comment_units(_value(row, "Activity regulation"), "ACTIVITY REGULATION")),
        _aspect(3, "sentence", _comment_units(_value(row, "Developmental stage"), "DEVELOPMENTAL STAGE")),
        _aspect(4, "sentence", _comment_units(_value(row, "Induction"), "INDUCTION")),
        _go_aspect(5, _value(row, ASPECT_NAMES[5]), "biological process"),
        _go_aspect(6, _value(row, ASPECT_NAMES[6]), "cellular component"),
        _go_aspect(7, _value(row, ASPECT_NAMES[7]), "molecular function"),
        _disease_aspect(_value(row, "Involvement in disease")),
        _subcellular_aspect(_value(row, "Subcellular location [CC]")),
        _aspect(10, "sentence", _comment_units(_value(row, "Domain [CC]"), "DOMAIN")),
        _aspect(11, "sentence", _comment_units(_value(row, "Pathway"), "PATHWAY")),
        _cofactor_aspect(_value(row, "Cofactor")),
        _catalytic_aspect(_value(row, "Catalytic activity")),
        _ec_aspect(_value(row, "EC number"), ec_descriptions),
        _aspect(15, "sentence", _comment_units(_value(row, "Caution"), "CAUTION")),
        _aspect(16, "sentence", _comment_units(_value(row, "Function [CC]"), "FUNCTION")),
        _aspect(17, "sentence", _comment_units(_value(row, "Miscellaneous [CC]"), "MISCELLANEOUS")),
        _aspect(18, "sentence", _comment_units(_value(row, "Polymorphism"), "POLYMORPHISM")),
        _aspect(19, "sentence", _comment_units(_value(row, "Post-translational modification"), "PTM")),
        _proteome_aspect(_value(row, "Proteomes")),
        _aspect(21, "sentence", _comment_units(_value(row, "Sequence similarities"), "SIMILARITY")),
        _aspect(22, "sentence", _comment_units(_value(row, "Subunit structure"), "SUBUNIT")),
        _aspect(23, "sentence", _comment_units(_value(row, "Tissue specificity"), "TISSUE SPECIFICITY")),
    ]
    if tuple(aspect.name for aspect in aspects) != ASPECT_NAMES:
        raise AssertionError("Internal aspect order does not match ASPECT_NAMES")
    return tuple(aspects)


def make_unit_id(
    accession: str,
    aspect_index: int,
    unit_type: str,
    unit_index: int,
) -> str:
    return f"{accession}:a{aspect_index:02d}:{unit_type}:{unit_index:04d}"
