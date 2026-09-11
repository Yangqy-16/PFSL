"""Exact-offset, bounded hierarchical masking for frozen ProTrek SAE features."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .feature_atlas_config import (
    AtlasPaths,
    AtlasSourceBinding,
    FeatureAtlasConfig,
    validate_source_binding,
)
from .feature_atlas_io import (
    AtlasIOError,
    file_metadata,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
    write_torch_atomic,
)
from .feature_atlas_selection import validate_selection_manifest
from .protrek_inference import ProTrekTextEmbedder, TokenizedText, load_text_embedder
from .sae_training_data import build_data_index
from .sparse_feature_activations import (
    _resolve_device_identity,
    load_frozen_sae,
    validate_sparse_feature_activations,
)


SCHEMA_VERSION = 1
WINDOW_LENGTHS = (1, 2, 4, 8)
EVIDENCE_GRADES = frozenset(
    {
        "strong_span",
        "distributed_evidence",
        "topk_competition_sensitive",
        "unstable_span",
        "insufficient_texts",
    }
)


class SpanAttributionError(ValueError):
    """Raised when a masking plan or registered attribution is invalid."""


@dataclass(frozen=True)
class MaskedText:
    original_text: str
    text: str
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    start: int
    end: int
    substring: str
    masked_token_count: int
    original_token_count: int
    output_token_count: int
    operation: str = "mask"

    def __post_init__(self) -> None:
        if self.operation not in {"mask", "sufficiency", "deletion"}:
            raise ValueError("masked text operation is invalid")
        source_length = len(self.offsets) if self.operation == "deletion" else len(self.token_ids)
        if not 0 <= self.start < self.end <= source_length:
            raise ValueError("masked token range is invalid")
        if self.masked_token_count < 0 or (
            self.masked_token_count == 0 and self.operation != "sufficiency"
        ):
            raise ValueError("masked_token_count must be positive")
        if self.operation != "deletion" and self.original_token_count != self.output_token_count:
            raise ValueError("equal-length masking changed the non-special token count")


def _content_indices(tokenized: TokenizedText) -> tuple[int, ...]:
    return tuple(
        index
        for index, special in enumerate(tokenized.special_tokens_mask)
        if special == 0
    )


def _validate_span(tokenized: TokenizedText, start: int, end: int) -> None:
    if type(start) is not int or type(end) is not int:
        raise TypeError("token span boundaries must be built-in ints")
    if not 0 <= start < end <= len(tokenized.token_ids):
        raise SpanAttributionError("token span lies outside the tokenized text")
    if any(tokenized.special_tokens_mask[index] for index in range(start, end)):
        raise SpanAttributionError("token span cannot contain special tokens")


def _render_replacements(
    tokenized: TokenizedText,
    replacement_indices: set[int],
    replacement: str = "[MASK]",
) -> str:
    pieces: list[str] = []
    cursor = 0
    for index in _content_indices(tokenized):
        start, end = tokenized.offsets[index]
        pieces.append(tokenized.text[cursor:start])
        pieces.append(replacement if index in replacement_indices else tokenized.text[start:end])
        cursor = end
    pieces.append(tokenized.text[cursor:])
    return "".join(pieces)


def mask_token_span(tokenized: TokenizedText, start: int, end: int) -> MaskedText:
    """Replace every target non-special token with exactly one ``[MASK]``."""

    _validate_span(tokenized, start, end)
    selected = set(range(start, end))
    token_ids = tuple(
        tokenized.mask_token_id if index in selected else token_id
        for index, token_id in enumerate(tokenized.token_ids)
    )
    begin, finish = tokenized.offsets[start][0], tokenized.offsets[end - 1][1]
    content_count = len(_content_indices(tokenized))
    return MaskedText(
        original_text=tokenized.text,
        text=_render_replacements(tokenized, selected),
        token_ids=token_ids,
        offsets=tokenized.offsets,
        special_tokens_mask=tokenized.special_tokens_mask,
        start=start,
        end=end,
        substring=tokenized.text[begin:finish],
        masked_token_count=end - start,
        original_token_count=content_count,
        output_token_count=content_count,
    )


def build_sufficiency_text(tokenized: TokenizedText, start: int, end: int) -> MaskedText:
    _validate_span(tokenized, start, end)
    retained = set(range(start, end))
    replaced = set(_content_indices(tokenized)) - retained
    token_ids = tuple(
        tokenized.mask_token_id if index in replaced else token_id
        for index, token_id in enumerate(tokenized.token_ids)
    )
    begin, finish = tokenized.offsets[start][0], tokenized.offsets[end - 1][1]
    count = len(_content_indices(tokenized))
    return MaskedText(
        original_text=tokenized.text,
        text=_render_replacements(tokenized, replaced),
        token_ids=token_ids,
        offsets=tokenized.offsets,
        special_tokens_mask=tokenized.special_tokens_mask,
        start=start,
        end=end,
        substring=tokenized.text[begin:finish],
        masked_token_count=len(replaced),
        original_token_count=count,
        output_token_count=count,
        operation="sufficiency",
    )


_sufficiency_text = build_sufficiency_text


def _deletion_text(tokenized: TokenizedText, start: int, end: int) -> MaskedText:
    _validate_span(tokenized, start, end)
    begin, finish = tokenized.offsets[start][0], tokenized.offsets[end - 1][1]
    count = len(_content_indices(tokenized))
    return MaskedText(
        original_text=tokenized.text,
        text=tokenized.text[:begin] + tokenized.text[finish:],
        token_ids=tokenized.token_ids[:start] + tokenized.token_ids[end:],
        offsets=tokenized.offsets,
        special_tokens_mask=tokenized.special_tokens_mask,
        start=start,
        end=end,
        substring=tokenized.text[begin:finish],
        masked_token_count=end - start,
        original_token_count=count,
        output_token_count=count - (end - start),
        operation="deletion",
    )


def segment_phrases(tokenized: TokenizedText) -> tuple[tuple[int, int], ...]:
    """Return deterministic token ranges at sentence and biological-clause boundaries."""

    content = _content_indices(tokenized)
    if not content:
        return ()
    ranges: list[tuple[int, int]] = []
    start = content[0]
    closers = frozenset({";", ":", ".", "!", "?", ")"})
    clause_words = frozenset({"and", "but", "whereas", "which", "that"})
    for index in content:
        token = tokenized.tokens[index].lower()
        if token == "(" and start < index:
            ranges.append((start, index))
            start = index
        elif token in clause_words and start < index:
            ranges.append((start, index))
            start = index
        if token in closers:
            ranges.append((start, index + 1))
            start = index + 1
    final_end = content[-1] + 1
    if start < final_end:
        ranges.append((start, final_end))
    return tuple((left, right) for left, right in ranges if left < right)


def sliding_token_windows(span: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    start, end = span
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise SpanAttributionError("window source span is invalid")
    return tuple(
        (left, left + length)
        for length in WINDOW_LENGTHS
        if length <= end - start
        for left in range(start, end - length + 1)
    )


def select_shortest_explanatory_span(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not candidates:
        raise SpanAttributionError("cannot select from an empty span list")
    checked: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            start, end, drop = (
                int(candidate["start"]),
                int(candidate["end"]),
                float(candidate["drop"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SpanAttributionError("span candidate is invalid") from error
        if end <= start or not math.isfinite(drop):
            raise SpanAttributionError("span candidate has invalid bounds/drop")
        checked.append(dict(candidate))
    largest = max(float(item["drop"]) for item in checked)
    eligible = [item for item in checked if float(item["drop"]) >= 0.8 * largest]
    return min(
        eligible,
        key=lambda item: (
            int(item["end"]) - int(item["start"]),
            -float(item["drop"]),
            int(item["start"]),
            int(item["end"]),
        ),
    )


def feature_signals(
    post_topk: torch.Tensor, dense_preactivation: torch.Tensor, feature_index: int
) -> dict[str, object]:
    if (
        not isinstance(post_topk, torch.Tensor)
        or not isinstance(dense_preactivation, torch.Tensor)
        or post_topk.ndim != 2
        or dense_preactivation.shape != post_topk.shape
        or post_topk.shape[0] != 1
        or not 0 <= feature_index < post_topk.shape[1]
    ):
        raise SpanAttributionError("feature signal tensors are invalid")
    post = float(post_topk[0, feature_index])
    dense = float(F.relu(dense_preactivation[0, feature_index]))
    if not math.isfinite(post) or not math.isfinite(dense):
        raise SpanAttributionError("feature signals must be finite")
    return {
        "post_topk_activation": post,
        "relu_dense_preactivation": dense,
        "support_member": post > 0.0,
    }


def necessity(original_activation: float, masked_activation: float) -> float:
    return (float(original_activation) - float(masked_activation)) / max(
        float(original_activation), 1e-8
    )


def sufficiency(original_activation: float, retained_activation: float) -> float:
    return float(retained_activation) / max(float(original_activation), 1e-8)


def empirical_p_value(observed: float, controls: Sequence[float]) -> float:
    if not controls:
        return 1.0
    if not math.isfinite(float(observed)) or any(not math.isfinite(float(x)) for x in controls):
        raise SpanAttributionError("empirical test values must be finite")
    return (1 + sum(float(value) >= float(observed) for value in controls)) / (
        len(controls) + 1
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    checked = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in checked):
        raise SpanAttributionError("p-values must be finite probabilities")
    count = len(checked)
    ordered = sorted(range(count), key=lambda index: (checked[index], index))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        index = ordered[rank_index]
        rank = rank_index + 1
        raw = checked[index] if rank == count else checked[index] * count / rank
        running = min(running, raw)
        adjusted[index] = min(1.0, running)
    return adjusted


def evidence_grade(metrics: Mapping[str, object]) -> str:
    def number(name: str) -> float:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise SpanAttributionError(f"{name} is missing or invalid") from error
        if not math.isfinite(value):
            raise SpanAttributionError(f"{name} must be finite")
        return value

    train_count = int(metrics.get("train_text_count", 0))
    validation_count = int(metrics.get("validation_text_count", 0))
    if train_count < 3 or validation_count < 3:
        return "insufficient_texts"
    direction = number("validation_direction_fraction")
    drop = number("activation_drop_fraction")
    percentile = number("random_percentile")
    fdr = number("fdr")
    agreement = metrics.get("mask_deletion_agreement") is True
    if direction < 0.70 or not agreement:
        return "unstable_span"
    dense_drop = number("dense_drop_fraction")
    support_dropout = number("support_dropout_fraction")
    if support_dropout >= 0.50 and drop >= 0.50 and dense_drop < 0.50:
        return "topk_competition_sensitive"
    if drop >= 0.50 and percentile > 0.95 and fdr < 0.05:
        return "strong_span"
    return "distributed_evidence"


def directory_tree_binding(directory: Path) -> dict[str, object]:
    """Return a deterministic content binding for a frozen config tree."""

    if not isinstance(directory, Path) or not directory.is_dir():
        raise SpanAttributionError("frozen config tree is missing")
    resolved = directory.resolve()
    entries: list[dict[str, object]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        if path.is_symlink():
            raise SpanAttributionError("frozen config tree cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SpanAttributionError("frozen config tree contains a non-file entry")
        metadata = file_metadata(path)
        entries.append(
            {
                "bytes": metadata["bytes"],
                "path": path.relative_to(resolved).as_posix(),
                "sha256": metadata["sha256"],
            }
        )
    if not entries:
        raise SpanAttributionError("frozen config tree is empty")
    serialized = json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "entries": entries,
        "path": str(resolved),
        "tree_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def build_protrek_model_binding(
    embedder: object, *, allow_test_fixture_unbound: bool = False
) -> dict[str, object]:
    """Bind cached embeddings to exact ProTrek checkpoint and config bytes."""

    if type(allow_test_fixture_unbound) is not bool:
        raise TypeError("fixture binding gate must be a built-in bool")
    fixture = getattr(embedder, "fixture_model_binding", None)
    if allow_test_fixture_unbound and fixture == {"fingerprint": "fixture-unbound"}:
        return {"fingerprint": "fixture-unbound"}
    provenance = getattr(embedder, "provenance", None)
    protrek = provenance.get("protrek") if isinstance(provenance, dict) else None
    required = {"checkpoint", "protein_config", "text_config", "structure_config"}
    if not isinstance(protrek, dict) or not required <= set(protrek):
        raise SpanAttributionError("embedder lacks exact ProTrek model provenance")
    raw_checkpoint = protrek.get("checkpoint")
    if type(raw_checkpoint) is not str or raw_checkpoint == "unknown":
        raise SpanAttributionError("embedder checkpoint provenance is unknown")
    checkpoint = Path(raw_checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise SpanAttributionError("frozen ProTrek checkpoint is missing")
    checkpoint_metadata = file_metadata(checkpoint)
    binding: dict[str, object] = {
        "checkpoint": {
            "bytes": checkpoint_metadata["bytes"],
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_metadata["sha256"],
        }
    }
    for name in ("protein_config", "text_config", "structure_config"):
        raw = protrek.get(name)
        if type(raw) is not str or raw == "unknown":
            raise SpanAttributionError(f"embedder {name} provenance is unknown")
        binding[name] = directory_tree_binding(Path(raw))
    serialized = json.dumps(binding, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    binding["fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return binding


class EmbeddingCache:
    """Strict SHA-256 keyed CPU float32 embedding cache."""

    def __init__(
        self, directory: Path, *, model_binding_fingerprint: str = "fixture-unbound"
    ) -> None:
        if not isinstance(directory, Path):
            raise TypeError("cache directory must be a Path")
        if type(model_binding_fingerprint) is not str or not model_binding_fingerprint:
            raise TypeError("model binding fingerprint must be a non-empty built-in str")
        self.directory = directory
        self.model_binding_fingerprint = model_binding_fingerprint

    @staticmethod
    def key(text: str) -> str:
        if type(text) is not str:
            raise TypeError("cached text must be a built-in str")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def path_for_text(self, text: str) -> Path:
        return self.directory / f"{self.key(text)}.pt"

    def contains(self, text: str) -> bool:
        return self.path_for_text(text).is_file()

    def _load(self, text: str) -> torch.Tensor:
        path = self.path_for_text(text)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise SpanAttributionError(f"invalid cached embedding: {path}") from error
        expected = self.key(text)
        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "embedding", "model_binding_fingerprint", "text_sha256"
            }
            or payload.get("text_sha256") != expected
            or payload.get("model_binding_fingerprint")
            != self.model_binding_fingerprint
            or not isinstance(payload.get("embedding"), torch.Tensor)
        ):
            raise SpanAttributionError(f"cached embedding schema mismatch: {path}")
        embedding = payload["embedding"]
        if (
            embedding.dtype != torch.float32
            or embedding.ndim != 1
            or embedding.device.type != "cpu"
            or not bool(torch.isfinite(embedding).all())
        ):
            raise SpanAttributionError(f"cached embedding tensor is invalid: {path}")
        return embedding

    def encode(
        self,
        texts: Sequence[str],
        encoder: Callable[[Sequence[str], int], torch.Tensor],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive built-in int")
        unique_missing: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = self.key(text)
            if key not in seen and not self.contains(text):
                seen.add(key)
                unique_missing.append(text)
        if unique_missing:
            self.directory.mkdir(parents=True, exist_ok=True)
            encoded = encoder(unique_missing, batch_size)
            if (
                not isinstance(encoded, torch.Tensor)
                or encoded.dtype != torch.float32
                or encoded.ndim != 2
                or encoded.shape[0] != len(unique_missing)
                or not bool(torch.isfinite(encoded).all())
            ):
                raise SpanAttributionError("embedder returned invalid uncached embeddings")
            for text, row in zip(unique_missing, encoded.detach().cpu()):
                write_torch_atomic(
                    self.path_for_text(text),
                    {
                        "embedding": row.clone(),
                        "model_binding_fingerprint": self.model_binding_fingerprint,
                        "text_sha256": self.key(text),
                    },
                )
        return torch.stack([self._load(text) for text in texts])


def _sae_output(sae: object, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model_device = embedding.device
    parameters = getattr(sae, "parameters", None)
    if callable(parameters):
        try:
            model_device = next(iter(parameters())).device
        except StopIteration:
            pass
    embedding = embedding.to(model_device)
    try:
        output = sae(embedding, update_dead_stats=False)  # type: ignore[operator]
    except TypeError:
        output = sae(embedding)  # type: ignore[operator]
    activations = getattr(output, "activations", None)
    preactivations = getattr(output, "preactivations", None)
    if isinstance(output, Mapping):
        activations = output.get("activations")
        preactivations = output.get("preactivations")
    if not isinstance(activations, torch.Tensor) or not isinstance(preactivations, torch.Tensor):
        raise SpanAttributionError("SAE output lacks activations/preactivations")
    return activations, preactivations


def _signal_for(sae: object, embedding: torch.Tensor, feature_index: int) -> dict[str, object]:
    activations, preactivations = _sae_output(sae, embedding.unsqueeze(0))
    return feature_signals(activations, preactivations, feature_index)


def _drop(original: Mapping[str, object], perturbed: Mapping[str, object]) -> float:
    return necessity(
        float(original["post_topk_activation"]),
        float(perturbed["post_topk_activation"]),
    )


def _embedding_metrics(original: torch.Tensor, perturbed: torch.Tensor) -> tuple[float, float]:
    return (
        float(F.cosine_similarity(original.unsqueeze(0), perturbed.unsqueeze(0))[0]),
        float(torch.linalg.vector_norm(original - perturbed)),
    )


def _deduplicated_texts(row: Mapping[str, object], split: str) -> list[dict[str, object]]:
    top_texts = row.get("top_texts")
    values = top_texts.get(split) if isinstance(top_texts, dict) else None
    if not isinstance(values, list):
        raise SpanAttributionError("atlas top_texts split is invalid")
    selected: list[dict[str, object]] = []
    seen_accessions: set[str] = set()
    seen_texts: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise SpanAttributionError("atlas top text is invalid")
        accession, text = value.get("accession"), value.get("text")
        normalized = " ".join(str(text).lower().split())
        if (
            type(accession) is str
            and type(text) is str
            and normalized
            and accession not in seen_accessions
            and normalized not in seen_texts
        ):
            selected.append(dict(value))
            seen_accessions.add(accession)
            seen_texts.add(normalized)
            if len(selected) == 10:
                break
    return selected


def _source_binding(manifest: Mapping[str, object]) -> AtlasSourceBinding:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise SpanAttributionError("selection provenance is invalid")
    try:
        return AtlasSourceBinding(
            **{name: provenance[name] for name in AtlasSourceBinding.__dataclass_fields__}
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SpanAttributionError("selection lacks a complete Task 1 source binding") from error


def _load_atlas_rows(path: Path) -> dict[int, dict[str, object]]:
    try:
        rows = list(strict_json_lines(path))
    except (AtlasIOError, OSError, ValueError) as error:
        raise SpanAttributionError(f"invalid frozen atlas: {path}") from error
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        feature = row.get("feature_index")
        if type(feature) is not int or feature in result:
            raise SpanAttributionError("atlas feature identities are invalid")
        result[feature] = row
    return result


def _selected_features(selection: Mapping[str, object]) -> list[int]:
    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise SpanAttributionError("frozen selection selected list is invalid")
    features = []
    for item in selected:
        feature = item.get("feature_index") if isinstance(item, dict) else None
        if type(feature) is not int:
            raise SpanAttributionError("frozen selection feature identity is invalid")
        features.append(feature)
    if len(features) != len(set(features)):
        raise SpanAttributionError("frozen selected feature identities are not unique")
    return sorted(features)


# The execution contract below deliberately separates the worst-case frozen
# allowlist from conditional hierarchical encoding.  These definitions replace
# the original fixture core above while retaining the public primitive API.


PERTURBATION_RECORD_KEYS = frozenset(
    {
        "accession", "attention_mask", "control_kind", "direction", "drop_fraction",
        "embedding_cosine", "embedding_l2", "end", "evaluable", "feature_index",
        "kind", "necessity", "offsets", "original_dense_preactivation",
        "original_embedding_sha256", "original_post_topk_activation",
        "original_support_member", "original_text", "parent_id",
        "perturbed_dense_preactivation", "perturbed_embedding_sha256",
        "perturbed_post_topk_activation", "perturbed_support_member", "record_id",
        "rendered_text", "selection_used_activation", "special_tokens_mask", "split",
        "start", "substring", "sufficiency", "support_dropout", "text_sha256",
        "token_ids", "tokens", "unit_id",
    }
)


def text_sha256(text: str) -> str:
    if type(text) is not str:
        raise TypeError("text must be a built-in str")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlannedPerturbation:
    perturbation_id: str
    sample_id: str
    kind: str
    control_kind: str | None
    parent_id: str | None
    phrase_index: int | None
    start: int | None
    end: int | None
    substring: str
    rendered_text: str
    tokenized: TokenizedText


@dataclass(frozen=True)
class PlannedSample:
    sample_id: str
    feature_index: int
    split: str
    accession: str
    unit_id: str
    text: str
    text_sha256: str
    tokenized: TokenizedText
    perturbations: tuple[PlannedPerturbation, ...]


@dataclass(frozen=True)
class FrozenFeaturePlan:
    feature_index: int
    seed: int
    permutations: int
    samples: tuple[PlannedSample, ...]
    allowed_sha256: tuple[str, ...]
    allowed_texts: tuple[tuple[str, str], ...]


def _perturbation_id(
    feature_index: int,
    sample_id: str,
    kind: str,
    start: int | None,
    end: int | None,
    control_kind: str | None,
    parent_id: str | None,
) -> str:
    raw = json.dumps(
        [feature_index, sample_id, kind, start, end, control_kind, parent_id],
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalized_tokens(tokenized: TokenizedText, start: int, end: int) -> tuple[str, ...]:
    return tuple(
        tokenized.tokens[index].lower().removeprefix("##")
        for index in range(start, end)
        if tokenized.special_tokens_mask[index] == 0
        and any(character.isalnum() for character in tokenized.tokens[index])
    )


def _planned_equal_length(
    *,
    embedder: object,
    feature_index: int,
    sample_id: str,
    tokenized: TokenizedText,
    kind: str,
    start: int,
    end: int,
    phrase_index: int | None,
    control_kind: str | None = None,
    parent_id: str | None = None,
) -> PlannedPerturbation:
    masked = (
        build_sufficiency_text(tokenized, start, end)
        if kind == "sufficiency"
        else mask_token_span(tokenized, start, end)
    )
    verifier = getattr(embedder, "verify_rendered_perturbation", None)
    if not callable(verifier):
        raise SpanAttributionError("embedder lacks perturbation retokenization validation")
    observed = verifier(tokenized, masked.text, masked.token_ids)
    if observed.token_ids != masked.token_ids:
        raise SpanAttributionError("executed equal-length perturbation differs from plan")
    identifier = _perturbation_id(
        feature_index, sample_id, kind, start, end, control_kind, parent_id
    )
    return PlannedPerturbation(
        perturbation_id=identifier,
        sample_id=sample_id,
        kind=kind,
        control_kind=control_kind,
        parent_id=parent_id,
        phrase_index=phrase_index,
        start=start,
        end=end,
        substring=masked.substring,
        rendered_text=masked.text,
        tokenized=observed,
    )


def _planned_deletion(
    *,
    embedder: object,
    feature_index: int,
    sample_id: str,
    tokenized: TokenizedText,
    start: int,
    end: int,
    phrase_index: int,
    parent_id: str,
) -> PlannedPerturbation:
    deleted = _deletion_text(tokenized, start, end)
    observed = embedder.tokenize_with_offsets(deleted.text)
    return PlannedPerturbation(
        perturbation_id=_perturbation_id(
            feature_index, sample_id, "deletion", start, end, None, parent_id
        ),
        sample_id=sample_id,
        kind="deletion",
        control_kind=None,
        parent_id=parent_id,
        phrase_index=phrase_index,
        start=start,
        end=end,
        substring=deleted.substring,
        rendered_text=deleted.text,
        tokenized=observed,
    )


def _alternative_spans(
    tokenized: TokenizedText,
    start: int,
    end: int,
    *,
    feature_index: int,
    sample_sha256: str,
    seed: int,
) -> tuple[tuple[int, int, str], ...]:
    """Enumerate same-length alternatives in stable nearest-position strata."""

    content = _content_indices(tokenized)
    if not content:
        return ()
    width = end - start
    last = content[-1] + 1
    candidates = []
    for left in content:
        right = left + width
        if right > last or left == start:
            continue
        if any(tokenized.special_tokens_mask[index] for index in range(left, right)):
            continue
        if not (right <= start or left >= end):
            continue
        relative_distance = abs(
            left / max(1, last - width) - start / max(1, last - width)
        )
        stratum = "near" if relative_distance <= 0.25 else "mid" if relative_distance <= 0.5 else "far"
        digest = hashlib.sha256(
            f"{seed}|{feature_index}|{sample_sha256}|{left}|{right}".encode("utf-8")
        ).hexdigest()
        candidates.append((relative_distance, stratum, digest, left, right))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return tuple((left, right, stratum) for _, stratum, _, left, right in candidates)


def _build_sample_plan(
    feature_index: int,
    evidence: Mapping[str, object],
    embedder: object,
    *,
    seed: int,
) -> PlannedSample:
    text = evidence.get("text")
    split = evidence.get("split")
    accession = evidence.get("accession")
    unit_id = evidence.get("unit_id")
    if (
        type(text) is not str
        or split not in {"train", "validation"}
        or type(accession) is not str
        or type(unit_id) is not str
    ):
        raise SpanAttributionError("feature plan evidence row is invalid")
    tokenized = embedder.tokenize_with_offsets(text)
    sample_sha = text_sha256(text)
    sample_id = hashlib.sha256(
        f"{feature_index}|{split}|{accession}|{unit_id}|{sample_sha}".encode("utf-8")
    ).hexdigest()
    perturbations: list[PlannedPerturbation] = []
    for phrase_index, (phrase_start, phrase_end) in enumerate(segment_phrases(tokenized)):
        phrase = _planned_equal_length(
            embedder=embedder,
            feature_index=feature_index,
            sample_id=sample_id,
            tokenized=tokenized,
            kind="phrase_mask",
            start=phrase_start,
            end=phrase_end,
            phrase_index=phrase_index,
        )
        perturbations.append(phrase)
        for start, end in sliding_token_windows((phrase_start, phrase_end)):
            window = _planned_equal_length(
                embedder=embedder,
                feature_index=feature_index,
                sample_id=sample_id,
                tokenized=tokenized,
                kind="window_mask",
                start=start,
                end=end,
                phrase_index=phrase_index,
                parent_id=phrase.perturbation_id,
            )
            perturbations.append(window)
            perturbations.append(
                _planned_equal_length(
                    embedder=embedder,
                    feature_index=feature_index,
                    sample_id=sample_id,
                    tokenized=tokenized,
                    kind="sufficiency",
                    start=start,
                    end=end,
                    phrase_index=phrase_index,
                    parent_id=window.perturbation_id,
                )
            )
            perturbations.append(
                _planned_deletion(
                    embedder=embedder,
                    feature_index=feature_index,
                    sample_id=sample_id,
                    tokenized=tokenized,
                    start=start,
                    end=end,
                    phrase_index=phrase_index,
                    parent_id=window.perturbation_id,
                )
            )
            alternatives = _alternative_spans(
                tokenized,
                start,
                end,
                feature_index=feature_index,
                sample_sha256=sample_sha,
                seed=seed,
            )
            for control_start, control_end, stratum in alternatives:
                perturbations.append(
                    _planned_equal_length(
                        embedder=embedder,
                        feature_index=feature_index,
                        sample_id=sample_id,
                        tokenized=tokenized,
                        kind="random_control",
                        start=control_start,
                        end=control_end,
                        phrase_index=phrase_index,
                        control_kind=f"length_position:{stratum}",
                        parent_id=window.perturbation_id,
                    )
                )
    identities = [item.perturbation_id for item in perturbations]
    if len(identities) != len(set(identities)):
        raise SpanAttributionError("feature plan has duplicate perturbation identities")
    return PlannedSample(
        sample_id=sample_id,
        feature_index=feature_index,
        split=split,
        accession=accession,
        unit_id=unit_id,
        text=text,
        text_sha256=sample_sha,
        tokenized=tokenized,
        perturbations=tuple(perturbations),
    )


def build_feature_plan(
    feature_index: int,
    texts_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    embedder: object,
    *,
    seed: int,
    permutations: int,
) -> FrozenFeaturePlan:
    if type(feature_index) is not int or feature_index < 0:
        raise TypeError("feature_index must be a nonnegative built-in int")
    if type(seed) is not int or seed < 0 or type(permutations) is not int or permutations <= 0:
        raise ValueError("seed/permutations are invalid")
    samples = tuple(
        _build_sample_plan(feature_index, evidence, embedder, seed=seed)
        for split in ("train", "validation")
        for evidence in texts_by_split.get(split, ())
    )
    allowed: dict[str, str] = {}
    for sample in samples:
        allowed[sample.text_sha256] = sample.text
        for item in sample.perturbations:
            key = text_sha256(item.rendered_text)
            previous = allowed.setdefault(key, item.rendered_text)
            if previous != item.rendered_text:
                raise SpanAttributionError("SHA-256 collision in feature plan")
    return FrozenFeaturePlan(
        feature_index=feature_index,
        seed=seed,
        permutations=permutations,
        samples=samples,
        allowed_sha256=tuple(sorted(allowed)),
        allowed_texts=tuple(sorted(allowed.items())),
    )


def preflight_feature_plans(
    plans: Sequence[FrozenFeaturePlan],
    cache: EmbeddingCache,
    *,
    max_encodings: int,
) -> dict[str, object]:
    allowed: dict[str, str] = {}
    for plan in plans:
        for key, text in plan.allowed_texts:
            previous = allowed.setdefault(key, text)
            if previous != text:
                raise SpanAttributionError("SHA-256 collision across feature plans")
    uncached = sum(not cache.contains(text) for text in allowed.values())
    if uncached > max_encodings:
        raise SpanAttributionError(
            f"projected unique uncached encoding count {uncached} exceeds hard limit {max_encodings:,}"
        )
    return {
        "allowed_sha256": sorted(allowed),
        "executed_sha256": [],
        "max_encodings": max_encodings,
        "projected_unique_uncached": uncached,
        "unique_allowed": len(allowed),
    }


def _reconciled_family_budget(
    frozen_family_budget: Mapping[str, object],
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reconciled = json.loads(json.dumps(frozen_family_budget, sort_keys=True))
    executed: set[str] = set()
    for payload in feature_payloads:
        local = payload.get("budget")
        if not isinstance(local, dict):
            raise SpanAttributionError("registered feature lacks a local budget")
        executed.update(str(value) for value in local.get("executed_sha256", ()))
    if not executed <= set(reconciled.get("allowed_sha256", ())):
        raise SpanAttributionError("registered feature budgets escape family allowlist")
    reconciled["executed_sha256"] = sorted(executed)
    return reconciled


def _encode_with_budget(
    texts: Sequence[str],
    embedder: object,
    cache: EmbeddingCache,
    batch_size: int,
    budget: dict[str, object],
) -> dict[str, torch.Tensor]:
    unique = list(dict.fromkeys(texts))
    requested = {text_sha256(text) for text in unique}
    allowed = set(budget["allowed_sha256"])
    if not requested <= allowed:
        outside = sorted(requested - allowed)
        raise SpanAttributionError(f"execution requested text outside frozen allowlist: {outside[0]}")
    tensor = cache.encode(unique, embedder.encode, batch_size=batch_size)
    executed = set(budget["executed_sha256"])
    executed.update(requested)
    budget["executed_sha256"] = sorted(executed)
    return dict(zip(unique, tensor))


def _embedding_sha256(embedding: torch.Tensor) -> str:
    return hashlib.sha256(
        embedding.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def _direction(value: float) -> str:
    return "decrease" if value > 0.0 else "increase" if value < 0.0 else "unchanged"


def _record_from_spec(
    *,
    plan: FrozenFeaturePlan,
    sample: PlannedSample,
    spec: PlannedPerturbation | None,
    original_embedding: torch.Tensor,
    perturbed_embedding: torch.Tensor,
    sae: object,
    selection_used_activation: bool,
    evaluable: bool = True,
) -> dict[str, object]:
    original_signal = _signal_for(sae, original_embedding, plan.feature_index)
    perturbed_signal = _signal_for(sae, perturbed_embedding, plan.feature_index)
    kind = "original" if spec is None else spec.kind
    drop = _drop(original_signal, perturbed_signal)
    tokenized = sample.tokenized if spec is None else spec.tokenized
    record = {
        "accession": sample.accession,
        "attention_mask": list(tokenized.attention_mask),
        "control_kind": None if spec is None else spec.control_kind,
        "direction": _direction(drop),
        "drop_fraction": drop,
        "embedding_cosine": _embedding_metrics(original_embedding, perturbed_embedding)[0],
        "embedding_l2": _embedding_metrics(original_embedding, perturbed_embedding)[1],
        "end": None if spec is None else spec.end,
        "evaluable": evaluable,
        "feature_index": plan.feature_index,
        "kind": kind,
        "necessity": drop if kind in {"phrase_mask", "window_mask", "random_control", "deletion"} else None,
        "offsets": [list(offset) for offset in tokenized.offsets],
        "original_dense_preactivation": float(original_signal["relu_dense_preactivation"]),
        "original_embedding_sha256": _embedding_sha256(original_embedding),
        "original_post_topk_activation": float(original_signal["post_topk_activation"]),
        "original_support_member": bool(original_signal["support_member"]),
        "original_text": sample.text,
        "parent_id": None if spec is None else spec.parent_id,
        "perturbed_dense_preactivation": float(perturbed_signal["relu_dense_preactivation"]),
        "perturbed_embedding_sha256": _embedding_sha256(perturbed_embedding),
        "perturbed_post_topk_activation": float(perturbed_signal["post_topk_activation"]),
        "perturbed_support_member": bool(perturbed_signal["support_member"]),
        "record_id": (
            _perturbation_id(plan.feature_index, sample.sample_id, "original", None, None, None, None)
            if spec is None else spec.perturbation_id
        ),
        "rendered_text": sample.text if spec is None else spec.rendered_text,
        "selection_used_activation": selection_used_activation,
        "special_tokens_mask": list(tokenized.special_tokens_mask),
        "split": sample.split,
        "start": None if spec is None else spec.start,
        "substring": sample.text if spec is None else spec.substring,
        "sufficiency": (
            sufficiency(
                float(original_signal["post_topk_activation"]),
                float(perturbed_signal["post_topk_activation"]),
            ) if kind == "sufficiency" else None
        ),
        "support_dropout": bool(original_signal["support_member"] and not perturbed_signal["support_member"]),
        "text_sha256": text_sha256(sample.text if spec is None else spec.rendered_text),
        "token_ids": list(tokenized.token_ids),
        "tokens": list(tokenized.tokens),
        "unit_id": sample.unit_id,
    }
    if set(record) != PERTURBATION_RECORD_KEYS:
        raise AssertionError("internal perturbation record schema drift")
    return record


def _specs(sample: PlannedSample, kind: str) -> list[PlannedPerturbation]:
    return [item for item in sample.perturbations if item.kind == kind]


def _children(
    sample: PlannedSample,
    parent_ids: set[str],
    kind: str,
) -> list[PlannedPerturbation]:
    return [
        item for item in sample.perturbations
        if item.kind == kind and item.parent_id in parent_ids
    ]


def _freeze_transfer_rule(
    selected: Sequence[tuple[PlannedSample, PlannedPerturbation]],
) -> dict[str, object]:
    if not selected:
        return {
            "normalized_token_pattern": [],
            "relative_position": 0.0,
            "tie_rule": "exact-pattern,overlap,relative-position,start",
            "token_width": 1,
        }
    candidates = []
    for sample, spec in selected:
        assert spec.start is not None and spec.end is not None
        pattern = _normalized_tokens(sample.tokenized, spec.start, spec.end)
        relative = spec.start / max(1, len(sample.tokenized.token_ids) - 1)
        candidates.append((pattern, spec.end - spec.start, relative))
    counts = Counter(pattern for pattern, _, _ in candidates)
    pattern = min(counts, key=lambda value: (-counts[value], len(value), value))
    matching = [item for item in candidates if item[0] == pattern]
    width = min((item[1] for item in matching), default=max(1, len(pattern)))
    relative = sum(item[2] for item in matching) / len(matching)
    return {
        "normalized_token_pattern": list(pattern),
        "relative_position": relative,
        "tie_rule": "exact-pattern,overlap,relative-position,start",
        "token_width": width,
    }


def apply_frozen_rule(
    rule: Mapping[str, object],
    tokenized: TokenizedText,
    phrase_local_candidates: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    pattern = tuple(str(value) for value in rule.get("normalized_token_pattern", ()))
    width = int(rule.get("token_width", max(1, len(pattern))))
    relative = float(rule.get("relative_position", 0.0))
    candidates = []
    seen: set[tuple[int, int]] = set()
    for candidate in phrase_local_candidates:
        if (
            not isinstance(candidate, tuple)
            or len(candidate) != 2
            or any(type(value) is not int for value in candidate)
        ):
            raise SpanAttributionError("phrase-local candidate schema is invalid")
        start, end = candidate
        if (start, end) in seen:
            continue
        seen.add((start, end))
        _validate_span(tokenized, start, end)
        if end - start != width:
            continue
        observed = _normalized_tokens(tokenized, start, end)
        exact = observed == pattern
        overlap = len(set(observed) & set(pattern)) / max(1, len(set(observed) | set(pattern)))
        distance = abs(start / max(1, len(tokenized.token_ids) - 1) - relative)
        candidates.append((not exact, -overlap, distance, start, end))
    if not candidates:
        raise SpanAttributionError("validation text has no span compatible with frozen rule")
    *_, start, end = min(candidates)
    return start, end


def matched_permutation_p_value(
    observed_by_text: Mapping[str, float],
    controls_by_text: Mapping[str, Sequence[float]],
    *,
    feature_index: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    eligible = sorted(
        key for key in observed_by_text
        if key in controls_by_text and len(controls_by_text[key]) > 0
    )
    if not eligible:
        return {"extreme_count": 0, "p_value": 1.0, "permutations": permutations, "status": "unevaluable"}
    observed = sum(float(observed_by_text[key]) for key in eligible) / len(eligible)
    seed_material = f"{seed}|{feature_index}|{'|'.join(eligible)}"
    rng = random.Random(int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16))
    extreme = 0
    for _ in range(permutations):
        null_mean = sum(
            float(controls_by_text[key][rng.randrange(len(controls_by_text[key]))])
            for key in eligible
        ) / len(eligible)
        extreme += null_mean >= observed
    return {
        "extreme_count": extreme,
        "p_value": (extreme + 1) / (permutations + 1),
        "permutations": permutations,
        "status": "evaluated",
    }


def execute_feature_plan(
    plan: FrozenFeaturePlan,
    embedder: object,
    sae: object,
    cache: EmbeddingCache,
    *,
    batch_size: int,
    budget: dict[str, object],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    selected_train: list[tuple[PlannedSample, PlannedPerturbation]] = []
    selected_records = {"train": [], "validation": []}
    top_phrase_ids: dict[str, list[str]] = {}
    text_embeddings: dict[str, torch.Tensor] = {}

    first_stage_texts = []
    for sample in plan.samples:
        first_stage_texts.append(sample.text)
        first_stage_texts.extend(spec.rendered_text for spec in _specs(sample, "phrase_mask"))
    text_embeddings.update(
        _encode_with_budget(first_stage_texts, embedder, cache, batch_size, budget)
    )
    for sample in plan.samples:
        original = text_embeddings[sample.text]
        records.append(
            _record_from_spec(
                plan=plan, sample=sample, spec=None,
                original_embedding=original, perturbed_embedding=original,
                sae=sae, selection_used_activation=False,
            )
        )
        phrase_records = []
        for spec in _specs(sample, "phrase_mask"):
            record = _record_from_spec(
                plan=plan, sample=sample, spec=spec,
                original_embedding=original,
                perturbed_embedding=text_embeddings[spec.rendered_text],
                sae=sae,
                selection_used_activation=sample.split == "train",
            )
            records.append(record)
            phrase_records.append((record, spec))
        if sample.split != "train":
            continue
        ranked = sorted(
            phrase_records,
            key=lambda item: (-float(item[0]["drop_fraction"]), item[1].phrase_index, item[1].perturbation_id),
        )[:2]
        top_ids = {spec.perturbation_id for _, spec in ranked}
        top_phrase_ids[sample.sample_id] = sorted(top_ids)
        windows = _children(sample, top_ids, "window_mask")
        lower_ids = {window.perturbation_id for window in windows}
        lower_specs = [
            spec for spec in sample.perturbations
            if spec in windows or spec.parent_id in lower_ids
        ]
        text_embeddings.update(
            _encode_with_budget(
                [spec.rendered_text for spec in lower_specs],
                embedder, cache, batch_size, budget,
            )
        )
        window_records = []
        for spec in lower_specs:
            record = _record_from_spec(
                plan=plan, sample=sample, spec=spec,
                original_embedding=original,
                perturbed_embedding=text_embeddings[spec.rendered_text],
                sae=sae,
                selection_used_activation=spec.kind == "window_mask",
            )
            records.append(record)
            if spec.kind == "window_mask":
                window_records.append((record, spec))
        if window_records:
            selected_record = select_shortest_explanatory_span(
                [
                    {"start": spec.start, "end": spec.end, "drop": record["drop_fraction"], "id": spec.perturbation_id}
                    for record, spec in window_records
                ]
            )
            chosen = next(spec for _, spec in window_records if spec.perturbation_id == selected_record["id"])
        elif ranked:
            chosen = ranked[0][1]
        else:
            continue
        selected_train.append((sample, chosen))
        selected_records["train"].append(
            next(record for record in records if record["record_id"] == chosen.perturbation_id)
        )

    rule = _freeze_transfer_rule(selected_train)
    validation_observed: dict[str, float] = {}
    validation_controls: dict[str, list[float]] = {}
    for sample in (item for item in plan.samples if item.split == "validation"):
        start, end = apply_frozen_rule(
            rule,
            sample.tokenized,
            sorted(
                {
                    (int(spec.start), int(spec.end))
                    for spec in _specs(sample, "window_mask")
                    if spec.start is not None and spec.end is not None
                }
            ),
        )
        matching = [
            spec for spec in _specs(sample, "window_mask")
            if spec.start == start and spec.end == end
        ]
        if not matching:
            raise SpanAttributionError("frozen validation span is outside planned candidate set")
        selected = min(matching, key=lambda spec: spec.perturbation_id)
        related = [
            spec for spec in sample.perturbations
            if spec.perturbation_id == selected.perturbation_id
            or spec.parent_id == selected.perturbation_id
        ]
        text_embeddings.update(
            _encode_with_budget(
                [spec.rendered_text for spec in related],
                embedder, cache, batch_size, budget,
            )
        )
        original = text_embeddings[sample.text]
        local_records = []
        has_controls = any(spec.kind == "random_control" for spec in related)
        for spec in related:
            record = _record_from_spec(
                plan=plan, sample=sample, spec=spec,
                original_embedding=original,
                perturbed_embedding=text_embeddings[spec.rendered_text],
                sae=sae,
                selection_used_activation=False,
                evaluable=has_controls,
            )
            records.append(record)
            local_records.append(record)
        selected_record = next(record for record in local_records if record["record_id"] == selected.perturbation_id)
        selected_records["validation"].append(selected_record)
        validation_observed[sample.text_sha256] = float(selected_record["drop_fraction"])
        validation_controls[sample.text_sha256] = [
            float(record["drop_fraction"])
            for record in local_records if record["kind"] == "random_control"
        ]

    permutation = matched_permutation_p_value(
        validation_observed,
        validation_controls,
        feature_index=plan.feature_index,
        permutations=plan.permutations,
        seed=plan.seed,
    )
    evaluable_selected = [
        record for record in selected_records["validation"] if record["evaluable"]
    ]
    drops = [float(record["drop_fraction"]) for record in evaluable_selected]
    controls_flat = [value for values in validation_controls.values() for value in values]
    mean_drop = sum(drops) / len(drops) if drops else 0.0
    dense_drops = [
        necessity(
            float(record["original_dense_preactivation"]),
            float(record["perturbed_dense_preactivation"]),
        ) for record in evaluable_selected
    ]
    deletion_by_parent = {
        record["parent_id"]: record for record in records
        if record["split"] == "validation" and record["kind"] == "deletion"
    }
    metrics = {
        "activation_drop_fraction": mean_drop,
        "dense_drop_fraction": sum(dense_drops) / len(dense_drops) if dense_drops else 0.0,
        "fdr": float(permutation["p_value"]),
        "mask_deletion_agreement": bool(evaluable_selected) and all(
            _direction(float(record["drop_fraction"]))
            == _direction(float(deletion_by_parent[record["record_id"]]["drop_fraction"]))
            for record in evaluable_selected
            if record["record_id"] in deletion_by_parent
        ),
        "permutation": permutation,
        "random_percentile": (
            sum(value < mean_drop for value in controls_flat) / len(controls_flat)
            if controls_flat else 0.0
        ),
        "support_dropout_fraction": (
            sum(bool(record["support_dropout"]) for record in evaluable_selected) / len(evaluable_selected)
            if evaluable_selected else 0.0
        ),
        "train_text_count": len(selected_records["train"]),
        "validation_direction_fraction": (
            sum(float(record["drop_fraction"]) > 0.0 for record in evaluable_selected) / len(evaluable_selected)
            if evaluable_selected else 0.0
        ),
        "validation_text_count": len(evaluable_selected),
    }
    grade_metrics = {key: value for key, value in metrics.items() if key != "permutation"}
    algorithm = {
        **rule,
        "permutations": plan.permutations,
        "phrase_top_k": 2,
        "seed": plan.seed,
        "shortest_drop_fraction": 0.8,
        "validation_selection_uses_activation": False,
        "window_lengths": list(WINDOW_LENGTHS),
    }
    records.sort(key=lambda record: str(record["record_id"]))
    return {
        "algorithm": algorithm,
        "budget": dict(budget),
        "evidence_grade": evidence_grade(grade_metrics),
        "feature_index": plan.feature_index,
        "metrics": metrics,
        "records": records,
        "search": {"top_two_phrase_ids": top_phrase_ids},
        "selected_records": selected_records,
    }


def attribute_feature_texts(
    feature_index: int,
    texts_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    embedder: ProTrekTextEmbedder,
    sae: object,
    cache: EmbeddingCache,
    *,
    batch_size: int = 64,
    seed: int = 42,
    max_encodings: int = 150_000,
    permutations: int = 1000,
) -> dict[str, object]:
    plan = build_feature_plan(
        feature_index, texts_by_split, embedder, seed=seed, permutations=permutations
    )
    budget = preflight_feature_plans([plan], cache, max_encodings=max_encodings)
    return execute_feature_plan(
        plan, embedder, sae, cache, batch_size=batch_size, budget=budget
    )


def select_feature_texts(row: Mapping[str, object], split: str) -> list[dict[str, object]]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    return _deduplicated_texts(row, split)


def _close(left: object, right: float, label: str) -> None:
    if type(left) is not float or not math.isfinite(left) or not math.isclose(
        left, right, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SpanAttributionError(f"{label} derived value does not recompute")


def _validate_record(
    record: object,
    *,
    feature_index: int,
    expected_top_texts: Mapping[str, Mapping[tuple[str, str], str]] | None,
) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != PERTURBATION_RECORD_KEYS:
        raise SpanAttributionError("perturbation record has invalid exact schema")
    if record.get("feature_index") != feature_index:
        raise SpanAttributionError("perturbation record feature link changed")
    split, accession, unit_id = record.get("split"), record.get("accession"), record.get("unit_id")
    if (
        split not in {"train", "validation"}
        or type(accession) is not str
        or not accession
        or type(unit_id) is not str
        or not unit_id
    ):
        raise SpanAttributionError("perturbation record identity is invalid")
    if expected_top_texts is not None:
        expected_text = expected_top_texts.get(str(split), {}).get((accession, unit_id))
        if expected_text is None:
            raise SpanAttributionError("perturbation record is not bound to a Task 8 top text")
        if record.get("original_text") != expected_text:
            raise SpanAttributionError("perturbation original text differs from exact Task 8 text")
    kind = record.get("kind")
    if kind not in {"original", "phrase_mask", "window_mask", "sufficiency", "deletion", "random_control"}:
        raise SpanAttributionError("perturbation kind is invalid")
    for name in ("original_text", "rendered_text", "substring"):
        if type(record.get(name)) is not str or (name != "rendered_text" and not record[name]):
            raise SpanAttributionError(f"perturbation {name} is invalid")
    if record.get("text_sha256") != text_sha256(str(record["rendered_text"])):
        raise SpanAttributionError("perturbation rendered-text SHA does not recompute")
    for name in ("record_id", "original_embedding_sha256", "perturbed_embedding_sha256"):
        value = record.get(name)
        if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise SpanAttributionError(f"perturbation {name} is invalid")
    token_ids, tokens = record.get("token_ids"), record.get("tokens")
    offsets = record.get("offsets")
    special = record.get("special_tokens_mask")
    attention = record.get("attention_mask")
    if (
        not isinstance(token_ids, list)
        or not isinstance(tokens, list)
        or not isinstance(offsets, list)
        or not isinstance(special, list)
        or not isinstance(attention, list)
        or not len(token_ids) == len(tokens) == len(offsets) == len(special) == len(attention)
        or not token_ids
    ):
        raise SpanAttributionError("perturbation token arrays are invalid")
    previous_end = 0
    for offset, is_special, attended in zip(offsets, special, attention):
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
            or type(is_special) is not int
            or is_special not in (0, 1)
            or attended != 1
        ):
            raise SpanAttributionError("perturbation offset/special/attention schema is invalid")
        start, end = offset
        if is_special:
            if offset != [0, 0]:
                raise SpanAttributionError("perturbation special offset is not exact")
        elif start < previous_end or end <= start or end > len(record["rendered_text"]):
            raise SpanAttributionError("perturbation offsets are not ordered/in bounds")
        else:
            previous_end = end
    original_post = record.get("original_post_topk_activation")
    perturbed_post = record.get("perturbed_post_topk_activation")
    original_dense = record.get("original_dense_preactivation")
    perturbed_dense = record.get("perturbed_dense_preactivation")
    for name, value in (
        ("original post-TopK", original_post), ("perturbed post-TopK", perturbed_post),
        ("original dense", original_dense), ("perturbed dense", perturbed_dense),
        ("embedding cosine", record.get("embedding_cosine")),
        ("embedding L2", record.get("embedding_l2")),
        ("drop fraction", record.get("drop_fraction")),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise SpanAttributionError(f"perturbation {name} is invalid")
    expected_drop = necessity(float(original_post), float(perturbed_post))
    _close(record["drop_fraction"], expected_drop, "drop fraction")
    expected_support_dropout = bool(record.get("original_support_member")) and not bool(
        record.get("perturbed_support_member")
    )
    if (
        type(record.get("original_support_member")) is not bool
        or type(record.get("perturbed_support_member")) is not bool
        or record.get("support_dropout") is not expected_support_dropout
        or record.get("direction") != _direction(expected_drop)
        or type(record.get("selection_used_activation")) is not bool
        or type(record.get("evaluable")) is not bool
    ):
        raise SpanAttributionError("perturbation support/direction derived fields do not recompute")
    if kind in {"phrase_mask", "window_mask", "random_control", "deletion"}:
        _close(record.get("necessity"), expected_drop, "necessity")
        if record.get("sufficiency") is not None:
            raise SpanAttributionError("non-sufficiency record has sufficiency value")
    elif kind == "sufficiency":
        if record.get("necessity") is not None:
            raise SpanAttributionError("sufficiency record has necessity value")
        _close(
            record.get("sufficiency"),
            sufficiency(float(original_post), float(perturbed_post)),
            "sufficiency",
        )
    elif record.get("necessity") is not None or record.get("sufficiency") is not None:
        raise SpanAttributionError("original record has intervention ratios")
    start, end = record.get("start"), record.get("end")
    if kind == "original":
        if start is not None or end is not None or record.get("parent_id") is not None:
            raise SpanAttributionError("original record intervention coordinates are invalid")
        _close(record["embedding_cosine"], 1.0, "original cosine")
        _close(record["embedding_l2"], 0.0, "original L2")
    elif type(start) is not int or type(end) is not int or end <= start:
        raise SpanAttributionError("intervention token coordinates are invalid")
    if kind == "random_control":
        if type(record.get("control_kind")) is not str or not str(record["control_kind"]).startswith("length_position:"):
            raise SpanAttributionError("random control matching stratum is invalid")
    elif record.get("control_kind") is not None:
        raise SpanAttributionError("non-control record has a control kind")
    return record


def _sample_id_from_record(record: Mapping[str, object]) -> str:
    material = (
        f"{record['feature_index']}|{record['split']}|{record['accession']}|"
        f"{record['unit_id']}|{text_sha256(str(record['original_text']))}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _rule_from_selected_records(
    records: Sequence[Mapping[str, object]],
    all_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if not records:
        return _freeze_transfer_rule([])
    candidates: list[tuple[tuple[str, ...], int, float]] = []
    originals = {
        _sample_id_from_record(record): record
        for record in (all_records or ())
        if record["kind"] == "original"
    }
    for record in records:
        start, end = int(record["start"]), int(record["end"])
        source = originals.get(_sample_id_from_record(record), record)
        tokens = source["tokens"]
        special = source["special_tokens_mask"]
        pattern = tuple(
            str(tokens[index]).lower().removeprefix("##")
            for index in range(start, end)
            if special[index] == 0
            and any(character.isalnum() for character in str(tokens[index]))
        )
        candidates.append(
            (pattern, end - start, start / max(1, len(source["token_ids"]) - 1))
        )
    counts = Counter(pattern for pattern, _, _ in candidates)
    pattern = min(counts, key=lambda value: (-counts[value], len(value), value))
    matching = [item for item in candidates if item[0] == pattern]
    return {
        "normalized_token_pattern": list(pattern),
        "relative_position": sum(item[2] for item in matching) / len(matching),
        "tie_rule": "exact-pattern,overlap,relative-position,start",
        "token_width": min(item[1] for item in matching),
    }


def _validate_record_graph(
    records: Sequence[Mapping[str, object]],
    by_id: Mapping[str, Mapping[str, object]],
) -> None:
    if [record["record_id"] for record in records] != sorted(by_id):
        raise SpanAttributionError("feature record ordering is not canonical")
    allowed_parent = {
        "original": None,
        "phrase_mask": None,
        "window_mask": "phrase_mask",
        "sufficiency": "window_mask",
        "deletion": "window_mask",
        "random_control": "window_mask",
    }
    for record in records:
        parent_id = record["parent_id"]
        expected_parent_kind = allowed_parent[str(record["kind"])]
        if expected_parent_kind is None:
            if parent_id is not None:
                raise SpanAttributionError("feature record parent graph is invalid")
        elif (
            type(parent_id) is not str
            or parent_id not in by_id
            or by_id[parent_id]["kind"] != expected_parent_kind
            or any(
                by_id[parent_id][name] != record[name]
                for name in ("feature_index", "split", "accession", "unit_id", "original_text")
            )
        ):
            raise SpanAttributionError("feature record parent graph is invalid")
        expected_id = _perturbation_id(
            int(record["feature_index"]),
            _sample_id_from_record(record),
            str(record["kind"]),
            record["start"],
            record["end"],
            record["control_kind"],
            parent_id,
        )
        if record["record_id"] != expected_id:
            raise SpanAttributionError("feature record identity does not reconstruct")


def _replay_conditional_execution(
    plan: FrozenFeaturePlan,
    by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected_ids: set[str] = set()
    executed_sha256: set[str] = set()
    selected_train: list[tuple[PlannedSample, PlannedPerturbation]] = []
    selected_train_ids: list[str] = []
    selected_validation_ids: list[str] = []
    top_phrase_ids: dict[str, list[str]] = {}

    for sample in plan.samples:
        original_id = _perturbation_id(
            plan.feature_index, sample.sample_id, "original", None, None, None, None
        )
        expected_ids.add(original_id)
        executed_sha256.add(sample.text_sha256)
        phrases = _specs(sample, "phrase_mask")
        expected_ids.update(spec.perturbation_id for spec in phrases)
        executed_sha256.update(text_sha256(spec.rendered_text) for spec in phrases)
        if any(spec.perturbation_id not in by_id for spec in phrases):
            raise SpanAttributionError("conditional execution lacks planned phrase records")
        if sample.split != "train":
            continue
        ranked = sorted(
            phrases,
            key=lambda spec: (
                -float(by_id[spec.perturbation_id]["drop_fraction"]),
                spec.phrase_index,
                spec.perturbation_id,
            ),
        )[:2]
        top_ids = {spec.perturbation_id for spec in ranked}
        top_phrase_ids[sample.sample_id] = sorted(top_ids)
        windows = _children(sample, top_ids, "window_mask")
        lower_ids = {spec.perturbation_id for spec in windows}
        lower_specs = [
            spec for spec in sample.perturbations
            if spec in windows or spec.parent_id in lower_ids
        ]
        expected_ids.update(spec.perturbation_id for spec in lower_specs)
        executed_sha256.update(text_sha256(spec.rendered_text) for spec in lower_specs)
        if any(spec.perturbation_id not in by_id for spec in windows):
            raise SpanAttributionError("conditional execution lacks planned training windows")
        if windows:
            selected = select_shortest_explanatory_span([
                {
                    "drop": by_id[spec.perturbation_id]["drop_fraction"],
                    "end": spec.end,
                    "id": spec.perturbation_id,
                    "start": spec.start,
                }
                for spec in windows
            ])
            chosen = next(
                spec for spec in windows if spec.perturbation_id == selected["id"]
            )
        elif ranked:
            chosen = ranked[0]
        else:
            continue
        selected_train.append((sample, chosen))
        selected_train_ids.append(chosen.perturbation_id)

    rule = _freeze_transfer_rule(selected_train)
    for sample in (item for item in plan.samples if item.split == "validation"):
        start, end = apply_frozen_rule(
            rule,
            sample.tokenized,
            sorted({
                (int(spec.start), int(spec.end))
                for spec in _specs(sample, "window_mask")
                if spec.start is not None and spec.end is not None
            }),
        )
        matching = [
            spec for spec in _specs(sample, "window_mask")
            if spec.start == start and spec.end == end
        ]
        if not matching:
            raise SpanAttributionError("conditional validation rule escaped frozen plan")
        chosen = min(matching, key=lambda spec: spec.perturbation_id)
        related = [
            spec for spec in sample.perturbations
            if spec.perturbation_id == chosen.perturbation_id
            or spec.parent_id == chosen.perturbation_id
        ]
        expected_ids.update(spec.perturbation_id for spec in related)
        executed_sha256.update(text_sha256(spec.rendered_text) for spec in related)
        selected_validation_ids.append(chosen.perturbation_id)

    planned_records: dict[str, dict[str, object]] = {}
    for sample in plan.samples:
        original_id = _perturbation_id(
            plan.feature_index, sample.sample_id, "original", None, None, None, None
        )
        if original_id in expected_ids:
            planned_records[original_id] = {
                "attention_mask": list(sample.tokenized.attention_mask),
                "end": None,
                "kind": "original",
                "offsets": [list(offset) for offset in sample.tokenized.offsets],
                "original_text": sample.text,
                "parent_id": None,
                "rendered_text": sample.text,
                "special_tokens_mask": list(sample.tokenized.special_tokens_mask),
                "start": None,
                "substring": sample.text,
                "token_ids": list(sample.tokenized.token_ids),
                "tokens": list(sample.tokenized.tokens),
            }
        for spec in sample.perturbations:
            if spec.perturbation_id not in expected_ids:
                continue
            planned_records[spec.perturbation_id] = {
                "attention_mask": list(spec.tokenized.attention_mask),
                "end": spec.end,
                "kind": spec.kind,
                "offsets": [list(offset) for offset in spec.tokenized.offsets],
                "original_text": sample.text,
                "parent_id": spec.parent_id,
                "rendered_text": spec.rendered_text,
                "special_tokens_mask": list(spec.tokenized.special_tokens_mask),
                "start": spec.start,
                "substring": spec.substring,
                "token_ids": list(spec.tokenized.token_ids),
                "tokens": list(spec.tokenized.tokens),
            }

    return {
        "executed_sha256": sorted(executed_sha256),
        "record_ids": expected_ids,
        "planned_records": planned_records,
        "rule": rule,
        "selected_train_ids": selected_train_ids,
        "selected_validation_ids": selected_validation_ids,
        "top_phrase_ids": top_phrase_ids,
    }


def validate_feature_payload(
    payload: object,
    *,
    expected_top_texts: Mapping[str, Mapping[tuple[str, str], str]] | None = None,
    expected_plan: FrozenFeaturePlan | None = None,
    expected_fdr: float | None = None,
    cache: EmbeddingCache | None = None,
    sae: object | None = None,
) -> dict[str, object]:
    exact = {
        "algorithm", "budget", "evidence_grade", "feature_index", "metrics",
        "records", "search", "selected_records",
    }
    if not isinstance(payload, dict) or set(payload) != exact:
        raise SpanAttributionError("feature payload has invalid exact schema")
    feature = payload.get("feature_index")
    if type(feature) is not int or feature < 0:
        raise SpanAttributionError("feature payload identity is invalid")
    algorithm = payload.get("algorithm")
    algorithm_keys = {
        "normalized_token_pattern", "permutations", "phrase_top_k", "relative_position",
        "seed", "shortest_drop_fraction", "tie_rule", "token_width",
        "validation_selection_uses_activation", "window_lengths",
    }
    if (
        not isinstance(algorithm, dict)
        or set(algorithm) != algorithm_keys
        or algorithm.get("phrase_top_k") != 2
        or algorithm.get("shortest_drop_fraction") != 0.8
        or algorithm.get("validation_selection_uses_activation") is not False
        or algorithm.get("window_lengths") != list(WINDOW_LENGTHS)
        or type(algorithm.get("permutations")) is not int
        or algorithm["permutations"] <= 0
        or type(algorithm.get("seed")) is not int
    ):
        raise SpanAttributionError("feature algorithm exact schema/domain is invalid")
    budget = payload.get("budget")
    if not isinstance(budget, dict) or set(budget) != {
        "allowed_sha256", "executed_sha256", "max_encodings",
        "projected_unique_uncached", "unique_allowed",
    }:
        raise SpanAttributionError("feature budget exact schema is invalid")
    if not set(budget["executed_sha256"]) <= set(budget["allowed_sha256"]):
        raise SpanAttributionError("feature execution escaped its frozen allowlist")
    for key in ("max_encodings", "projected_unique_uncached", "unique_allowed"):
        if type(budget.get(key)) is not int or budget[key] < 0:
            raise SpanAttributionError("feature budget count domain is invalid")
    if (
        budget["max_encodings"] <= 0
        or budget["unique_allowed"] != len(set(budget["allowed_sha256"]))
        or budget["projected_unique_uncached"] > budget["unique_allowed"]
        or len(set(budget["executed_sha256"])) != len(budget["executed_sha256"])
    ):
        raise SpanAttributionError("feature budget counts do not reconstruct")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SpanAttributionError("feature perturbation records are invalid")
    checked = [
        _validate_record(
            record, feature_index=feature, expected_top_texts=expected_top_texts
        ) for record in records
    ]
    if (cache is None) != (sae is None):
        raise SpanAttributionError("cache and SAE must be supplied together")
    if cache is not None and sae is not None:
        for record in checked:
            original_embedding = cache._load(str(record["original_text"]))
            perturbed_embedding = cache._load(str(record["rendered_text"]))
            if (
                record["original_embedding_sha256"] != _embedding_sha256(original_embedding)
                or record["perturbed_embedding_sha256"] != _embedding_sha256(perturbed_embedding)
            ):
                raise SpanAttributionError("record embedding SHA differs from registered cache")
            cosine, l2 = _embedding_metrics(original_embedding, perturbed_embedding)
            _close(record["embedding_cosine"], cosine, "cache embedding cosine")
            _close(record["embedding_l2"], l2, "cache embedding L2")
            original_signal = _signal_for(sae, original_embedding, feature)
            perturbed_signal = _signal_for(sae, perturbed_embedding, feature)
            _close(
                record["original_post_topk_activation"],
                float(original_signal["post_topk_activation"]),
                "SAE original post-TopK",
            )
            _close(
                record["perturbed_post_topk_activation"],
                float(perturbed_signal["post_topk_activation"]),
                "SAE perturbed post-TopK",
            )
            _close(
                record["original_dense_preactivation"],
                float(original_signal["relu_dense_preactivation"]),
                "SAE original dense",
            )
            _close(
                record["perturbed_dense_preactivation"],
                float(perturbed_signal["relu_dense_preactivation"]),
                "SAE perturbed dense",
            )
            if (
                record["original_support_member"] is not original_signal["support_member"]
                or record["perturbed_support_member"] is not perturbed_signal["support_member"]
            ):
                raise SpanAttributionError("record support differs from frozen SAE recomputation")
    by_id = {record["record_id"]: record for record in checked}
    if len(by_id) != len(checked):
        raise SpanAttributionError("feature perturbation record IDs are duplicated")
    _validate_record_graph(checked, by_id)
    replay = (
        _replay_conditional_execution(expected_plan, by_id)
        if expected_plan is not None
        else None
    )
    if replay is not None and set(by_id) != replay["record_ids"]:
        raise SpanAttributionError("conditional execution record family does not reconstruct")
    if replay is not None and any(
        any(record.get(name) != value for name, value in planned.items())
        for record_id, planned in replay["planned_records"].items()
        for record in (by_id[record_id],)
    ):
        raise SpanAttributionError("conditional record differs from exact planned span")
    executed_from_records = (
        replay["executed_sha256"]
        if replay is not None
        else sorted({text_sha256(str(record["rendered_text"])) for record in checked})
    )
    if budget["executed_sha256"] != executed_from_records:
        raise SpanAttributionError("feature budget executed set does not reconstruct from records")
    if expected_plan is not None:
        if (
            expected_plan.feature_index != feature
            or expected_plan.seed != algorithm["seed"]
            or expected_plan.permutations != algorithm["permutations"]
            or list(expected_plan.allowed_sha256) != budget["allowed_sha256"]
            or len(expected_plan.allowed_sha256) != budget["unique_allowed"]
        ):
            raise SpanAttributionError("feature budget/algorithm differs from frozen plan")
    selected = payload.get("selected_records")
    if not isinstance(selected, dict) or set(selected) != {"train", "validation"}:
        raise SpanAttributionError("feature selected record exact schema is invalid")
    for split, values in selected.items():
        if not isinstance(values, list):
            raise SpanAttributionError("feature selected records must be lists")
        for record in values:
            record_id = record.get("record_id") if isinstance(record, dict) else None
            if record_id not in by_id or by_id[record_id] != record or record.get("split") != split:
                raise SpanAttributionError("feature selected record does not link to canonical record")
            if split == "validation" and record.get("selection_used_activation") is not False:
                raise SpanAttributionError("validation selection used activation")
    if replay is not None and (
        [record["record_id"] for record in selected["train"]]
        != replay["selected_train_ids"]
        or [record["record_id"] for record in selected["validation"]]
        != replay["selected_validation_ids"]
    ):
        raise SpanAttributionError("conditional selected records do not reconstruct")
    expected_rule = (
        replay["rule"]
        if replay is not None
        else _rule_from_selected_records(selected["train"], checked)
    )
    for name, value in expected_rule.items():
        if algorithm.get(name) != value:
            raise SpanAttributionError("feature algorithm transfer rule does not reconstruct")
    metrics = payload.get("metrics")
    metric_keys = {
        "activation_drop_fraction", "dense_drop_fraction", "fdr",
        "mask_deletion_agreement", "permutation", "random_percentile",
        "support_dropout_fraction", "train_text_count",
        "validation_direction_fraction", "validation_text_count",
    }
    if not isinstance(metrics, dict) or set(metrics) != metric_keys:
        raise SpanAttributionError("feature metrics exact schema is invalid")
    validation = [record for record in selected["validation"] if record["evaluable"]]
    drops = [float(record["drop_fraction"]) for record in validation]
    mean_drop = sum(drops) / len(drops) if drops else 0.0
    _close(metrics["activation_drop_fraction"], mean_drop, "activation metric")
    dense = [
        necessity(float(record["original_dense_preactivation"]), float(record["perturbed_dense_preactivation"]))
        for record in validation
    ]
    _close(metrics["dense_drop_fraction"], sum(dense) / len(dense) if dense else 0.0, "dense metric")
    if metrics.get("train_text_count") != len(selected["train"]) or metrics.get("validation_text_count") != len(validation):
        raise SpanAttributionError("feature text-count metrics do not recompute")
    direction_fraction = sum(float(record["drop_fraction"]) > 0 for record in validation) / len(validation) if validation else 0.0
    _close(metrics["validation_direction_fraction"], direction_fraction, "direction metric")
    support_fraction = sum(bool(record["support_dropout"]) for record in validation) / len(validation) if validation else 0.0
    _close(metrics["support_dropout_fraction"], support_fraction, "support metric")
    controls_by_parent: dict[str, list[dict[str, object]]] = {}
    deletions = {}
    for record in checked:
        if record["kind"] == "random_control":
            controls_by_parent.setdefault(str(record["parent_id"]), []).append(record)
        elif record["kind"] == "deletion":
            deletions[record["parent_id"]] = record
    observed_by_text = {
        text_sha256(str(record["original_text"])): float(record["drop_fraction"])
        for record in validation
    }
    controls_by_text = {
        text_sha256(str(record["original_text"])): [
            float(control["drop_fraction"]) for control in controls_by_parent.get(str(record["record_id"]), [])
        ] for record in validation
    }
    expected_permutation = matched_permutation_p_value(
        observed_by_text, controls_by_text,
        feature_index=feature,
        permutations=int(algorithm["permutations"]),
        seed=int(algorithm["seed"]),
    )
    if metrics.get("permutation") != expected_permutation:
        raise SpanAttributionError("feature permutation result does not recompute")
    expected_fdr_value = expected_permutation["p_value"] if expected_fdr is None else expected_fdr
    _close(metrics["fdr"], float(expected_fdr_value), "FDR")
    controls_flat = [value for values in controls_by_text.values() for value in values]
    percentile = sum(value < mean_drop for value in controls_flat) / len(controls_flat) if controls_flat else 0.0
    _close(metrics["random_percentile"], percentile, "random percentile")
    agreement = bool(validation) and all(
        record["record_id"] in deletions
        and _direction(float(record["drop_fraction"]))
        == _direction(float(deletions[record["record_id"]]["drop_fraction"]))
        for record in validation
    )
    if metrics.get("mask_deletion_agreement") is not agreement:
        raise SpanAttributionError("mask/deletion agreement does not recompute")
    grade_metrics = {key: value for key, value in metrics.items() if key != "permutation"}
    if payload.get("evidence_grade") != evidence_grade(grade_metrics):
        raise SpanAttributionError("feature evidence grade does not recompute")
    search = payload.get("search")
    if not isinstance(search, dict) or set(search) != {"top_two_phrase_ids"}:
        raise SpanAttributionError("feature search exact schema is invalid")
    top_two = search["top_two_phrase_ids"]
    if not isinstance(top_two, dict):
        raise SpanAttributionError("feature search phrase map is invalid")
    if expected_plan is not None:
        if top_two != replay["top_phrase_ids"]:
            raise SpanAttributionError("feature search top phrase graph does not reconstruct")
    else:
        phrase_ids = {
            str(record["record_id"]) for record in checked if record["kind"] == "phrase_mask"
        }
        if any(
            type(sample_id) is not str
            or not isinstance(ids, list)
            or ids != sorted(set(ids))
            or len(ids) > 2
            or not set(ids) <= phrase_ids
            for sample_id, ids in top_two.items()
        ):
            raise SpanAttributionError("feature search phrase links are invalid")
    return payload


STAGE_MANIFEST_KEYS = frozenset(
    {
        "bindings", "budget", "cache", "complete", "completed_features", "features",
        "final_outputs", "provisional_outputs", "schema_version",
    }
)


def new_stage_manifest(
    bindings: Mapping[str, object],
    *,
    features: Sequence[int],
    budget: Mapping[str, object],
) -> dict[str, object]:
    required_bindings = {
        "config_fingerprint", "source_binding", "task4_binding", "task8_binding",
        "tokenizer_binding", "device_identity", "model_binding", "test_fixture_unbound"
    }
    if set(bindings) != required_bindings:
        raise SpanAttributionError("stage bindings have invalid exact schema")
    if any(type(feature) is not int or feature < 0 for feature in features) or list(features) != sorted(set(features)):
        raise SpanAttributionError("stage feature identities are invalid")
    return {
        "bindings": json.loads(json.dumps(bindings, sort_keys=True)),
        "budget": json.loads(json.dumps(budget, sort_keys=True)),
        "cache": [],
        "complete": False,
        "completed_features": [],
        "features": list(features),
        "final_outputs": [],
        "provisional_outputs": [],
        "schema_version": SCHEMA_VERSION,
    }


def publish_incremental_manifest(output_dir: Path, manifest: Mapping[str, object]) -> None:
    if set(manifest) != STAGE_MANIFEST_KEYS:
        raise SpanAttributionError("incremental manifest has invalid exact schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "manifest.json", dict(manifest))


def _output_metadata(path: Path, feature_index: int) -> dict[str, object]:
    metadata = file_metadata(path)
    return {
        "bytes": metadata["bytes"],
        "feature_index": feature_index,
        "path": path.name,
        "sha256": metadata["sha256"],
    }


def publish_provisional_feature(
    output_dir: Path,
    manifest: Mapping[str, object],
    payload: Mapping[str, object],
) -> dict[str, object]:
    if manifest.get("complete") is not False:
        raise SpanAttributionError("cannot append provisional output to complete stage")
    feature = payload.get("feature_index")
    if type(feature) is not int or feature not in manifest.get("features", ()):
        raise SpanAttributionError("provisional feature is outside frozen family")
    if feature in manifest.get("completed_features", ()):
        raise SpanAttributionError("provisional feature is already registered")
    validate_feature_payload(payload)
    path = output_dir / f"feature_{feature:04d}.raw.json"
    write_json_atomic(path, dict(payload))
    updated = json.loads(json.dumps(manifest, sort_keys=True))
    updated["completed_features"] = sorted([*updated["completed_features"], feature])
    updated["provisional_outputs"] = sorted(
        [*updated["provisional_outputs"], _output_metadata(path, feature)],
        key=lambda item: item["feature_index"],
    )
    publish_incremental_manifest(output_dir, updated)
    return updated


def _final_payload(raw: Mapping[str, object], q_value: float) -> dict[str, object]:
    final = json.loads(json.dumps(raw, sort_keys=True))
    final["metrics"]["fdr"] = q_value
    grade_metrics = {
        key: value for key, value in final["metrics"].items() if key != "permutation"
    }
    final["evidence_grade"] = evidence_grade(grade_metrics)
    validate_feature_payload(final, expected_fdr=q_value)
    return final


def finalize_feature_family(
    output_dir: Path,
    manifest: Mapping[str, object],
    raw_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Write unregistered final views, then register all of them data-last."""

    if manifest.get("complete") is not False:
        raise SpanAttributionError("family finalization requires an incomplete manifest")
    features = list(manifest.get("features", ()))
    raw_by_feature = {int(payload["feature_index"]): payload for payload in raw_payloads}
    if sorted(raw_by_feature) != features or manifest.get("completed_features") != features:
        raise SpanAttributionError("family finalization lacks immutable provisional outputs")
    q_values = benjamini_hochberg(
        [float(raw_by_feature[feature]["metrics"]["permutation"]["p_value"]) for feature in features]
    )
    final_metadata = []
    try:
        for feature, q_value in zip(features, q_values):
            expected = _final_payload(raw_by_feature[feature], q_value)
            path = output_dir / f"feature_{feature:04d}.json"
            if path.exists():
                observed = strict_json_load(path)
                if observed != expected:
                    raise SpanAttributionError(
                        f"unregistered final feature {feature} is not adoptable"
                    )
            else:
                write_json_atomic(path, expected)
            final_metadata.append(_output_metadata(path, feature))
    except SpanAttributionError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise SpanAttributionError(f"family final publication interrupted: {error}") from error
    complete = json.loads(json.dumps(manifest, sort_keys=True))
    complete["complete"] = True
    complete["final_outputs"] = final_metadata
    publish_incremental_manifest(output_dir, complete)
    return complete


def validate_stage_bindings(
    manifest: Mapping[str, object], current_bindings: Mapping[str, object]
) -> None:
    recorded = manifest.get("bindings")
    if not isinstance(recorded, dict):
        raise SpanAttributionError("stage binding is missing")
    labels = {
        "config_fingerprint": "configuration binding",
        "source_binding": "Task 1 binding",
        "task4_binding": "Task 4 binding",
        "task8_binding": "Task 8 binding",
        "tokenizer_binding": "tokenizer binding",
        "device_identity": "device binding",
        "model_binding": "ProTrek model binding",
        "test_fixture_unbound": "test fixture binding gate",
    }
    for name, label in labels.items():
        if recorded.get(name) != current_bindings.get(name):
            raise SpanAttributionError(f"{label} changed on resume")


def adopt_registered_cache(
    output_dir: Path,
    manifest: Mapping[str, object],
    allowed_sha256: set[str],
) -> dict[str, object]:
    updated = json.loads(json.dumps(manifest, sort_keys=True))
    cache_dir = output_dir / "cache"
    registrations = []
    bindings = manifest.get("bindings")
    model_binding = bindings.get("model_binding") if isinstance(bindings, dict) else None
    model_fingerprint = (
        model_binding.get("fingerprint") if isinstance(model_binding, dict) else None
    )
    if type(model_fingerprint) is not str:
        raise SpanAttributionError("stage lacks a cache model binding")
    if cache_dir.exists():
        for path in sorted(cache_dir.iterdir()):
            if not path.is_file() or path.suffix != ".pt" or path.stem not in allowed_sha256:
                raise SpanAttributionError("cache contains an unallowlisted artifact")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise SpanAttributionError(f"invalid cache artifact: {path}") from error
            embedding = payload.get("embedding") if isinstance(payload, dict) else None
            if (
                not isinstance(payload, dict)
                or set(payload) != {
                    "embedding", "model_binding_fingerprint", "text_sha256"
                }
                or payload.get("text_sha256") != path.stem
                or payload.get("model_binding_fingerprint") != model_fingerprint
                or not isinstance(embedding, torch.Tensor)
                or embedding.dtype != torch.float32
                or embedding.ndim != 1
                or not bool(torch.isfinite(embedding).all())
            ):
                raise SpanAttributionError("cache artifact schema/content is invalid")
            metadata = file_metadata(path)
            metadata["path"] = f"cache/{path.name}"
            metadata["model_binding_fingerprint"] = model_fingerprint
            metadata["text_sha256"] = path.stem
            registrations.append(metadata)
    updated["cache"] = registrations
    return updated


def replace_registered_outputs(
    output_dir: Path,
    manifest: Mapping[str, object],
    replacements: Mapping[str, Path],
) -> dict[str, object]:
    output_groups = ("provisional_outputs", "final_outputs")
    registered = {
        item.get("path")
        for group in output_groups
        for item in manifest.get(group, ())
        if isinstance(item, dict)
    }
    if any(path not in registered for path in replacements):
        raise SpanAttributionError("overwrite requested an unregistered output")
    recovery = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.recovery-", dir=output_dir.parent)
    )
    replaced: list[str] = []
    preserve = False
    try:
        for relative, source in replacements.items():
            target = output_dir / relative
            if not target.is_file() or not source.is_file():
                raise SpanAttributionError("registered overwrite input is missing")
            shutil.copy2(target, recovery / relative)
            temporary = output_dir / f".{Path(relative).name}.replace-{os.getpid()}"
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            replaced.append(relative)
    except BaseException as publish_error:
        try:
            for relative in reversed(replaced):
                os.replace(recovery / relative, output_dir / relative)
        except BaseException as restore_error:
            preserve = True
            raise SpanAttributionError(
                "registered overwrite and rollback failed; "
                f"recovery: {recovery}; publish error: {publish_error}; restore error: {restore_error}"
            ) from restore_error
        if isinstance(publish_error, (KeyboardInterrupt, SystemExit, SpanAttributionError)):
            raise
        raise SpanAttributionError(f"registered overwrite failed: {publish_error}") from publish_error
    finally:
        if not preserve:
            shutil.rmtree(recovery, ignore_errors=True)
    updated = json.loads(json.dumps(manifest, sort_keys=True))
    output_by_path = {
        item["path"]: item
        for group in output_groups
        for item in updated[group]
        if isinstance(item, dict)
    }
    for relative in replacements:
        metadata = file_metadata(output_dir / relative)
        output_by_path[relative].update(
            {"bytes": metadata["bytes"], "sha256": metadata["sha256"]}
        )
    return updated


def _replace_complete_stage(
    output_dir: Path,
    old_manifest: Mapping[str, object],
    replacements: Mapping[str, Path],
    new_manifest: Mapping[str, object],
) -> None:
    registered = {
        item["path"]
        for group in ("provisional_outputs", "final_outputs")
        for item in old_manifest.get(group, ())
        if isinstance(item, dict)
    }
    if set(replacements) != registered:
        raise SpanAttributionError("complete overwrite replacement allowlist changed")
    recovery = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.overwrite-recovery-", dir=output_dir.parent)
    )
    replaced: list[str] = []
    manifest_replaced = False
    preserve = False
    try:
        shutil.copy2(output_dir / "manifest.json", recovery / "manifest.json")
        for relative in sorted(registered):
            source, target = replacements[relative], output_dir / relative
            shutil.copy2(target, recovery / relative)
            temporary = output_dir / f".{Path(relative).name}.overwrite-{os.getpid()}"
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            replaced.append(relative)
        staged_manifest = output_dir / f".manifest.overwrite-{os.getpid()}.json"
        write_json_atomic(staged_manifest, dict(new_manifest))
        os.replace(staged_manifest, output_dir / "manifest.json")
        manifest_replaced = True
    except BaseException as publish_error:
        try:
            if manifest_replaced or (recovery / "manifest.json").exists():
                shutil.copy2(recovery / "manifest.json", output_dir / ".manifest.restore")
                os.replace(output_dir / ".manifest.restore", output_dir / "manifest.json")
            for relative in reversed(replaced):
                shutil.copy2(recovery / relative, output_dir / f".{Path(relative).name}.restore")
                os.replace(
                    output_dir / f".{Path(relative).name}.restore", output_dir / relative
                )
        except BaseException as restore_error:
            preserve = True
            raise SpanAttributionError(
                "complete overwrite and rollback failed; "
                f"recovery: {recovery}; publish error: {publish_error}; restore error: {restore_error}"
            ) from restore_error
        if isinstance(publish_error, (KeyboardInterrupt, SystemExit, SpanAttributionError)):
            raise
        raise SpanAttributionError(f"complete registered overwrite failed: {publish_error}") from publish_error
    finally:
        if not preserve:
            shutil.rmtree(recovery, ignore_errors=True)


def _tokenizer_files_binding(
    paths: AtlasPaths,
    embedder: object,
    model_binding: Mapping[str, object],
    *,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    # AtlasPaths does not own ProTrek weights; the exact path is exposed by the
    # embedder provenance and must resolve to a local frozen directory.
    provenance = getattr(embedder, "provenance", None)
    protrek = provenance.get("protrek") if isinstance(provenance, dict) else None
    raw = protrek.get("text_config") if isinstance(protrek, dict) else None
    if allow_test_fixture_unbound and model_binding == {"fingerprint": "fixture-unbound"}:
        return {
            "fast_class": type(getattr(embedder, "_offset_tokenizer", None)).__name__,
            "fixture_unbound": True,
            "slow_class": type(
                getattr(getattr(getattr(embedder, "model", None), "text_encoder", None), "tokenizer", None)
            ).__name__,
            "text_config_tree_sha256": "fixture-unbound",
        }
    if type(raw) is not str or raw == "unknown":
        raise SpanAttributionError("embedder lacks exact tokenizer text-config provenance")
    directory = Path(raw)
    if not directory.is_dir():
        raise SpanAttributionError("frozen tokenizer text-config directory is missing")
    files = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            continue
        metadata = file_metadata(path)
        files[path.name] = {"bytes": metadata["bytes"], "sha256": metadata["sha256"]}
    required = {"vocab.txt", "tokenizer_config.json"}
    if not required <= set(files):
        raise SpanAttributionError("frozen tokenizer files are incomplete")
    fast = getattr(embedder, "_offset_tokenizer", None)
    slow = getattr(getattr(getattr(embedder, "model", None), "text_encoder", None), "tokenizer", None)
    if fast is None or slow is None:
        raise SpanAttributionError("embedder tokenizer adapters are unavailable")
    return {
        "fast_class": type(fast).__name__,
        "files": files,
        "slow_class": type(slow).__name__,
        "text_config": str(directory.resolve()),
        "text_config_tree_sha256": model_binding["text_config"]["tree_sha256"],
    }


def _validate_production_context(
    selection_path: Path,
    atlas_path: Path,
    units_path: Path,
    inference_path: Path,
    *,
    device: str,
    config: FeatureAtlasConfig,
    atlas_paths: AtlasPaths | None,
    sparse_activation_dir: Path | None,
    feature_stats_path: Path | None,
    embedder: ProTrekTextEmbedder | None,
    sae: object | None,
    allow_test_fixture_unbound: bool,
) -> dict[str, object]:
    """Run the complete Task 1/4/8, tokenizer, model, and device gate."""

    try:
        selection_raw = strict_json_load(selection_path)
        binding = _source_binding(selection_raw)
        paths = atlas_paths or AtlasPaths(units_path=units_path)
        validate_source_binding(binding, paths)
        selection = validate_selection_manifest(
            selection_path,
            atlas_path,
            inference_path,
            config,
            provenance=binding,
        )
        rows = _load_atlas_rows(atlas_path)
        features = _selected_features(selection)
        if sparse_activation_dir is None:
            if not rows:
                raise SpanAttributionError("frozen atlas has no rows")
            upstream = rows[min(rows)].get("upstream_artifacts")
            sparse_metadata = upstream.get("sparse_manifest") if isinstance(upstream, dict) else None
            sparse_manifest_path = sparse_metadata.get("path") if isinstance(sparse_metadata, dict) else None
            if type(sparse_manifest_path) is not str:
                raise SpanAttributionError("Task 8 atlas lacks Task 4 sparse-manifest binding")
            sparse_activation_dir = Path(sparse_manifest_path).parent
        feature_stats_path = feature_stats_path or paths.run_dir / "feature_stats.pt"
        source_index = build_data_index(paths.units_path, paths.text_manifest_path, paths.activation_dir)
        validate_sparse_feature_activations(
            source_index,
            inference_path,
            feature_stats_path,
            sparse_activation_dir,
            atlas_paths=paths,
            source_binding=binding,
        )
        resolved_device, device_identity = _resolve_device_identity(device)
        if embedder is None:
            embedder, _ = load_text_embedder(device=str(resolved_device))
        if sae is None:
            sae = load_frozen_sae(inference_path, resolved_device)
        model_binding = build_protrek_model_binding(
            embedder, allow_test_fixture_unbound=allow_test_fixture_unbound
        )
        if model_binding != {"fingerprint": "fixture-unbound"}:
            try:
                embedder_device = torch.device(getattr(embedder, "device"))
                sae_device = next(sae.parameters()).device
            except (AttributeError, StopIteration, TypeError, RuntimeError) as error:
                raise SpanAttributionError("model runtime device identity is unavailable") from error
            if embedder_device != resolved_device or sae_device != resolved_device:
                raise SpanAttributionError("model runtime device differs from requested device")
        tokenizer_binding = _tokenizer_files_binding(
            paths,
            embedder,
            model_binding,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
        )
        sparse_manifest = sparse_activation_dir / "manifest.json"
        task4_binding = {
            "feature_stats_sha256": sha256_file(feature_stats_path),
            "inference_sha256": sha256_file(inference_path),
            "sparse_manifest_sha256": sha256_file(sparse_manifest),
        }
        task8_binding = {
            "atlas_sha256": sha256_file(atlas_path),
            "selection_fingerprint": selection["selection_fingerprint"],
            "selection_sha256": sha256_file(selection_path),
        }
        bindings = {
            "config_fingerprint": config.fingerprint(),
            "device_identity": device_identity,
            "source_binding": asdict(binding),
            "task4_binding": task4_binding,
            "task8_binding": task8_binding,
            "tokenizer_binding": tokenizer_binding,
            "model_binding": model_binding,
            "test_fixture_unbound": allow_test_fixture_unbound,
        }
    except SpanAttributionError:
        raise
    except Exception as error:
        raise SpanAttributionError(f"Task 1/4/8/tokenizer/device gate failed: {error}") from error
    return {
        "bindings": bindings,
        "embedder": embedder,
        "features": features,
        "rows": rows,
        "sae": sae,
    }


def _feature_expected_top_texts(
    row: Mapping[str, object],
) -> dict[str, dict[tuple[str, str], str]]:
    return {
        split: {
            (str(item["accession"]), str(item["unit_id"])): str(item["text"])
            for item in select_feature_texts(row, split)
        }
        for split in ("train", "validation")
    }


def _validate_registered_stage(
    output_dir: Path,
    manifest: Mapping[str, object],
    *,
    rows: Mapping[int, Mapping[str, object]],
    sae: object | None = None,
    plans_by_feature: Mapping[int, FrozenFeaturePlan] | None = None,
) -> list[dict[str, object]]:
    if set(manifest) != STAGE_MANIFEST_KEYS or manifest.get("schema_version") != SCHEMA_VERSION:
        raise SpanAttributionError("stage manifest has invalid exact schema")
    features = manifest.get("features")
    completed = manifest.get("completed_features")
    provisional_outputs = manifest.get("provisional_outputs")
    final_outputs = manifest.get("final_outputs")
    if (
        not isinstance(features, list)
        or features != sorted(set(features))
        or not isinstance(completed, list)
        or completed != sorted(set(completed))
        or not set(completed) <= set(features)
        or not isinstance(provisional_outputs, list)
        or len(provisional_outputs) != len(completed)
        or not isinstance(final_outputs, list)
        or (manifest.get("complete") is False and final_outputs)
        or (manifest.get("complete") is True and len(final_outputs) != len(features))
    ):
        raise SpanAttributionError("stage feature registration is invalid")
    raw_by_feature = {}
    bindings = manifest.get("bindings")
    model_binding = bindings.get("model_binding") if isinstance(bindings, dict) else None
    model_fingerprint = (
        model_binding.get("fingerprint") if isinstance(model_binding, dict) else None
    )
    validation_cache = (
        EmbeddingCache(
            output_dir / "cache", model_binding_fingerprint=str(model_fingerprint)
        )
        if sae is not None
        else None
    )
    for metadata in provisional_outputs:
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "feature_index", "path", "sha256"}:
            raise SpanAttributionError("stage output registration schema is invalid")
        feature = metadata["feature_index"]
        if feature in raw_by_feature or metadata["path"] != f"feature_{feature:04d}.raw.json":
            raise SpanAttributionError("stage output registration identity is invalid")
        path = output_dir / metadata["path"]
        observed = _output_metadata(path, feature)
        if observed != metadata:
            raise SpanAttributionError("stage feature output checksum changed")
        payload = strict_json_load(path)
        raw_by_feature[feature] = payload
    if sorted(raw_by_feature) != completed:
        raise SpanAttributionError("stage completed features and outputs disagree")
    for feature in completed:
        if feature not in rows:
            raise SpanAttributionError("stage output feature is absent from Task 8 atlas")
        validate_feature_payload(
            raw_by_feature[feature],
            expected_top_texts=_feature_expected_top_texts(rows[feature]),
            expected_plan=(plans_by_feature or {}).get(feature),
            expected_fdr=float(raw_by_feature[feature]["metrics"]["permutation"]["p_value"]),
            cache=validation_cache,
            sae=sae,
        )
    recorded_budget = manifest.get("budget")
    if not isinstance(recorded_budget, dict):
        raise SpanAttributionError("stage family budget is invalid")
    frozen_budget = json.loads(json.dumps(recorded_budget, sort_keys=True))
    frozen_budget["executed_sha256"] = []
    if _reconciled_family_budget(
        frozen_budget, [raw_by_feature[feature] for feature in completed]
    ) != recorded_budget:
        raise SpanAttributionError("stage family budget does not reconcile from raw features")
    if manifest.get("complete") is True and completed != features:
        raise SpanAttributionError("complete stage is missing selected features")
    final_payloads = []
    if manifest.get("complete") is True:
        q_values = benjamini_hochberg([
            float(raw_by_feature[feature]["metrics"]["permutation"]["p_value"])
            for feature in features
        ])
        for feature, q_value, metadata in zip(features, q_values, final_outputs):
            if (
                not isinstance(metadata, dict)
                or metadata.get("feature_index") != feature
                or metadata.get("path") != f"feature_{feature:04d}.json"
                or _output_metadata(output_dir / metadata["path"], feature) != metadata
            ):
                raise SpanAttributionError("final feature registration is invalid")
            payload = strict_json_load(output_dir / metadata["path"])
            if payload != _final_payload(raw_by_feature[feature], q_value):
                raise SpanAttributionError("final payload differs from immutable provisional evidence")
            validate_feature_payload(
                payload,
                expected_top_texts=_feature_expected_top_texts(rows[feature]),
                expected_plan=(plans_by_feature or {}).get(feature),
                expected_fdr=q_value,
                cache=validation_cache,
                sae=sae,
            )
            final_payloads.append(payload)
    budget = manifest.get("budget")
    allowed = set(budget.get("allowed_sha256", ())) if isinstance(budget, dict) else set()
    adopted = adopt_registered_cache(output_dir, manifest, allowed)
    if adopted["cache"] != manifest.get("cache"):
        raise SpanAttributionError("stage cache registry differs from validated cache files")
    return final_payloads if manifest.get("complete") else list(raw_by_feature.values())


def _overwrite_complete_family(
    output_dir: Path,
    old_manifest: Mapping[str, object],
    *,
    bindings: Mapping[str, object],
    plans: Sequence[FrozenFeaturePlan],
    rows: Mapping[int, Mapping[str, object]],
    embedder: object,
    sae: object,
    cache: EmbeddingCache,
    budget: Mapping[str, object],
    batch_size: int,
) -> dict[str, object]:
    """Recompute a complete family, then replace its registered files atomically."""

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.overwrite-stage-", dir=output_dir.parent))
    try:
        replacement = new_stage_manifest(
            bindings,
            features=[plan.feature_index for plan in plans],
            budget=budget,
        )
        replacement["cache"] = json.loads(json.dumps(old_manifest["cache"], sort_keys=True))
        publish_incremental_manifest(staging, replacement)
        raw_payloads = []
        for plan in plans:
            local_budget = preflight_feature_plans(
                [plan], cache, max_encodings=int(budget["max_encodings"])
            )
            payload = execute_feature_plan(
                plan,
                embedder,
                sae,
                cache,
                batch_size=batch_size,
                budget=local_budget,
            )
            validate_feature_payload(
                payload,
                expected_top_texts=_feature_expected_top_texts(rows[plan.feature_index]),
                expected_plan=plan,
            )
            replacement["budget"] = _reconciled_family_budget(
                budget, [*raw_payloads, payload]
            )
            replacement = publish_provisional_feature(staging, replacement, payload)
            raw_payloads.append(payload)
        replacement = finalize_feature_family(staging, replacement, raw_payloads)
        replacements = {
            item["path"]: staging / item["path"]
            for group in ("provisional_outputs", "final_outputs")
            for item in replacement[group]
        }
        _replace_complete_stage(output_dir, old_manifest, replacements, replacement)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    installed = strict_json_load(output_dir / "manifest.json")
    _validate_registered_stage(
        output_dir,
        installed,
        rows=rows,
        sae=sae,
        plans_by_feature={plan.feature_index: plan for plan in plans},
    )
    return installed


def attribute_feature_spans(
    selection_path: Path,
    atlas_path: Path,
    units_path: Path,
    inference_path: Path,
    output_dir: Path,
    *,
    device: str,
    batch_size: int = 64,
    overwrite: bool = False,
    config: FeatureAtlasConfig | None = None,
    atlas_paths: AtlasPaths | None = None,
    sparse_activation_dir: Path | None = None,
    feature_stats_path: Path | None = None,
    embedder: ProTrekTextEmbedder | None = None,
    sae: object | None = None,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    """Run a binding-first, preflight-first, incrementally registered stage."""

    if not all(isinstance(path, Path) for path in (selection_path, atlas_path, units_path, inference_path, output_dir)):
        raise TypeError("span attribution paths must be pathlib.Path values")
    if (
        type(overwrite) is not bool
        or type(allow_test_fixture_unbound) is not bool
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise TypeError("overwrite/batch_size have invalid exact types")
    config = config or FeatureAtlasConfig()
    # This gate intentionally precedes every resume/overwrite return path.
    context = _validate_production_context(
        selection_path,
        atlas_path,
        units_path,
        inference_path,
        device=device,
        config=config,
        atlas_paths=atlas_paths,
        sparse_activation_dir=sparse_activation_dir,
        feature_stats_path=feature_stats_path,
        embedder=embedder,
        sae=sae,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    bindings = context["bindings"]
    rows = context["rows"]
    features = context["features"]
    embedder = context["embedder"]
    sae = context["sae"]
    plans = [
        build_feature_plan(
            feature,
            {
                split: select_feature_texts(rows[feature], split)
                for split in ("train", "validation")
            },
            embedder,
            seed=config.seed,
            permutations=config.permutations,
        )
        for feature in features
    ]
    cache = EmbeddingCache(
        output_dir / "cache",
        model_binding_fingerprint=bindings["model_binding"]["fingerprint"],
    )
    # No output/cache directory is created before this complete worst-case plan passes.
    budget = preflight_feature_plans(
        plans, cache, max_encodings=config.max_span_encodings
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = strict_json_load(manifest_path)
        except (AtlasIOError, OSError, ValueError) as error:
            raise SpanAttributionError("invalid existing span manifest") from error
        validate_stage_bindings(manifest, bindings)
        if manifest.get("features") != features:
            raise SpanAttributionError("frozen selected feature family changed")
        recorded_budget = manifest.get("budget")
        if not isinstance(recorded_budget, dict) or recorded_budget.get("allowed_sha256") != budget["allowed_sha256"]:
            raise SpanAttributionError("frozen feature-aware preflight plan changed")
        family_budget = json.loads(json.dumps(recorded_budget, sort_keys=True))
        _validate_registered_stage(
            output_dir,
            manifest,
            rows=rows,
            sae=sae,
            plans_by_feature={plan.feature_index: plan for plan in plans},
        )
        if manifest.get("complete") is True and not overwrite:
            return manifest
        if manifest.get("complete") is True and overwrite:
            return _overwrite_complete_family(
                output_dir,
                manifest,
                bindings=bindings,
                plans=plans,
                rows=rows,
                embedder=embedder,
                sae=sae,
                cache=cache,
                budget=budget,
                batch_size=batch_size,
            )
    else:
        if output_dir.exists() and any(
            path.name.startswith("feature_") or path.name == "cache"
            for path in output_dir.iterdir()
        ):
            raise SpanAttributionError("unregistered span artifacts cannot be adopted without a manifest")
        family_budget = dict(budget)
        manifest = new_stage_manifest(bindings, features=features, budget=family_budget)
        publish_incremental_manifest(output_dir, manifest)
    manifest = adopt_registered_cache(output_dir, manifest, set(budget["allowed_sha256"]))
    publish_incremental_manifest(output_dir, manifest)
    completed = set(manifest["completed_features"])
    for plan in plans:
        if plan.feature_index in completed:
            continue
        local_budget = preflight_feature_plans(
            [plan], cache, max_encodings=config.max_span_encodings
        )
        result = execute_feature_plan(
            plan,
            embedder,
            sae,
            cache,
            batch_size=batch_size,
            budget=local_budget,
        )
        validate_feature_payload(
            result,
            expected_top_texts=_feature_expected_top_texts(rows[plan.feature_index]),
            expected_plan=plan,
        )
        manifest = adopt_registered_cache(
            output_dir, manifest, set(budget["allowed_sha256"])
        )
        registered_payloads = [
            strict_json_load(output_dir / item["path"])
            for item in manifest["provisional_outputs"]
        ]
        manifest["budget"] = _reconciled_family_budget(
            family_budget, [*registered_payloads, result]
        )
        manifest = publish_provisional_feature(output_dir, manifest, result)
        completed.add(plan.feature_index)
    payloads = [
        strict_json_load(output_dir / f"feature_{feature:04d}.raw.json")
        for feature in features
    ]
    manifest = finalize_feature_family(output_dir, manifest, payloads)
    _validate_registered_stage(
        output_dir,
        manifest,
        rows=rows,
        sae=sae,
        plans_by_feature={plan.feature_index: plan for plan in plans},
    )
    return manifest


def validate_span_attribution(
    output_dir: Path,
    *,
    current_bindings: Mapping[str, object] | None = None,
    atlas_rows: Mapping[int, Mapping[str, object]] | None = None,
    sae: object | None = None,
    plans_by_feature: Mapping[int, FrozenFeaturePlan] | None = None,
) -> dict[str, object]:
    try:
        manifest = strict_json_load(output_dir / "manifest.json")
    except (AtlasIOError, OSError, ValueError) as error:
        raise SpanAttributionError("invalid span attribution manifest") from error
    if current_bindings is not None:
        validate_stage_bindings(manifest, current_bindings)
    if atlas_rows is None:
        raise SpanAttributionError("span validation requires canonical Task 8 atlas rows")
    _validate_registered_stage(
        output_dir,
        manifest,
        rows=atlas_rows,
        sae=sae,
        plans_by_feature=plans_by_feature,
    )
    return manifest
