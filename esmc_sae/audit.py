"""Experiment-1 binding, reconstruction, no-op, and coverage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from esmc_sae.artifacts import save_json
from esmc_sae.config import ProjectConfig, load_config
from esmc_sae.extract import (
    DecoderBridge,
    PREPROCESSING_DESCRIPTION,
    SparseProteinFeatures,
    _selected_proteins,
    _sequence_sha256,
    _tensor_sha256,
)
from esmc_sae.manifest import verify_manifest_target


AUDIT_SCHEMA_VERSION = 1
EXTRACTION_SCHEMA_VERSION = 2
EXPECTED_PROVENANCE_FINGERPRINT = (
    "3156f48000d1da2d5a1e7b80c9607a8b"
    "9698b115561ed04fc81dc4b66d23b52a"
)
REPORT_NAME = "experiment1_audit.json"
DECODE_CHUNK_RESIDUES = 32
TOP_CONCENTRATION_LIMIT = 20
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AuditFailure(RuntimeError):
    """A Checkpoint-A failure that must stop downstream work."""

    def __init__(self, gate: str, detail: str | None = None):
        if detail is None:
            detail = gate
            gate = "decoder_provenance"
        super().__init__(detail)
        self.gate = gate


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if any(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def _validate_sparse_decode_contract(
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
) -> None:
    indices = sparse.indices
    values = sparse.values
    if indices.ndim != 2 or values.shape != indices.shape:
        raise ValueError("sparse indices and values must share shape [L, K]")
    if indices.shape[1] != decoder.topk:
        raise ValueError("sparse width does not match decoder Top-K")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparse indices must use an integer dtype")
    if not torch.is_floating_point(values):
        raise ValueError("sparse values must use a floating dtype")
    if bool((indices < -1).any()) or bool(
        (indices >= decoder.codebook_dim).any()
    ):
        raise ValueError("sparse feature index is outside the codebook")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sparse feature values must be finite")


def _sparse_decode(
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
    *,
    values: torch.Tensor | None = None,
    include_bias: bool,
) -> torch.Tensor:
    _validate_sparse_decode_contract(sparse, decoder)
    selected_values = sparse.values if values is None else values
    if selected_values.shape != sparse.indices.shape:
        raise ValueError("decode values must match sparse indices")

    device = decoder.W_dec.device
    compute_dtype = _compute_dtype(decoder.W_dec, selected_values)
    length, width = sparse.indices.shape
    outputs: list[torch.Tensor] = []
    for start in range(0, length, DECODE_CHUNK_RESIDUES):
        stop = min(start + DECODE_CHUNK_RESIDUES, length)
        indices = sparse.indices[start:stop].to(
            device=device,
            dtype=torch.int64,
        )
        active = indices >= 0
        safe_indices = indices.masked_fill(~active, 0)
        weights = selected_values[start:stop].to(
            device=device,
            dtype=compute_dtype,
        )
        weights = weights.masked_fill(~active, 0)
        rows = decoder.W_dec.index_select(
            0,
            safe_indices.reshape(-1),
        ).reshape(stop - start, width, decoder.d_model)
        decoded = torch.einsum(
            "lk,lkd->ld",
            weights,
            rows.to(dtype=compute_dtype),
        )
        if include_bias:
            decoded = decoded + decoder.b_dec.to(
                device=device,
                dtype=compute_dtype,
            )
        outputs.append(decoded)
    if not outputs:
        return torch.empty(
            (0, decoder.d_model),
            device=device,
            dtype=compute_dtype,
        )
    return torch.cat(outputs, dim=0)


def reconstruct_normalized(
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
) -> torch.Tensor:
    """Decode fixed-width sparse activations in normalized SAE coordinates."""

    return _sparse_decode(
        sparse,
        decoder,
        include_bias=True,
    )


def apply_feature_delta(
    h: torch.Tensor,
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
    removed_ids: frozenset[int],
) -> torch.Tensor:
    """Apply an SAE decoder delta while preserving the reconstruction residual."""

    if not removed_ids:
        return h
    if h.ndim != 2 or h.shape != (
        sparse.indices.shape[0],
        decoder.d_model,
    ):
        raise ValueError("hidden state and sparse features are not aligned")
    invalid_ids = sorted(
        feature_id
        for feature_id in removed_ids
        if feature_id < 0 or feature_id >= decoder.codebook_dim
    )
    if invalid_ids:
        raise ValueError(f"removed feature ID is outside codebook: {invalid_ids[0]}")

    removed = torch.zeros_like(sparse.indices, dtype=torch.bool)
    for feature_id in sorted(removed_ids):
        removed |= sparse.indices == feature_id
    delta_values = torch.where(
        removed,
        -sparse.values,
        torch.zeros_like(sparse.values),
    )
    decoded_delta = _sparse_decode(
        sparse,
        decoder,
        values=delta_values,
        include_bias=False,
    )
    compute_dtype = _compute_dtype(h, decoded_delta)
    hidden = h.to(dtype=compute_dtype)
    scale = hidden.std(dim=-1, keepdim=True) + decoder.eps
    result = hidden + scale * decoded_delta.to(
        device=h.device,
        dtype=compute_dtype,
    )
    return result.to(dtype=h.dtype)


def _select_audit_sample(
    proteins: Sequence[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Choose unique shortest, median-nearest, and longest strata."""

    ordered = sorted(proteins, key=lambda item: (len(item[1]), item[0]))
    if len(ordered) <= 30:
        return ordered
    median_length = statistics.median(len(sequence) for _, sequence in ordered)
    nearest = sorted(
        ordered,
        key=lambda item: (
            abs(len(item[1]) - median_length),
            len(item[1]),
            item[0],
        ),
    )[:10]
    chosen = {
        gene_id: sequence
        for gene_id, sequence in (
            ordered[:10] + nearest + ordered[-10:]
        )
    }
    return sorted(
        chosen.items(),
        key=lambda item: (len(item[1]), item[0]),
    )


def _validated_checksum_mapping(
    value: object,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise AuditFailure(
            "decoder_provenance",
            f"decoder {label} are invalid",
        )
    validated: dict[str, str] = {}
    for key, checksum in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.strip() != key
            or not isinstance(checksum, str)
            or _SHA256_PATTERN.fullmatch(checksum) is None
        ):
            raise AuditFailure(
                "decoder_provenance",
                f"decoder {label} are invalid",
            )
        validated[key] = checksum
    return validated


def _canonical_provenance_fingerprint(
    payload: Mapping[str, object],
    source_checksums: Mapping[str, str],
    implementation_checksums: Mapping[str, str],
) -> str:
    provenance = {
        "schema_version": payload.get("provenance_schema_version"),
        "source_file_checksums": dict(source_checksums),
        "implementation_source_checksums": dict(implementation_checksums),
        "transformers_version": payload.get("transformers_version"),
        "runtime_binding": payload.get("runtime_binding"),
        "compute_dtypes": payload.get("compute_dtypes"),
        "local_files_only": True,
        "layer": payload.get("layer"),
        "topk": payload.get("topk"),
        "codebook_dim": payload.get("codebook_dim"),
        "d_model": payload.get("d_model"),
        "eps": payload.get("eps"),
        "normalize_sae": False,
        "sae_preprocessing": payload.get("sae_preprocessing"),
    }
    try:
        serialized = json.dumps(
            provenance,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise AuditFailure(
            "decoder_provenance",
            "decoder canonical provenance is not serializable",
        ) from error
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _decoder_from_payload(
    payload: object,
    cfg: ProjectConfig,
    device: torch.device,
) -> DecoderBridge:
    if not isinstance(payload, Mapping):
        raise AuditFailure(
            "decoder_provenance",
            "decoder bridge payload is not a mapping",
        )
    W_dec = payload.get("W_dec")
    b_dec = payload.get("b_dec")
    if not isinstance(W_dec, torch.Tensor) or not isinstance(
        b_dec,
        torch.Tensor,
    ):
        raise AuditFailure(
            "decoder_provenance",
            "decoder bridge tensors are absent",
        )
    if W_dec.shape != (
        cfg.model.codebook_dim,
        cfg.model.d_model,
    ) or b_dec.shape != (cfg.model.d_model,):
        raise AuditFailure(
            "decoder_provenance",
            "decoder tensor shape differs from config",
        )
    if W_dec.dtype != torch.float16 or b_dec.dtype != torch.float16:
        raise AuditFailure(
            "decoder_provenance",
            "decoder tensor dtype must be float16",
        )
    checksums = {
        "W_dec": _tensor_sha256(W_dec),
        "b_dec": _tensor_sha256(b_dec),
    }
    if payload.get("decoder_tensor_checksums") != checksums:
        raise AuditFailure(
            "decoder_provenance",
            "decoder tensor checksum mismatch",
        )
    expected_metadata = {
        "provenance_schema_version": EXTRACTION_SCHEMA_VERSION,
        "provenance_fingerprint": EXPECTED_PROVENANCE_FINGERPRINT,
        "layer": cfg.model.layer,
        "topk": cfg.model.topk,
        "codebook_dim": cfg.model.codebook_dim,
        "d_model": cfg.model.d_model,
        "eps": cfg.model.sae_eps,
    }
    mismatch = [
        key
        for key, expected in expected_metadata.items()
        if payload.get(key) != expected
    ]
    if mismatch:
        raise AuditFailure(
            "decoder_provenance",
            f"decoder provenance mismatch: {mismatch[0]}",
        )
    source_checksums = _validated_checksum_mapping(
        payload.get("source_file_checksums"),
        "source checksums",
    )
    implementation_checksums = _validated_checksum_mapping(
        payload.get("implementation_source_checksums"),
        "implementation checksums",
    )
    if payload.get("sae_preprocessing") != PREPROCESSING_DESCRIPTION:
        raise AuditFailure(
            "decoder_provenance",
            "decoder preprocessing provenance mismatch",
        )
    canonical_fingerprint = _canonical_provenance_fingerprint(
        payload,
        source_checksums,
        implementation_checksums,
    )
    if (
        canonical_fingerprint != payload.get("provenance_fingerprint")
        or canonical_fingerprint != EXPECTED_PROVENANCE_FINGERPRINT
    ):
        raise AuditFailure(
            "decoder_provenance",
            "decoder canonical provenance fingerprint mismatch",
        )
    if not bool(torch.isfinite(W_dec).all()) or not bool(
        torch.isfinite(b_dec).all()
    ):
        raise AuditFailure("finite", "decoder tensors contain non-finite values")
    return DecoderBridge(
        W_dec=W_dec.to(device=device),
        b_dec=b_dec.to(device=device),
        layer=cfg.model.layer,
        topk=cfg.model.topk,
        codebook_dim=cfg.model.codebook_dim,
        d_model=cfg.model.d_model,
        eps=cfg.model.sae_eps,
    )


def _load_trusted_decoder_payload(cfg: ProjectConfig) -> object:
    """Verify the external Task 2 file anchor before parsing bridge bytes."""

    verify_manifest_target(
        cfg.paths.project_root / "manifests/task02.sha256",
        cfg.paths.decoder_bridge,
        code_root=cfg.paths.code_root,
        label="Task 2 decoder bridge",
    )
    return torch.load(
        cfg.paths.decoder_bridge,
        map_location="cpu",
        weights_only=True,
    )


def _load_cache_pair(
    cfg: ProjectConfig,
    gene_id: int,
    sequence: str,
) -> tuple[torch.Tensor, SparseProteinFeatures]:
    dense_path = cfg.paths.dense_cache / f"{gene_id}.pt"
    sparse_path = cfg.paths.sparse_cache / f"{gene_id}.pt"
    if not dense_path.is_file() or not sparse_path.is_file():
        raise AuditFailure("alignment", f"cache pair is missing for gene {gene_id}")
    dense = torch.load(dense_path, map_location="cpu", weights_only=True)
    sparse_payload = torch.load(
        sparse_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(dense, Mapping) or not isinstance(
        sparse_payload,
        Mapping,
    ):
        raise AuditFailure("alignment", f"cache payload invalid for gene {gene_id}")

    expected_metadata = {
        "gene_id": gene_id,
        "sequence_sha256": _sequence_sha256(sequence),
        "provenance_schema_version": EXTRACTION_SCHEMA_VERSION,
        "provenance_fingerprint": EXPECTED_PROVENANCE_FINGERPRINT,
    }
    for payload in (dense, sparse_payload):
        mismatch = [
            key
            for key, expected in expected_metadata.items()
            if payload.get(key) != expected
        ]
        if mismatch:
            gate = (
                "decoder_provenance"
                if mismatch[0].startswith("provenance_")
                else "alignment"
            )
            raise AuditFailure(
                gate,
                f"cache metadata mismatch for gene {gene_id}: {mismatch[0]}",
            )

    h = dense.get("h60")
    sparse_tree = sparse_payload.get("sparse")
    if not isinstance(h, torch.Tensor) or not isinstance(
        sparse_tree,
        Mapping,
    ):
        raise AuditFailure("alignment", f"cache tensors absent for gene {gene_id}")
    indices = sparse_tree.get("indices")
    values = sparse_tree.get("values")
    if not isinstance(indices, torch.Tensor) or not isinstance(
        values,
        torch.Tensor,
    ):
        raise AuditFailure("alignment", f"sparse tensors absent for gene {gene_id}")

    length = len(sequence)
    if (
        h.shape != (length, cfg.model.d_model)
        or indices.shape != (length, cfg.model.topk)
        or values.shape != (length, cfg.model.topk)
        or indices.dtype != torch.int32
        or h.dtype != torch.float16
        or values.dtype != torch.float16
    ):
        raise AuditFailure(
            "alignment",
            f"shape/dtype alignment failed for gene {gene_id}",
        )
    if not bool(torch.isfinite(h).all()) or not bool(
        torch.isfinite(values).all()
    ):
        raise AuditFailure("finite", f"non-finite cache value for gene {gene_id}")
    if bool((indices < 0).any()) or bool(
        (indices >= cfg.model.codebook_dim).any()
    ):
        raise AuditFailure(
            "alignment",
            f"Top-K feature index invalid for gene {gene_id}",
        )
    sorted_indices = torch.sort(indices, dim=1).values
    if bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
        raise AuditFailure(
            "alignment",
            f"Top-K feature IDs are not unique for gene {gene_id}",
        )
    return h, SparseProteinFeatures(indices=indices, values=values)


def _finite_float(value: torch.Tensor, label: str) -> float:
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise AuditFailure("finite", f"non-finite reconstruction metric: {label}")
    return scalar


def _reconstruction_metrics(
    h: torch.Tensor,
    sparse: SparseProteinFeatures,
    decoder: DecoderBridge,
) -> dict[str, float]:
    hidden = h.to(device=decoder.W_dec.device, dtype=torch.float32)
    scale = hidden.std(dim=-1, keepdim=True) + decoder.eps
    normalized = (hidden - hidden.mean(dim=-1, keepdim=True)) / scale
    reconstructed = reconstruct_normalized(sparse, decoder)
    if reconstructed.shape != normalized.shape or not bool(
        torch.isfinite(reconstructed).all()
    ):
        raise AuditFailure("finite", "reconstruction is non-finite or misaligned")
    residual = normalized - reconstructed
    target_variance = normalized.var(unbiased=False)
    residual_variance = residual.var(unbiased=False)
    explained_variance = 1.0 - residual_variance / target_variance
    return {
        "mse": _finite_float(residual.square().mean(), "mse"),
        "cosine_similarity": _finite_float(
            F.cosine_similarity(
                normalized,
                reconstructed,
                dim=-1,
                eps=1e-8,
            ).mean(),
            "cosine_similarity",
        ),
        "explained_variance": _finite_float(
            explained_variance,
            "explained_variance",
        ),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "max": ordered[-1],
    }


def _length_correlation(
    lengths: Sequence[int],
    active_counts: Sequence[int],
) -> float | None:
    if len(lengths) < 2 or len(set(lengths)) < 2 or len(set(active_counts)) < 2:
        return None
    x = torch.tensor(lengths, dtype=torch.float64)
    y = torch.tensor(active_counts, dtype=torch.float64)
    value = float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())
    return value if math.isfinite(value) else None


def _feature_coverage_counts(
    feature_total: torch.Tensor,
    feature_presence: torch.Tensor,
) -> dict[str, int]:
    return {
        "dead_feature_count": int((feature_total == 0).sum().item()),
        "rare_feature_count": int((feature_presence == 1).sum().item()),
    }


def _duplicate_groups(
    proteins: Sequence[tuple[int, str]],
) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for gene_id, sequence in proteins:
        groups[_sequence_sha256(sequence)].append(gene_id)
    return [
        {"sequence_sha256": checksum, "gene_ids": sorted(gene_ids)}
        for checksum, gene_ids in sorted(groups.items())
        if len(gene_ids) > 1
    ]


def _initial_report(config_path: Path) -> dict[str, object]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "status": "RUNNING",
        "stop_gate": {
            "passed": False,
            "checks": {
                "alignment": None,
                "finite": None,
                "decoder_provenance": None,
                "no_op_exact": None,
                "gpu_execution": None,
            },
            "no_op_tolerance": 0.0,
            "failure": None,
        },
    }


def run_audit(config_path: Path) -> int:
    report = _initial_report(config_path)
    fallback_path = Path(__file__).resolve().parent / "reports" / REPORT_NAME
    report_path = fallback_path
    try:
        cfg = load_config(config_path)
        report_path = cfg.paths.reports / REPORT_NAME
        proteins = _selected_proteins(cfg)
        sample = _select_audit_sample(proteins)
        sample_ids = frozenset(gene_id for gene_id, _ in sample)
        if not proteins:
            raise AuditFailure("alignment", "configured splits contain no proteins")
        if not torch.cuda.is_available():
            raise AuditFailure("gpu_execution", "CUDA is unavailable")
        device = torch.device("cuda")
        report["stop_gate"]["checks"]["gpu_execution"] = True

        try:
            bridge_payload = _load_trusted_decoder_payload(cfg)
        except ValueError as error:
            raise AuditFailure("decoder_provenance", str(error)) from error
        decoder = _decoder_from_payload(bridge_payload, cfg, device)
        report["stop_gate"]["checks"]["decoder_provenance"] = True
        report["binding"] = {
            "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
            "provenance_fingerprint": EXPECTED_PROVENANCE_FINGERPRINT,
            "decoder_tensor_checksums": bridge_payload[
                "decoder_tensor_checksums"
            ],
            "layer": decoder.layer,
            "topk": decoder.topk,
            "codebook_dim": decoder.codebook_dim,
            "d_model": decoder.d_model,
            "eps": decoder.eps,
            "device": str(device),
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

        feature_total = torch.zeros(decoder.codebook_dim, dtype=torch.int64)
        feature_presence = torch.zeros(decoder.codebook_dim, dtype=torch.int64)
        feature_max_count = torch.zeros(decoder.codebook_dim, dtype=torch.int64)
        feature_max_gene = torch.full(
            (decoder.codebook_dim,),
            -1,
            dtype=torch.int64,
        )
        protein_coverage: list[dict[str, object]] = []
        reconstruction_rows: list[dict[str, object]] = []
        no_op_errors: list[float] = []

        for ordinal, (gene_id, sequence) in enumerate(proteins, start=1):
            h, sparse = _load_cache_pair(cfg, gene_id, sequence)
            active_ids = sparse.indices.reshape(-1).to(torch.int64)
            counts = torch.bincount(
                active_ids,
                minlength=decoder.codebook_dim,
            )
            unique_ids = torch.nonzero(counts, as_tuple=False).flatten()
            feature_total += counts
            feature_presence[unique_ids] += 1
            better = counts > feature_max_count
            tied_smaller = (
                (counts == feature_max_count)
                & (counts > 0)
                & ((feature_max_gene < 0) | (gene_id < feature_max_gene))
            )
            replace = better | tied_smaller
            feature_max_count[replace] = counts[replace]
            feature_max_gene[replace] = gene_id
            protein_coverage.append(
                {
                    "gene_id": gene_id,
                    "length": len(sequence),
                    "active_feature_count": int(unique_ids.numel()),
                    "activation_count": int(active_ids.numel()),
                    "max_activation": float(sparse.values.max().item()),
                }
            )

            if gene_id in sample_ids:
                metrics = _reconstruction_metrics(h, sparse, decoder)
                unchanged = apply_feature_delta(
                    h,
                    sparse,
                    decoder,
                    frozenset(),
                )
                no_op_error = float((unchanged - h).abs().max().item())
                if no_op_error != 0.0:
                    raise AuditFailure(
                        "no_op_exact",
                        f"no-op is not exact for gene {gene_id}",
                    )
                no_op_errors.append(no_op_error)
                reconstruction_rows.append(
                    {
                        "gene_id": gene_id,
                        "length": len(sequence),
                        "no_op_max_abs_error": no_op_error,
                        **metrics,
                    }
                )
            print(
                f"AUDIT {ordinal}/{len(proteins)} gene_id={gene_id} "
                f"length={len(sequence)}",
                flush=True,
            )
            del h, sparse, active_ids, counts, unique_ids

        report["stop_gate"]["checks"]["alignment"] = True
        report["stop_gate"]["checks"]["finite"] = True
        report["stop_gate"]["checks"]["no_op_exact"] = (
            bool(no_op_errors) and max(no_op_errors) == 0.0
        )
        if not report["stop_gate"]["checks"]["no_op_exact"]:
            raise AuditFailure("no_op_exact", "no proteins received a no-op audit")

        lengths = [int(row["length"]) for row in protein_coverage]
        active_counts = [
            int(row["active_feature_count"]) for row in protein_coverage
        ]
        active_feature_ids = torch.nonzero(
            feature_total,
            as_tuple=False,
        ).flatten()
        concentration_rows = []
        for feature_id in active_feature_ids.tolist():
            total = int(feature_total[feature_id].item())
            maximum = int(feature_max_count[feature_id].item())
            concentration_rows.append(
                {
                    "feature_id": feature_id,
                    "protein_id": int(feature_max_gene[feature_id].item()),
                    "protein_share": maximum / total,
                    "protein_activation_count": maximum,
                    "domain_activation_count": total,
                }
            )
        concentration_rows.sort(
            key=lambda row: (
                -float(row["protein_share"]),
                -int(row["domain_activation_count"]),
                int(row["feature_id"]),
            )
        )
        report["domain"] = {
            "configured_protein_count": len(proteins),
            "cache_pair_count": len(protein_coverage),
            "reconstruction_sample_strategy": (
                "10 shortest + 10 nearest median length + 10 longest; "
                "ties by length then gene ID; unique proteins"
            ),
            "reconstruction_sample_gene_ids": [
                gene_id for gene_id, _ in sample
            ],
            "duplicate_sequence_checksum_groups": _duplicate_groups(proteins),
        }
        report["coverage"] = {
            "per_protein": protein_coverage,
            "active_feature_count_summary": _summary(active_counts),
            "max_activation_summary": _summary(
                [float(row["max_activation"]) for row in protein_coverage]
            ),
            "length_vs_active_feature_count_pearson": _length_correlation(
                lengths,
                active_counts,
            ),
            **_feature_coverage_counts(feature_total, feature_presence),
            "rare_feature_definition": "active in exactly one domain protein",
            "top_concentration_by_protein_id": concentration_rows[
                :TOP_CONCENTRATION_LIMIT
            ],
        }
        metric_names = ("mse", "cosine_similarity", "explained_variance")
        report["reconstruction"] = {
            "coordinate_system": "official per-residue z-score SAE coordinates",
            "sample_size": len(reconstruction_rows),
            "metrics_summary": {
                name: _summary(
                    [float(row[name]) for row in reconstruction_rows]
                )
                for name in metric_names
            },
            "per_protein": reconstruction_rows,
        }
        report["stop_gate"]["passed"] = True
        report["status"] = "PASS"
    except AuditFailure as error:
        checks = report["stop_gate"]["checks"]
        if error.gate in checks:
            checks[error.gate] = False
        report["stop_gate"]["failure"] = {
            "gate": error.gate,
            "detail": str(error),
        }
        report["status"] = "BLOCKED"
    except Exception as error:
        report["stop_gate"]["failure"] = {
            "gate": "unexpected_error",
            "detail": f"{type(error).__name__}: {error}",
        }
        report["status"] = "BLOCKED"

    save_json(report_path, report)
    print(
        f"AUDIT_RESULT status={report['status']} report={report_path}",
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_audit(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
