"""Create compact mean-pooled layer-60 caches for all-context training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Sequence

import pandas as pd
import torch

from esmc_sae.artifacts import save_cpu_torch, save_json
from esmc_sae.config import load_config
from esmc_sae.extract import (
    _append_manifest,
    _build_provenance,
    _dtype_name,
    _effective_runtime_binding,
    _load_sequences,
    _normalize_sequence,
    _sequence_sha256,
    _timestamp,
    load_local_models,
)
from esmc_sae.manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]
GENES_FILE = ROOT / "data/SLKB/CASE_BRCA2_PARP1_ALL_CONTEXT/genes.csv"
POOL_ROOT = ROOT / "esmc_sae/cache/layer60_mean_all_context"
FULL_DENSE_ROOT = ROOT / "esmc_sae/cache/layer60_dense"
MVP_CONFIG = ROOT / "esmc_sae/configs/mvp_a549_c1_fold0.yaml"
CASE_ROOT = (
    ROOT / "esmc_sae/case_study/ovcar8_brca2_parp1_all_context"
)
AUDIT_FILE = CASE_ROOT / "reports/pooled_cache_audit.json"


def load_gene_manifest(path: Path) -> pd.DataFrame:
    """Load a nonempty, unique case-specific gene manifest."""

    genes = pd.read_csv(path)
    expected = {"gene_id", "input_length", "sequence_sha256"}
    if not expected <= set(genes.columns):
        raise ValueError(f"gene manifest lacks {sorted(expected)}")
    if genes.empty or genes["gene_id"].duplicated().any():
        raise ValueError("gene manifest must be nonempty with unique gene IDs")
    return genes.sort_values("gene_id", kind="stable").reset_index(drop=True)


def pooled_payload_from_dense(
    dense: dict[str, object],
    *,
    gene_id: int,
    sequence_sha256: str,
) -> dict[str, object]:
    """Convert a trusted full-residue cache without re-encoding."""

    if dense.get("gene_id") != gene_id:
        raise ValueError("dense cache gene ID changed")
    if dense.get("sequence_sha256") != sequence_sha256:
        raise ValueError("dense cache sequence hash changed")
    fingerprint = dense.get("provenance_fingerprint")
    hidden = dense.get("h60")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 2
        or not torch.isfinite(hidden).all()
    ):
        raise ValueError("dense cache payload is invalid")
    return {
        "schema_version": 1,
        "gene_id": gene_id,
        "sequence_sha256": sequence_sha256,
        "provenance_fingerprint": fingerprint,
        "source": "full_residue_cache",
        "mean_h60": hidden.float().mean(dim=0),
    }


def extract_mean_layer60(
    sequence: str,
    model: object,
    tokenizer: object,
    *,
    expected_width: int = 2560,
) -> torch.Tensor:
    """Extract residue layer 60 once and return only its float32 mean."""

    normalized = _normalize_sequence(sequence)
    inputs = tokenizer(normalized, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {
            name: tensor.to(device)
            for name, tensor in inputs.items()
        }
    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=True,
            compute_sae=False,
        )
    token_count = int(inputs["attention_mask"].sum().item())
    if token_count != len(normalized) + 2:
        raise ValueError("token alignment mismatch")
    hidden = output.hidden_states[60][0, 1:-1]
    if hidden.shape != (len(normalized), expected_width):
        raise ValueError(f"layer-60 shape mismatch: {tuple(hidden.shape)}")
    mean = hidden.detach().float().mean(dim=0).cpu().contiguous()
    if not torch.isfinite(mean).all():
        raise ValueError("layer-60 mean contains non-finite values")
    return mean


def pooled_cache_action(path: Path, *, current: bool) -> str:
    """Never replace an existing pooled cache, valid or invalid."""

    exists = Path(path).exists()
    if not exists:
        if current:
            raise RuntimeError("absent pooled cache cannot be current")
        return "extract"
    if current:
        return "skip"
    raise RuntimeError("existing pooled cache is invalid; refusing overwrite")


def _pooled_is_current(
    path: Path,
    *,
    gene_id: int,
    sequence_sha256: str,
    provenance_fingerprint: str,
    expected_width: int = 2560,
) -> bool:
    if not Path(path).is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mean = payload["mean_h60"]
        return bool(
            payload["schema_version"] == 1
            and payload["gene_id"] == gene_id
            and payload["sequence_sha256"] == sequence_sha256
            and payload["provenance_fingerprint"] == provenance_fingerprint
            and payload["source"] in {"full_residue_cache", "direct_esmc"}
            and isinstance(mean, torch.Tensor)
            and mean.shape == (expected_width,)
            and mean.dtype == torch.float32
            and mean.device.type == "cpu"
            and torch.isfinite(mean).all()
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _trusted_fingerprint(project: object) -> str:
    payload = torch.load(
        project.paths.decoder_bridge,
        map_location="cpu",
        weights_only=True,
    )
    fingerprint = payload.get("provenance_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("trusted decoder bridge lacks provenance fingerprint")
    return fingerprint


def populate_pooled_cache(
    *,
    genes_file: Path = GENES_FILE,
    pool_root: Path = POOL_ROOT,
    full_dense_root: Path = FULL_DENSE_ROOT,
    config_path: Path = MVP_CONFIG,
    case_root: Path = CASE_ROOT,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, object]:
    """Populate one deterministic shard, reusing full caches when present."""

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard selection")
    genes = load_gene_manifest(genes_file).sort_values(
        ["input_length", "gene_id"],
        kind="stable",
    )
    rows = [
        row
        for ordinal, row in enumerate(genes.itertuples(index=False))
        if ordinal % num_shards == shard_index
    ]
    project = load_config(config_path)
    sequences = _load_sequences(project.paths.sequences)
    trusted_fingerprint = _trusted_fingerprint(project)
    Path(pool_root).mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(case_root)
        / "manifests"
        / f"pooled_cache_shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"
    )

    model = tokenizer = None
    extracted = skipped = reused_full = direct = 0
    for ordinal, row in enumerate(rows, start=1):
        gene_id = int(row.gene_id)
        sequence = _normalize_sequence(sequences[gene_id])
        digest = _sequence_sha256(sequence)
        if digest != row.sequence_sha256 or len(sequence) != int(row.input_length):
            raise RuntimeError(f"gene manifest changed for {gene_id}")
        target = Path(pool_root) / f"{gene_id}.pt"
        current = _pooled_is_current(
            target,
            gene_id=gene_id,
            sequence_sha256=digest,
            provenance_fingerprint=trusted_fingerprint,
        )
        action = pooled_cache_action(target, current=current)
        if action == "skip":
            skipped += 1
            continue

        started = time.monotonic()
        dense_path = Path(full_dense_root) / f"{gene_id}.pt"
        if dense_path.is_file():
            dense = torch.load(
                dense_path,
                map_location="cpu",
                weights_only=True,
            )
            payload = pooled_payload_from_dense(
                dense,
                gene_id=gene_id,
                sequence_sha256=digest,
            )
            if payload["provenance_fingerprint"] != trusted_fingerprint:
                raise RuntimeError(f"dense provenance changed for {gene_id}")
            reused_full += 1
        else:
            if model is None:
                import transformers

                model, sae, sae_layer, tokenizer = load_local_models(project)
                provenance, fingerprint = _build_provenance(
                    project,
                    transformers_version=transformers.__version__,
                    runtime_binding=_effective_runtime_binding(model),
                    model_compute_dtype=_dtype_name(
                        next(model.parameters()).dtype
                    ),
                    sae_compute_dtype=_dtype_name(sae_layer.W_enc.dtype),
                )
                del sae, sae_layer, provenance
                if fingerprint != trusted_fingerprint:
                    raise RuntimeError("runtime provenance differs from trusted cache")
            mean = extract_mean_layer60(sequence, model, tokenizer)
            payload = {
                "schema_version": 1,
                "gene_id": gene_id,
                "sequence_sha256": digest,
                "provenance_fingerprint": trusted_fingerprint,
                "source": "direct_esmc",
                "mean_h60": mean,
            }
            direct += 1
        save_cpu_torch(target, payload)
        elapsed = time.monotonic() - started
        extracted += 1
        _append_manifest(
            manifest,
            {
                "elapsed_seconds": round(elapsed, 6),
                "gene_id": gene_id,
                "pooled_cache": str(target),
                "pooled_cache_sha256": file_sha256(target),
                "provenance_fingerprint": trusted_fingerprint,
                "sequence_sha256": digest,
                "source": payload["source"],
                "timestamp": _timestamp(),
            },
        )
        print(
            f"POOLED_CACHE {ordinal}/{len(rows)} gene_id={gene_id} "
            f"source={payload['source']} seconds={elapsed:.3f}",
            flush=True,
        )
    return {
        "schema_version": 1,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "selected": len(rows),
        "extracted": extracted,
        "skipped": skipped,
        "reused_full": reused_full,
        "direct_esmc": direct,
        "provenance_fingerprint": trusted_fingerprint,
        "manifest": str(manifest),
    }


def audit_pooled_cache(
    *,
    genes_path: Path = GENES_FILE,
    pooled_root: Path = POOL_ROOT,
    expected_width: int = 2560,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Audit every pooled tensor and require one provenance fingerprint."""

    genes = pd.read_csv(genes_path).sort_values("gene_id")
    records = []
    fingerprints = set()
    sources: Counter[str] = Counter()
    for row in genes.itertuples(index=False):
        gene_id = int(row.gene_id)
        path = Path(pooled_root) / f"{gene_id}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mean = payload.get("mean_h60")
        if (
            payload.get("schema_version") != 1
            or payload.get("gene_id") != gene_id
            or payload.get("sequence_sha256") != row.sequence_sha256
            or not isinstance(mean, torch.Tensor)
            or mean.shape != (expected_width,)
            or mean.dtype != torch.float32
            or mean.device.type != "cpu"
            or not torch.isfinite(mean).all()
        ):
            raise RuntimeError(f"invalid pooled cache: {path}")
        fingerprint = payload.get("provenance_fingerprint")
        source = payload.get("source")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise RuntimeError(f"invalid pooled provenance: {path}")
        if source not in {"full_residue_cache", "direct_esmc"}:
            raise RuntimeError(f"invalid pooled source: {path}")
        fingerprints.add(fingerprint)
        sources[source] += 1
        records.append(
            {
                "gene_id": gene_id,
                "sha256": file_sha256(path),
                "source": source,
            }
        )
    if len(fingerprints) != 1:
        raise RuntimeError("pooled caches do not share one provenance")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "cache_count": len(records),
        "expected_width": expected_width,
        "all_cpu_float32_finite": True,
        "provenance_fingerprint": next(iter(fingerprints)),
        "source_counts": dict(sorted(sources.items())),
        "records": records,
    }
    if output_path is not None:
        save_json(Path(output_path), result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--extract", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--genes-file", type=Path, default=GENES_FILE)
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    parser.add_argument("--full-dense-root", type=Path, default=FULL_DENSE_ROOT)
    parser.add_argument("--case-root", type=Path, default=CASE_ROOT)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_FILE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.extract:
        result = populate_pooled_cache(
            genes_file=args.genes_file,
            pool_root=args.pool_root,
            full_dense_root=args.full_dense_root,
            case_root=args.case_root,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    else:
        result = audit_pooled_cache(
            genes_path=args.genes_file,
            pooled_root=args.pool_root,
            output_path=args.audit_output,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
