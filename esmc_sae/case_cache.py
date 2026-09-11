"""Complete and audit the shared ESMC/SAE cache for the OVCAR8 case."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
from typing import Sequence

import pandas as pd
import torch

from esmc_sae.artifacts import save_cpu_torch, save_json
from esmc_sae.audit import (
    _decoder_from_payload,
    _load_cache_pair,
    _load_trusted_decoder_payload,
    _reconstruction_metrics,
    apply_feature_delta,
)
from esmc_sae.config import load_config
from esmc_sae.extract import (
    _append_manifest,
    _build_provenance,
    _cache_is_current,
    _cache_payloads,
    _dtype_name,
    _effective_runtime_binding,
    _load_sequences,
    _normalize_sequence,
    _sequence_sha256,
    _timestamp,
    export_decoder_bridge,
    extract_one,
    load_local_models,
)
from esmc_sae.manifest import file_sha256


CODE_ROOT = Path("/data/qingyuyang/SL/PFSL")
GENES_FILE = CODE_ROOT / "data/SLKB/CASE_BRCA2_PARP1/genes.csv"
MVP_CONFIG = CODE_ROOT / "esmc_sae/configs/mvp_a549_c1_fold0.yaml"
CASE_ROOT = CODE_ROOT / "esmc_sae/case_study/ovcar8_brca2_parp1"
EXTRACTION_MANIFEST = CASE_ROOT / "manifests/cache_extraction.jsonl"
AUDIT_FILE = CASE_ROOT / "reports/cache_audit.json"


def load_original_missing_gene_ids(path: Path = GENES_FILE) -> list[int]:
    """Return the deterministic genes absent when the split was frozen."""

    genes = pd.read_csv(path)
    required = {"gene_id", "sequence_length", "sequence_sha256", "cache_status"}
    if set(genes.columns) != required:
        raise ValueError("case gene manifest columns differ")
    if len(genes) != 62 or genes["gene_id"].duplicated().any():
        raise ValueError("case gene manifest must contain 62 unique genes")
    if not set(genes["cache_status"]) <= {"present", "missing"}:
        raise ValueError("case gene manifest has an invalid cache status")
    return sorted(
        int(value)
        for value in genes.loc[
            genes["cache_status"] == "missing",
            "gene_id",
        ]
    )


def cache_action(
    dense_path: Path,
    sparse_path: Path,
    *,
    current: bool,
) -> str:
    """Choose skip/extract without ever replacing an existing cache file."""

    dense_exists = Path(dense_path).exists()
    sparse_exists = Path(sparse_path).exists()
    if dense_exists != sparse_exists:
        raise RuntimeError("partial cache pair exists; refusing overwrite")
    if dense_exists:
        if not current:
            raise RuntimeError("existing cache pair is invalid; refusing overwrite")
        return "skip"
    if current:
        raise RuntimeError("cache cannot be current when both files are absent")
    return "extract"


def extract_missing_case_cache(
    *,
    genes_file: Path = GENES_FILE,
    config_path: Path = MVP_CONFIG,
) -> dict[str, object]:
    """Extract only the 36 originally missing OVCAR8 proteins."""

    project = load_config(config_path)
    missing_ids = load_original_missing_gene_ids(genes_file)
    if len(missing_ids) != 36:
        raise RuntimeError(f"expected 36 originally missing genes, got {len(missing_ids)}")
    sequences = _load_sequences(project.paths.sequences)
    missing_sequences = sorted(set(missing_ids).difference(sequences))
    if missing_sequences:
        raise RuntimeError(f"case gene lacks sequence: {missing_sequences[0]}")

    import transformers

    model, sae, sae_layer, tokenizer = load_local_models(project)
    provenance, fingerprint = _build_provenance(
        project,
        transformers_version=transformers.__version__,
        runtime_binding=_effective_runtime_binding(model),
        model_compute_dtype=_dtype_name(next(model.parameters()).dtype),
        sae_compute_dtype=_dtype_name(sae_layer.W_enc.dtype),
    )
    del sae
    export_decoder_bridge(
        project.paths.decoder_bridge,
        sae_layer,
        project,
        provenance,
    )

    rows = sorted(
        (
            (gene_id, _normalize_sequence(sequences[gene_id]))
            for gene_id in missing_ids
        ),
        key=lambda item: (len(item[1]), item[0]),
    )
    extracted = 0
    skipped = 0
    timings: list[float] = []
    for ordinal, (gene_id, sequence) in enumerate(rows, start=1):
        sequence_sha256 = _sequence_sha256(sequence)
        dense_path = project.paths.dense_cache / f"{gene_id}.pt"
        sparse_path = project.paths.sparse_cache / f"{gene_id}.pt"
        current = _cache_is_current(
            dense_path,
            sparse_path,
            gene_id=gene_id,
            sequence_sha256=sequence_sha256,
            provenance_fingerprint=fingerprint,
        )
        action = cache_action(dense_path, sparse_path, current=current)
        if action == "skip":
            skipped += 1
            print(
                f"CASE_CACHE SKIP {ordinal}/{len(rows)} gene_id={gene_id}",
                flush=True,
            )
            continue

        started = time.monotonic()
        cache = replace(
            extract_one(sequence, model, sae_layer, tokenizer),
            gene_id=gene_id,
        )
        dense_payload, sparse_payload = _cache_payloads(cache, fingerprint)
        save_cpu_torch(dense_path, dense_payload)
        save_cpu_torch(sparse_path, sparse_payload)
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        extracted += 1
        _append_manifest(
            EXTRACTION_MANIFEST,
            {
                "dense_cache": str(dense_path),
                "elapsed_seconds": round(elapsed, 6),
                "gene_id": gene_id,
                "length": len(sequence),
                "provenance_fingerprint": fingerprint,
                "sequence_sha256": sequence_sha256,
                "sparse_cache": str(sparse_path),
                "status": "extracted",
                "timestamp": _timestamp(),
            },
        )
        print(
            f"CASE_CACHE DONE {ordinal}/{len(rows)} gene_id={gene_id} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )
    return {
        "schema_version": 1,
        "selected": len(rows),
        "extracted": extracted,
        "skipped": skipped,
        "median_seconds": (
            float(statistics.median(timings)) if timings else 0.0
        ),
        "provenance_fingerprint": fingerprint,
        "manifest": str(EXTRACTION_MANIFEST),
    }


def audit_case_cache(
    *,
    genes_file: Path = GENES_FILE,
    config_path: Path = MVP_CONFIG,
) -> dict[str, object]:
    """Validate all 62 OVCAR8 cache pairs against the trusted bridge."""

    project = load_config(config_path)
    genes = pd.read_csv(genes_file).sort_values("gene_id")
    sequences = _load_sequences(project.paths.sequences)
    decoder_payload = _load_trusted_decoder_payload(project)
    decoder = _decoder_from_payload(
        decoder_payload,
        project,
        torch.device("cpu"),
    )
    rows = []
    target_metrics = {}
    for gene_id in (int(value) for value in genes["gene_id"]):
        sequence = _normalize_sequence(sequences[gene_id])
        hidden, sparse = _load_cache_pair(project, gene_id, sequence)
        unchanged = apply_feature_delta(
            hidden,
            sparse,
            decoder,
            frozenset(),
        )
        no_op_error = float((unchanged - hidden).abs().max().item())
        if no_op_error != 0.0:
            raise RuntimeError(f"SAE no-op failed for gene {gene_id}")
        dense_path = project.paths.dense_cache / f"{gene_id}.pt"
        sparse_path = project.paths.sparse_cache / f"{gene_id}.pt"
        rows.append(
            {
                "gene_id": gene_id,
                "length": len(sequence),
                "dense_sha256": file_sha256(dense_path),
                "sparse_sha256": file_sha256(sparse_path),
                "no_op_max_abs_error": no_op_error,
            }
        )
        if gene_id in (675, 142):
            target_metrics[str(gene_id)] = _reconstruction_metrics(
                hidden,
                sparse,
                decoder,
            )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "cache_pairs": len(rows),
        "all_finite_aligned": True,
        "all_no_op_exact": True,
        "brca2_length": next(row["length"] for row in rows if row["gene_id"] == 675),
        "parp1_length": next(row["length"] for row in rows if row["gene_id"] == 142),
        "target_reconstruction": target_metrics,
        "decoder_bridge": str(project.paths.decoder_bridge),
        "decoder_bridge_sha256": file_sha256(project.paths.decoder_bridge),
        "provenance_fingerprint": decoder_payload["provenance_fingerprint"],
        "proteins": rows,
    }
    if (
        result["cache_pairs"] != 62
        or result["brca2_length"] != 2000
        or result["parp1_length"] != 1014
    ):
        raise RuntimeError("locked case cache audit counts changed")
    save_json(AUDIT_FILE, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--extract", action="store_true")
    action.add_argument("--audit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = (
        extract_missing_case_cache()
        if args.extract
        else audit_case_cache()
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
