"""One-pass ESMC layer-60 and official SAE feature extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import pickle
import platform
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch

from esmc_sae.artifacts import save_cpu_torch
from esmc_sae.config import ProjectConfig, load_config


VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTUVWY")
EXTRACTION_SCHEMA_VERSION = 2
PREPROCESSING_DESCRIPTION = (
    "Official ESMC SAE preprocessing applies per-residue z-score "
    "normalization across d_model using std + eps before encoding."
)


@dataclass(frozen=True)
class SparseProteinFeatures:
    indices: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class ProteinCache:
    gene_id: int
    sequence_sha256: str
    h60: torch.Tensor
    sparse: SparseProteinFeatures

    @property
    def feature_indices(self) -> torch.Tensor:
        """Read-only compatibility view of ``sparse.indices``."""

        return self.sparse.indices

    @property
    def feature_values(self) -> torch.Tensor:
        """Read-only compatibility view of ``sparse.values``."""

        return self.sparse.values


@dataclass(frozen=True)
class DecoderBridge:
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    layer: int
    topk: int
    codebook_dim: int
    d_model: int
    eps: float


def _normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a string")
    normalized = sequence.strip().upper()
    invalid = sorted(set(normalized).difference(VALID_AMINO_ACIDS))
    if not normalized or invalid:
        detail = invalid if invalid else ["<empty>"]
        raise ValueError(f"invalid amino acid sequence: {detail}")
    return normalized


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _model_device(model: object) -> torch.device | None:
    device = getattr(model, "device", None)
    if device is None:
        return None
    return torch.device(device)


def _fixed_width_sparse(
    raw: torch.Tensor,
    *,
    token_count: int,
    residue_count: int,
    topk: int = 64,
) -> SparseProteinFeatures:
    if raw.layout != torch.sparse_coo:
        raise ValueError("SAE output must be a sparse COO tensor")
    sparse = raw.coalesce()
    coords = sparse.indices()
    values = sparse.values()
    if coords.ndim != 2 or coords.shape[0] != 2:
        raise ValueError("SAE output must have two sparse coordinate axes")

    keep = (coords[0] >= 1) & (coords[0] < token_count - 1)
    residue_rows = (coords[0, keep] - 1).to(
        device="cpu", dtype=torch.int64
    )
    feature_ids = coords[1, keep].to(device="cpu", dtype=torch.int64)
    feature_values = values[keep].to(device="cpu", dtype=torch.float16)

    indices = torch.full((residue_count, topk), -1, dtype=torch.int32)
    magnitudes = torch.zeros((residue_count, topk), dtype=torch.float16)
    for residue_row in range(residue_count):
        row_mask = residue_rows == residue_row
        row_features = feature_ids[row_mask][:topk]
        row_values = feature_values[row_mask][:topk]
        count = int(row_features.numel())
        if count:
            indices[residue_row, :count] = row_features.to(torch.int32)
            magnitudes[residue_row, :count] = row_values
    return SparseProteinFeatures(indices=indices, values=magnitudes)


def extract_one(sequence: str, model, sae_layer, tokenizer) -> ProteinCache:
    """Extract layer-60 and raw official-SAE features with one model call."""

    del sae_layer
    normalized = _normalize_sequence(sequence)
    inputs = tokenizer(normalized, return_tensors="pt")
    device = _model_device(model)
    if device is not None:
        inputs = {
            name: tensor.to(device)
            for name, tensor in inputs.items()
        }

    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=True,
            compute_sae=True,
            normalize_sae=False,
        )
        token_count = int(inputs["attention_mask"].sum().item())
        expected_token_count = len(normalized) + 2
        if token_count != expected_token_count:
            raise ValueError(
                "token alignment mismatch: expected one BOS and one EOS token"
            )
        h60 = (
            output.hidden_states[60][0, 1:-1]
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
        )
        if h60.shape != (len(normalized), 2560):
            raise ValueError(
                f"layer-60 shape mismatch: got {tuple(h60.shape)}"
            )

        raw = output.sae_outputs["layer60"]
        sparse = _fixed_width_sparse(
            raw,
            token_count=token_count,
            residue_count=len(normalized),
            topk=64,
        )
    return ProteinCache(
        gene_id=-1,
        sequence_sha256=_sequence_sha256(normalized),
        h60=h60,
        sparse=sparse,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(cfg: ProjectConfig) -> dict[str, Path]:
    model_names = (
        "config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    files = {
        f"esmc/{name}": cfg.paths.esmc_model / name
        for name in model_names
    }
    for path in sorted(cfg.paths.esmc_model.glob("model-*.safetensors")):
        files[f"esmc/{path.name}"] = path
    files.update(
        {
            "sae/config.json": cfg.paths.esmc_sae / "config.json",
            "sae/layer_60.safetensors": (
                cfg.paths.esmc_sae / "layer_60.safetensors"
            ),
        }
    )
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local model source file: {missing[0]}")
    return files


def _implementation_source_files() -> dict[str, Path]:
    from transformers.models.esmc import (
        configuration_esmc,
        configuration_esmc_sae,
        modeling_esmc,
        modeling_esmc_sae,
    )

    modules = {
        "transformers/configuration_esmc.py": configuration_esmc,
        "transformers/configuration_esmc_sae.py": configuration_esmc_sae,
        "transformers/modeling_esmc.py": modeling_esmc,
        "transformers/modeling_esmc_sae.py": modeling_esmc_sae,
    }
    files: dict[str, Path] = {}
    for name, module in modules.items():
        source = inspect.getsourcefile(module)
        if source is None:
            raise FileNotFoundError(
                f"cannot locate transformers implementation source: {name}"
            )
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing transformers implementation source: {path}"
            )
        files[name] = path
    return files


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _effective_runtime_binding(model) -> dict[str, object]:
    from transformers.models.esmc import modeling_esmc

    esmc_model = getattr(model, "esmc", model)
    use_varlen_flash = bool(
        getattr(esmc_model, "_use_flash_attn", False)
    )
    xformers_available = bool(modeling_esmc._xformers_available)
    flash_attn_available = bool(modeling_esmc._flash_attn_available)
    if use_varlen_flash:
        attention = "flash_attention_2_varlen"
    elif xformers_available:
        attention = "xformers_memory_efficient_attention"
    elif flash_attn_available:
        attention = "flash_attention_2"
    else:
        attention = "torch_scaled_dot_product_attention"

    device = torch.device(model.device)
    device_binding: dict[str, object] = {"type": device.type}
    if device.type == "cuda":
        device_binding.update(
            {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )

    cuda_backends = torch.backends.cuda
    runtime: dict[str, object] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device_binding,
        "backends": {
            "attention": attention,
            "layernorm_mlp": (
                "transformer_engine"
                if modeling_esmc._te_available
                else "pytorch"
            ),
            "rope": (
                "flash_attn_triton"
                if modeling_esmc._flash_attn_rotary_available
                else "pytorch"
            ),
            "availability": {
                "transformer_engine": bool(modeling_esmc._te_available),
                "xformers": xformers_available,
                "flash_attn_2": flash_attn_available,
                "flash_attn_rotary": bool(
                    modeling_esmc._flash_attn_rotary_available
                ),
            },
            "torch_sdpa_policy": {
                "flash_enabled": bool(cuda_backends.flash_sdp_enabled()),
                "memory_efficient_enabled": bool(
                    cuda_backends.mem_efficient_sdp_enabled()
                ),
                "math_enabled": bool(cuda_backends.math_sdp_enabled()),
                "cudnn_enabled": bool(cuda_backends.cudnn_sdp_enabled()),
            },
        },
        "matmul": {
            "float32_precision": torch.get_float32_matmul_precision(),
            "cuda_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "model_attn_implementation": getattr(
            model.config,
            "_attn_implementation",
            None,
        ),
    }
    return runtime


def _build_provenance(
    cfg: ProjectConfig,
    *,
    transformers_version: str,
    runtime_binding: Mapping[str, object],
    model_compute_dtype: str,
    sae_compute_dtype: str,
) -> tuple[dict[str, object], str]:
    print("PROVENANCE hashing local model sources", flush=True)
    checksums = {
        name: _sha256_file(path)
        for name, path in _source_files(cfg).items()
    }
    implementation_checksums = {
        name: _sha256_file(path)
        for name, path in _implementation_source_files().items()
    }
    provenance: dict[str, object] = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_file_checksums": checksums,
        "implementation_source_checksums": implementation_checksums,
        "transformers_version": transformers_version,
        "runtime_binding": dict(runtime_binding),
        "compute_dtypes": {
            "model": model_compute_dtype,
            "sae": sae_compute_dtype,
        },
        "local_files_only": True,
        "layer": cfg.model.layer,
        "topk": cfg.model.topk,
        "codebook_dim": cfg.model.codebook_dim,
        "d_model": cfg.model.d_model,
        "eps": cfg.model.sae_eps,
        "normalize_sae": False,
        "sae_preprocessing": PREPROCESSING_DESCRIPTION,
    }
    serialized = json.dumps(
        provenance,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    provenance["fingerprint"] = fingerprint
    print(f"PROVENANCE fingerprint={fingerprint}", flush=True)
    return provenance, fingerprint


def load_local_models(cfg: ProjectConfig):
    """Load and freeze the local ESMC model, tokenizer, and layer-60 SAE."""

    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ESMC extraction")
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        cfg.paths.esmc_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.paths.esmc_model,
        local_files_only=True,
    )
    sae = AutoModel.from_pretrained(
        cfg.paths.esmc_sae,
        local_files_only=True,
        device=model.device,
    )
    sae.to(dtype=torch.bfloat16)
    sae.initialize_layers([60], device=model.device)
    sae.eval()
    model.add_sae_models([sae.layers["60"]])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ESMC parameters must be frozen")
    if any(parameter.requires_grad for parameter in sae.parameters()):
        raise RuntimeError("SAE parameters must be frozen")
    return model, sae, sae.layers["60"], tokenizer


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_dtype_name(value.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bridge_payload(
    bridge: DecoderBridge,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    decoder_tensor_checksums = {
        "W_dec": _tensor_sha256(bridge.W_dec),
        "b_dec": _tensor_sha256(bridge.b_dec),
    }
    return {
        "W_dec": bridge.W_dec,
        "b_dec": bridge.b_dec,
        "provenance_schema_version": provenance["schema_version"],
        "layer": bridge.layer,
        "topk": bridge.topk,
        "codebook_dim": bridge.codebook_dim,
        "d_model": bridge.d_model,
        "eps": bridge.eps,
        "source_file_checksums": provenance["source_file_checksums"],
        "implementation_source_checksums": provenance[
            "implementation_source_checksums"
        ],
        "transformers_version": provenance["transformers_version"],
        "runtime_binding": provenance["runtime_binding"],
        "compute_dtypes": provenance["compute_dtypes"],
        "sae_preprocessing": provenance["sae_preprocessing"],
        "provenance_fingerprint": provenance["fingerprint"],
        "decoder_tensor_checksums": decoder_tensor_checksums,
    }


def export_decoder_bridge(
    path: Path,
    sae_layer,
    cfg: ProjectConfig,
    provenance: Mapping[str, object],
) -> None:
    """Export or validate the immutable official-SAE decoder bridge."""

    bridge = DecoderBridge(
        W_dec=sae_layer.W_dec.detach().to(
            device="cpu", dtype=torch.float16
        ).contiguous(),
        b_dec=sae_layer.b_dec.detach().to(
            device="cpu", dtype=torch.float16
        ).contiguous(),
        layer=cfg.model.layer,
        topk=cfg.model.topk,
        codebook_dim=cfg.model.codebook_dim,
        d_model=cfg.model.d_model,
        eps=cfg.model.sae_eps,
    )
    payload = _bridge_payload(bridge, provenance)
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(existing, dict)
            or not isinstance(existing.get("W_dec"), torch.Tensor)
            or not isinstance(existing.get("b_dec"), torch.Tensor)
        ):
            raise ValueError("existing decoder bridge tensor contract differs")
        if (
            existing["W_dec"].shape
            != (cfg.model.codebook_dim, cfg.model.d_model)
            or existing["b_dec"].shape != (cfg.model.d_model,)
            or existing["W_dec"].dtype != torch.float16
            or existing["b_dec"].dtype != torch.float16
        ):
            raise ValueError("existing decoder bridge tensor contract differs")
        observed_checksums = {
            "W_dec": _tensor_sha256(existing["W_dec"]),
            "b_dec": _tensor_sha256(existing["b_dec"]),
        }
        if existing.get("decoder_tensor_checksums") != observed_checksums:
            raise ValueError("existing decoder tensor checksum mismatch")
        metadata_keys = (
            "provenance_schema_version",
            "layer",
            "topk",
            "codebook_dim",
            "d_model",
            "eps",
            "source_file_checksums",
            "implementation_source_checksums",
            "transformers_version",
            "runtime_binding",
            "compute_dtypes",
            "sae_preprocessing",
            "provenance_fingerprint",
            "decoder_tensor_checksums",
        )
        if any(existing.get(key) != payload[key] for key in metadata_keys):
            raise ValueError("existing decoder bridge provenance differs")
        print(f"BRIDGE valid_existing path={path}", flush=True)
        return
    save_cpu_torch(path, payload)
    print(f"BRIDGE exported path={path}", flush=True)


def _load_sequences(path: Path) -> dict[int, str]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    sequences: dict[int, str] = {}
    for row in rows:
        if (
            not isinstance(row, (tuple, list))
            or len(row) != 2
            or type(row[0]) is not int
            or not isinstance(row[1], str)
        ):
            raise ValueError("all_id_seq.pkl has an invalid row")
        gene_id, sequence = row
        if gene_id in sequences:
            raise ValueError(f"duplicate sequence gene ID: {gene_id}")
        sequences[gene_id] = sequence
    return sequences


def _split_gene_ids(cfg: ProjectConfig) -> set[int]:
    directory = (
        cfg.paths.split_root
        / cfg.experiment.cell_line
        / cfg.experiment.scene
    )
    gene_ids: set[int] = set()
    for split in ("train", "val", "test"):
        path = directory / f"sl_{split}_{cfg.experiment.fold}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"0", "1"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(f"split CSV lacks gene columns: {path}")
            for row in reader:
                gene_ids.add(int(row["0"]))
                gene_ids.add(int(row["1"]))
    return gene_ids


def _selected_proteins(cfg: ProjectConfig) -> list[tuple[int, str]]:
    sequences = _load_sequences(cfg.paths.sequences)
    gene_ids = _split_gene_ids(cfg)
    missing = sorted(gene_ids.difference(sequences))
    if missing:
        raise ValueError(f"split gene ID lacks sequence: {missing[0]}")
    selected = (
        (gene_id, _normalize_sequence(sequences[gene_id]))
        for gene_id in gene_ids
    )
    return sorted(
        selected,
        key=lambda item: (len(item[1]), item[0]),
    )


def _cache_payloads(
    cache: ProteinCache,
    provenance_fingerprint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    common: dict[str, object] = {
        "gene_id": cache.gene_id,
        "sequence_sha256": cache.sequence_sha256,
        "provenance_schema_version": EXTRACTION_SCHEMA_VERSION,
        "provenance_fingerprint": provenance_fingerprint,
    }
    dense = {**common, "h60": cache.h60}
    sparse = {
        **common,
        "sparse": {
            "indices": cache.sparse.indices,
            "values": cache.sparse.values,
        },
    }
    return dense, sparse


def _cache_is_current(
    dense_path: Path,
    sparse_path: Path,
    *,
    gene_id: int,
    sequence_sha256: str,
    provenance_fingerprint: str,
) -> bool:
    if not dense_path.is_file() or not sparse_path.is_file():
        return False
    try:
        dense = torch.load(dense_path, map_location="cpu", weights_only=True)
        sparse = torch.load(sparse_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
        return False
    expected = {
        "gene_id": gene_id,
        "sequence_sha256": sequence_sha256,
        "provenance_schema_version": EXTRACTION_SCHEMA_VERSION,
        "provenance_fingerprint": provenance_fingerprint,
    }
    return all(
        all(payload.get(key) == value for key, value in expected.items())
        for payload in (dense, sparse)
    )


def _append_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--gene-id", type=int)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--smoke-shortest", type=_positive_integer)
    return parser


def _choose_rows(
    rows: list[tuple[int, str]],
    args: argparse.Namespace,
) -> list[tuple[int, str]]:
    if args.gene_id is not None:
        selected = [row for row in rows if row[0] == args.gene_id]
        if not selected:
            raise ValueError(
                f"gene ID is absent from configured splits: {args.gene_id}"
            )
        return selected
    if args.smoke_shortest is not None:
        return rows[: args.smoke_shortest]
    return rows


def run_extraction(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.cache.extraction_batch_size != 1:
        raise ValueError("extraction_batch_size must equal 1")
    rows = _choose_rows(_selected_proteins(cfg), args)

    import transformers

    model, sae, sae_layer, tokenizer = load_local_models(cfg)
    runtime_binding = _effective_runtime_binding(model)
    provenance, fingerprint = _build_provenance(
        cfg,
        transformers_version=transformers.__version__,
        runtime_binding=runtime_binding,
        model_compute_dtype=_dtype_name(next(model.parameters()).dtype),
        sae_compute_dtype=_dtype_name(sae_layer.W_enc.dtype),
    )
    del sae
    export_decoder_bridge(
        cfg.paths.decoder_bridge,
        sae_layer,
        cfg,
        provenance,
    )

    manifest_path = cfg.paths.project_root / "manifests" / "extraction.jsonl"
    timings: list[float] = []
    extracted = 0
    skipped = 0
    eta_reported = False
    total = len(rows)
    print(
        f"EXTRACTION selected={total} batch_size=1 "
        f"order=length_then_gene_id",
        flush=True,
    )
    for ordinal, (gene_id, sequence) in enumerate(rows, start=1):
        sequence_sha256 = _sequence_sha256(sequence)
        dense_path = cfg.paths.dense_cache / f"{gene_id}.pt"
        sparse_path = cfg.paths.sparse_cache / f"{gene_id}.pt"
        if _cache_is_current(
            dense_path,
            sparse_path,
            gene_id=gene_id,
            sequence_sha256=sequence_sha256,
            provenance_fingerprint=fingerprint,
        ):
            skipped += 1
            _append_manifest(
                manifest_path,
                {
                    "gene_id": gene_id,
                    "sequence_sha256": sequence_sha256,
                    "provenance_fingerprint": fingerprint,
                    "length": len(sequence),
                    "status": "skipped_current",
                    "timestamp": _timestamp(),
                },
            )
            print(
                f"SKIP {ordinal}/{total} gene_id={gene_id} "
                f"length={len(sequence)}",
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
            manifest_path,
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
            f"DONE {ordinal}/{total} gene_id={gene_id} "
            f"length={len(sequence)} seconds={elapsed:.3f}",
            flush=True,
        )
        if len(timings) == 10 and not eta_reported:
            remaining = total - ordinal
            median_seconds = statistics.median(timings)
            eta_seconds = remaining * median_seconds
            print(
                f"ETA_ONCE remaining={remaining} "
                f"median_seconds={median_seconds:.3f} "
                f"eta_seconds={eta_seconds:.3f}",
                flush=True,
            )
            eta_reported = True

    print(
        f"EXTRACTION_COMPLETE selected={total} extracted={extracted} "
        f"skipped={skipped} provenance={fingerprint}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_extraction(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
