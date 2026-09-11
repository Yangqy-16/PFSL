"""Create the locked OVCAR8 BRCA2-PARP1 five-fold case-study data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from esmc_sae.manifest import file_sha256


CODE_ROOT = Path("/data/qingyuyang/SL/PFSL")
DEFAULT_SOURCE = CODE_ROOT / "data/SLKB/cell_line/table/OVCAR8.csv"
DEFAULT_SEQUENCES = CODE_ROOT / "data/SLKB/all_id_seq.pkl"
DEFAULT_OUTPUT = CODE_ROOT / "data/SLKB/CASE_BRCA2_PARP1"
DENSE_CACHE = CODE_ROOT / "esmc_sae/cache/layer60_dense"
SPARSE_CACHE = CODE_ROOT / "esmc_sae/cache/sae_raw"
TARGET_GENES = frozenset({675, 142})
TARGET_PAIR = (142, 675)
LOCKED_COUNTS = {
    "source_rows": 966,
    "quarantined_rows": 47,
    "eligible_rows": 919,
    "eligible_positive": 34,
    "eligible_negative": 885,
    "eligible_gene_count": 60,
    "all_gene_count": 62,
}


def _write_idempotent(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sequence_map(path: Path) -> dict[int, str]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    result: dict[int, str] = {}
    for row in rows:
        if (
            not isinstance(row, (tuple, list))
            or len(row) != 2
            or type(row[0]) is not int
            or not isinstance(row[1], str)
        ):
            raise ValueError("all_id_seq.pkl contains an invalid row")
        if row[0] in result:
            raise ValueError(f"duplicate sequence gene ID: {row[0]}")
        result[row[0]] = row[1]
    return result


def _canonical_pairs(frame: pd.DataFrame) -> pd.Series:
    return frame[["0", "1"]].apply(
        lambda row: tuple(sorted((int(row["0"]), int(row["1"])))),
        axis=1,
    )


def _validate_source(source: pd.DataFrame) -> pd.Series:
    if list(source.columns) != ["0", "1", "2"]:
        raise ValueError("OVCAR8 source columns must equal 0,1,2")
    if any(
        not pd.api.types.is_integer_dtype(source[column])
        for column in ("0", "1", "2")
    ):
        raise ValueError("OVCAR8 source columns must be integer")
    if not set(source["2"]) <= {0, 1}:
        raise ValueError("OVCAR8 source labels must be binary")
    canonical = _canonical_pairs(source)
    if canonical.duplicated().any():
        raise ValueError("OVCAR8 source contains duplicate unordered pairs")
    target = source.loc[canonical == TARGET_PAIR]
    if len(target) != 1 or int(target.iloc[0]["2"]) != 1:
        raise ValueError("BRCA2-PARP1 must occur exactly once with label 1")
    return canonical


def _require_locked_counts(counts: dict[str, int]) -> None:
    for name, expected in LOCKED_COUNTS.items():
        if counts[name] != expected:
            raise RuntimeError(
                f"{name} changed: expected {expected}, observed {counts[name]}"
            )


def build_case_splits(
    source_csv: Path,
    sequences_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Write deterministic folds and return the byte-stable split manifest."""

    if type(seed) is not int or seed != 42:
        raise ValueError("case split seed must equal 42")
    source_csv = Path(source_csv)
    sequences_path = Path(sequences_path)
    output_dir = Path(output_dir)
    source = pd.read_csv(source_csv)
    canonical = _validate_source(source)
    source = source.copy()
    source.insert(
        0,
        "row_id",
        [f"ovcar8_{index:06d}" for index in range(len(source))],
    )

    quarantine_mask = source["0"].isin(TARGET_GENES) | source["1"].isin(
        TARGET_GENES
    )
    eligible = source.loc[~quarantine_mask].reset_index(drop=True)
    eligible_genes = set(eligible["0"]) | set(eligible["1"])
    all_genes = set(source["0"]) | set(source["1"])
    counts = {
        "source_rows": len(source),
        "quarantined_rows": int(quarantine_mask.sum()),
        "eligible_rows": len(eligible),
        "eligible_positive": int(eligible["2"].sum()),
        "eligible_negative": int((eligible["2"] == 0).sum()),
        "eligible_gene_count": len(eligible_genes),
        "all_gene_count": len(all_genes),
    }
    _require_locked_counts(counts)
    if TARGET_GENES & eligible_genes:
        raise RuntimeError("supervised case-gene leakage after quarantine")

    target_row = source.loc[canonical == TARGET_PAIR, ["row_id", "0", "1", "2"]]
    locked_case = output_dir / "locked_case.csv"
    _write_idempotent(locked_case, _csv_bytes(target_row.reset_index(drop=True)))

    fold_rows: list[dict[str, object]] = []
    validation_ids: list[str] = []
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    for fold, (train_index, val_index) in enumerate(
        splitter.split(eligible, eligible["2"])
    ):
        train = eligible.iloc[train_index].reset_index(drop=True)
        validation = eligible.iloc[val_index].reset_index(drop=True)
        train_path = output_dir / f"sl_train_{fold}.csv"
        val_path = output_dir / f"sl_val_{fold}.csv"
        test_path = output_dir / f"sl_test_{fold}.csv"
        _write_idempotent(train_path, _csv_bytes(train))
        validation_bytes = _csv_bytes(validation)
        _write_idempotent(val_path, validation_bytes)
        _write_idempotent(test_path, validation_bytes)
        validation_ids.extend(str(value) for value in validation["row_id"])
        fold_rows.append(
            {
                "fold": fold,
                "train_file": str(train_path),
                "train_rows": len(train),
                "train_positive": int(train["2"].sum()),
                "train_sha256": file_sha256(train_path),
                "val_file": str(val_path),
                "val_rows": len(validation),
                "val_positive": int(validation["2"].sum()),
                "val_sha256": file_sha256(val_path),
                "test_file": str(test_path),
                "test_sha256": file_sha256(test_path),
            }
        )
    if len(validation_ids) != 919 or len(set(validation_ids)) != 919:
        raise RuntimeError("validation folds do not exactly cover eligible rows")

    sequences = _sequence_map(sequences_path)
    missing_sequences = sorted(all_genes.difference(sequences))
    if missing_sequences:
        raise RuntimeError(f"case gene lacks sequence: {missing_sequences[0]}")
    gene_rows = []
    for gene_id in sorted(all_genes):
        sequence = sequences[gene_id]
        cache_present = (
            (DENSE_CACHE / f"{gene_id}.pt").is_file()
            and (SPARSE_CACHE / f"{gene_id}.pt").is_file()
        )
        gene_rows.append(
            {
                "gene_id": gene_id,
                "sequence_length": len(sequence),
                "sequence_sha256": hashlib.sha256(
                    sequence.encode("ascii")
                ).hexdigest(),
                "cache_status": "present" if cache_present else "missing",
            }
        )
    genes = pd.DataFrame(gene_rows)
    genes_path = output_dir / "genes.csv"
    _write_idempotent(genes_path, _csv_bytes(genes))
    status_counts = genes["cache_status"].value_counts().to_dict()
    cache_present = int(status_counts.get("present", 0))
    cache_missing = int(status_counts.get("missing", 0))
    if cache_present + cache_missing != len(all_genes):
        raise RuntimeError(f"invalid cache coverage: {status_counts}")

    manifest_path = output_dir / "split_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "case_id": "ovcar8_brca2_parp1",
        "seed": seed,
        **counts,
        "target": {
            "cell_line": "OVCAR8",
            "gene_a": 675,
            "gene_b": 142,
            "label": 1,
            "brca2_input_length": 2000,
            "parp1_input_length": 1014,
        },
        "source_file": str(source_csv),
        "source_sha256": file_sha256(source_csv),
        "case_file": str(locked_case),
        "case_sha256": file_sha256(locked_case),
        "gene_file": str(genes_path),
        "gene_sha256": file_sha256(genes_path),
        "cache_present": cache_present,
        "cache_missing": cache_missing,
        "folds": fold_rows,
        "validation_exact_coverage": True,
        "supervised_case_gene_disjoint": True,
        "manifest_file": str(manifest_path),
    }
    encoded = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _write_idempotent(manifest_path, encoded)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_case_splits(
        args.source,
        args.sequences,
        args.output_dir,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
