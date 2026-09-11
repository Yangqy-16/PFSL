"""Build one target-held-out, pair-grouped all-context split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence
import zlib

import pandas as pd

from esmc_sae.case_split import (
    _csv_bytes,
    _sequence_map,
    _write_idempotent,
)
from esmc_sae.manifest import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/SLKB/SLKB_processed.csv"
DEFAULT_SEQUENCES = ROOT / "data/SLKB/all_id_seq.pkl"
DEFAULT_OMICS = Path("/data/qingyuyang/SL/MVGCNiSL/data/cellline_feats")
DEFAULT_OUTPUT = ROOT / "data/SLKB/CASE_BRCA2_PARP1_ALL_CONTEXT"
DEFAULT_TARGET_GENE_A = 675
DEFAULT_TARGET_GENE_B = 142
DEFAULT_TARGET_CONTEXT = "OVCAR8"
DEFAULT_CASE_ID = "ovcar8_brca2_parp1_all_context"
OMICS_KINDS = ("cnv", "ess", "exp", "mut")


def complete_omics_contexts(
    source: pd.DataFrame,
    omics_root: Path,
) -> list[str]:
    """Return source contexts having every required omics file."""

    root = Path(omics_root)
    return [
        context
        for context in sorted(str(value) for value in source["context"].unique())
        if all((root / f"{context}_{kind}.txt").is_file() for kind in OMICS_KINDS)
    ]


def _pair_bucket(gene_a: int, gene_b: int) -> int:
    canonical = tuple(sorted((int(gene_a), int(gene_b))))
    return zlib.crc32(f"{canonical[0]}:{canonical[1]}".encode("ascii")) % 10


def _validate_source(source: pd.DataFrame) -> None:
    expected = ["context", "gene1", "gene2", "class", "score"]
    if list(source.columns) != expected:
        raise ValueError(f"SLKB_processed columns must equal {expected}")
    if not set(source["class"]) <= {0, 1}:
        raise ValueError("class must be binary")


def build_all_context_split(
    source_csv: Path,
    sequences_path: Path,
    omics_root: Path,
    output_dir: Path,
    *,
    target_gene_a: int = DEFAULT_TARGET_GENE_A,
    target_gene_b: int = DEFAULT_TARGET_GENE_B,
    target_context: str = DEFAULT_TARGET_CONTEXT,
    case_id: str = DEFAULT_CASE_ID,
    holdout_scope: str = "gene",
) -> dict[str, object]:
    """Create immutable train/validation/test data and a source audit."""

    target_gene_a = int(target_gene_a)
    target_gene_b = int(target_gene_b)
    if holdout_scope not in {"gene", "pair"}:
        raise ValueError("holdout_scope must be 'gene' or 'pair'")
    if target_gene_a == target_gene_b:
        raise ValueError("target genes must be distinct")
    target_genes = frozenset({target_gene_a, target_gene_b})
    target_low, target_high = sorted(target_genes)
    source_csv = Path(source_csv)
    sequences_path = Path(sequences_path)
    output_dir = Path(output_dir)
    source = pd.read_csv(source_csv)
    _validate_source(source)
    source = source.copy()
    source.insert(
        0,
        "row_id",
        [f"slkb_{index:06d}" for index in range(len(source))],
    )
    source["a"] = source[["gene1", "gene2"]].min(axis=1).astype(int)
    source["b"] = source[["gene1", "gene2"]].max(axis=1).astype(int)

    target = source[
        (source["context"] == target_context)
        & (source["a"] == target_low)
        & (source["b"] == target_high)
    ]
    if len(target) != 1 or int(target.iloc[0]["class"]) != 1:
        raise RuntimeError(
            f"locked {target_context} {target_gene_a}-{target_gene_b} "
            "positive case is not unique"
        )
    locked_frame = pd.DataFrame(
        {
            "row_id": [str(target.iloc[0]["row_id"])],
            "context": [target_context],
            "0": [target_gene_a],
            "1": [target_gene_b],
            "2": [1],
        }
    )

    contexts = complete_omics_contexts(source, omics_root)
    eligible = source[source["context"].isin(contexts)].copy()
    complete_context_source_rows = len(eligible)
    if holdout_scope == "gene":
        quarantine = eligible["gene1"].isin(target_genes) | eligible[
            "gene2"
        ].isin(target_genes)
    else:
        quarantine = (eligible["a"] == target_low) & (
            eligible["b"] == target_high
        )
    target_holdout_rows = int(quarantine.sum())
    target_gene_quarantine_rows = (
        target_holdout_rows if holdout_scope == "gene" else 0
    )
    eligible = eligible.loc[~quarantine].copy()

    key = ["context", "a", "b"]
    label_counts = eligible.groupby(key, sort=False)["class"].nunique()
    conflicts = label_counts[label_counts > 1]
    conflict_index = set(conflicts.index)
    row_keys = list(eligible[key].itertuples(index=False, name=None))
    conflict_mask = pd.Series(
        [row_key in conflict_index for row_key in row_keys],
        index=eligible.index,
    )
    conflict_rows_dropped = int(conflict_mask.sum())
    eligible = eligible.loc[~conflict_mask].copy()
    before_deduplication = len(eligible)
    eligible = eligible.drop_duplicates(key, keep="first").copy()
    same_label_duplicate_rows_dropped = before_deduplication - len(eligible)

    frame = pd.DataFrame(
        {
            "row_id": eligible["row_id"].astype(str),
            "context": eligible["context"].astype(str),
            "0": eligible["a"].astype(int),
            "1": eligible["b"].astype(int),
            "2": eligible["class"].astype(int),
        }
    )
    buckets = [
        _pair_bucket(gene_a, gene_b)
        for gene_a, gene_b in frame[["0", "1"]].itertuples(
            index=False,
            name=None,
        )
    ]
    validation_mask = pd.Series(buckets, index=frame.index) == 0
    train = frame.loc[~validation_mask].sort_values(
        "row_id", kind="stable"
    ).reset_index(drop=True)
    validation = frame.loc[validation_mask].sort_values(
        "row_id", kind="stable"
    ).reset_index(drop=True)

    train_pairs = set(train[["0", "1"]].itertuples(index=False, name=None))
    validation_pairs = set(
        validation[["0", "1"]].itertuples(index=False, name=None)
    )
    if train_pairs & validation_pairs:
        raise RuntimeError("unordered pair leakage across train/validation")
    supervised_genes = set(frame["0"]) | set(frame["1"])
    target_gene_a_supervised_rows = int(
        ((frame["0"] == target_gene_a) | (frame["1"] == target_gene_a)).sum()
    )
    target_gene_b_supervised_rows = int(
        ((frame["0"] == target_gene_b) | (frame["1"] == target_gene_b)).sum()
    )
    supervised_target_gene_disjoint = not bool(target_genes & supervised_genes)
    supervised_target_pair_disjoint = (
        (target_low, target_high) not in train_pairs
        and (target_low, target_high) not in validation_pairs
    )
    if not supervised_target_pair_disjoint:
        raise RuntimeError("target pair leaked into train/validation")

    train_path = output_dir / "sl_train.csv"
    validation_path = output_dir / "sl_val.csv"
    test_path = output_dir / "sl_test.csv"
    locked_path = output_dir / "locked_case.csv"
    _write_idempotent(train_path, _csv_bytes(train))
    validation_bytes = _csv_bytes(validation)
    _write_idempotent(validation_path, validation_bytes)
    _write_idempotent(test_path, validation_bytes)
    _write_idempotent(locked_path, _csv_bytes(locked_frame))

    sequences = _sequence_map(sequences_path)
    genes = sorted(set(frame["0"]) | set(frame["1"]))
    missing_sequences = sorted(set(genes).difference(sequences))
    if missing_sequences:
        raise RuntimeError(f"eligible gene lacks sequence: {missing_sequences[0]}")
    gene_frame = pd.DataFrame(
        {
            "gene_id": genes,
            "input_length": [len(sequences[gene_id]) for gene_id in genes],
            "sequence_sha256": [
                hashlib.sha256(sequences[gene_id].encode("ascii")).hexdigest()
                for gene_id in genes
            ],
        }
    )
    genes_path = output_dir / "genes.csv"
    _write_idempotent(genes_path, _csv_bytes(gene_frame))

    manifest_path = output_dir / "split_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "case_id": case_id,
        "target_context": target_context,
        "target_gene_a": target_gene_a,
        "target_gene_b": target_gene_b,
        "holdout_scope": holdout_scope,
        "source_file": str(source_csv),
        "source_sha256": file_sha256(source_csv),
        "complete_contexts": contexts,
        "complete_context_source_rows": complete_context_source_rows,
        "target_holdout_rows": target_holdout_rows,
        "target_gene_quarantine_rows": target_gene_quarantine_rows,
        "target_gene_a_supervised_rows": target_gene_a_supervised_rows,
        "target_gene_b_supervised_rows": target_gene_b_supervised_rows,
        "conflict_groups_dropped": len(conflict_index),
        "conflict_rows_dropped": conflict_rows_dropped,
        "same_label_duplicate_rows_dropped": same_label_duplicate_rows_dropped,
        "eligible_rows": len(frame),
        "eligible_positive": int(frame["2"].sum()),
        "eligible_negative": int((frame["2"] == 0).sum()),
        "eligible_genes": len(genes),
        "train_rows": len(train),
        "train_positive": int(train["2"].sum()),
        "validation_rows": len(validation),
        "validation_positive": int(validation["2"].sum()),
        "validation_bucket": 0,
        "pair_hash": "crc32(canonical_gene_a:canonical_gene_b) % 10",
        "pair_group_overlap_zero": True,
        "supervised_target_pair_disjoint": supervised_target_pair_disjoint,
        "supervised_target_gene_disjoint": supervised_target_gene_disjoint,
        "train_file": str(train_path),
        "train_sha256": file_sha256(train_path),
        "validation_file": str(validation_path),
        "validation_sha256": file_sha256(validation_path),
        "test_file": str(test_path),
        "test_sha256": file_sha256(test_path),
        "locked_case_file": str(locked_path),
        "locked_case_sha256": file_sha256(locked_path),
        "genes_file": str(genes_path),
        "genes_sha256": file_sha256(genes_path),
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
    parser.add_argument("--omics-root", type=Path, default=DEFAULT_OMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-gene-a", type=int, default=DEFAULT_TARGET_GENE_A)
    parser.add_argument("--target-gene-b", type=int, default=DEFAULT_TARGET_GENE_B)
    parser.add_argument("--target-context", default=DEFAULT_TARGET_CONTEXT)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument(
        "--holdout-scope",
        choices=("gene", "pair"),
        default="gene",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_all_context_split(
        args.source,
        args.sequences,
        args.omics_root,
        args.output_dir,
        target_gene_a=args.target_gene_a,
        target_gene_b=args.target_gene_b,
        target_context=args.target_context,
        case_id=args.case_id,
        holdout_scope=args.holdout_scope,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
