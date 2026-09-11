# PFSL

PFSL is a synthetic lethality (SL) prediction codebase using protein language
model representations and additional biological modalities. It includes the
ESM4SL model components and an ESM-C/SAE workflow for studying dependencies of
the ESM-C branch.

## Repository layout

- `esm4sl/`: the main ESM4SL model and training components.
- `train.py`, `test.py`, `tent.py`: established root entry points.
- `script/`: training, evaluation, transfer, and analysis launch scripts.
- `data_preprocess/`: data preparation notebooks and utilities.
- `pfsl/` and `pfsl_process/`: protein-function SL components and SAE utilities.
- `esmc_sae/`: ESM-C/SAE interpretability workflow; see `esmc_sae/README.md`.
- `esmc_sae/tests/`: ESM-C interpretability unit and regression tests.
- `tests/pfsl_process/sae/`: ProTrek-SAE unit and regression tests.
- `coach_pl/`: vendored upstream training framework source.

## Environment

Create the project environment from the repository root:

```bash
bash environment.sh
```

## Data preprocessing

Data-preparation notebooks and utilities are under `data_preprocess/`.

## Main training entry point

A basic ESM4SL run can be launched with:

```bash
bash script/attn/train.sh
```

After training, inspect its TensorBoard output with:

```bash
tensorboard --logdir output/attn/<name_of_your_run>
```

For multi-cell-line or multi-scene runs, see
`script/attn/specific_all_in_one.py`. The corresponding `script/mlp/`
launchers provide the ESM-2+MLP baseline.

## Local artifacts

Datasets, pretrained weights, embeddings, caches, checkpoints, logs, and
experiment outputs are intentionally excluded from Git. Existing local files
remain in their repository-relative locations, and active configurations use
the `PFSL` root. Historical artifacts are not rewritten, so any embedded
absolute path containing the former directory name may require manual
resolution. A clone requires the corresponding private data and model assets
to be restored separately.

The checksum-bound `esmc_sae/configs/mvp_a549_c1_fold0.yaml` is retained as a
frozen historical configuration and therefore still records the former root.

## Tests

```bash
python -m pytest esmc_sae/tests -q
python -m pytest tests/pfsl_process/sae -q
```

## Vendored dependency

`coach_pl/` is vendored from https://github.com/DuskNgai/coach-pl. Its upstream
README and LICENSE are retained in that directory.

## References and attribution

The training framework is based on
[coach-pl](https://github.com/DuskNgai/coach-pl), and portions of the protein
language model integration are credited to
[ESM](https://github.com/facebookresearch/esm).
