# PF4SL (PFSL codebase)

PF4SL predicts synthetic lethality from UniProt protein functional text,
protein sequence representations, and omics measurements from the current
cell line (CNV, ESS, expression, and mutation). The current paper model projects
and concatenates these modalities without PPI neighbor aggregation.

## Retained source

- `pfsl/next/fusion.py`: the current `NoGraphFusion` model.
- `pfsl/omics_scaling.py`: normalization fitted on training genes only.
- `script/train_pfsl_next.py`: current training and evaluation entry point,
  including structural modality ablations.
- `script/prepare_ppi.py`, `script/prepare_next_bank.py`: shared feature-bank
  preparation. These historical bank formats also contain graph and reference
  features; the `no_graph_concat` model does not aggregate neighbors or use the
  reference-cell dependency profile.
- `train.py`, `test.py`, `script/pf.py`, `pfsl/configuration/prosl.yaml`:
  original configuration-based entry points and their compatibility modules.
- `data_preprocess/preprocess.ipynb`, `data_preprocess/utils.py`,
  `data_preprocess/ccle_filter.ipynb`: dataset splitting and preparation.
- `pfsl_process/pf.ipynb`, `pfsl_process/text_pre.py`,
  `pfsl_process/utils.py`: original UniProt text preparation helpers.
- `script/slkb_baseline_common.py`: shared split validation/loading used by
  the main runner's multi-source option; it contains no baseline model.
- `tests/test_strict_protocol.py`: normalization and structural ablation checks.
- `coach_pl/`: vendored training framework, with upstream README and LICENSE.

Some graph, DA, and legacy model modules remain because the existing entry
points import them. Keeping these dependencies preserves the training logic;
their presence does not enable them in the current paper model.

## Environment and local assets

The experiments use the `DTI_new` environment:

```bash
conda activate DTI_new
```

`environment.sh` is the historical environment recipe and uses an older
environment name. Dataset preparation also retains local asset paths; adjust
these paths when moving the project to another machine.

Data and pretrained representations are restored separately from Git. The
current runner expects:

- Frozen feature bank: `data/ppi/next_v1.pt` (or pass `--bank`).
- Gene mapping: `data/mapping/name2id.pkl`.
- Specific splits: `data/SLKB/specific_1:1/<cell>/<scene>/sl_{train,val,test}_<fold>.csv`;
  original-ratio splits use `data/SLKB/specific/`.
- Current-cell omics: `../MVGCNiSL/data/cellline_feats/<cell>_{cnv,ess,exp,mut}.txt`.
  This directory supplies data files; the model does not import MVGCN-iSL code.

## Current paper model

From the repository root, for one balanced A549 gene-holdout fold:

```bash
python script/train_pfsl_next.py --stage no_graph_concat --fusion-ablation full \
  --ablation-mode remove --omics-normalization train_genes \
  --source A549 --scene C3 --fold 0 --data-root data/SLKB/specific_1:1 \
  --device cuda --output output/pf4sl/A549/C3/fold0
```

`C1` denotes pair-holdout and `C3` denotes gene-holdout. For original-ratio
experiments, set `--data-root data/SLKB/specific`. To remove a modality, replace
`full` with `no_text`, `no_seq`, or `no_omics`; `--ablation-mode remove` removes
that branch. Keep `--omics-normalization train_genes` explicit: the runner
retains historical defaults for compatibility. This normalization option
currently supports specific experiments, not cross-cell transfer.

Each run requires an empty output directory and writes the checkpoint,
configuration, normalization statistics, metrics, and pair-level predictions
there. Existing experiment outputs are not overwritten by this command.

```bash
python script/train_pfsl_next.py --help
python -m pytest tests/test_strict_protocol.py -q
```

## Git scope

`.gitignore` uses an explicit source allowlist. SAE/LLM interpretability,
baseline adapters, exploratory experiments, manuscript files, datasets,
pretrained weights, cached features, and results stay local. Ignoring a file
does not delete it. Add a matching exception to `.gitignore` when introducing
a new main-model source file. Do not use `git add -f` to publish ignored assets.

## Attribution

The training framework is vendored from
[coach-pl](https://github.com/DuskNgai/coach-pl); its upstream README and LICENSE
are retained. Protein language model integration includes components based on
[ESM](https://github.com/facebookresearch/esm).
