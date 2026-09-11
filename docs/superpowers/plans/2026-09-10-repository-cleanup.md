# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `ESM4SL_modified` to `PFSL` and prepare it as a clear, private, source-oriented Git repository without changing implementation logic, runtime interfaces, or experiment results.

**Architecture:** Keep the internal source tree and all public Python module paths intact while changing the repository root to `/data/qingyuyang/SL/PFSL`. Apply only a literal old-root-to-new-root substitution in runtime-bearing source, configurations, tests, and notebooks; preserve historical artifacts unchanged and create no compatibility symlink. Separate source visibility from local scientific artifacts through a precise root `.gitignore`, document the existing workflow instead of moving it, and convert the broken `coach_pl` submodule pointer into ordinary attributed vendored source.

**Tech Stack:** Git, Bash, Python, pytest, Markdown

**Spec:** `docs/superpowers/specs/2026-09-10-repository-cleanup-design.md`

## Global Constraints

- Do not change model, data-processing, training, prediction, attribution, intervention, calibration, or reporting logic.
- Permit only mechanical `ESM4SL_modified` to `PFSL` path substitutions in active source files, configurations, tests, notebooks, and current repository documentation where required by the root-directory rename.
- Do not delete datasets, checkpoints, caches, logs, reports, or previous experiment outputs.
- Preserve all `esmc_sae.*` import paths and `python -m esmc_sae.*` commands.
- Preserve root entry points such as `train.py`, `test.py`, and `tent.py`.
- Keep `esmc_sae/tests/` and `tests/pfsl_process/sae/` as the two existing test-suite locations.
- Keep runtime cache audits, manifest verification, schema validation, and fail-closed artifact checks in production modules.
- Store `coach_pl` as ordinary vendored source while preserving its implementation, README, LICENSE, and upstream attribution.
- Do not move local data, weights, caches, or result directories.
- Do not rewrite historical reports, manifests, logs, or experiment outputs.
- Do not create an `ESM4SL_modified` compatibility symlink after the rename.
- Do not stage, commit, push, create a remote repository, or expose credentials.
- Rename the root to `/data/qingyuyang/SL/PFSL` and initialize it with branch name `main` only after the pre-rename cleanup is complete.

---

### Task 1: Establish a regression baseline

**Files:**
- Read: `docs/superpowers/specs/2026-09-10-repository-cleanup-design.md`
- Read: `esmc_sae/tests/*.py`
- Read: `tests/pfsl_process/sae/*.py`
- Create after all tasks: `docs/repository-cleanup-report.md`

**Interfaces:**
- Consumes: the unchanged working tree and its currently installed Python environment.
- Produces: recorded pre-change results for both test suites and a list of existing failures, if any.

- [x] **Step 1: Confirm that this task does not have an available Git worktree**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: failure with `fatal: not a git repository`; perform the approved cleanup in place because Git initialization is itself a deliverable.

- [x] **Step 2: Run the ESMC-SAE baseline suite**

Run:

```bash
python -m pytest esmc_sae/tests -q
```

Expected: all currently passing tests remain green; record passed, skipped, failed, and elapsed counts verbatim for the final report.

- [x] **Step 3: Run the ProTrek-SAE baseline suite**

Run:

```bash
python -m pytest tests/pfsl_process/sae -q
```

Expected: all currently passing tests remain green; record passed, skipped, failed, and elapsed counts verbatim for the final report.

- [x] **Step 4: Inventory production Python files and test definitions**

Run:

```bash
find esmc_sae -maxdepth 1 -type f -name '*.py' -print | sort
find esmc_sae/tests tests/pfsl_process/sae -type f -name 'test_*.py' -print | sort
rg -n '^def test_|^class Test' esmc_sae --glob '*.py' --glob '!esmc_sae/tests/**'
find coach_pl data_preprocess esm4sl esmc_sae pfsl pfsl_process script tests \
  -type f -name '*.py' -print0 |
  sort -z |
  xargs -0 sha256sum > /tmp/esm4sl_repository_cleanup_python_before.sha256
sha256sum train.py test.py tent.py \
  >> /tmp/esm4sl_repository_cleanup_python_before.sha256
```

Expected: production modules are inventoried; the `rg` command returns no embedded test definitions; the hash snapshot covers every Python implementation and test file in the source-oriented repository. Runtime validators such as `audit_pooled_cache`, `audit_case_cache`, `validate_evidence_packet`, `validate_llm_mechanism`, and `verify_manifest_target` are not moved because they enforce runtime experiment contracts.

---

### Task 2: Define the source/artifact boundary and vendor `coach_pl`

**Files:**
- Modify: `.gitignore`
- Delete: `.gitmodules`
- Delete: `coach_pl/.git`
- Preserve unchanged: `coach_pl/LICENSE`
- Preserve unchanged: `coach_pl/README.md`
- Preserve unchanged: all other files below `coach_pl/`

**Interfaces:**
- Consumes: existing source and local artifact paths.
- Produces: Git ignore rules that retain source/configuration/documentation and exclude local data, weights, results, caches, credentials, and generated files; a normal vendored `coach_pl` directory with no broken nested Git pointer.

- [x] **Step 1: Replace broad legacy ignore rules with explicit categories**

Replace `.gitignore` with:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.tox/
.nox/
.coverage
.coverage.*
htmlcov/

# Notebooks and editors
.ipynb_checkpoints/
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Environments and local configuration
.env
.env.*
!.env.example
.venv/
venv/
env/
*.pem
*.key
*.p12
*.pfx
credentials*.json
secrets*.json

# Local credential-bearing source (do not commit until the key is removed)
/pfsl_process/cl_llm.py

# Runtime logs and experiment tracking
*.log
logs/
runs/
lightning_logs/
wandb/
tb_logs/

# Project datasets and generated results
/data/
/sc_data/
/output/
/all_methods_count/
/all_methods_metrics/
/all_methods_rank_count/
/pfsl_metric_count/

# ESMC-SAE local artifacts and historical experiment records
/.superpowers/
/esmc_sae/.superpowers/
/esmc_sae/cache/
/esmc_sae/case_study/
/esmc_sae/docs/
/esmc_sae/experiment2/
/esmc_sae/experiment3/
/esmc_sae/interpret_data/
/esmc_sae/manifests/
/esmc_sae/reports/
/esmc_sae/runs/

# Local pretrained models, embeddings, and derived analyses
/pfsl_process/ESMC/
/pfsl_process/PubmedBERT/
/pfsl_process/cl_embed/
/pfsl_process/retrieval/
/pfsl_process/att_map/
/pfsl_process/comparison/
/pfsl_process/func_aspect_ablation/
/pfsl_process/func_aspect_ablation_use/
/pfsl_process/new_ablation/

# Model weights and serialized/large data
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.msgpack
*.pkl
*.pickle
*.npy
*.npz
*.h5
*.hdf5
*.h5ad
*.arrow
*.feather
*.parquet
*.jsonl
*.jsonl.gz
*.csv.gz
*.tsv.gz
*.tar
*.tar.gz
*.tgz
*.zip
*.7z
```

This removes the legacy broad patterns `script`, `*lora*`, `*old*`, and `*hw3*`, which currently hide source scripts and notebooks merely because of their names.

- [x] **Step 2: Verify representative ignore behavior before Git initialization**

Run:

```bash
git check-ignore -q data
git check-ignore -q output
git check-ignore -q esmc_sae/cache
git check-ignore -q esmc_sae/reports
git check-ignore -q pfsl_process/ESMC
```

Expected before Git initialization: each command fails only because there is no repository. Repeat these checks in Task 5 after `git init`.

- [x] **Step 3: Remove the obsolete submodule metadata**

Delete only:

```text
.gitmodules
coach_pl/.git
```

Before deletion, verify `coach_pl/.git` contains the broken pointer `gitdir: ../.git/modules/coach_pl` and that `.git/modules/coach_pl` does not exist. Do not edit or delete any other file inside `coach_pl`.

- [x] **Step 4: Verify vendored source integrity**

Run:

```bash
test -f coach_pl/LICENSE
test -f coach_pl/README.md
test ! -e coach_pl/.git
test ! -e .gitmodules
find coach_pl -type f -name '*.py' -print | sort
```

Expected: license, upstream README, and implementation files remain present; only the broken Git metadata is absent.

---

### Task 3: Add repository navigation without moving code

**Files:**
- Modify: `README.md`
- Create: `esmc_sae/README.md`
- Create: `tests/README.md`
- Preserve unchanged: `esmc_sae/*.py`
- Preserve unchanged: `esmc_sae/tests/*.py`
- Preserve unchanged: `tests/pfsl_process/sae/*.py`

**Interfaces:**
- Consumes: the existing root entry points, source packages, experiment modules, and test suites.
- Produces: a GitHub-facing root guide, an ESMC-SAE workflow map, and a test-suite guide.

- [x] **Step 1: Expand the root README**

Replace the minimal README with the following complete content:

````markdown
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
````

- [x] **Step 2: Create the ESMC-SAE workflow map**

Create `esmc_sae/README.md` with the following module groups:

````markdown
# ESMC-SAE Interpretability

This package implements the ESM-C branch interpretability workflow. Production
modules remain at package root to preserve established imports, CLI commands,
manifests, reports, and regression-test monkeypatch targets.

## Experiment 1: representation extraction and SAE audit

- `config.py`: validated experiment configuration.
- `extract.py`: ESM-C hidden-state and SAE activation extraction.
- `audit.py`: reconstruction, normalization, sparsity, and provenance audit.
- `dataset.py`, `train_dense.py`, `sl_adapter.py`: dense SL prediction data,
  training, and frozen prediction adapter.

The checksum-bound `configs/mvp_a549_c1_fold0.yaml` remains byte-identical to
the original experiment configuration, including its historical absolute root.

## Experiment 2: local attribution and intervention

- `attribution.py`: active-feature first-order scoring.
- `intervention.py`: single-feature and paired-feature interventions.
- `calibrate.py`: validation-set threshold calibration.
- `heldout.py`: blinded held-out explanation workflow.
- `pair_report.py`: pair-level explanation reports.

## Experiment 3: semantics and mechanism generation

- `metadata.py`: official SAE feature metadata retrieval and caching.
- `mechanism.py`: evidence-packet validation and mechanism reports.
- `mechanism_eval.py`: mechanism-output evaluation.

## Case-study workflows

- `case_split.py`, `case_cache.py`, `case_train.py`, `case_predict.py`: original
  case-study split, cache, training, and prediction path.
- `all_context_split.py`, `all_context_cache.py`, `all_context_dataset.py`,
  `all_context_train.py`, `all_context_predict.py`: all-context training path
  with a locked case.
- `all_context_case_mechanism.py`: case intervention, SAE semantics, constrained
  LLM request construction, response validation, and illustrative reporting.

## Shared infrastructure

- `artifacts.py`: artifact serialization.
- `manifest.py`: checksum verification.
- `runtime.py`: managed local job execution.
- `tests/`: unit and regression tests for this package.

## Test command

```bash
python -m pytest esmc_sae/tests -q
```
````

Do not claim that ESM-C is the complete multimodal model explanation; explicitly describe this package as explaining dependencies of the ESM-C branch.

- [x] **Step 3: Create the test-suite guide**

Create `tests/README.md` with:

````markdown
# Test Suites

The repository has two independent pytest suites:

- `esmc_sae/tests/` verifies the ESM-C/SAE SL interpretability workflow,
  including configuration, extraction contracts, attribution, intervention,
  calibration, held-out evaluation, reporting, and case-study workflows.
- `tests/pfsl_process/sae/` verifies the ProTrek-SAE pipeline, including corpus
  construction, activation extraction, SAE training, feature discovery,
  feature descriptions, and feature-atlas generation.

Run them from the repository root:

```bash
python -m pytest esmc_sae/tests -q
python -m pytest tests/pfsl_process/sae -q
```

Runtime audit and validation functions remain in production modules because
they protect scientific artifact contracts; they are not test-suite code.
````

- [x] **Step 4: Verify documentation paths**

Run:

```bash
test -f README.md
test -f esmc_sae/README.md
test -f tests/README.md
test -f coach_pl/LICENSE
test -f coach_pl/README.md
test -f script/attn/train.sh
test -f esmc_sae/all_context_case_mechanism.py
test -d esmc_sae/tests
test -d tests/pfsl_process/sae
```

Expected: every referenced local path exists.

- [x] **Step 5: Confirm that Python implementation files are unchanged**

Compare the pre-change Python inventory and content hashes with a new post-documentation snapshot.

Run:

```bash
find coach_pl data_preprocess esm4sl esmc_sae pfsl pfsl_process script tests \
  -type f -name '*.py' -print0 |
  sort -z |
  xargs -0 sha256sum > /tmp/esm4sl_repository_cleanup_python_after.sha256
sha256sum train.py test.py tent.py \
  >> /tmp/esm4sl_repository_cleanup_python_after.sha256
diff -u \
  /tmp/esm4sl_repository_cleanup_python_before.sha256 \
  /tmp/esm4sl_repository_cleanup_python_after.sha256
```

Expected: `diff` exits with status 0 and prints no differences at this
pre-rename checkpoint.

---

### Task 4: Migrate the repository root from `ESM4SL_modified` to `PFSL`

**Files:**
- Mechanically modify:
  - `data_preprocess/baseline.ipynb`
  - `data_preprocess/preprocess.ipynb`
  - `data_preprocess/utils.py`
  - `esm4sl/configuration/prosl.yaml`
  - `esm4sl/configuration/unified_token.yaml`
  - `esm4sl/dataset/prosl.py`
  - `esmc_sae/case_cache.py`
  - `esmc_sae/case_split.py`
  - `esmc_sae/config.py`
  - `esmc_sae/tests/test_all_context_split.py`
  - `esmc_sae/tests/test_all_context_train_predict.py`
  - `esmc_sae/tests/test_attribution.py`
  - `esmc_sae/tests/test_case_cache.py`
  - `esmc_sae/tests/test_case_predict.py`
  - `esmc_sae/tests/test_case_split.py`
  - `esmc_sae/tests/test_case_train.py`
  - `esmc_sae/tests/test_config_runtime.py`
  - `esmc_sae/tests/test_dataset_train.py`
  - `pfsl/text_data.py`
  - `pfsl/train.py`
  - `pfsl_process/analysis.ipynb`
  - `pfsl_process/check.ipynb`
  - `pfsl_process/cl_embed.py`
  - `pfsl_process/cl_llm.py`
  - `pfsl_process/esmc.py`
  - `pfsl_process/reg_data.ipynb`
  - `pfsl_process/result_count.ipynb`
  - `pfsl_process/split.ipynb`
  - `script/pf.py`
  - `script/pf_reg.py`
  - `script/tent.py`
- Rename: `/data/qingyuyang/SL/ESM4SL_modified` to `/data/qingyuyang/SL/PFSL`
- Preserve unchanged: `esmc_sae/configs/mvp_a549_c1_fold0.yaml`
- Preserve unchanged: historical files below `esmc_sae/.superpowers/`, `esmc_sae/case_study/`, `esmc_sae/docs/`, `esmc_sae/experiment2/`, `esmc_sae/experiment3/`, `esmc_sae/manifests/`, `esmc_sae/reports/`, and `esmc_sae/runs/`

**Interfaces:**
- Consumes: the prepared pre-rename tree and old absolute root `/data/qingyuyang/SL/ESM4SL_modified`.
- Produces: the same tree at `/data/qingyuyang/SL/PFSL`, runtime-bearing paths pointing at the new root, no old-name symlink, and historical artifacts retaining their original provenance text.

- [x] **Step 1: Capture the exact runtime-bearing path-migration set**

Run from `/data/qingyuyang/SL/ESM4SL_modified`:

```bash
rg -l --hidden --no-ignore "ESM4SL_modified" \
  esm4sl esmc_sae pfsl pfsl_process script data_preprocess tests \
  --glob '!esmc_sae/.superpowers/**' \
  --glob '!esmc_sae/cache/**' \
  --glob '!esmc_sae/case_study/**' \
  --glob '!esmc_sae/docs/**' \
  --glob '!esmc_sae/experiment2/**' \
  --glob '!esmc_sae/experiment3/**' \
  --glob '!esmc_sae/interpret_data/**' \
  --glob '!esmc_sae/manifests/**' \
  --glob '!esmc_sae/reports/**' \
  --glob '!esmc_sae/runs/**' \
  --glob '!esmc_sae/configs/mvp_a549_c1_fold0.yaml' \
  --glob '!pfsl_process/ESMC/**' \
  --glob '!pfsl_process/PubmedBERT/**' \
  --glob '!pfsl_process/cl_embed/**' \
  --glob '!pfsl_process/retrieval/**' \
  --glob '!**/__pycache__/**' |
  sort > /tmp/pfsl_path_migration_files.txt
diff -u \
  /tmp/pfsl_path_migration_files.txt \
  <(sed -n '/^  - `/s/^  - `\\(.*\\)`$/\\1/p' \
    docs/superpowers/plans/2026-09-10-repository-cleanup.md |
    sed -n '/^data_preprocess\\/baseline.ipynb$/,/^script\\/tent.py$/p')
```

Expected: the list contains exactly the 31 files declared above and `diff`
prints no differences.

- [x] **Step 2: Snapshot those 31 files before substitution**

Run:

```bash
test ! -e /tmp/pfsl_path_migration_before.tar
tar -cf /tmp/pfsl_path_migration_before.tar \
  -T /tmp/pfsl_path_migration_files.txt
tar -tf /tmp/pfsl_path_migration_before.tar |
  diff -u /tmp/pfsl_path_migration_files.txt -
```

Expected: the archive is created outside the repository and its member list
matches the migration list exactly.

- [x] **Step 3: Apply only the approved root-path substitution**

Run:

```bash
while IFS= read -r file; do
  sed -i \
    's|/data/qingyuyang/SL/ESM4SL_modified|/data/qingyuyang/SL/PFSL|g' \
    "$file"
done < /tmp/pfsl_path_migration_files.txt
```

Expected: only the literal absolute root changes; no file is reformatted or
moved in this step.

- [x] **Step 4: Prove that every edited file differs only by that substitution**

Run:

```bash
while IFS= read -r file; do
  tar -xOf /tmp/pfsl_path_migration_before.tar "$file" |
    sed 's|/data/qingyuyang/SL/ESM4SL_modified|/data/qingyuyang/SL/PFSL|g' |
    cmp -s - "$file" || exit 1
done < /tmp/pfsl_path_migration_files.txt
```

Expected: exit status 0. This is the no-logic-change gate for all Python,
configuration, test, and notebook edits introduced by the rename.

- [x] **Step 5: Confirm old runtime paths are gone while historical records remain untouched**

Run the same scoped `rg` command from Step 1 without `-l`.

Expected: no result in migrated runtime-bearing source, configuration, tests,
or notebooks. The frozen YAML, migration design/plan, and historical artifact
directories are excluded intentionally because they retain the former name as
documentation or provenance.

- [x] **Step 6: Rename the root directory**

Preflight:

```bash
test -d /data/qingyuyang/SL/ESM4SL_modified
test ! -e /data/qingyuyang/SL/PFSL
test ! -L /data/qingyuyang/SL/PFSL
```

Then run:

```bash
mv /data/qingyuyang/SL/ESM4SL_modified /data/qingyuyang/SL/PFSL
```

Expected: the move completes within the same parent filesystem without
copying or deleting project contents.

- [x] **Step 7: Verify the final root and absence of a compatibility symlink**

Run from `/data/qingyuyang/SL/PFSL`:

```bash
test "$(pwd -P)" = /data/qingyuyang/SL/PFSL
test -d /data/qingyuyang/SL/PFSL
test ! -e /data/qingyuyang/SL/ESM4SL_modified
test ! -L /data/qingyuyang/SL/ESM4SL_modified
test -f /data/qingyuyang/SL/PFSL/README.md
test -f /data/qingyuyang/SL/PFSL/esmc_sae/config.py
```

Expected: only the new root exists; no file or symlink occupies the old path.

---

### Task 5: Initialize Git and audit the candidate repository

**Files:**
- Create: `.git/` through `git init -b main .`
- Inspect only: all files not excluded by `.gitignore`

**Interfaces:**
- Consumes: cleaned source tree, documentation, and ignore rules.
- Produces: an unstaged Git repository on branch `main` whose candidate files fit GitHub limits and contain no detected credential values.

- [x] **Step 1: Initialize Git without staging files**

Run:

```bash
git init -b main .
git branch --show-current
git status --short --untracked-files=normal
```

Expected: current branch is `main`; no index entries or commits exist; source and documentation appear as untracked while excluded artifacts do not.

- [x] **Step 2: Test representative ignored and retained paths**

Run:

```bash
git check-ignore -q data/SLKB
git check-ignore -q output
git check-ignore -q esmc_sae/cache
git check-ignore -q esmc_sae/case_study
git check-ignore -q esmc_sae/docs
git check-ignore -q esmc_sae/reports
git check-ignore -q esmc_sae/manifests
git check-ignore -q esmc_sae/bridge/sae_layer60_decoder.pt
git check-ignore -q pfsl_process/ESMC
git check-ignore -q pfsl_process/cl_embed
git check-ignore -q .env
test -z "$(git check-ignore README.md || true)"
test -z "$(git check-ignore script/attn/train.sh || true)"
test -z "$(git check-ignore esmc_sae/attribution.py || true)"
test -z "$(git check-ignore esmc_sae/tests/test_attribution.py || true)"
test -z "$(git check-ignore tests/pfsl_process/sae/test_topk_sae.py || true)"
test -z "$(git check-ignore coach_pl/LICENSE || true)"
```

Expected: every artifact/credential check succeeds as ignored; every source/documentation check succeeds as unignored.

- [x] **Step 3: Audit unignored file sizes**

Run:

```bash
git ls-files --others --cached --exclude-standard -z |
  xargs -0 -r stat -c '%s %n' |
  sort -nr
```

Expected: inspect every entry above 50 MB; no candidate file exceeds 100 MB. If a large entry is generated data or a local model artifact, add the narrowest justified ignore rule and repeat the audit. Do not ignore source code solely to satisfy the limit.

- [x] **Step 4: Scan candidate text files for likely credential assignments**

Generate the candidate file list with:

```bash
git ls-files --others --cached --exclude-standard -z
```

Scan candidate text files for likely assignments while printing filenames only:

```bash
git ls-files --others --cached --exclude-standard -z |
  xargs -0 -r rg -l -I --hidden --no-ignore -i \
  "(OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|PRIVATE_KEY|PASSWORD|SECRET|TOKEN)[[:space:]]*[:=][[:space:]]*['\"]?[A-Za-z0-9_./+=-]{8,}"
```

Review every returned filename. Treat environment-variable reads, CLI option names, dummy test values, and documentation examples as references, not leaked values. Do not print suspected secret values into the terminal or report. If an actual value is found, stop, add a narrow ignore rule for the containing local file, and report only its path and variable name.

- [x] **Step 5: Verify repository state**

Run:

```bash
git status --short --untracked-files=normal
git diff --cached --quiet
git log -1 --oneline
```

Expected: files remain unstaged; the cached diff is empty; `git log` reports that the branch has no commits.

---

### Task 6: Run final regression gates and report every change

**Files:**
- Create: `docs/repository-cleanup-report.md`
- Inspect only: all Python source files
- Inspect only: all CLI-bearing `esmc_sae` modules

**Interfaces:**
- Consumes: baseline results from Task 1 and the final cleaned tree.
- Produces: evidence that behavior is unchanged plus an English cleanup report containing all changes and audit outcomes.

- [x] **Step 1: Byte-compile candidate Python source without polluting the repository**

Run:

```bash
python -m compileall -q \
  coach_pl \
  data_preprocess \
  esm4sl \
  esmc_sae \
  pfsl \
  pfsl_process \
  script \
  train.py test.py tent.py
```

Expected: exit status 0. Generated `__pycache__/` files remain ignored.

- [x] **Step 2: Confirm documented ESMC-SAE CLI modules resolve**

Run:

```bash
python -m esmc_sae.train_dense --help
python -m esmc_sae.audit --help
python -m esmc_sae.attribution --help
python -m esmc_sae.calibrate --help
python -m esmc_sae.heldout --help
python -m esmc_sae.pair_report --help
python -m esmc_sae.case_split --help
python -m esmc_sae.case_cache --help
python -m esmc_sae.case_train --help
python -m esmc_sae.case_predict --help
python -m esmc_sae.all_context_split --help
python -m esmc_sae.all_context_cache --help
python -m esmc_sae.all_context_train --help
python -m esmc_sae.all_context_predict --help
python -m esmc_sae.all_context_case_mechanism --help
```

Expected: every command exits successfully and prints usage without starting an experiment.

- [x] **Step 3: Re-run the ESMC-SAE suite**

Run:

```bash
python -m pytest esmc_sae/tests -q
```

Expected: `171 passed, 17 failed, 1 skipped`. The failures are confined to the
formal-config consumers in `test_attribution.py` (1),
`test_calibrate_heldout.py` (5), `test_config_runtime.py` (1),
`test_dataset_train.py` (7), and `test_metadata_pair_report.py` (3). They occur
because the checksum-bound historical YAML retains the former root while active
configuration validation requires `/data/qingyuyang/SL/PFSL`. Any other failure
stops the cleanup.

- [x] **Step 4: Re-run the ProTrek-SAE suite**

Run:

```bash
python -m pytest tests/pfsl_process/sae -q
```

Expected: results match or improve on the Task 1 baseline; no cleanup-induced failures.

- [x] **Step 5: Write the cleanup report**

Create `docs/repository-cleanup-report.md` in English with:

```markdown
# Repository Cleanup Report

**Date:** 2026-09-10
**Scope:** `/data/qingyuyang/SL/PFSL`

## Outcome

## Files changed

## Files deliberately not moved

## Git ignore coverage

## Vendored `coach_pl`

## Verification results

## Remaining user actions
```

List every created, modified, and deleted file. State explicitly that no Python
module moved and no implementation logic changed, no data/output file was
deleted, Git has no commit or remote, and the user still controls review, first
commit, private remote creation, and push.

For the root migration, distinguish the 31 files whose only content change is
the approved literal path substitution from all other Python implementation
files, which must remain byte-identical. State that the root directory was
renamed, no compatibility symlink was created, and historical artifact contents
were not rewritten.

- [x] **Step 6: Perform final repository checks**

Run:

```bash
git status --short --untracked-files=normal
git diff --cached --quiet
git remote -v
git rev-list --all --count
```

Expected: all candidate changes are visible but unstaged; the index is empty; no remote is configured; commit count is 0.

- [x] **Step 7: Present the result**

Report:

- every filesystem change, grouped as modified, added, and removed;
- the root rename from `ESM4SL_modified` to `PFSL` and the exact 31-file
  path-substitution list;
- both test-suite results before and after cleanup;
- CLI/compile, file-size, ignore-rule, and credential-scan outcomes;
- the absolute clickable paths below `/data/qingyuyang/SL/PFSL` to `README.md`,
  `esmc_sae/README.md`, `tests/README.md`, and
  `docs/repository-cleanup-report.md`;
- confirmation that no implementation logic changed and no Git commit, remote, or push was created.
