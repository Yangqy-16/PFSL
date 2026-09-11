# Repository Cleanup Report

**Date:** 2026-09-11  
**Scope:** `/data/qingyuyang/SL/PFSL`

## Outcome

The project root was renamed from `ESM4SL_modified` to `PFSL` and initialized
as an empty Git repository on branch `main`. The source layout, Python module
names, public imports, command entry points, model logic, and scientific
outputs were not reorganized.

The cleanup has one accepted compatibility limitation. The checksum-bound
historical configuration `esmc_sae/configs/mvp_a549_c1_fold0.yaml` retains its
original absolute root and SHA-256. Without an old-name compatibility symlink,
17 formal tests that load this configuration fail at the fixed-path validation
gate. This limitation is reported explicitly rather than changing the frozen
configuration or its integrity chain.

## Files changed

### Repository files

- Modified `.gitignore` with explicit rules for local data, weights, outputs,
  caches, experiment records, credentials, and generated files.
- Modified `README.md` with the PFSL name, repository map, environment and test
  commands, local-artifact policy, and vendored dependency attribution.
- Added `esmc_sae/README.md` with the Experiment 1, Experiment 2, Experiment 3,
  case-study, and shared-infrastructure module map.
- Added `tests/README.md` with the two independent pytest suites.
- Added
  `docs/superpowers/specs/2026-09-10-repository-cleanup-design.md`.
- Added
  `docs/superpowers/plans/2026-09-10-repository-cleanup.md`.
- Added this report.
- Removed the obsolete root `.gitmodules`.
- Removed only the broken `coach_pl/.git` pointer.
- Created local `.git/` metadata with branch `main`.

The implementation, README, and LICENSE below `coach_pl/` remain present. The
vendored source is attributed to <https://github.com/DuskNgai/coach-pl>.

### Literal root-path migration

The following 31 files received only this literal substitution:

```text
/data/qingyuyang/SL/ESM4SL_modified
→ /data/qingyuyang/SL/PFSL
```

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

An archived pre-migration copy was used to compare each file with the expected
literal substitution. All 31 comparisons passed. These files include 22 Python
files; the other 167 Python files in the source-oriented repository remained
byte-identical to the pre-cleanup hash baseline.

## Files deliberately not moved or rewritten

- No Python module or test file was moved.
- `train.py`, `test.py`, and `tent.py` remain root entry points.
- `esmc_sae/tests/` and `tests/pfsl_process/sae/` remain the two test locations.
- Runtime audit, schema, manifest, cache, and fail-closed validation functions
  remain in production modules because they enforce experiment contracts.
- Datasets, pretrained models, embeddings, checkpoints, caches, logs, reports,
  manifests, and previous experiment outputs remain in their original
  repository-relative locations.
- Historical reports, manifests, logs, and outputs were not rewritten.
- `esmc_sae/configs/mvp_a549_c1_fold0.yaml` is byte-identical to its original
  content and retains SHA-256
  `3450666b8d49bf5f73bcb0cf001d00d26954305aa0f39481af4144342b939cae`.
- No `ESM4SL_modified` compatibility symlink was created.

## Git ignore coverage

The final `.gitignore` excludes:

- `data/`, `sc_data/`, `output/`, metric/result directories, and local
  experiment records;
- ESMC-SAE caches, runs, case studies, experiment outputs, reports, manifests,
  and local task-control records;
- local ESM-C, PubMedBERT, cell-line embedding, retrieval, and derived-analysis
  directories;
- checkpoints, tensors, serialized objects, array/HDF5/Arrow files, compressed
  datasets, and JSONL corpora;
- Python caches, pytest caches, notebook checkpoints, editor files, logs,
  TensorBoard/W&B files, coverage output, environments, keys, certificates,
  credential files, and `.env` files.
- the exact local source file `pfsl_process/cl_llm.py`, which contains a
  hard-coded API credential and must not be committed until that credential is
  removed.

Other source code, shell scripts, notebooks, configuration files, repository
documentation, and licenses remain candidates for the first commit.

## Verification results

### Baseline before cleanup

- Correct interpreter:
  `/home/qingyuyang/anaconda3/envs/DTI_new/bin/python` (Python 3.10).
- ESMC-SAE: **188 passed, 1 skipped**.
- ProTrek-SAE: **727 passed**.
- Python content baseline: **189 files**.
- Embedded production `test_*` definitions: **0**.

An initial diagnostic run with the base Python 3.9 interpreter failed during
collection because it lacked `fvcore` and did not support the project's type
syntax. No dependency was installed and no source change was made; all formal
results above and below use the existing `DTI_new` environment.

### Final state

- Root rename: **PASS**.
- Old root absent: **PASS**.
- Old-name compatibility symlink absent: **PASS**.
- Literal-only comparison for 31 migrated files: **PASS**.
- Byte identity for 167 non-migrated Python files: **PASS**.
- Frozen historical YAML byte identity and SHA-256: **PASS**.
- Forced Python byte-compilation: **PASS**.
- Documented ESMC-SAE CLI help checks: **15/15 passed**.
- ProTrek-SAE: **727 passed**.
- ESMC-SAE: **171 passed, 17 failed, 1 skipped**.
- Candidate files above 50 MiB: **0**.
- Candidate files above 100 MiB: **0**.
- Largest candidate file: `pfsl_process/dss_subloc.csv`, 1,923,033 bytes.
- Git ignore checks for representative data, output, cache, report, model,
  checkpoint, and environment paths: **PASS**.
- Credential scan: **protected after review**. A likely live API credential was
  detected in the `api_key` argument in `pfsl_process/cl_llm.py`. Its value was
  not printed or modified; the whole local file is excluded by an exact
  `.gitignore` rule. Other matches were ordinary protein-model token variables,
  environment-variable reads, or the audit documentation itself.
- Lint and static type checking: **not configured in the repository and not
  added by this cleanup**.

The 17 accepted ESMC-SAE failures are confined to formal consumers of the
frozen YAML:

- `test_attribution.py`: 1;
- `test_calibrate_heldout.py`: 5;
- `test_config_runtime.py`: 1;
- `test_dataset_train.py`: 7;
- `test_metadata_pair_report.py`: 3.

All fail because the historical YAML records
`/data/qingyuyang/SL/ESM4SL_modified`, while active configuration validation
requires `/data/qingyuyang/SL/PFSL`. Updating the YAML would invalidate its
recorded checksum; updating the complete checksum chain would rewrite
historical experiment records.

## Cleanup scope not applied

The anti-slop review did not delete dead code, merge duplicate logic, rename
symbols, introduce abstractions, or add tests. Those passes were intentionally
not applicable because the approved scope prohibited implementation changes.

## Remaining user actions

1. Review the untracked candidate files and `.gitignore`.
2. Stage and create the first commit when satisfied.
3. Create a private GitHub repository named `PFSL`.
4. Add its remote and push the first commit.
5. Restore excluded private datasets and model assets separately when using a
   fresh clone.
6. Rotate the API credential stored in `pfsl_process/cl_llm.py`, then replace
   the embedded value with an environment-variable lookup before deciding
   whether to track that file.

No file is staged, no commit exists, no remote is configured, and nothing has
been pushed.
