# Repository Cleanup Design

**Date:** 2026-09-10  
**Current path:** `/data/qingyuyang/SL/ESM4SL_modified`  
**Final path and repository name:** `/data/qingyuyang/SL/PFSL`

## Objective

Rename the existing project to `PFSL` and prepare it as a private,
source-oriented Git repository without changing implementation logic, runtime
module interfaces, or scientific results.

## Constraints

- Do not change model, data-processing, training, prediction, attribution,
  intervention, calibration, or reporting logic.
- Permit only mechanical `ESM4SL_modified` to `PFSL` path substitutions in
  active source files, configurations, tests, notebooks, and current
  repository documentation where required by the root-directory rename.
- Do not delete datasets, checkpoints, caches, logs, reports, or previous
  experiment outputs.
- Preserve all `esmc_sae.*` import paths and `python -m esmc_sae.*` commands.
- Preserve root entry points such as `train.py`, `test.py`, and `tent.py`.
- Do not rewrite historical reports, manifests, logs, or experiment outputs.
- Do not create an `ESM4SL_modified` compatibility symlink after the rename.
- Do not upload, push, create a remote repository, or expose credentials.
- Keep Git history preparation independent of experimental artifact storage.

## Selected Approach

Use a conservative, publish-ready cleanup rather than a package namespace
rewrite.

### Production and test boundaries

The existing `esmc_sae/tests/` directory remains the dedicated unit and
regression test location for the ESMC-SAE interpretability code. There are no
embedded `test_*` functions in the production modules.

Runtime functions such as cache audits, manifest verification, schema
validation, and fail-closed artifact checks remain in production modules.
They enforce formal experiment contracts and are not test-suite code.

No production Python module is moved. This preserves:

- imports used by the implementation and tests;
- monkeypatch targets used by regression tests;
- established `python -m esmc_sae.<module>` command forms;
- package-relative references used by experiment plans and reports.

An `esmc_sae/README.md` file will group the existing modules by responsibility
so the workflow can be understood without a risky namespace migration.

### Repository-level organization

The physical locations of data, model outputs, caches, and legacy result
directories relative to the repository root remain unchanged. The repository
root itself changes from `ESM4SL_modified` to `PFSL`.

Active source files, configurations, tests, notebooks, and current repository
documentation containing the old absolute root will receive a literal-only
path substitution. The substitution does not restructure functions, imports,
or workflows.

Historical reports, manifests, logs, and experiment outputs retain their
recorded `ESM4SL_modified` paths as provenance. Because no compatibility
symlink will be created, those embedded old absolute paths may no longer
resolve directly. They remain readable records, and no historical content or
checksum is rewritten to conceal that it was produced under the former path.

`esmc_sae/configs/mvp_a549_c1_fold0.yaml` is also retained byte-for-byte as a
frozen historical configuration. Its SHA-256 is bound by the frozen thresholds
and several task manifests. Migrating that file without rewriting the integrity
chain would make the recorded artifacts internally inconsistent.

The root README will be expanded to document:

- the `PFSL` project and repository name;
- the source package layout;
- the two test-suite locations;
- environment setup and entry points;
- which required datasets and weights are intentionally excluded from Git;
- the role and upstream origin of the vendored `coach_pl` source.

This provides a clear GitHub view while preserving the repository-relative
layout used by existing experiments. The explicitly accepted exception is that
historical embedded absolute paths retain the former root and may not resolve.

### `coach_pl` handling

`coach_pl` will be stored as ordinary vendored source. Its implementation,
README, and LICENSE remain unchanged.

The broken `coach_pl/.git` pointer and root `.gitmodules` file will be removed.
The root README will retain the upstream project URL and state that the code is
vendored.

### Git ignore policy

The repository will track source code, configuration files, shell scripts,
notebooks, licenses, and repository documentation.

The following categories will be ignored:

- datasets and derived data under `data/` and `sc_data/`;
- training and evaluation outputs;
- ESMC-SAE caches, runs, case-study outputs, experiment outputs, reports,
  manifests, and historical task-control records;
- local pretrained model directories and embedding stores;
- model checkpoints, tensor files, serialized Python objects, array stores,
  HDF5/Arrow files, compressed datasets, and large JSONL corpora;
- Python, pytest, notebook, IDE, OS, TensorBoard, W&B, and coverage artifacts;
- `.env` files, keys, certificates, and common credential files.

If the audit identifies a likely live credential inside a source file, preserve
the local file unchanged and add the narrowest exact-path ignore rule. Report
the file and credential field without printing the value. For this repository,
`pfsl_process/cl_llm.py` is excluded until its hard-coded `api_key` is rotated
and replaced with an environment-variable lookup.

The ignore policy will be audited after Git initialization. Every unignored
file larger than 50 MB will be reviewed, and no unignored file may exceed
GitHub's 100 MB single-file limit.

### Git initialization

After pre-rename verification and source preparation, rename the root directory
to `/data/qingyuyang/SL/PFSL`. Confirm that
`/data/qingyuyang/SL/ESM4SL_modified` no longer exists and do not create a
compatibility symlink.

Initialize the renamed directory as a Git repository with branch name `main`.
Git does not store a repository display name, so `PFSL` is established locally
by the directory and README names; the user will use `PFSL` when creating the
private GitHub repository. Do not stage, commit, push, or create a GitHub
remote. The user retains control of the first commit and private remote
creation.

## Documentation Deliverables

- Updated root `.gitignore`
- Updated root `README.md`
- Root directory renamed from `ESM4SL_modified` to `PFSL`
- New `esmc_sae/README.md`
- New `tests/README.md`
- This repository cleanup design
- A detailed implementation plan under `docs/superpowers/plans/`
- A final cleanup report listing every changed file and verification result

## Verification

Run these gates before and after the cleanup:

1. `python -m pytest esmc_sae/tests -q`
2. `python -m pytest tests/pfsl_process/sae -q`
3. Audit runtime-bearing source, configurations, tests, and notebooks before
   and after the rename to ensure that no active `ESM4SL_modified` path
   remains; migration documentation and historical artifacts are explicitly
   exempt because they record both the former and final paths
4. Confirm the old root path is absent and no compatibility symlink exists
5. Python byte-compilation for tracked `.py` files
6. Git ignore checks for representative datasets, checkpoints, caches,
   reports, environment files, and source files
7. Scan unignored files for size violations and common credential patterns
8. Confirm all documented ESMC-SAE module and CLI paths still resolve from the
   renamed root

The pre-rename baseline must pass. After the rename, 17 formal tests are
expected to fail because they load the frozen YAML, which still records the
former absolute root, while `config.py` correctly enforces the new active root.
The expected final result is `171 passed, 17 failed, 1 skipped`.

The affected tests are confined to formal-config consumers in:

- `test_attribution.py` (1 test);
- `test_calibrate_heldout.py` (5 parameterized cases/tests);
- `test_config_runtime.py` (1 test);
- `test_dataset_train.py` (7 parameterized cases/tests);
- `test_metadata_pair_report.py` (3 tests).

All other tests must retain their baseline result. Any additional failure stops
the cleanup for diagnosis. The 17 accepted failures are reported as a
root-rename compatibility limitation, not hidden or reclassified as passing.

## Non-goals

- Reformatting all Python source
- Removing apparently unused scientific code
- Changing hard-coded paths other than the literal root rename from
  `ESM4SL_modified` to `PFSL`
- Renaming public modules or commands
- Moving large local artifacts
- Rewriting historical reports, manifests, logs, or experiment outputs
- Creating a backward-compatible `ESM4SL_modified` symlink
- Introducing packaging, dependency, CI, or Git LFS configuration
- Publishing the repository
