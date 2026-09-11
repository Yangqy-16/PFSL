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
