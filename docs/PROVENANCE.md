# Provenance and attribution

## Original model

The final seasonal model and two-family search were developed in the team's
`keshav/ml_pipeline/seasonal_challenger.py`. This repository packages that work;
it does not claim to originate the forecasting method or its measured gains.

- Audited project commit: `a768cd417334d9d5835aa2be96e561163168e021`
- Original `seasonal_challenger.py` SHA256:
  `26a7f0659b60db21a429253fbe3899e58b233d3aea317ae0e36cfef6014cea95`
- Original passenger CSV SHA256:
  `046a7ac6e7b66200be426e317637a4463244ca85c5bda6282f759d8979cd841f`
- Independent audit: original project directory
  `abraham/gpt_agent_workspace/teammate_seasonal_audit_20261121/`
- Canonical metric definitions originated in
  `abraham/agent_workspace/shared_eval/metrics.py`.

## What changed in this package

1. Isolated the three numerical kernels and added a validated immutable config.
2. Restricted observations **before** pivoting, fixing future-only market lookahead.
3. Separated input history from target labeling and scoring.
4. Made demand validation split-specific, removing unnecessary coupling to future
   demand validity for prediction.
5. Added finite handling for an entirely zero historical reference network.
6. Added an installable package, command-line programs, provenance manifests,
   synthetic tests, CI, baseline scoring and per-route metric distributions.
7. Retained the original 198-setting development objective and original monthly
   seasonal-repeat baseline.
8. Kept raw inputs and route-level generated artifacts out of version control.

These changes preserve the selected model's reproduced forecasts on the project
dataset. Strict equality of floating-point arrays is checked against the original
canonical predictions in the local packaging verification. Aggregate reference
metrics are retained without distributing original passenger rows or predictions.

## Reproduction environment

Local verification uses Python 3.14.2, NumPy 2.4.3 and pandas 2.3.3.
Optional charts use Matplotlib 3.10.8. Supported dependency ranges are in
pyproject.toml. `requirements-repro.txt` records the exact tested environment.
GitHub Actions is configured for synthetic tests under Python 3.11–3.13;
those remote CI jobs have not been run by the local verification.

## Data and license

The project owner must determine the source-code license and any data sharing
permissions. No third-party source-code license is invented here. Private source
records are not bundled. Aggregate metrics, documentation and project attribution
do not substitute for permission to distribute the dataset.
