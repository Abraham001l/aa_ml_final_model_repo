# Package reproduction results

Verified locally on November 22, 2026, using the supplied passenger file identified
by SHA256 in `evaluation_run.json`. Raw data and route-level forecasts are not
included in version control. Generated local prediction files are under `outputs/`.

## Checks completed

- **14 unit-test methods passed**, including 36 adversarial checks per family
  (72 total), positive controls, future-only reverse-market injection, calendar-day
  normalization, unpaired routes, zero-reference behavior and metric definitions.
- The complete real-data **198-configuration × 3-origin search** selects the
  published route-recovery configuration: 3 / .70 / 12 / .25.
- The search command passed **36 real-data adversarial checks + 4 positive
  controls for each searched family**, and reran those checks for the selected
  configuration before recording results. Details: `development_leakage_checks.json`.
- Final evaluation passed **9 real-data adversarial checks + 1 positive control**
  at July 2024. Details: `evaluation_leakage_checks.json`.
- All **410,856 package predictions are exactly equal** to the private canonical
  forecasts after aligning route and month. Maximum absolute difference: **0**.
- All **594 development losses**, their rankings and selected settings remain
  exactly unchanged when all real final-test demand values are set to 9e12.
- Winning settings, all three fold WAPEs and all metrics in the three nested
  test-window summaries match the archived audited reference.
- Synthetic end-to-end search and evaluation, unlabeled forecast generation,
  route plotting and the four-step model illustration program were executed.

## Reproduced headline

| Quantity | Value |
|---|---:|
| Full 24-month WAPE | 0.17341753957268646 |
| Seasonal-repeat WAPE | 0.18285038364227046 |
| RMSE | 1408.5069910555176 |
| Volume bias | +0.04771288210081651 |
| Route-month observations | 410856 |

Ratios above are fractions, not percentages. These results retain the exposed-test
and tuned-development-score qualifications in the model card. The remote CI matrix
is configured but was not executed locally on all listed Python versions.

## Reproduce on the authorized original input

```sh
python scripts/reproduce.py --data data/pax_forecasting.csv --output outputs/new_reproduction
python scripts/check_reproduction.py --development outputs/new_reproduction/development --evaluation outputs/new_reproduction/evaluation
```

Every run saves fresh manifests and leakage checks. Output directories must not
already exist. The exact data hash is required only for checking against the
archived numerical reference; the forecasting API can accept other valid panels.
