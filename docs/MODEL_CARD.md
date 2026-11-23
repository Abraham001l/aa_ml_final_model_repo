# Model card

## Model and output

Adaptive Route-Recovery Forecasting is a deterministic, analytical seasonal model
for monthly origin–destination passenger volume. It predicts 24 months for each
directional market observed by the origin. It does not require learned weights,
airport geography, external data, fares or revenue. A market can include connecting
itineraries; a directional OD market is not synonymous with nonstop service.

## Intended use

An explainable demand baseline for analysts reviewing seasonal markets and demand
levels. Forecasts may inform subsequent itinerary/capacity/profitability analysis.
The model alone does not recommend flight schedules, enforce physical constraints,
estimate causal drivers or optimize financial outcomes. No prediction intervals
are supplied. We do not claim operational deployment or realized savings.

## Selection and evaluation

The original search compares 144 route-recovery and 54 network-recovery settings
on July 2021, January 2022 and July 2022 origins, each with a 24-month target.
The selected configuration is route recovery, window 3, recent weight .7,
damping 12 and reverse pooling .25. Development labels end by July 2024.
The final forecast is issued at July 2024 for August 2024–July 2026.

Three overlapping development folds were used repeatedly to select settings and
models. Their tuned mean WAPE 19.08% is not independent nested cross-validation.
The project documentation records previous inspection of the final window. Code
tests establish automatic selection independence from final passenger values;
they do not prove the historical model family or design constants were chosen
before any human saw that window. A fresh time-stamped evaluation is needed.

## Results and scale

Full-period WAPE 17.3418%, baseline 18.2850%, RMSE 1,408.51 PAX, volume bias +4.77%.
There are 410,856 route-month observations across 17,119 routes. Year-one WAPE
13.89% rises to 21.02% in year two. Median per-route WAPE is 24.15%, with an
interquartile range of 16.70–35.91%. Large markets dominate network WAPE; an
individual small route should not inherit the headline error assumption.
These are historical point comparisons, not confidence intervals or a paired
significance claim. Shorter evaluation windows are overlapping diagnostics.

## Boundary hardening

Original code pivoted the whole file, retaining future-only route membership
before truncating columns. Reverse availability could therefore leak information
about which routes would appear later. A synthetic example changes a known
forward route by 25%. This package discovers routes only after filtering history.
On the supplied data all 17,119 routes were already known at each evaluated
origin, so the fix has no effect on the reproduced numbers.

Only the split being used is demand-validated. Future NaN, negative or infinite
demand cannot block history-only prediction. Evaluation appropriately rejects
invalid labels when it attempts to score them. Target calendar information such
as month and days-in-month is known in advance and is an allowed input.

## Limitations

- A complete 2019 reference and consecutive global monthly coverage are required.
- Absent route-month rows mean zero by evaluation convention, not necessarily
  zero underlying travel demand.
- Markets first appearing after the origin are outside the forecast universe.
- With no active network-recovery adjustment, the seasonal curve repeats. The
  model cannot anticipate new shocks, new annual trends or route launches.
- The recent-profile multiplier is identically one in this implementation. Do
  not describe it as a separately estimated short-term momentum model.
- The network gate also changes seasonal trust, and route structural share changes
  the blend for weak-2019 markets. A universal 30/70 statement is inaccurate.
- Thresholds 0.86/1.06 and smoothing constants 2/100 are fixed choices. They were
  not tuned in the published grid.
- All-zero 2019 network demand has a documented finite extension in this package;
  that corner case was absent from the audited benchmark.

## Data and publication

Input data is project-provided and is not distributed here. No data redistribution
or new source-code licensing rights are asserted. This repository documents
implementation and reproduction, not the provenance of the underlying passenger
estimates beyond their supplied fields and file hash. See PROVENANCE.md.
