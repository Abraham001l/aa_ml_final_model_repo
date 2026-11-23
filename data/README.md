# Input data (not included)

Obtain the project-authorized `pax_forecasting.csv` and either place it here or
pass its absolute path with `--data`. This directory is gitignored except for this
document. Do not upload passenger records without permission from their owner.

The pipe-delimited CSV requires `ORG_AIRPRT`, `DES_AIRPRT`, `DEP_YEAR`,
`DEP_MONTH`, `OD_PAX`. An extra leading index column is harmless. Demand must be
finite and nonnegative within the split being used. Each directional route/month
must be unique. The model requires a full January–December 2019 reference and
consecutive historical calendar months through the forecast origin.

Absent route-month records become zero under the project's evaluation convention.
They are not interpolated. This convention does not establish zero latent demand.
Only routes observed by the origin enter forecasts. A→B and B→A are distinct.
Revenue, airport metadata, fares, macroeconomic and demographic data are not inputs.

On the original project layout, the input is `../abraham/global_data/pax_forecasting.csv`
when commands run from the repository root. The package itself does not depend on
that layout or any sibling repository.
