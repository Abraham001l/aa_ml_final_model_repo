"""Explicit history/label boundaries. No source files are modified."""
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["ORG_AIRPRT", "DES_AIRPRT"]
COLUMNS = KEYS + ["DEP_YEAR", "DEP_MONTH", "OD_PAX"]


def read_data(path):
    """Read the supplied pipe-delimited extract; demand validation is split-specific."""
    raw = pd.read_csv(Path(path), sep="|", usecols=COLUMNS)
    if raw[KEYS + ["DEP_YEAR", "DEP_MONTH"]].isna().any().any():
        raise ValueError("Missing airport code or calendar field.")
    dates = pd.to_datetime(dict(year=raw.DEP_YEAR, month=raw.DEP_MONTH, day=1), errors="raise")
    raw["month"] = pd.PeriodIndex(dates, freq="M")
    return raw


def _validate(part):
    if part.duplicated(KEYS + ["month"]).any():
        raise ValueError("Duplicate directional route-month observations.")
    part = part.copy()
    part["OD_PAX"] = pd.to_numeric(part.OD_PAX, errors="raise")
    values = part.OD_PAX.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand within the requested split must be finite and nonnegative.")
    return part


def build_history(raw, origin):
    """Filter dates BEFORE pivoting or discovering routes; ignore future demand values."""
    origin = pd.Period(origin, freq="M")
    if origin < pd.Period("2019-12", freq="M"):
        raise ValueError("A full 2019 reference is required.")
    past = _validate(raw.loc[raw.month.between(pd.Period("2019-01"), origin)])
    expected = pd.period_range("2019-01", origin, freq="M")
    if not pd.PeriodIndex(sorted(past.month.unique())).equals(expected):
        raise ValueError("History must include every calendar month from January 2019 through origin.")
    panel = past.pivot(index=KEYS, columns="month", values="OD_PAX")
    return panel.reindex(columns=expected).sort_index().fillna(0).astype(float)


def build_targets(raw, history, horizon=24):
    """Build labels separately on the history-only route universe; absent rows mean zero."""
    months = pd.period_range(history.columns[-1] + 1, periods=horizon, freq="M")
    labels = raw.loc[raw.month.isin(months)]
    if not pd.PeriodIndex(sorted(labels.month.unique())).equals(months):
        raise ValueError("Evaluation requires all 24 target months. Use predict for unlabeled future forecasts.")
    route_index = pd.MultiIndex.from_frame(labels[KEYS])
    labels = _validate(labels.loc[route_index.isin(history.index)])
    panel = labels.pivot(index=KEYS, columns="month", values="OD_PAX")
    return panel.reindex(index=history.index, columns=months).fillna(0).astype(float)


def long_forecast(frame, actual=None):
    n, horizon = frame.shape
    out = pd.DataFrame({
        "ORG_AIRPRT": np.repeat(frame.index.get_level_values(0), horizon),
        "DES_AIRPRT": np.repeat(frame.index.get_level_values(1), horizon),
        "month": np.tile(frame.columns.astype(str), n),
        "horizon": np.tile(np.arange(1, horizon + 1), n),
        "y_pred": frame.to_numpy().ravel(),
    })
    if actual is not None:
        if not frame.index.equals(actual.index) or not frame.columns.equals(actual.columns):
            raise ValueError("Actuals and predictions must have identical route/date alignment.")
        out["y_true"] = actual.to_numpy().ravel()
    return out
