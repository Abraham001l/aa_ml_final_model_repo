"""Canonical project metrics; ratios are fractions, not percentages."""
import numpy as np


def compute_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("Actuals and forecasts must be nonempty, equally shaped arrays.")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all() or (yt < 0).any() or (yp < 0).any():
        raise ValueError("Metrics require finite nonnegative actuals and predictions.")
    absolute = np.abs(yt - yp)
    denom = np.abs(yt) + np.abs(yp)
    smape = np.divide(2 * absolute, denom, out=np.zeros_like(absolute), where=denom > 0)
    total = float(yt.sum())
    return {
        "n": int(yt.size), "sum_y_true": total, "sum_y_pred": float(yp.sum()),
        "wape": float(absolute.sum() / total) if total else None,
        "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "smape": float(smape.mean()),
        "rmsle": float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2))),
        "bias": float((yp - yt).sum() / total) if total else None,
    }
