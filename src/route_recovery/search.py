"""Chronological development search matching the original 198 configurations."""
from itertools import product
import numpy as np
import pandas as pd
from .data import build_history, build_targets
from .model import Config, seasonal_components, recovery_components, directional_adjustment

DEV_ORIGINS = ("2021-07", "2022-01", "2022-07")


def tune(raw):
    # Restrict input before even constructing folds. Final labels cannot enter search.
    raw = raw.loc[raw.month <= pd.Period("2024-07", freq="M")]
    rows = []
    for origin in DEV_ORIGINS:
        history = build_history(raw, origin)
        actual = build_targets(raw, history).to_numpy()
        total = actual.sum()
        if total <= 0:
            raise ValueError("Development fold has no positive target demand.")
        for family, windows, recoveries in [
            ("network recovery", [3, 6, 9], [.5, 1.]),
            ("route recovery", [1, 2, 3, 6], [2., 6., 12., 24.]),
        ]:
            for window, recent, recovery in product(windows, [.35, .7, 1.], recoveries):
                if family == "route recovery":
                    own = recovery_components(history, window, recovery, recent)
                    paired = directional_adjustment(history, own)
                else:
                    own, paired = seasonal_components(history, window, recent, recovery)
                for pool in [0., .25, .5]:
                    pred = own + pool * (paired - own)
                    rows.append(dict(origin=origin, family=family, window=window,
                                     recent=recent, recovery=recovery, pool=pool,
                                     wape=float(np.abs(pred - actual).sum() / total)))
    losses = pd.DataFrame(rows)
    params = ["family", "window", "recent", "recovery", "pool"]
    # Mean across folds, not pooled across all fold labels; original selection objective.
    ranked = losses.groupby(params).wape.mean().sort_values(kind="stable").reset_index()
    best = ranked.iloc[0]
    config = Config(family=best.family, window=int(best.window), recent=float(best.recent),
                    recovery=float(best.recovery), pool=float(best.pool))
    return config, losses, ranked
