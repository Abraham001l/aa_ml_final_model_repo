"""Adversarial checks against future values, rows and route membership."""
import numpy as np
import pandas as pd
from .data import build_history
from .model import Config, forecast_history

ORIGINS = ("2021-07", "2022-01", "2022-07", "2024-07")


def run_checks(raw, config=Config(), origins=ORIGINS):
    records = []
    for origin in origins:
        cutoff = pd.Period(origin, freq="M")
        future = raw.month > cutoff
        base = forecast_history(build_history(raw, origin), config)
        for variant in ["huge", "zero", "random", "nan", "negative", "infinite",
                        "drop_future", "future_reverse", "row_order"]:
            changed = raw.copy()
            if variant in {"huge", "zero", "nan", "negative", "infinite"}:
                value = {"huge": 9e12, "zero": 0., "nan": np.nan,
                         "negative": -9e12, "infinite": np.inf}[variant]
                changed.loc[future, "OD_PAX"] = value
            elif variant == "random":
                changed.loc[future, "OD_PAX"] = np.random.default_rng(42).uniform(0, 1e9, int(future.sum()))
            elif variant == "drop_future":
                changed = changed.loc[~future]
            elif variant == "future_reverse":
                # Prefer adding a future-only reverse of a known one-way market.
                route = next(((b, a) for a, b in base.index if (b, a) not in base.index), ("ZZZ", "YYY"))
                extra = {"ORG_AIRPRT": route[0], "DES_AIRPRT": route[1],
                         "month": cutoff + 1, "OD_PAX": 9e12,
                         "DEP_YEAR": (cutoff + 1).year, "DEP_MONTH": (cutoff + 1).month}
                changed = pd.concat([changed, pd.DataFrame([extra])], ignore_index=True)
            else:
                changed = changed.sample(frac=1, random_state=19)
            candidate = forecast_history(build_history(changed, origin), config)
            if not base.index.equals(candidate.index) or not base.columns.equals(candidate.columns):
                raise AssertionError(f"Route/date lookahead at {origin}: {variant}")
            np.testing.assert_array_equal(base.to_numpy(), candidate.to_numpy(),
                                          err_msg=f"Future lookahead at {origin}: {variant}")
            records.append({"origin": origin, "variant": variant, "passed": True})
        # Positive control: the test must notice changes in genuinely known demand.
        changed = raw.copy()
        known = changed.month <= cutoff
        changed.loc[known, "OD_PAX"] = pd.to_numeric(changed.loc[known, "OD_PAX"]) * 1.4 + 10
        candidate = forecast_history(build_history(changed, origin), config)
        if np.array_equal(base.to_numpy(), candidate.to_numpy()):
            raise AssertionError("Positive control failed: changing history should affect predictions.")
    return {"passed": True, "adversarial_checks": len(records),
            "positive_controls": len(origins), "config": config.to_dict(), "checks": records}
