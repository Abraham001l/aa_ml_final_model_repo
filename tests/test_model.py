import unittest
import numpy as np
import pandas as pd
from route_recovery.data import build_history, build_targets, long_forecast
from route_recovery.metrics import compute_metrics
from route_recovery.model import Config, forecast_history
from route_recovery.search import tune
from route_recovery.verification import run_checks


def synthetic():
    rows = []
    for a, b, scale in [("AAA", "BBB", 110), ("BBB", "AAA", 95), ("AAA", "CCC", 40)]:
        for month in pd.period_range("2019-01", "2026-07", freq="M"):
            daily = scale * (1 + .18 * np.sin(2 * np.pi * (month.month - 1) / 12))
            if month.year >= 2020:
                daily *= .55 + .09 * (month.year - 2020)
            rows.append(dict(ORG_AIRPRT=a, DES_AIRPRT=b, month=month,
                             DEP_YEAR=month.year, DEP_MONTH=month.month,
                             OD_PAX=daily * month.days_in_month))
    return pd.DataFrame(rows)


class LeakageTests(unittest.TestCase):
    def test_36_adversarial_checks_each_family(self):
        for config in [Config(), Config(family="network recovery", recovery=.5)]:
            result = run_checks(synthetic(), config)
            self.assertEqual(result["adversarial_checks"], 36)
            self.assertEqual(result["positive_controls"], 4)

    def test_all_594_development_losses_ignore_final_targets(self):
        raw = synthetic()
        first, losses, ranked = tune(raw)
        raw.loc[raw.month > pd.Period("2024-07"), "OD_PAX"] = 9e12
        second, poisoned, reranked = tune(raw)
        self.assertEqual(first, second)
        self.assertEqual(len(ranked), 198)
        self.assertEqual(len(losses), 594)
        pd.testing.assert_frame_equal(losses, poisoned, check_exact=True)
        pd.testing.assert_frame_equal(ranked, reranked, check_exact=True)

    def test_future_only_reverse_does_not_enable_pooling(self):
        raw = synthetic()
        origin = "2024-07"
        before = forecast_history(build_history(raw, origin))
        extra = dict(ORG_AIRPRT="CCC", DES_AIRPRT="AAA", month=pd.Period("2024-08"),
                     DEP_YEAR=2024, DEP_MONTH=8, OD_PAX=90000.)
        after = forecast_history(build_history(pd.concat([raw, pd.DataFrame([extra])]), origin))
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        self.assertNotIn(("CCC", "AAA"), after.index)


class ModelTests(unittest.TestCase):
    def test_shape_dates_finite_nonnegative(self):
        h = build_history(synthetic(), "2024-07")
        forecast = forecast_history(h)
        self.assertEqual(forecast.shape, (3, 24))
        self.assertEqual(str(forecast.columns[0]), "2024-08")
        self.assertEqual(str(forecast.columns[-1]), "2026-07")
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())
        self.assertTrue((forecast.to_numpy() >= 0).all())

    def test_unpaired_route_retains_own_forecast(self):
        h = build_history(synthetic(), "2024-07")
        no_pool = forecast_history(h, Config(pool=0))
        pooled = forecast_history(h)
        np.testing.assert_array_equal(no_pool.loc[("AAA", "CCC")], pooled.loc[("AAA", "CCC")])

    def test_constant_daily_demand_and_leap_month(self):
        raw = synthetic()
        raw["OD_PAX"] = raw.month.map(lambda p: p.days_in_month) * 10.
        h = build_history(raw, "2022-07")
        forecast = forecast_history(h)
        np.testing.assert_allclose(forecast.to_numpy(),
                                   np.tile(forecast.columns.days_in_month * 10., (3, 1)), rtol=1e-14)
        self.assertEqual(forecast.loc[("AAA", "BBB"), pd.Period("2024-02")], 290.)

    def test_zero_reference_network_is_finite(self):
        raw = synthetic()
        raw.loc[raw.DEP_YEAR == 2019, "OD_PAX"] = 0.
        for config in [Config(), Config(family="network recovery", recovery=.5)]:
            self.assertTrue(np.isfinite(forecast_history(build_history(raw, "2024-07"), config)).all().all())
        raw["OD_PAX"] = 0.
        np.testing.assert_array_equal(forecast_history(build_history(raw, "2024-07")).to_numpy(), 0.)

    def test_duplicate_and_invalid_history_rejected(self):
        raw = synthetic()
        with self.assertRaises(ValueError):
            build_history(pd.concat([raw, raw.iloc[:1]]), "2024-07")
        raw.loc[0, "OD_PAX"] = -1
        with self.assertRaises(ValueError):
            build_history(raw, "2024-07")

    def test_invalid_config(self):
        for kwargs in [dict(window=0), dict(window=1.5), dict(pool=2), dict(recovery=0),
                       dict(recent=np.nan), dict(family="unknown")]:
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_missing_labels_are_distinct_from_missing_target_calendar(self):
        raw = synthetic()
        h = build_history(raw, "2024-07")
        target = pd.Period("2024-08")
        # Other routes establish calendar coverage, but an absent route row scores zero.
        one_missing = raw.loc[~(raw.month.eq(target) & raw.ORG_AIRPRT.eq("BBB"))]
        labels = build_targets(one_missing, h)
        self.assertEqual(labels.loc[("BBB", "AAA"), target], 0.)
        with self.assertRaises(ValueError):
            build_targets(raw.loc[raw.month != target], h)

    def test_prediction_only_needs_no_future_observations(self):
        raw = synthetic()
        raw = raw.loc[raw.month <= pd.Period("2024-07")]
        self.assertEqual(len(long_forecast(forecast_history(build_history(raw, "2024-07")))), 72)

    def test_input_not_modified(self):
        raw = synthetic()
        original = raw.copy(deep=True)
        forecast_history(build_history(raw, "2024-07"))
        pd.testing.assert_frame_equal(raw, original)


class MetricTests(unittest.TestCase):
    def test_weighted_error_is_not_average_route_error(self):
        metric = compute_metrics([100, 10], [90, 0])
        self.assertAlmostEqual(metric["wape"], 20/110)
        self.assertAlmostEqual(metric["bias"], -20/110)
        self.assertEqual(metric["n"], 2)

    def test_zero_target_denominator_and_perfect_scores(self):
        self.assertIsNone(compute_metrics([0, 0], [0, 0])["wape"])
        self.assertEqual(compute_metrics([0, 1], [0, 1])["smape"], 0.)
        with self.assertRaises(ValueError):
            compute_metrics([1], [float("nan")])


if __name__ == "__main__":
    unittest.main()
