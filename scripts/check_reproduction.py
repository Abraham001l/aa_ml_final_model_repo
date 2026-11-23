"""Check local search/evaluation outputs against the included audited aggregate reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", default="outputs/development")
    parser.add_argument("--evaluation", default="outputs/evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "reports/reference/audited_metrics.json").read_text())
    expected_config = json.loads((root / "configs/final.json").read_text())
    development, evaluation = Path(args.development), Path(args.evaluation)
    for folder in [development, evaluation]:
        run = json.loads((folder / "run.json").read_text())
        if run["input_sha256"] != reference["data_sha256"]:
            raise ValueError("Input differs from the audited dataset; reference scores are not applicable.")
    selected = json.loads((development / "selected_config.json").read_text())
    if selected != expected_config:
        raise AssertionError("Selected settings differ from the audited winner.")
    scores = json.loads((evaluation / "metrics.json").read_text())
    mapping = {"full_24m": "full_24m", "first_21m": "clean_pre_anomaly_21m",
               "first_17m": "clean_through_2025_17m"}
    for produced, old in mapping.items():
        # Accept historical reference window names through their numeric suffix.
        if old not in reference["holdout"]:
            old = next(k for k in reference["holdout"] if str(21 if produced == "first_21m" else 17) in k)
        for metric, value in reference["holdout"][old]["market_dir_overall"].items():
            np.testing.assert_allclose(scores[produced]["model"][metric], value, rtol=1e-12, atol=1e-12)
    losses = pd.read_csv(development / "development_losses.csv")
    if len(losses) != 594:
        raise AssertionError("Expected 594 development losses.")
    mask = np.ones(len(losses), dtype=bool)
    for key, value in selected.items():
        mask &= losses[key].eq(value)
    selected_rows = losses.loc[mask].sort_values("origin")
    if len(selected_rows) != 3:
        raise AssertionError("Expected exactly three selected-configuration fold scores.")
    for (_, row), old in zip(selected_rows.iterrows(), reference["folds"].values()):
        np.testing.assert_allclose(row.wape, old["wape"], rtol=1e-12, atol=1e-12)
    print("PASS: selected settings, all three fold WAPEs, and all three final-window metric suites match the audited reference.")
