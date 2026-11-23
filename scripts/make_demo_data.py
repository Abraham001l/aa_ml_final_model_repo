"""Create synthetic data to try the workflow. This does NOT reproduce project accuracy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo.csv")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new path.")
    rng = np.random.default_rng(2024)
    rows = []
    airports = ["AAA", "BBB", "CCC", "DDD"]
    for i, a in enumerate(airports):
        for j, b in enumerate(airports):
            if a == b:
                continue
            scale = 10 * (i + 1) * (j + 2)
            for month in pd.period_range("2019-01", "2026-07", freq="M"):
                year_factor = 1 if month.year == 2019 else min(1.12, .4 + .14 * (month.year - 2020))
                seasonal = 1 + .20 * np.sin(2 * np.pi * (month.month - 1) / 12)
                pax = max(0., scale * year_factor * seasonal * month.days_in_month * rng.uniform(.9, 1.1))
                rows.append(dict(ORG_AIRPRT=a, DES_AIRPRT=b, DEP_YEAR=month.year,
                                 DEP_MONTH=month.month, OD_PAX=round(pax, 2)))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="|", index=False)
    print(f"Created synthetic example: {output}. Not real airports, passengers or benchmark evidence.")
