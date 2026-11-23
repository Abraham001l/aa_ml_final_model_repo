"""Run development search, freeze its selected settings, then evaluate the final window."""
import argparse
from pathlib import Path
from route_recovery.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/reproduction")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        parser.error("Choose a new output directory.")
    data = str(Path(args.data).resolve())
    main(["tune", "--data", data, "--output", str(root / "development")])
    main(["evaluate", "--data", data, "--config", str(root / "development/selected_config.json"),
          "--origin", "2024-07", "--output", str(root / "evaluation")])
