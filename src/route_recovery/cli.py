"""Separate parameter selection, forecasting, evaluation and verification commands."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from .data import read_data, build_history, build_targets, long_forecast
from .metrics import compute_metrics
from .model import Config, forecast_history
from .search import tune
from .verification import run_checks


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance(args, config, **extra):
    digest = hashlib.sha256()
    with Path(args.data).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(package_version=__version__, python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, command=args.command,
                input_filename=Path(args.data).name, input_sha256=digest.hexdigest(),
                config=config.to_dict(), **extra)


def evaluate(raw, history, prediction):
    actual = build_targets(raw, history)
    own = long_forecast(prediction, actual)
    # Original baseline repeats the latest monthly raw totals, not daily-normalized PAX.
    baseline = np.tile(history.iloc[:, -12:].to_numpy(), (1, 2))
    report = {}
    for name, count in [("full_24m", 24), ("first_21m", 21), ("first_17m", 17)]:
        yt, yp = actual.iloc[:, :count].to_numpy(), prediction.iloc[:, :count].to_numpy()
        report[name] = {"model": compute_metrics(yt, yp),
                        "seasonal_repeat": compute_metrics(yt, baseline[:, :count])}
    report["year1"] = {"model": compute_metrics(actual.iloc[:, :12], prediction.iloc[:, :12])}
    report["year2"] = {"model": compute_metrics(actual.iloc[:, 12:], prediction.iloc[:, 12:])}
    per_route = []
    for index, key in enumerate(history.index):
        metrics = compute_metrics(actual.iloc[index].to_numpy(), prediction.iloc[index].to_numpy())
        per_route.append(dict(ORG_AIRPRT=key[0], DES_AIRPRT=key[1], **metrics))
    route_metrics = pd.DataFrame(per_route)
    valid = route_metrics.wape.dropna()
    report["route_distribution"] = dict(routes=len(route_metrics), positive_total_routes=len(valid),
                                       median_wape=float(valid.median()) if len(valid) else None,
                                       q25_wape=float(valid.quantile(.25)) if len(valid) else None,
                                       q75_wape=float(valid.quantile(.75)) if len(valid) else None)
    return report, own, route_metrics


def plot(args):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter
    predictions = pd.read_csv(args.predictions)
    route = predictions.loc[predictions.ORG_AIRPRT.eq(args.origin_airport.upper()) &
                            predictions.DES_AIRPRT.eq(args.destination_airport.upper())].sort_values("month")
    if route.empty:
        raise ValueError("Route does not appear in prediction file.")
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = pd.to_datetime(route.month)
    if "y_true" in route:
        ax.plot(dates, route.y_true, label="Actual")
    ax.plot(dates, route.y_pred, linestyle="--", label="Forecast")
    ax.set(title=f"{args.origin_airport.upper()} → {args.destination_airport.upper()}: monthly passengers",
           xlabel="Month", ylabel="Passengers (OD_PAX)", ylim=(0, None))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive Route-Recovery Forecasting")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["tune", "predict", "evaluate", "verify"]:
        sub = subs.add_parser(name)
        sub.add_argument("--data", required=True, help="Path to pipe-delimited pax_forecasting.csv")
        sub.add_argument("--output", required=True, help="New output directory (must not exist)")
        if name != "tune":
            sub.add_argument("--config", help="Settings JSON; omitted means published final settings")
        if name in {"predict", "evaluate"}:
            sub.add_argument("--origin", default="2024-07", help="Last observed month, YYYY-MM")
    sub = subs.add_parser("plot")
    sub.add_argument("--predictions", required=True)
    sub.add_argument("--origin-airport", default="IAH")
    sub.add_argument("--destination-airport", default="LAX")
    sub.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        plot(args)
        return
    output = Path(args.output)
    if output.exists():
        parser.error("Output directory already exists. Choose a fresh directory to preserve run provenance.")
    raw = read_data(args.data)
    config = Config(**json.loads(Path(args.config).read_text())) if getattr(args, "config", None) else Config()
    if args.command == "predict":
        history = build_history(raw, args.origin)
        prediction = forecast_history(history, config)
        output.mkdir(parents=True)
        long_forecast(prediction).to_csv(output / "predictions.csv", index=False)
        write_json(output / "run.json", provenance(args, config, origin=args.origin,
                   routes=len(history), purpose="Forecast generation without target access"))
    elif args.command == "verify":
        checks = run_checks(raw, config)
        output.mkdir(parents=True)
        write_json(output / "leakage_checks.json", checks)
        write_json(output / "run.json", provenance(args, config))
        print(f"Passed {checks['adversarial_checks']} adversarial checks and {checks['positive_controls']} positive controls.")
    elif args.command == "tune":
        # Verify both searched mathematical families before recording development scores.
        first = run_checks(raw, Config())
        second = run_checks(raw, Config(family="network recovery", recovery=.5))
        config, losses, ranked = tune(raw)
        selected = run_checks(raw, config)
        output.mkdir(parents=True)
        losses.to_csv(output / "development_losses.csv", index=False)
        ranked.to_csv(output / "ranked_configurations.csv", index=False)
        write_json(output / "selected_config.json", config.to_dict())
        write_json(output / "leakage_checks.json", dict(route_family=first, network_family=second, selected=selected))
        write_json(output / "run.json", provenance(args, config, configurations=len(ranked),
                   development_origins=["2021-07", "2022-01", "2022-07"],
                   last_development_label="2024-07", target_window_exposure="Previously inspected project test"))
        print(f"Selected from {len(ranked)} configurations: {config.to_dict()}")
        print(f"Mean development WAPE: {ranked.iloc[0].wape:.6f} (tuned, not nested CV)")
    else:
        checks = run_checks(raw, config, origins=(args.origin,))
        history = build_history(raw, args.origin)
        prediction = forecast_history(history, config)
        report, predictions, route_metrics = evaluate(raw, history, prediction)
        output.mkdir(parents=True)
        write_json(output / "leakage_checks.json", checks)
        write_json(output / "metrics.json", report)
        predictions.to_csv(output / "predictions.csv", index=False)
        route_metrics.to_csv(output / "route_metrics.csv", index=False)
        write_json(output / "run.json", provenance(args, config, origin=args.origin,
                   target_start=str(prediction.columns[0]), target_end=str(prediction.columns[-1]),
                   historical_routes=len(history), evaluation_status="Historical, previously inspected project window"))
        print(f"Full 24-month WAPE: {report['full_24m']['model']['wape']:.6f}")
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
