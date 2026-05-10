from pathlib import Path
import argparse
import importlib
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from registry import MODEL_REGISTRY


def load_model_class_and_loss(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_name]
    module = importlib.import_module(model_info["module"])
    model_class = getattr(module, model_info["class_name"])
    loss_fn = getattr(module, model_info["loss_name"])
    return model_class, loss_fn


def normalised_quantile_loss(y_true, y_pred, quantile):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_true - y_pred
    weighted_errors = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    normaliser = np.mean(np.abs(y_true))
    if normaliser == 0:
        return float("nan")
    return float(2.0 * np.mean(weighted_errors) / normaliser)

def compute_quantile_calibration_metrics(merged_df):
    """
    Compute simple calibration diagnostics for predicted quantiles.
    """
    target = merged_df["target"].to_numpy()
    p10 = merged_df["p10"].to_numpy()
    p50 = merged_df["p50"].to_numpy()
    p90 = merged_df["p90"].to_numpy()

    observed_q10 = float(np.mean(target <= p10))
    observed_q50 = float(np.mean(target <= p50))
    observed_q90 = float(np.mean(target <= p90))

    interval_80_coverage = float(np.mean((target >= p10) & (target <= p90)))

    return {
        "observed_q10": observed_q10,
        "observed_q50": observed_q50,
        "observed_q90": observed_q90,
        "interval_80_coverage": interval_80_coverage,
    }

def build_targets_dataframe(test_dataset, formatter):
    rows = []
    encoder_steps = formatter.get_fixed_params()["num_encoder_steps"]

    for raw_sample in test_dataset.samples:
        sample_id = raw_sample["identifier"][0, 0]
        full_times = raw_sample["time"][:, 0]
        forecast_origin = full_times[encoder_steps - 1]
        target_times = full_times[encoder_steps:]
        targets = raw_sample["outputs"].numpy().reshape(-1)

        for step, value in enumerate(targets, start=1):
            rows.append(
                {
                    "id": sample_id,
                    "forecast_origin": forecast_origin,
                    "target_time": target_times[step - 1],
                    "horizon": step,
                    "target": value,
                }
            )

    targets_df = pd.DataFrame(rows)
    targets_df = formatter.format_predictions(targets_df)
    return targets_df


def evaluate_validation_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["outputs"].to(device)

            predictions = model(inputs)
            loss = loss_fn(targets, predictions)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return total_loss / num_batches


def make_plot(merged_df, model_name, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)

    first_id = merged_df["id"].iloc[0]
    first_horizon = 1
    plot_df = (
        merged_df[(merged_df["id"] == first_id) & (merged_df["horizon"] == first_horizon)]
        .sort_values("target_time")
        .head(200)
    )

    if plot_df.empty:
        return None

    x = plot_df["target_time_dt"]
    plt.figure(figsize=(10, 5))
    plt.plot(x, plot_df["target"].values, label="target")
    plt.plot(x, plot_df["p50"].values, label="p50")
    plt.fill_between(
        x=x,
        y1=plot_df["p10"].values,
        y2=plot_df["p90"].values,
        alpha=0.25,
        label="p10-p90",
    )
    plt.title(f"{model_name}: first series, horizon 1")
    plt.xlabel("Time")
    plt.ylabel("Power usage (kW)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"{model_name}_forecast_plot.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path

def make_forecast_window_plot(merged_df, model_name, plots_dir):
    """
    Plot one complete 24-hour forecast window for a single forecast origin.

    This is useful for thesis/report visualization because it shows
    how the quantile forecast behaves across the whole prediction horizon.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pick first available forecast window
    first_row = merged_df.iloc[0]
    selected_id = first_row["id"]
    selected_origin = first_row["forecast_origin"]

    plot_df = (
        merged_df[
            (merged_df["id"] == selected_id) &
            (merged_df["forecast_origin"] == selected_origin)
        ]
        .sort_values("horizon")
        .copy()
    )

    if plot_df.empty:
        return None

    x = plot_df["target_time_dt"]

    plt.figure(figsize=(10, 5))
    plt.plot(x, plot_df["target"].values, marker="o", label="target")
    plt.plot(x, plot_df["p50"].values, marker="o", label="p50")
    plt.fill_between(
        x=x,
        y1=plot_df["p10"].values,
        y2=plot_df["p90"].values,
        alpha=0.25,
        label="p10-p90",
    )
    forecast_date = plot_df["target_time_dt"].iloc[0].strftime("%d-%m-%Y")

    plt.title(f"{model_name}: example 24-hour forecast window ({forecast_date})")
    plt.xlabel("Hour")
    plt.ylabel("Power usage (kW)")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    plt.xticks(rotation=0)
    plt.legend(loc="best")
    plt.tight_layout()

    plot_path = plots_dir / f"{model_name}_forecast_window_plot.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path

def make_reliability_plot(merged_df, model_name, plots_dir):
    """
    Reliability diagram for predicted quantiles.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    nominal = np.array([0.1, 0.5, 0.9], dtype=float)
    observed = np.array([
        np.mean(merged_df["target"].to_numpy() <= merged_df["p10"].to_numpy()),
        np.mean(merged_df["target"].to_numpy() <= merged_df["p50"].to_numpy()),
        np.mean(merged_df["target"].to_numpy() <= merged_df["p90"].to_numpy()),
    ], dtype=float)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.plot(nominal, observed, marker="o", label="model")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed proportion")
    plt.title(f"{model_name}: quantile reliability diagram")
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"{model_name}_reliability_plot.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path

def make_interval_calibration_plot(merged_df, model_name, plots_dir):
    """
    Plot observed vs expected coverage/error rates for the p10-p90 interval.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    target = merged_df["target"].to_numpy()
    p10 = merged_df["p10"].to_numpy()
    p90 = merged_df["p90"].to_numpy()

    observed_values = [
        float(np.mean((target >= p10) & (target <= p90))),  # expected 0.80
        float(np.mean(target < p10)),                       # expected 0.10
        float(np.mean(target > p90)),                       # expected 0.10
    ]
    expected_values = [0.80, 0.10, 0.10]
    labels = ["inside p10-p90", "below p10", "above p90"]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, expected_values, width=width, label="expected")
    plt.bar(x + width / 2, observed_values, width=width, label="observed")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Proportion")
    plt.title(f"{model_name}: interval calibration check")
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"{model_name}_interval_calibration_plot.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.model if args.seed is None else f"{args.model}_seed_{args.seed}"
    data_path = Path(args.data_path)
    checkpoint_path = Path(args.checkpoint_path)
    predictions_path = Path(args.predictions_path)
    metrics_path = Path(args.metrics_path)
    plots_dir = Path(args.plots_dir)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed electricity data...")
    df = pd.read_csv(data_path)

    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Building datasets...")
    valid_dataset = TFTDataset(valid_df, formatter)
    test_dataset = TFTDataset(test_df, formatter)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model_class, loss_fn = load_model_class_and_loss(args.model)

    print(f"Using device: {device}")
    model = model_class(formatter).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Running evaluation for: {run_name}")
    validation_loss = evaluate_validation_loss(model, valid_loader, loss_fn, device)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    targets_df = build_targets_dataframe(test_dataset, formatter)

    merge_keys = ["id", "forecast_origin", "target_time", "horizon"]
    merged_df = predictions_df.merge(targets_df, on=merge_keys, how="inner")

    start_date = pd.Timestamp("2011-01-01 00:00:00")

    merged_df["forecast_origin_dt"] = (
        start_date + pd.to_timedelta(merged_df["forecast_origin"], unit="h")
    )

    merged_df["target_time_dt"] = (
        start_date + pd.to_timedelta(merged_df["target_time"], unit="h")
    )

    if merged_df.empty:
        raise ValueError("Merged evaluation dataframe is empty.")

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "validation_loss": float(validation_loss),
        "test_nql_p10": normalised_quantile_loss(merged_df["target"], merged_df["p10"], 0.1),
        "test_nql_p50": normalised_quantile_loss(merged_df["target"], merged_df["p50"], 0.5),
        "test_nql_p90": normalised_quantile_loss(merged_df["target"], merged_df["p90"], 0.9),
        "num_prediction_rows": int(len(merged_df)),
    }

    calibration_metrics = compute_quantile_calibration_metrics(merged_df)
    metrics.update(calibration_metrics)

    metrics["test_nql_mean"] = float(
        np.nanmean([metrics["test_nql_p10"], metrics["test_nql_p50"], metrics["test_nql_p90"]])
    )

    plot_path = make_plot(merged_df, run_name, plots_dir)
    if plot_path is not None:
        metrics["example_plot"] = str(plot_path)

    forecast_window_plot_path = make_forecast_window_plot(merged_df, run_name, plots_dir)
    if forecast_window_plot_path is not None:
        metrics["forecast_window_plot"] = str(forecast_window_plot_path)

    reliability_plot_path = make_reliability_plot(merged_df, run_name, plots_dir)
    if reliability_plot_path is not None:
        metrics["reliability_plot"] = str(reliability_plot_path)

    interval_calibration_plot_path = make_interval_calibration_plot(merged_df, run_name, plots_dir)
    if interval_calibration_plot_path is not None:
        metrics["interval_calibration_plot"] = str(interval_calibration_plot_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", type=str, default="data/electricity_processed.csv")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    main(args)