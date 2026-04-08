from pathlib import Path
import argparse
import importlib
import json

import matplotlib.pyplot as plt
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
                    "forecast_time": target_times[step - 1],
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

    x = np.arange(len(plot_df))
    plt.figure(figsize=(12, 5))
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
    plt.xlabel("ordered prediction point")
    plt.ylabel("power usage")
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"{model_name}_forecast_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    validation_loss = evaluate_validation_loss(model, valid_loader, loss_fn, device)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    targets_df = build_targets_dataframe(test_dataset, formatter)

    merge_keys = ["id", "forecast_time", "forecast_origin", "target_time", "horizon"]
    merged_df = predictions_df.merge(targets_df, on=merge_keys, how="inner")

    if merged_df.empty:
        raise ValueError("Merged evaluation dataframe is empty.")

    metrics = {
        "model": args.model,
        "validation_loss": float(validation_loss),
        "test_nql_p10": normalised_quantile_loss(merged_df["target"], merged_df["p10"], 0.1),
        "test_nql_p50": normalised_quantile_loss(merged_df["target"], merged_df["p50"], 0.5),
        "test_nql_p90": normalised_quantile_loss(merged_df["target"], merged_df["p90"], 0.9),
        "num_prediction_rows": int(len(merged_df)),
    }

    metrics["test_nql_mean"] = float(
        np.nanmean([metrics["test_nql_p10"], metrics["test_nql_p50"], metrics["test_nql_p90"]])
    )

    plot_path = make_plot(merged_df, args.model, plots_dir)
    if plot_path is not None:
        metrics["example_plot"] = str(plot_path)

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

    args = parser.parse_args()
    main(args)