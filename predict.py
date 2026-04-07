from pathlib import Path
import argparse
import importlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset


MODEL_REGISTRY = {
    "tft_baseline": "models.tft_baseline",
    "mlp_features": "models.mlp_features",
    "no_attention": "models.no_attention",
    "no_lstm": "models.no_lstm",
    "transformer_only": "models.transformer_only",
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    module = importlib.import_module(MODEL_REGISTRY[model_name])
    return getattr(module, "TemporalFusionTransformer")


def quantile_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Normalized quantile loss, similar in spirit to TFT evaluation.
    y_true, y_pred shape: (n_samples, horizon)
    """
    errors = y_true - y_pred
    q_loss = np.maximum(q * errors, (q - 1.0) * errors)
    denominator = np.mean(np.abs(y_true))

    if denominator < 1e-8:
        return float(np.mean(q_loss))
    return float(2.0 * np.mean(q_loss) / denominator)


def rmse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def build_prediction_dataframe(
    model,
    test_loader,
    test_dataset,
    formatter,
    device,
    model_name: str,
):
    model.eval()

    all_rows = []
    sample_index = 0
    encoder_steps = formatter.get_fixed_params()["num_encoder_steps"]

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["outputs"].cpu().numpy()
            predictions = model(inputs).cpu().numpy()

            batch_size = predictions.shape[0]
            decoder_steps = predictions.shape[1]

            for b in range(batch_size):
                raw_sample = test_dataset.samples[sample_index]

                sample_id = raw_sample["identifier"][0, 0]
                sample_times = raw_sample["time"][encoder_steps:, 0]

                # Assumes output_size = 1 and quantiles ordered as p10, p50, p90
                p10 = predictions[b, :, 0]
                p50 = predictions[b, :, 1]
                p90 = predictions[b, :, 2]
                target = targets[b, :, 0]

                for step in range(decoder_steps):
                    all_rows.append(
                        {
                            "model": model_name,
                            "id": sample_id,
                            "forecast_time": sample_times[step],
                            "horizon_step": step + 1,
                            "target": target[step],
                            "p10": p10[step],
                            "p50": p50[step],
                            "p90": p90[step],
                        }
                    )

                sample_index += 1

    predictions_df = pd.DataFrame(all_rows)
    return predictions_df


def inverse_scale_predictions(predictions_df: pd.DataFrame, formatter: ElectricityFormatter):
    """
    Inverse-scale target and quantile columns back to original power_usage scale.
    """
    out = predictions_df.copy()

    for col in ["target", "p10", "p50", "p90"]:
        tmp = out[["forecast_time", "id", col]].rename(columns={"id": "identifier"})
        inv = formatter.format_predictions(tmp)
        out[col] = inv[col].values

    return out


def compute_metrics(predictions_df: pd.DataFrame):
    y_true = predictions_df["target"].to_numpy()
    p10 = predictions_df["p10"].to_numpy()
    p50 = predictions_df["p50"].to_numpy()
    p90 = predictions_df["p90"].to_numpy()

    metrics = {
        "n_rows": int(len(predictions_df)),
        "rmse_p50": rmse_numpy(y_true, p50),
        "mae_p50": mae_numpy(y_true, p50),
        "q_loss_p10": quantile_loss_numpy(y_true, p10, 0.1),
        "q_loss_p50": quantile_loss_numpy(y_true, p50, 0.5),
        "q_loss_p90": quantile_loss_numpy(y_true, p90, 0.9),
        "interval_coverage_10_90": float(np.mean((y_true >= p10) & (y_true <= p90))),
        "mean_interval_width_10_90": float(np.mean(p90 - p10)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model architecture to use.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/electricity_processed.csv",
        help="Path to processed electricity csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If omitted, defaults to checkpoints/{model}_best.pt",
    )
    parser.add_argument(
        "--predictions_output",
        type=str,
        default=None,
        help="Path to save predictions csv. Default: predictions/{model}_test_predictions.csv",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Path to save metrics json. Default: predictions/{model}_test_metrics.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for prediction.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(f"checkpoints/{args.model}_best.pt")
    )
    predictions_output = (
        Path(args.predictions_output)
        if args.predictions_output is not None
        else Path(f"predictions/{args.model}_test_predictions.csv")
    )
    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output is not None
        else Path(f"predictions/{args.model}_test_metrics.json")
    )

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)

    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Building test dataset...")
    test_dataset = TFTDataset(test_df, formatter)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    ModelClass = get_model_class(args.model)
    model = ModelClass(formatter).to(DEVICE)

    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    print("Running predictions...")
    predictions_df = build_prediction_dataframe(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        formatter=formatter,
        device=DEVICE,
        model_name=args.model,
    )

    print("Inverse-scaling predictions and targets...")
    predictions_df = inverse_scale_predictions(predictions_df, formatter)

    print("Computing metrics...")
    metrics = compute_metrics(predictions_df)
    metrics["model"] = args.model
    metrics["checkpoint"] = str(checkpoint_path)

    print(f"Saving predictions to: {predictions_output}")
    predictions_df.to_csv(predictions_output, index=False)

    print(f"Saving metrics to: {metrics_output}")
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(pd.Series(metrics))


if __name__ == "__main__":
    main()