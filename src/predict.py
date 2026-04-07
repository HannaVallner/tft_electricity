from pathlib import Path
import argparse
import importlib

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from registry import MODEL_REGISTRY


def load_model_class_and_loss(model_name):
    """
    Load model class dynamically from registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_name]
    module = importlib.import_module(model_info["module"])
    model_class = getattr(module, model_info["class_name"])
    return model_class


def inverse_scale_predictions(predictions_df, formatter):
    """
    Convert normalized predictions back to original power_usage scale.
    """
    return formatter.format_predictions(predictions_df)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path)
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading processed electricity data...")
    df = pd.read_csv(data_path)
    print("Full data shape:", df.shape)

    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Building test dataset...")
    test_dataset = TFTDataset(test_df, formatter)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_class = load_model_class_and_loss(args.model)

    print(f"Using device: {device}")
    model = model_class(formatter).to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_rows = []
    sample_index = 0

    print("Running predictions...")

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            predictions = model(inputs)  # (batch, decoder_steps, num_quantile_outputs)
            predictions = predictions.cpu().numpy()

            batch_size = predictions.shape[0]
            decoder_steps = predictions.shape[1]

            for b in range(batch_size):
                raw_sample = test_dataset.samples[sample_index]

                sample_id = raw_sample["identifier"][0, 0]
                sample_times = raw_sample["time"][formatter.get_fixed_params()["num_encoder_steps"]:, 0]

                p10 = predictions[b, :, 0]
                p50 = predictions[b, :, 1]
                p90 = predictions[b, :, 2]

                for step in range(decoder_steps):
                    all_rows.append(
                        {
                            "id": sample_id,
                            "forecast_time": sample_times[step],
                            "p10": p10[step],
                            "p50": p50[step],
                            "p90": p90[step],
                        }
                    )

                sample_index += 1

    predictions_df = pd.DataFrame(all_rows)

    print("Inverse-scaling predictions...")
    predictions_df = inverse_scale_predictions(predictions_df, formatter)

    print(f"Saving predictions to: {output_path}")
    predictions_df.to_csv(output_path, index=False)

    print("Done.")
    print(predictions_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model architecture produced the checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/electricity_processed.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()
    main(args)