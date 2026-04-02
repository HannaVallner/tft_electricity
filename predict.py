from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from model import TemporalFusionTransformer


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATA_PATH = Path("data/electricity_processed.csv")
MODEL_PATH = Path("checkpoints/tft_electricity_small_test_best.pt")
OUTPUT_PATH = Path("predictions/test_predictions.csv")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_scale_predictions(predictions_df, formatter):
    """
    Convert normalized predictions back to original power_usage scale.

    The formatter expects:
    - id
    - forecast_time
    - prediction columns
    """
    return formatter.format_predictions(predictions_df)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading processed electricity data...")
    df = pd.read_csv(DATA_PATH)
    print("Full data shape:", df.shape)

    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Building test dataset...")
    test_dataset = TFTDataset(test_df, formatter)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Using device: {DEVICE}")
    model = TemporalFusionTransformer(formatter).to(DEVICE)

    print(f"Loading checkpoint from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_rows = []

    print("Running predictions...")
    sample_index = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(DEVICE)
            predictions = model(inputs)  # shape: (batch, 24, 3)

            predictions = predictions.cpu().numpy()

            batch_size = predictions.shape[0]
            decoder_steps = predictions.shape[1]

            for b in range(batch_size):
                raw_sample = test_dataset.samples[sample_index]

                # Raw metadata for this sample
                sample_id = raw_sample["identifier"][0, 0]
                sample_times = raw_sample["time"][formatter.get_fixed_params()["num_encoder_steps"]:, 0]

                p10 = predictions[b, :, 0]
                p50 = predictions[b, :, 1]
                p90 = predictions[b, :, 2]

                for step in range(decoder_steps):
                    all_rows.append({
                        "id": sample_id,
                        "forecast_time": sample_times[step],
                        "p10": p10[step],
                        "p50": p50[step],
                        "p90": p90[step],
                    })

                sample_index += 1

    predictions_df = pd.DataFrame(all_rows)

    print("Inverse-scaling predictions...")
    predictions_df = inverse_scale_predictions(predictions_df, formatter)

    print(f"Saving predictions to: {OUTPUT_PATH}")
    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print(predictions_df.head())


if __name__ == "__main__":
    main()