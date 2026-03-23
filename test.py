from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from model import TemporalFusionTransformer, quantile_loss


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATA_PATH = Path("data/electricity_processed.csv")
MODEL_SAVE_PATH = Path("checkpoints/tft_electricity_small_test_best.pt")

# Small debug run settings
NUM_TRAIN_IDS = 20
NUM_VALID_IDS = 20

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
PRINT_EVERY = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation data and return average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["outputs"].to(device)

            predictions = model(inputs)
            loss = quantile_loss(targets, predictions)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return total_loss / num_batches


def train_one_epoch(model, dataloader, optimizer, device, epoch_index):
    """
    Train model for one epoch and return average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs = batch["inputs"].to(device)
        targets = batch["outputs"].to(device)

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = quantile_loss(targets, predictions)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_batches
            print(
                f"Epoch {epoch_index + 1} | "
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Train Loss: {avg_loss:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    if num_batches == 0:
        raise ValueError("Training dataloader produced zero batches.")

    return total_loss / num_batches


def select_subset_of_ids(df, num_ids, random_state=42):
    """
    Keep only a subset of unique IDs.

    This is much better than random row sampling because it preserves
    contiguous sequences for each time series.
    """
    unique_ids = df["id"].drop_duplicates()

    if num_ids > len(unique_ids):
        raise ValueError(
            f"Requested {num_ids} ids, but only {len(unique_ids)} are available."
        )

    selected_ids = unique_ids.sample(n=num_ids, random_state=random_state)

    subset = df[df["id"].isin(selected_ids)].copy()
    subset = subset.sort_values(["id", "hours_from_start"]).reset_index(drop=True)

    return subset


def main():
    # --------------------------------------------------------
    # Prepare output folder
    # --------------------------------------------------------
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------
    print("Loading processed electricity data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Full data shape: {df.shape}")

    # --------------------------------------------------------
    # Format + split
    # --------------------------------------------------------
    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Full train dataframe shape:", train_df.shape)
    print("Full valid dataframe shape:", valid_df.shape)
    print("Full test dataframe shape:", test_df.shape)

    # --------------------------------------------------------
    # Keep only a small subset of IDs for debugging
    # --------------------------------------------------------
    print(f"Selecting small subset of train IDs: {NUM_TRAIN_IDS}")
    train_df = select_subset_of_ids(train_df, NUM_TRAIN_IDS, random_state=42)

    print(f"Selecting small subset of valid IDs: {NUM_VALID_IDS}")
    valid_df = select_subset_of_ids(valid_df, NUM_VALID_IDS, random_state=42)

    print("Small train dataframe shape:", train_df.shape)
    print("Small valid dataframe shape:", valid_df.shape)

    print("Train unique IDs:", train_df["id"].nunique())
    print("Valid unique IDs:", valid_df["id"].nunique())

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------
    print("Building datasets...")
    train_dataset = TFTDataset(train_df, formatter)
    valid_dataset = TFTDataset(valid_df, formatter)

    train_dataset.summary()
    valid_dataset.summary()

    # --------------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------------
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    print(f"Using device: {DEVICE}")
    model = TemporalFusionTransformer(formatter).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_valid_loss = float("inf")

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch_index=epoch,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            device=DEVICE,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS} completed | "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to: {MODEL_SAVE_PATH}")

    print("Training finished.")
    print(f"Best validation loss: {best_valid_loss:.6f}")


if __name__ == "__main__":
    main()