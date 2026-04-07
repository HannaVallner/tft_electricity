from pathlib import Path
import argparse
import importlib
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from registry import MODEL_REGISTRY


def load_model_class_and_loss(model_name):
    """
    Load model class and loss function dynamically from the registry.
    """
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


def select_subset_of_ids(df, num_ids, random_state=42):
    """
    Keep only a subset of unique IDs.
    Useful for quick experiments/debugging.
    """
    unique_ids = df["id"].drop_duplicates()

    if num_ids is None:
        return df.copy()

    if num_ids > len(unique_ids):
        raise ValueError(
            f"Requested {num_ids} ids, but only {len(unique_ids)} are available."
        )

    selected_ids = unique_ids.sample(n=num_ids, random_state=random_state)

    subset = df[df["id"].isin(selected_ids)].copy()
    subset = subset.sort_values(["id", "hours_from_start"]).reset_index(drop=True)

    return subset


def evaluate(model, dataloader, loss_fn, device):
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
            loss = loss_fn(targets, predictions)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return total_loss / num_batches


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch_index, print_every):
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
        loss = loss_fn(targets, predictions)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % print_every == 0:
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class, loss_fn = load_model_class_and_loss(args.model)

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"{args.model}_best.pt"

    print("Loading processed electricity data...")
    df = pd.read_csv(data_path)
    print(f"Full data shape: {df.shape}")

    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Full train dataframe shape:", train_df.shape)
    print("Full valid dataframe shape:", valid_df.shape)
    print("Full test dataframe shape:", test_df.shape)

    train_df = select_subset_of_ids(train_df, args.num_train_ids, random_state=42)
    valid_df = select_subset_of_ids(valid_df, args.num_valid_ids, random_state=42)

    print("Train dataframe shape after ID subset:", train_df.shape)
    print("Valid dataframe shape after ID subset:", valid_df.shape)
    print("Train unique IDs:", train_df["id"].nunique())
    print("Valid unique IDs:", valid_df["id"].nunique())

    print("Building datasets...")
    train_dataset = TFTDataset(train_df, formatter)
    valid_dataset = TFTDataset(valid_df, formatter)

    train_dataset.summary()
    valid_dataset.summary()

    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Using device: {device}")
    model = model_class(formatter).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_valid_loss = float("inf")

    print(f"Starting training for model: {args.model}")
    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch_index=epoch,
            print_every=args.print_every,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch + 1}/{args.num_epochs} completed | "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to: {checkpoint_path}")

    print("Training finished.")
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"Best checkpoint saved at: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model architecture to train.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/electricity_processed.csv",
        help="Path to processed electricity csv.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_train_ids",
        type=int,
        default=None,
        help="Optional subset of train IDs for quick experiments.",
    )
    parser.add_argument(
        "--num_valid_ids",
        type=int,
        default=None,
        help="Optional subset of validation IDs for quick experiments.",
    )

    args = parser.parse_args()
    main(args)