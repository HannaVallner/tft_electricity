from pathlib import Path
import argparse
import importlib
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset


MODEL_REGISTRY = {
    "baseline": "models.tft_baseline",
    "no_lstm": "models.tft_no_lstm",
    "no_attention": "models.tft_no_attention",
    "mlp_features": "models.tft_mlp_features",
    "transformer_only": "models.tft_transformer_only",
}


def load_model_class_and_loss(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    module = importlib.import_module(MODEL_REGISTRY[model_name])

    if not hasattr(module, "TemporalFusionTransformer"):
        raise ValueError(f"{MODEL_REGISTRY[model_name]} is missing TemporalFusionTransformer")

    if not hasattr(module, "quantile_loss"):
        raise ValueError(f"{MODEL_REGISTRY[model_name]} is missing quantile_loss")

    return module.TemporalFusionTransformer, module.quantile_loss


def evaluate(model, dataloader, loss_fn, device):
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


def select_subset_of_ids(df, num_ids, random_state=42):
    unique_ids = df["id"].drop_duplicates()

    if num_ids > len(unique_ids):
        raise ValueError(
            f"Requested {num_ids} ids, but only {len(unique_ids)} are available."
        )

    selected_ids = unique_ids.sample(n=num_ids, random_state=random_state)
    subset = df[df["id"].isin(selected_ids)].copy()
    subset = subset.sort_values(["id", "hours_from_start"]).reset_index(drop=True)
    return subset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class, loss_fn = load_model_class_and_loss(args.model)

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / f"{args.model}_best.pt"

    print("Loading processed electricity data...")
    df = pd.read_csv(data_path)
    print(f"Full data shape: {df.shape}")

    print("Formatting data...")
    formatter = ElectricityFormatter()
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Full train dataframe shape:", train_df.shape)
    print("Full valid dataframe shape:", valid_df.shape)
    print("Full test dataframe shape:", test_df.shape)

    if args.num_train_ids is not None:
        print(f"Selecting subset of train IDs: {args.num_train_ids}")
        train_df = select_subset_of_ids(train_df, args.num_train_ids, random_state=42)

    if args.num_valid_ids is not None:
        print(f"Selecting subset of valid IDs: {args.num_valid_ids}")
        valid_df = select_subset_of_ids(valid_df, args.num_valid_ids, random_state=42)

    print("Train dataframe shape:", train_df.shape)
    print("Valid dataframe shape:", valid_df.shape)

    print("Building datasets...")
    train_dataset = TFTDataset(train_df, formatter)
    valid_dataset = TFTDataset(valid_df, formatter)

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
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to: {model_save_path}")

    print("Training finished.")
    print(f"Best validation loss: {best_valid_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", type=str, default="data/electricity_processed.csv")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--num_train_ids", type=int, default=None)
    parser.add_argument("--num_valid_ids", type=int, default=None)

    args = parser.parse_args()
    main(args)