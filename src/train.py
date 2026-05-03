from pathlib import Path
import argparse
import importlib
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_formatter import ElectricityFormatter
from dataset import TFTDataset
from registry import MODEL_REGISTRY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    if not hasattr(module, model_info["class_name"]):
        raise ValueError(f"{model_info['module']} is missing {model_info['class_name']}")
    if not hasattr(module, model_info["loss_name"]):
        raise ValueError(f"{model_info['module']} is missing {model_info['loss_name']}")

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


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch_index, print_every, max_gradient_norm=None):
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

        if max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

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
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    formatter = ElectricityFormatter()
    default_model_params = formatter.get_default_model_params()
    default_fixed_params = formatter.get_fixed_params()

    model_class, loss_fn = load_model_class_and_loss(args.model)

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / f"{args.model}_seed_{args.seed}_best.pt"
    history_path = metrics_dir / f"{args.model}_seed_{args.seed}_training_history.json"

    print("Loading processed electricity data...")
    df = pd.read_csv(data_path)
    print(f"Full data shape: {df.shape}")

    print("Formatting data...")
    train_df, valid_df, test_df = formatter.split_data(df)

    print("Full train dataframe shape:", train_df.shape)
    print("Full valid dataframe shape:", valid_df.shape)
    print("Full test dataframe shape:", test_df.shape)

    if args.num_train_ids is not None:
        print(f"Selecting subset of train IDs: {args.num_train_ids}")
        train_df = select_subset_of_ids(train_df, args.num_train_ids, random_state=args.seed)
    
    if args.num_valid_ids is not None:
        print(f"Selecting subset of valid IDs: {args.num_valid_ids}")
        valid_df = select_subset_of_ids(valid_df, args.num_valid_ids, random_state=args.seed)

    print("Train dataframe shape:", train_df.shape)
    print("Valid dataframe shape:", valid_df.shape)

    print(f"Using seed: {args.seed}")
    
    print("Building datasets...")
    train_dataset = TFTDataset(train_df, formatter)
    valid_dataset = TFTDataset(valid_df, formatter)

    print("Creating dataloaders...")
    batch_size = args.batch_size if args.batch_size is not None else default_model_params["minibatch_size"]
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"Using device: {device}")
    model = model_class(formatter).to(device)
    learning_rate = args.learning_rate if args.learning_rate is not None else default_model_params["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = args.num_epochs if args.num_epochs is not None else default_fixed_params["num_epochs"]
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience is not None else default_fixed_params["early_stopping_patience"]
    max_gradient_norm = args.max_gradient_norm if args.max_gradient_norm is not None else default_model_params.get("max_gradient_norm", None)

    best_valid_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    training_history = []

    print(f"Starting training for model: {args.model}")
    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch_index=epoch,
            print_every=args.print_every,
            max_gradient_norm=max_gradient_norm,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch_time_seconds": epoch_time,
            }
        )

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} completed | "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to: {model_save_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}.")
            break

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "best_valid_loss": best_valid_loss,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs_requested": num_epochs,
                "early_stopping_patience": early_stopping_patience,
                "history": training_history,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print("Training finished.")
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_path", type=str, default="data/electricity_processed.csv")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--max_gradient_norm", type=float, default=None)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--num_train_ids", type=int, default=None)
    parser.add_argument("--num_valid_ids", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)