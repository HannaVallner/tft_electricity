from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from registry import MODEL_REGISTRY


def main(args):
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in MODEL_REGISTRY.keys():
        metrics_path = metrics_dir / f"{model_name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        else:
            print(f"Skipping {model_name}: metrics file not found at {metrics_path}")

    if not rows:
        raise ValueError("No metrics files found to compare.")

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("test_nql_mean").reset_index(drop=True)

    comparison_csv_path = metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison table to: {comparison_csv_path}")

    metric_columns = ["validation_loss", "test_nql_p10", "test_nql_p50", "test_nql_p90", "test_nql_mean"]
    for metric_name in metric_columns:
        if metric_name not in comparison_df.columns:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(comparison_df["model"], comparison_df[metric_name])
        plt.title(f"Model comparison: {metric_name}")
        plt.xlabel("model")
        plt.ylabel(metric_name)
        plt.xticks(rotation=30)
        plt.tight_layout()

        plot_path = plots_dir / f"compare_{metric_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to: {plot_path}")

    print(comparison_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    args = parser.parse_args()
    main(args)