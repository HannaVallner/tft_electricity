from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from registry import MODEL_REGISTRY

def save_latex_table(comparison_df, output_path):
    """
    Save a compact LaTeX table for thesis use.
    """
    latex_df = comparison_df.copy()

    columns_to_include = [
        col for col in [
            "rank",
            "model",
            "validation_loss",
            "test_nql_p10",
            "test_nql_p50",
            "test_nql_p90",
            "test_nql_mean",
            "observed_q10",
            "observed_q50",
            "observed_q90",
            "interval_80_coverage",
            "below_p10_rate",
            "above_p90_rate",
        ]
        if col in latex_df.columns
    ]

    latex_df = latex_df[columns_to_include]

    latex_table = latex_df.to_latex(
        index=False,
        float_format="%.6f",
        caption="Comparison of TFT model variants on the electricity forecasting task.",
        label="tab:model_comparison",
        bold_rows=False,
        escape=False,
    )

    best_idx = latex_df["test_nql_mean"].idxmin()

    for col in latex_df.columns:
        latex_df.loc[best_idx, col] = f"\\textbf{{{latex_df.loc[best_idx, col]}}}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)


def save_summary_plot(comparison_df, plots_dir):
    """
    Save one clean summary plot using mean test NQL.
    """
    metric_name = "test_nql_mean"
    if metric_name not in comparison_df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["model"], comparison_df[metric_name])
    plt.title("Overall model comparison by mean test normalized quantile loss")
    plt.xlabel("Model")
    plt.ylabel("Mean test NQL")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plot_path = plots_dir / "compare_summary_mean_nql.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def save_calibration_comparison_plot(comparison_df, metric_name, expected_value, plots_dir):
    """
    Save calibration comparison plot with expected reference line.
    """
    if metric_name not in comparison_df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["model"], comparison_df[metric_name])
    plt.axhline(expected_value, linestyle="--", label=f"expected = {expected_value:.2f}")
    plt.title(f"Calibration comparison: {metric_name}")
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"compare_{metric_name}_calibration.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

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
    comparison_df["rank"] = range(1, len(comparison_df) + 1)
    
    # Save CSV comparison
    comparison_csv_path = metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison table to: {comparison_csv_path}")

    # Save LaTeX comparison
    comparison_tex_path = metrics_dir / "model_comparison.tex"
    save_latex_table(comparison_df, comparison_tex_path)
    print(f"Saved LaTeX table to: {comparison_tex_path}")

    # Save individual metric plots
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

    # Save one summary plot
    summary_plot_path = save_summary_plot(comparison_df, plots_dir)
    if summary_plot_path is not None:
        print(f"Saved summary plot to: {summary_plot_path}")

    calibration_targets = {
        "observed_q10": 0.10,
        "observed_q50": 0.50,
        "observed_q90": 0.90,
        "interval_80_coverage": 0.80,
        "below_p10_rate": 0.10,
        "above_p90_rate": 0.10,
    }

    for metric_name, expected_value in calibration_targets.items():
        calibration_plot_path = save_calibration_comparison_plot(
            comparison_df=comparison_df,
            metric_name=metric_name,
            expected_value=expected_value,
            plots_dir=plots_dir,
        )
        if calibration_plot_path is not None:
            print(f"Saved calibration plot to: {calibration_plot_path}")

    print(comparison_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    args = parser.parse_args()
    main(args)