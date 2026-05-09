from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from registry import MODEL_REGISTRY

def format_mean_std(mean_value, std_value):
    if pd.isna(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"

def metric_label(metric_name):
    labels = {
        "validation_loss": "Validation loss",
        "test_nql_p10": "NQL at q=0.1",
        "test_nql_p50": "NQL at q=0.5",
        "test_nql_p90": "NQL at q=0.9",
        "test_nql_mean": "Mean NQL",
        "observed_q10": "Observed proportion at q=0.1",
        "observed_q50": "Observed proportion at q=0.5",
        "observed_q90": "Observed proportion at q=0.9",
        "interval_80_coverage": "80% interval coverage",
        "below_p10_rate": "Below p10 rate",
        "above_p90_rate": "Above p90 rate",
    }
    return labels.get(metric_name, metric_name)

def save_latex_table(summary_df, output_path):
    """
    Save a LaTeX table with mean ± std across seeds.
    """

    rows = []

    metrics = [
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

    best_idx = summary_df["test_nql_mean_mean"].idxmin()

    for idx, row in summary_df.iterrows():
        formatted_row = {
            "rank": int(row["rank"]),
            "model": row["model"],
            "num_seeds": int(row["num_seeds"]),
        }

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in summary_df.columns:
                formatted_row[metric] = format_mean_std(
                    row[mean_col],
                    row[std_col] if std_col in summary_df.columns else float("nan"),
                )

        rows.append(formatted_row)

    latex_df = pd.DataFrame(rows)

    best_model = summary_df.loc[summary_df["test_nql_mean_mean"].idxmin(), "model"]

    latex_df = latex_df.rename(
        columns={
            "rank": "Rank",
            "model": "Model",
            "num_seeds": "Seeds",
            "validation_loss": "Val. loss",
            "test_nql_p10": "NQL$_{0.1}$",
            "test_nql_p50": "NQL$_{0.5}$",
            "test_nql_p90": "NQL$_{0.9}$",
            "test_nql_mean": "Mean NQL",
            "observed_q10": "Obs. Q$_{0.1}$",
            "observed_q50": "Obs. Q$_{0.5}$",
            "observed_q90": "Obs. Q$_{0.9}$",
            "interval_80_coverage": "Cov$_{80}$",
        }
    )

    best_row_idx = latex_df.index[latex_df["Model"] == best_model][0]
    for col in latex_df.columns:
        latex_df.loc[best_row_idx, col] = f"\\textbf{{{latex_df.loc[best_row_idx, col]}}}"
    
    latex_table = latex_df.to_latex(
        index=False,
        caption="Comparison of TFT model variants on the electricity forecasting task across random seeds. Values are reported as mean $\\pm$ standard deviation.",
        label="tab:model_comparison_seeds",
        bold_rows=False,
        escape=False,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)



def save_summary_plot(summary_df, plots_dir):
    """
    Save summary plot using mean test NQL with std error bars.
    """
    metric_name = "test_nql_mean"
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    if mean_col not in summary_df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["model"],
        summary_df[mean_col],
        yerr=summary_df[std_col] if std_col in summary_df.columns else None,
        capsize=4,
    )
    plt.title("Overall model comparison by mean test normalized quantile loss")
    plt.xlabel("Model")
    plt.ylabel("Mean test NQL")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plot_path = plots_dir / "compare_summary_mean_nql_with_std.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def save_metric_plot(summary_df, metric_name, plots_dir):
    """
    Save model comparison plot for one metric using mean ± std.
    """
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    if mean_col not in summary_df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["model"],
        summary_df[mean_col],
        yerr=summary_df[std_col] if std_col in summary_df.columns else None,
        capsize=4,
    )
    plt.title(f"Model comparison: {metric_label(metric_name)}")
    plt.xlabel("Model")
    plt.ylabel(metric_label(metric_name))
    plt.xticks(rotation=30)
    plt.tight_layout()

    plot_path = plots_dir / f"compare_{metric_name}_mean_std.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def save_calibration_comparison_plot(summary_df, metric_name, expected_value, plots_dir):
    """
    Save calibration comparison plot with mean ± std and expected reference line.
    """
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    if mean_col not in summary_df.columns:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(
        summary_df["model"],
        summary_df[mean_col],
        yerr=summary_df[std_col] if std_col in summary_df.columns else None,
        capsize=4,
    )
    plt.axhline(expected_value, linestyle="--", label=f"expected = {expected_value:.2f}")
    plt.title(f"Calibration comparison: {metric_label(metric_name)}")
    plt.xlabel("Model")
    plt.ylabel(metric_label(metric_name))
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"compare_{metric_name}_calibration_mean_std.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def save_combined_reliability_plot(summary_df, plots_dir):
    nominal = [0.1, 0.5, 0.9]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")

    for _, row in summary_df.iterrows():
        observed = [
            row["observed_q10_mean"],
            row["observed_q50_mean"],
            row["observed_q90_mean"],
        ]
        plt.plot(nominal, observed, marker="o", label=row["model"])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed proportion")
    plt.title("Quantile reliability across model variants")
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / "combined_reliability_diagram_mean.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

def save_calibration_error_table(summary_df, output_path):
    """
    Save a compact LaTeX table showing calibration error.
    Lower values indicate better calibration.
    """

    table_df = summary_df[
        [
            "model",
            "test_nql_mean_mean",
            "q10_abs_error_mean",
            "q50_abs_error_mean",
            "q90_abs_error_mean",
            "cov80_abs_error_mean",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "model": "Model",
            "test_nql_mean_mean": "Mean NQL",
            "q10_abs_error_mean": "Q10 error",
            "q50_abs_error_mean": "Q50 error",
            "q90_abs_error_mean": "Q90 error",
            "cov80_abs_error_mean": "Cov80 error",
        }
    )

    numeric_cols = ["Mean NQL", "Q10 error", "Q50 error", "Q90 error", "Cov80 error"]

    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")

    latex_table = table_df.to_latex(
        index=False,
        caption="Calibration error summary across model variants. Lower values indicate closer agreement with the nominal quantile or interval coverage.",
        label="tab:calibration_error_summary",
        escape=False,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)


def main(args):
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name in MODEL_REGISTRY.keys():
        pattern = f"{model_name}_seed_*_metrics.json"
        matching_files = sorted(metrics_dir.glob(pattern))

        if not matching_files:
            print(f"Skipping {model_name}: no seeded metrics files found")
            continue

        for metrics_path in matching_files:
            with open(metrics_path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
    if not rows:
        raise ValueError("No metrics files found to compare.")

    seed_df = pd.DataFrame(rows)

    if "seed" not in seed_df.columns:
        raise ValueError("Metrics files do not contain a 'seed' field.")

    if "test_nql_mean" not in seed_df.columns:
        raise ValueError("Metrics files do not contain 'test_nql_mean'.")

    seed_df = seed_df.sort_values(["model", "seed"]).reset_index(drop=True)

    seed_csv_path = metrics_dir / "model_comparison_by_seed.csv"
    seed_df.to_csv(seed_csv_path, index=False)
    print(f"Saved seed-level comparison table to: {seed_csv_path}")

    numeric_cols = seed_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "seed"]

    summary_df = (
        seed_df
        .groupby("model")[numeric_cols]
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{stat}" for metric, stat in summary_df.columns
    ]

    num_seeds = (
        seed_df
        .groupby("model")["seed"]
        .nunique()
        .rename("num_seeds")
    )

    summary_df = summary_df.join(num_seeds)
    summary_df = summary_df.reset_index()

    summary_df = summary_df.sort_values("test_nql_mean_mean").reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)

    summary_df["q10_abs_error_mean"] = (summary_df["observed_q10_mean"] - 0.10).abs()
    summary_df["q50_abs_error_mean"] = (summary_df["observed_q50_mean"] - 0.50).abs()
    summary_df["q90_abs_error_mean"] = (summary_df["observed_q90_mean"] - 0.90).abs()
    summary_df["cov80_abs_error_mean"] = (summary_df["interval_80_coverage_mean"] - 0.80).abs()

    summary_csv_path = metrics_dir / "model_comparison_summary_mean_std.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary comparison table to: {summary_csv_path}")

    comparison_tex_path = metrics_dir / "model_comparison_summary_mean_std.tex"
    save_latex_table(summary_df, comparison_tex_path)
    print(f"Saved LaTeX table to: {comparison_tex_path}")

    calibration_error_tex_path = metrics_dir / "calibration_error_summary.tex"
    save_calibration_error_table(summary_df, calibration_error_tex_path)
    print(f"Saved calibration error table to: {calibration_error_tex_path}")

    metric_columns = [
        "validation_loss",
        "test_nql_p10",
        "test_nql_p50",
        "test_nql_p90",
        "test_nql_mean",
    ]

    for metric_name in metric_columns:
        plot_path = save_metric_plot(summary_df, metric_name, plots_dir)
        if plot_path is not None:
            print(f"Saved plot to: {plot_path}")

    summary_plot_path = save_summary_plot(summary_df, plots_dir)
    if summary_plot_path is not None:
        print(f"Saved summary plot to: {summary_plot_path}")

    reliability_plot_path = save_combined_reliability_plot(summary_df, plots_dir)
    print(f"Saved combined reliability plot to: {reliability_plot_path}")

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
            summary_df=summary_df,
            metric_name=metric_name,
            expected_value=expected_value,
            plots_dir=plots_dir,
        )
        if calibration_plot_path is not None:
            print(f"Saved calibration plot to: {calibration_plot_path}")

    print("\nSeed-level results:")
    print(seed_df)

    print("\nSummary results:")
    print(summary_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    args = parser.parse_args()
    main(args)