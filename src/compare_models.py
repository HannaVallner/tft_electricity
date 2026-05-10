from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from registry import MODEL_REGISTRY
from data_formatter import ElectricityFormatter
from dataset import TFTDataset


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
    ]

    best_idx = summary_df["test_nql_mean_mean"].idxmin()

    for idx, row in summary_df.iterrows():
        formatted_row = {
            "rank": int(row["rank"]),
            "model": row["model"],
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

    latex_df = latex_df.astype(str)
    best_row_idx = latex_df.index[latex_df["Model"] == best_model][0]
    for col in latex_df.columns:
        latex_df.loc[best_row_idx, col] = f"\\textbf{{{latex_df.loc[best_row_idx, col]}}}"
    
    latex_table = latex_df.to_latex(
        index=False,
        caption="Comparison of TFT model variants across random seeds.",
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

    plot_path = plots_dir / "compare_summary_mean_nql_with_std.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
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

    plot_path = plots_dir / f"compare_{metric_name}_mean_std.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path

def build_targets_dataframe(test_dataset, formatter):
    rows = []
    encoder_steps = formatter.get_fixed_params()["num_encoder_steps"]

    for raw_sample in test_dataset.samples:
        sample_id = raw_sample["identifier"][0, 0]
        full_times = raw_sample["time"][:, 0]
        forecast_origin = full_times[encoder_steps - 1]
        target_times = full_times[encoder_steps:]
        targets = raw_sample["outputs"].numpy().reshape(-1)

        for step, value in enumerate(targets, start=1):
            rows.append(
                {
                    "id": sample_id,
                    "forecast_origin": forecast_origin,
                    "target_time": target_times[step - 1],
                    "horizon": step,
                    "target": value,
                }
            )

    targets_df = pd.DataFrame(rows)
    targets_df = formatter.format_predictions(targets_df)
    return targets_df

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

    plot_path = plots_dir / f"compare_{metric_name}_calibration_mean_std.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
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

    plot_path = plots_dir / "combined_reliability_diagram_mean.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
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
            "q10_abs_error_mean",
            "q50_abs_error_mean",
            "q90_abs_error_mean",
            "cov80_abs_error_mean",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "model": "Model",
            "q10_abs_error_mean": "Q10 error",
            "q50_abs_error_mean": "Q50 error",
            "q90_abs_error_mean": "Q90 error",
            "cov80_abs_error_mean": "Cov80 error",
        }
    )

    numeric_cols = ["Q10 error", "Q50 error", "Q90 error", "Cov80 error"]

    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")

    latex_table = table_df.to_latex(
        index=False,
        caption="Calibration error summary across model variants.",
        label="tab:calibration_error_summary",
        escape=False,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

def find_representative_shared_seed(seed_df):
    """
    Select one shared seed whose average Mean NQL across models is closest
    to the overall average across seeds.
    """
    seed_means = seed_df.groupby("seed")["test_nql_mean"].mean()
    overall_mean = seed_means.mean()
    representative_seed = (seed_means - overall_mean).abs().idxmin()
    return int(representative_seed)

def save_representative_p50_comparison_plot(
    representative_seed,
    predictions_dir,
    plots_dir,
    data_path,
):
    models = ["baseline", "no_attention", "mlp_features", "no_lstm", "transformer_only"]

    formatter = ElectricityFormatter()
    df = pd.read_csv(data_path)
    _, _, test_df = formatter.split_data(df)
    test_dataset = TFTDataset(test_df, formatter)
    targets_df = build_targets_dataframe(test_dataset, formatter)

    prediction_frames = {}

    for model_name in models:
        prediction_path = predictions_dir / f"{model_name}_seed_{representative_seed}_predictions.csv"

        if not prediction_path.exists():
            print(f"Skipping p50 comparison plot: missing {prediction_path}")
            return None

        prediction_frames[model_name] = pd.read_csv(prediction_path)

    merge_keys = ["id", "forecast_origin", "target_time", "horizon"]

    base_merged = prediction_frames[models[0]].merge(
        targets_df,
        on=merge_keys,
        how="inner",
    )

    if base_merged.empty:
        print("Skipping p50 comparison plot: merged base dataframe is empty")
        return None

    first_row = base_merged.iloc[0]
    selected_id = first_row["id"]
    selected_origin = first_row["forecast_origin"]

    base_window = (
        base_merged[
            (base_merged["id"] == selected_id)
            & (base_merged["forecast_origin"] == selected_origin)
        ]
        .sort_values("horizon")
        .copy()
    )

    start_date = pd.Timestamp("2011-01-01 00:00:00")
    base_window["target_time_dt"] = (
        start_date + pd.to_timedelta(base_window["target_time"], unit="h")
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        base_window["target_time_dt"],
        base_window["target"].values,
        color="black",
        marker="o",
        linewidth=3.2,
        markersize=6,
        label="target",
        zorder=10,
    )

    for model_name in models:
        model_merged = prediction_frames[model_name].merge(
            targets_df,
            on=merge_keys,
            how="inner",
        )

        model_window = (
            model_merged[
                (model_merged["id"] == selected_id)
                & (model_merged["forecast_origin"] == selected_origin)
            ]
            .sort_values("horizon")
            .copy()
        )

        model_window["target_time_dt"] = (
            start_date + pd.to_timedelta(model_window["target_time"], unit="h")
        )

        plt.plot(
            model_window["target_time_dt"],
            model_window["p50"].values,
            marker="o",
            linewidth=1.6,
            alpha=0.85,
            label=model_name,
        )

    plt.title(f"Median forecast comparison for representative seed {representative_seed}")
    plt.xlabel("Time")
    plt.ylabel("Power usage (kW)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower left")
    plt.legend()
    plt.tight_layout()

    plot_path = plots_dir / f"representative_seed_{representative_seed}_p50_comparison.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return plot_path

def main(args):
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    predictions_dir = Path(args.predictions_dir)

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
    representative_seed = find_representative_shared_seed(seed_df)
    print(f"Representative shared seed: {representative_seed}")

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

    p50_comparison_plot_path = save_representative_p50_comparison_plot(
        representative_seed=representative_seed,
        predictions_dir=predictions_dir,
        plots_dir=plots_dir,
        data_path=Path(args.data_path),
    )

    if p50_comparison_plot_path is not None:
        print(f"Saved representative p50 comparison plot to: {p50_comparison_plot_path}")

    calibration_targets = {
        "observed_q10": 0.10,
        "observed_q50": 0.50,
        "observed_q90": 0.90,
        "interval_80_coverage": 0.80,
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
    parser.add_argument("--predictions_dir", type=str, default="outputs/predictions")
    parser.add_argument("--data_path", type=str, default="data/electricity_processed.csv")
    args = parser.parse_args()
    main(args)