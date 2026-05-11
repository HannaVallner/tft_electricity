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


def save_calibration_table(summary_df, output_path):
    """
    Save a LaTeX table with observed quantile calibration and
    80% interval coverage values as mean ± std across seeds.
    """

    table_df = summary_df[
        [
            "model",
            "observed_q10_mean",
            "observed_q10_std",
            "observed_q50_mean",
            "observed_q50_std",
            "observed_q90_mean",
            "observed_q90_std",
            "interval_80_coverage_mean",
            "interval_80_coverage_std",
        ]
    ].copy()

    rows = []

    for _, row in table_df.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Obs. Q$_{0.1}$": format_mean_std(
                    row["observed_q10_mean"],
                    row["observed_q10_std"],
                ),
                "Obs. Q$_{0.5}$": format_mean_std(
                    row["observed_q50_mean"],
                    row["observed_q50_std"],
                ),
                "Obs. Q$_{0.9}$": format_mean_std(
                    row["observed_q90_mean"],
                    row["observed_q90_std"],
                ),
                "Cov$_{80}$": format_mean_std(
                    row["interval_80_coverage_mean"],
                    row["interval_80_coverage_std"],
                ),
            }
        )

    latex_df = pd.DataFrame(rows)

    latex_table = latex_df.to_latex(
        index=False,
        caption="Quantile calibration and interval coverage metrics for TFT model variants across random seeds.",
        label="tab:calibration_summary",
        escape=False,
    )

    latex_table = latex_table.replace(
        "\\begin{table}",
        "\\begin{table}[H]"
    )

    latex_table = latex_table.replace(
        "\\begin{tabular}{lllll}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccc}"
    )

    latex_table = latex_table.replace(
        "\\end{tabular}",
        "\\end{tabular}%\n}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

def save_latex_table(summary_df, output_path):
    """
    Save a LaTeX table with forecasting performance metrics
    as mean ± std across seeds.
    """

    rows = []

    best_model = summary_df.loc[
        summary_df["test_nql_mean_mean"].idxmin(),
        "model"
    ]

    for _, row in summary_df.iterrows():
        formatted_row = {
            "Model": row["model"],
            "Mean NQL": format_mean_std(
                row["test_nql_mean_mean"],
                row["test_nql_mean_std"],
            ),
            "NQL$_{0.1}$": format_mean_std(
                row["test_nql_p10_mean"],
                row["test_nql_p10_std"],
            ),
            "NQL$_{0.5}$": format_mean_std(
                row["test_nql_p50_mean"],
                row["test_nql_p50_std"],
            ),
            "NQL$_{0.9}$": format_mean_std(
                row["test_nql_p90_mean"],
                row["test_nql_p90_std"],
            ),
        }

        rows.append(formatted_row)

    latex_df = pd.DataFrame(rows).astype(str)

    best_row_idx = latex_df.index[latex_df["Model"] == best_model][0]

    for col in latex_df.columns:
        latex_df.loc[best_row_idx, col] = (
            f"\\textbf{{{latex_df.loc[best_row_idx, col]}}}"
        )

    latex_table = latex_df.to_latex(
        index=False,
        caption="Forecasting performance of TFT model variants across random seeds.",
        label="tab:model_comparison",
        escape=False,
    )

    latex_table = latex_table.replace(
        "\\begin{table}",
        "\\begin{table}[H]"
    )

    latex_table = latex_table.replace(
        "\\begin{tabular}{lllll}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccc}"
    )

    latex_table = latex_table.replace(
        "\\end{tabular}",
        "\\end{tabular}%\n}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

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


def find_representative_shared_seed(seed_df):
    """
    Select one shared seed whose average Mean NQL across models is closest
    to the overall average across seeds.
    """
    seed_means = seed_df.groupby("seed")["test_nql_mean"].mean()
    overall_mean = seed_means.mean()
    representative_seed = (seed_means - overall_mean).abs().idxmin()
    return int(representative_seed)

def save_diagnostic_p50_comparison_plot(
    representative_seed,
    predictions_dir,
    plots_dir,
    data_path,
):
    """
    Save a p50 comparison plot for a diagnostic forecast window.

    The seed is selected as the shared seed whose average mean NQL across models
    is closest to the overall average across seeds. Within that seed, the plotted
    window is selected to show meaningful target variation and visible differences
    between model variants.
    """

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
            print(f"Skipping diagnostic p50 comparison plot: missing {prediction_path}")
            return None

        prediction_frames[model_name] = pd.read_csv(
            prediction_path,
            on_bad_lines="skip",
        )

    merge_keys = ["id", "forecast_origin", "target_time", "horizon"]

    for col in merge_keys:
        targets_df[col] = targets_df[col].astype(int)

    for model_name in models:
        for col in merge_keys:
            prediction_frames[model_name][col] = prediction_frames[model_name][col].astype(int)

    merged_frames = {}
    window_error_frames = []

    for model_name in models:
        model_merged = prediction_frames[model_name].merge(
            targets_df,
            on=merge_keys,
            how="inner",
        )

        if model_merged.empty:
            print(f"Skipping diagnostic p50 comparison plot: merged dataframe is empty for {model_name}")
            return None

        model_merged["abs_p50_error"] = (
            model_merged["target"] - model_merged["p50"]
        ).abs()

        model_window_errors = (
            model_merged
            .groupby(["id", "forecast_origin"])["abs_p50_error"]
            .mean()
            .reset_index(name=f"{model_name}_mae")
        )

        merged_frames[model_name] = model_merged
        window_error_frames.append(model_window_errors)

    combined_window_errors = window_error_frames[0]

    for error_df in window_error_frames[1:]:
        combined_window_errors = combined_window_errors.merge(
            error_df,
            on=["id", "forecast_origin"],
            how="inner",
        )

    mae_columns = [
        col for col in combined_window_errors.columns
        if col.endswith("_mae")
    ]

    combined_window_errors["mean_window_mae"] = (
        combined_window_errors[mae_columns].mean(axis=1)
    )

    combined_window_errors["robust_model_spread"] = (
        combined_window_errors[mae_columns].quantile(0.75, axis=1)
        - combined_window_errors[mae_columns].quantile(0.25, axis=1)
    )

    combined_window_errors["max_model_mae"] = (
        combined_window_errors[mae_columns].max(axis=1)
    )

    base_merged = merged_frames[models[0]]

    target_stats = (
        base_merged
        .groupby(["id", "forecast_origin"])
        .agg(
            target_mean=("target", "mean"),
            target_max=("target", "max"),
            target_min=("target", "min"),
        )
        .reset_index()
    )

    target_stats["target_range"] = (
        target_stats["target_max"] - target_stats["target_min"]
    )

    combined_window_errors = combined_window_errors.merge(
        target_stats,
        on=["id", "forecast_origin"],
        how="inner",
    )

    candidates = combined_window_errors[
        (combined_window_errors["target_range"] >= combined_window_errors["target_range"].quantile(0.70))
        & (combined_window_errors["target_mean"] >= combined_window_errors["target_mean"].quantile(0.40))
        & (combined_window_errors["max_model_mae"] <= combined_window_errors["max_model_mae"].quantile(0.90))
        & (combined_window_errors["robust_model_spread"] >= combined_window_errors["robust_model_spread"].quantile(0.85))
    ].copy()

    if candidates.empty:
        print("No diagnostic candidates found; falling back to high robust-spread window.")
        candidates = combined_window_errors[
            combined_window_errors["robust_model_spread"]
            >= combined_window_errors["robust_model_spread"].quantile(0.85)
        ].copy()

    selected_window = (
        candidates
        .sort_values(["mean_window_mae", "robust_model_spread"], ascending=[True, False])
        .iloc[0]
    )

    selected_id = selected_window["id"]
    selected_origin = selected_window["forecast_origin"]
    start_date = pd.Timestamp("2011-01-01 00:00:00")

    base_window = (
        base_merged[
            (base_merged["id"] == selected_id)
            & (base_merged["forecast_origin"] == selected_origin)
        ]
        .sort_values("horizon")
        .copy()
    )

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
        model_window = (
            merged_frames[model_name][
                (merged_frames[model_name]["id"] == selected_id)
                & (merged_frames[model_name]["forecast_origin"] == selected_origin)
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

    plt.title(f"Median forecast comparison for diagnostic window, seed {representative_seed}")
    plt.xlabel("Time")
    plt.ylabel("Power usage (kW)")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best")
    plt.tight_layout()

    plot_path = plots_dir / f"diagnostic_seed_{representative_seed}_p50_comparison.pdf"
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

    summary_csv_path = metrics_dir / "model_comparison_summary_mean_std.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary comparison table to: {summary_csv_path}")

    comparison_tex_path = metrics_dir / "model_comparison_summary_mean_std.tex"
    save_latex_table(summary_df, comparison_tex_path)
    print(f"Saved LaTeX table to: {comparison_tex_path}")

    calibration_tex_path = metrics_dir / "calibration_summary.tex"
    save_calibration_table(summary_df, calibration_tex_path)
    print(f"Saved calibration table to: {calibration_tex_path}")


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

    reliability_plot_path = save_combined_reliability_plot(summary_df, plots_dir)
    print(f"Saved combined reliability plot to: {reliability_plot_path}")

    p50_comparison_plot_path = save_diagnostic_p50_comparison_plot(
        representative_seed=representative_seed,
        predictions_dir=predictions_dir,
        plots_dir=plots_dir,
        data_path=Path(args.data_path),
    )

    if p50_comparison_plot_path is not None:
        print(f"Saved diagnostic p50 comparison plot to: {p50_comparison_plot_path}")
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