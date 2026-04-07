from pathlib import Path
import argparse
import json

import pandas as pd


DEFAULT_MODELS = [
    "tft_baseline",
    "mlp_features",
    "no_attention",
    "no_lstm",
    "transformer_only",
]


def load_metrics(metrics_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []

    for model_name in models:
        metrics_path = metrics_dir / f"{model_name}_test_metrics.json"
        if not metrics_path.exists():
            print(f"Skipping {model_name}: metrics file not found at {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            row = json.load(f)

        rows.append(row)

    if not rows:
        raise ValueError("No metrics files found.")

    df = pd.DataFrame(rows)
    return df


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Lower is better
    lower_better = [
        "rmse_p50",
        "mae_p50",
        "q_loss_p10",
        "q_loss_p50",
        "q_loss_p90",
        "mean_interval_width_10_90",
    ]

    # Closer to 0.8 is better for 10-90 interval coverage
    if "interval_coverage_10_90" in out.columns:
        out["coverage_error_10_90"] = (out["interval_coverage_10_90"] - 0.8).abs()
        lower_better.append("coverage_error_10_90")

    for col in lower_better:
        if col in out.columns:
            out[f"rank_{col}"] = out[col].rank(method="min", ascending=True)

    return out


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "model",
        "rmse_p50",
        "mae_p50",
        "q_loss_p10",
        "q_loss_p50",
        "q_loss_p90",
        "interval_coverage_10_90",
        "mean_interval_width_10_90",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    summary = df[cols].copy()

    if "q_loss_p50" in summary.columns:
        summary = summary.sort_values("q_loss_p50", ascending=True)
    elif "rmse_p50" in summary.columns:
        summary = summary.sort_values("rmse_p50", ascending=True)

    return summary.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="predictions",
        help="Directory containing *_test_metrics.json files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions/model_comparison.csv",
        help="Where to save the full comparison table.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="predictions/model_comparison_summary.csv",
        help="Where to save the simpler summary table.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="List of model names to compare.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    comparison_df = load_metrics(metrics_dir, args.models)
    comparison_df = add_rank_columns(comparison_df)

    summary_df = build_summary_table(comparison_df)

    comparison_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nFull comparison table:")
    print(comparison_df)

    print("\nSummary table:")
    print(summary_df)

    print(f"\nSaved full comparison to: {output_csv}")
    print(f"Saved summary comparison to: {summary_csv}")


if __name__ == "__main__":
    main()