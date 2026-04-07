from pathlib import Path
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# Static output path
OUTPUT_CSV_PATH = Path("data/electricity_processed.csv")


def get_electricity_data():
    """
    Download and preprocess the UCI Electricity dataset.
    """

    # Make sure the output folder exists
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Find data from source
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    print("Downloading electricity dataset...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Read raw text file into pandas dataframe
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        raw_filename = "LD2011_2014.txt"

        if raw_filename not in zf.namelist():
            raise FileNotFoundError(f"{raw_filename} was not found inside the downloaded zip.")

        with zf.open(raw_filename) as raw_file:
            print("Reading raw text file...")
            df = pd.read_csv(raw_file, sep=";", decimal=",", index_col=0)

    # Convert index to pandas datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Aggregate to hourly resolution
    print("Resampling to hourly data...")
    hourly_df = df.resample("1h").mean()

    # Treat zeros as missing values
    hourly_df = hourly_df.replace(0.0, np.nan)

    earliest_time = hourly_df.index.min()

    print("Processing each electricity meter...")
    per_client_frames = []

    for meter_id in hourly_df.columns:
        print(f"  Processing {meter_id}")

        series = hourly_df[meter_id].copy()

        # Find the first timestamp where the series becomes active
        forward_filled = series.ffill().dropna()

        # Find the last timestamp where the series is still active
        backward_filled = series.bfill().dropna()

        # Skip completely empty series
        if forward_filled.empty or backward_filled.empty:
            continue

        start_date = forward_filled.index.min()
        end_date = backward_filled.index.max()

        # Keep only the active period
        active_mask = (series.index >= start_date) & (series.index <= end_date)
        active_series = series.loc[active_mask]

        # Fill missing values inside the active period with zeros
        active_series = active_series.fillna(0.0)

        tmp = pd.DataFrame({"power_usage": active_series})
        date = tmp.index

        # Create relative time indices
        hours_from_start = ((date - earliest_time).total_seconds() // 3600).astype(int)
        tmp["hours_from_start"] = hours_from_start
        tmp["days_from_start"] = (date - earliest_time).days.astype(int)

        # Add features
        tmp["id"] = meter_id
        tmp["categorical_id"] = meter_id
        tmp["date"] = date
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month
        tmp["categorical_hour"] = tmp["hour"]
        tmp["categorical_day_of_week"] = tmp["day_of_week"]

        per_client_frames.append(tmp)

    # Safety check in case no series were processed
    if not per_client_frames:
        raise ValueError("No electricity meter series were processed.")

    # Concatenate all client dataframes together
    output = pd.concat(per_client_frames, axis=0).reset_index(drop=True)

    # Match benchmark range used in the original TFT code
    output = output[
        (output["days_from_start"] >= 1096) &
        (output["days_from_start"] < 1346)
    ].copy()

    # Sort for clean downstream processing
    output = output.sort_values(["id", "hours_from_start"]).reset_index(drop=True)

    # Save processed data
    print(f"Saving processed data to: {OUTPUT_CSV_PATH}")
    output.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")
    return output


if __name__ == "__main__":
    df = get_electricity_data()
    print(df.head())
    print(df.shape)