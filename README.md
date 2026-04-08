# Temporal Fusion Transformer (TFT) for Electricity Forecasting

## Project Overview
This project implements and evaluates the Temporal Fusion Transformer (TFT) architecture for multi-horizon electricity demand forecasting using the UCI Electricity dataset.

The main goals of this project are:
- Reproduce a PyTorch-based implementation of TFT
- Analyze key architectural components
- Compare multiple model variants through ablation studies

The models predict 24-hour ahead electricity consumption using a 168-hour historical window, producing probabilistic forecasts (quantiles: 0.1, 0.5, 0.9).

---

## Objectives
- Implement TFT in PyTorch
- Reproduce the original paper setup
- Evaluate the importance of:
  - LSTM layers
  - Attention mechanisms
  - Variable Selection Networks (VSN)
- Compare alternative model architectures

---

## Implemented Models
- baseline – Full TFT
- no_lstm – Removes LSTM component
- no_attention – Removes attention mechanism
- mlp_features – Replaces VSN with MLP
- transformer_only – Transformer-only architecture

---

## Project Structure
```
tft_electricity/
├── data/
├── models/
│   ├── baseline.py
│   ├── mlp_features.py
│   ├── no_attention.py
│   ├── no_lstm.py
│   ├── transformer_only.py
├── notebooks/
│   └── colab_runner.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── compare_models.py
│   ├── create_dataset.py
│   ├── data_formatter.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── layers.py
│   ├── predict.py
│   ├── registry.py
│   └── train.py
├── .gitignore
├── README.md
├── requirements.txt
```

---

## Setup

### Option 1: Google Colab (Recommended)
1. Open `notebooks/colab_runner.ipynb`
2. Enable GPU runtime
3. Run all cells

Outputs can optionally be saved to Google Drive.

---

### Option 2: Local Setup

Clone the repository:
```bash
git clone https://github.com/HannaVallner/tft_electricity.git
cd tft_electricity
```

Create and activate environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Important Note on Running Scripts

All executable scripts are located inside the `src/` directory.

To ensure imports (e.g. `models.*`) work correctly, run all scripts with:

```bash
PYTHONPATH=. python src/<script>.py
```

---

## Usage

### Generate Dataset
```bash
PYTHONPATH=. python src/create_dataset.py
```

### Train Models
```bash
PYTHONPATH=. python src/train.py --model baseline
PYTHONPATH=. python src/train.py --model no_lstm
PYTHONPATH=. python src/train.py --model no_attention
PYTHONPATH=. python src/train.py --model mlp_features
PYTHONPATH=. python src/train.py --model transformer_only
```

### Generate Predictions
```bash
PYTHONPATH=. python src/predict.py \
    --model baseline \
    --checkpoint_path outputs/checkpoints/baseline_best.pt \
    --output_path outputs/predictions/baseline.csv
```

### Evaluate Models
```bash
PYTHONPATH=. python src/evaluate.py \
    --model baseline \
    --checkpoint_path outputs/checkpoints/baseline_best.pt \
    --predictions_path outputs/predictions/baseline.csv \
    --metrics_path outputs/metrics/baseline_metrics.json
```

### Compare Models
```bash
PYTHONPATH=. python src/compare_models.py \
    --metrics_dir outputs/metrics \
    --plots_dir outputs/plots
```

---

## Evaluation
- Quantile Loss
- Normalized Quantile Loss (NQL)
- Calibration metrics
- Prediction interval coverage

Outputs:
- outputs/metrics/
- outputs/plots/
- outputs/predictions/

---

## Prediction Output Format

Each prediction row contains:

| Column | Description |
|--------|------------|
| id | Time series identifier |
| forecast_origin | Last observed timestep |
| target_time | Future timestep being predicted |
| horizon | Steps ahead (1–24) |
| p10, p50, p90 | Predicted quantiles |

---

## Forecast Setup
- Input window: 168 hours
- Forecast horizon: 24 hours
- Quantiles: 0.1, 0.5, 0.9

---

## References
- Temporal Fusion Transformer paper (Lim et al., 2021)
- Google Research implementation

---

## Possible Future Improvements
- Hyperparameter tuning
- Additional datasets
- Feature engineering improvements
- Model optimization

---

## Author
Hanna Vallner