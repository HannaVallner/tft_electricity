# Temporal Fusion Transformer (TFT) for Electricity Forecasting

## Project Overview
This project implements and evaluates Temporal Fusion Transformer (TFT) model variants for probabilistic household electricity consumption forecasting using the UCI Electricity dataset.

The project focuses on multi-horizon forecasting and architectural ablation analysis. Each model predicts 24 hours ahead using a 168-hour historical input window and produces probabilistic forecasts at the 0.1, 0.5, and 0.9 quantiles.

The main goals of this project are:
- implement TFT-based forecasting models in PyTorch;
- compare the full TFT architecture with several ablated variants;
- evaluate predictive accuracy using normalized quantile loss (NQL);
- evaluate probabilistic calibration using empirical quantile coverage and 80% interval coverage;
- compare model stability across multiple random seeds.

---

## Objectives

The project investigates the contribution of key TFT architectural components, including:
- recurrent LSTM layers;
- interpretable self-attention;
- Variable Selection Networks (VSNs);
- simplified MLP-based feature fusion;
- attention-centric transformer-style forecasting without the full TFT structure.

The experiments are designed to assess both forecasting accuracy and uncertainty estimation quality.

---

## Implemented Models
The following model variants are implemented:

| Model | Description |
|------|-------------|
| `baseline` | Full TFT model with LSTMs, attention, static context, and VSNs |
| `no_lstm` | Removes the LSTM encoder/decoder while keeping attention and VSNs |
| `no_attention` | Removes the attention mechanism while keeping recurrent modelling |
| `mlp_features` | Replaces Variable Selection Networks with simpler MLP-based feature processing |
| `transformer_only` | Removes recurrence, VSNs, and static context, keeping an attention-centric architecture |

---

## Forecast Setup

- Dataset: UCI Electricity
- Input window: 168 hours
- Forecast horizon: 24 hours
- Forecast type: probabilistic
- Quantiles: 0.1, 0.5, 0.9
- Evaluation: multiple random seeds per model

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

### Option 1: Google Colab 

The recommended way to run the full experiment pipeline is through the notebook:
```bash
notebooks/colab_runner.ipynb
```
Steps:
1. Open the notebook in Google Colab.
2. Enable GPU runtime.
3. Run all the notebook cells in order.

Outputs can optionally be saved to Google Drive. The notebook is used as an execution layer, while the main implementation logic remains in reusable Python scripts inside `src/` and `models/`.

---

### Option 2: Local Setup

Clone the repository:
```bash
git clone https://github.com/HannaVallner/tft_electricity.git
cd tft_electricity
```

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```
For Linux/macOS:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Generate Dataset
```bash
PYTHONPATH=. python src/create_dataset.py
```
This creates the processed electricity dataset used for training, validation, and testing.

### Train a Model

Example for the baseline model:
```bash
PYTHONPATH=. python src/train.py \
    --model baseline \
    --seed 0 \
    --save_dir outputs/checkpoints \
    --metrics_dir outputs/metrics
```
For Windows PowerShell:
```bash
$env:PYTHONPATH="."
python src/train.py `
    --model baseline `
    --seed 0 `
    --save_dir outputs/checkpoints `
    --metrics_dir outputs/metrics
```
For random-seed experiments, checkpoints are saved with seed-specific filenames, for example:
```bash
outputs/checkpoints/baseline_seed_0_best.pt
outputs/checkpoints/baseline_seed_1_best.pt
```

### Generate Predictions
```bash
PYTHONPATH=. python src/predict.py \
    --model baseline \
    --checkpoint_path outputs/checkpoints/baseline_seed_0_best.pt \
    --output_path outputs/predictions/baseline_seed_0_predictions.csv
```
Prediction files are saved in long format, with one row per forecast horizon step.

### Evaluate a Model
```bash
PYTHONPATH=. python src/evaluate.py \
    --model baseline \
    --checkpoint_path outputs/checkpoints/baseline_seed_0_best.pt \
    --predictions_path outputs/predictions/baseline_seed_0_predictions.csv \
    --metrics_path outputs/metrics/baseline_seed_0_metrics.json \
    --plots_dir outputs/plots \
    --seed 0
```
The evaluation script computes validation loss, normalized quantile loss, calibration metrics, and diagnostic plots.

### Compare Models Across Seeds
```bash
PYTHONPATH=. python src/compare_models.py \
    --metrics_dir outputs/metrics \
    --plots_dir outputs/plots \
    --predictions_dir outputs/predictions
```
This aggregates seed-level metrics across model variants and generates comparison tables and plots.

---

## References
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.
- Google Research Temporal Fusion Transformer implementation.
- UCI Electricity Load Diagrams dataset.

---

## Possible Future Improvements
- independent hyperparameter tuning for each model variant;
- additional datasets;
- additional contextual or calendar features;
- horizon-specific metric analysis;
- computational efficiency comparison;
- alternative probabilistic forecasting losses.

---

## Author
Hanna Vallner