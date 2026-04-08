# Temporal Fusion Transformer (TFT) for Electricity Forecasting

## Project Overview
This project implements and evaluates the Temporal Fusion Transformer (TFT) architecture for multi-horizon electricity demand forecasting using the UCI Electricity dataset.

The goal of this project is as follows:
- Reproduce a PyTorch-based version of TFT
- Analyze key architectural components
- Compare multiple model variants through ablation studies

The models predict 24-hour ahead electricity consumption using a 168-hour history window and output probabilistic forecasts (quantiles 0.1, 0.5, 0.9).

---

## Objectives
- Implement TFT in PyTorch
- Reproduce original paper setup
- Evaluate importance of:
  - LSTM
  - Attention
  - Variable Selection Networks
- Compare alternative architectures

---


## Implemented models
- baseline - Full TFT
- no_lstm - Removes LSTM
- no_attention - Removes attention
- mlp_features -  Replaces VSN with MLP
- transformer_only - Transformer-only model

---

## Project Structure
```
tft_electricity/
├── data/ *
├── models/
│   ├── baseline.py
│   ├── mlp_features.py
│   ├── no_attention.py
│   ├── no_lstm.py
│   ├── transformer_only.py
├── notebooks/
│   ├── colab_runner.ipynb
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
\* Generated after running the following:
```
python create_dataset.py
```
---

## Setup

### Option 1: Google Colab
1. Upload notebook
2. Enable GPU
3. Run all cells

Outputs saved to Google Drive.

---

### Option 2: Local setup

Clone repo:
```
git clone https://github.com/HannaVallner/tft_electricity.git
cd tft_electricity
```

Create environment:
```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## Usage

### Generate the dataset
```
python create_dataset.py
```

### Train models
```
python train.py --model baseline
python train.py --model no_lstm
python train.py --model no_attention
python train.py --model mlp_featurer
python train.py --model transformer_only
```

### Predict
```
python predict.py --model baseline --checkpoint_path outputs/checkpoints/baseline_best.pt --output_path outputs/predictions/baseline.csv

python predict.py --model mlp_features --checkpoint_path outputs/checkpoints/mlp_features_best.pt --output_path outputs/predictions/mlp_features.csv

python predict.py --model no_attention --checkpoint_path outputs/checkpoints/no_attention_best.pt --output_path outputs/predictions/no_attention.csv

python predict.py --model no_lstm --checkpoint_path outputs/checkpoints/no_lstm_best.pt --output_path outputs/predictions/no_lstm.csv

python predict.py --model transformer_only --checkpoint_path outputs/checkpoints/transformer_only_best.pt --output_path outputs/predictions/transformer_only.csv
```

### Evaluate
```
python evaluate.py --model baseline --checkpoint_path outputs/checkpoints/baseline_best.pt

python evaluate.py --model mlp_features --checkpoint_path outputs/checkpoints/mlp_features'
_best.pt

python evaluate.py --model no_attention --checkpoint_path outputs/checkpoints/no_attention_best.pt

python evaluate.py --model no_lstm --checkpoint_path outputs/checkpoints/no_lstm_best.pt

python evaluate.py --model transformer_only --checkpoint_path outputs/checkpoints/transformer_only_best.pt
```

### Compare models
```
python compare_models.py
```

---

## Evaluation
- Quantile Loss
- Normalized Quantile Loss

Outputs:
- metrics/
- plots/
- predictions/

---

## Forecast Setup
- Input: 168 hours
- Output: 24 hours
- Quantiles: 0.1, 0.5, 0.9

---

## References
- Temporal Fusion Transformer paper
- Google Research implementation

---

## Possible future developments
- Hyperparameter tuning
- More datasets
- Improved features