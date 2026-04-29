# NVIDIA Stock Direction Predictor

A supervised learning project that predicts whether NVIDIA's stock price will close **higher or lower the following day**, using historical price and volume data.

---

## Overview

Predicting stock price direction is a classic binary classification problem. This project builds and evaluates three machine learning models — Logistic Regression, Random Forest, and Gradient Boosting — then combines them into a Voting Ensemble. The final model is tested against a simple Buy & Hold benchmark via a strategy backtest.

---

## Results

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | 53.75% | 0.49 |
| Random Forest | 51.84% | 0.52 |
| Gradient Boosting | 53.24% | 0.49 |
| **Ensemble** | **53.90%** | **0.49** |

All models perform close to chance (~0.50 AUC), which is consistent with the Efficient Market Hypothesis — publicly available price data alone is unlikely to yield a reliable edge. The confusion matrix reveals the models default to predicting "Up" nearly every day, failing to identify down days. This is discussed in depth in the notebook.

---

## Methodology

- **Target:** Binary — 1 if next day's close > today's close, 0 otherwise
- **Features:** Daily return, volume change, volatility, 10/50-day moving averages, lagged returns, momentum, MA spread, rolling return mean/std
- **Validation:** Time Series Split (no data leakage)
- **Tuning:** GridSearchCV over key hyperparameters per model
- **Ensemble:** Soft-voting VotingClassifier across all three tuned models
- **Backtest:** Cumulative ML strategy return vs. Buy & Hold over the test period

---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- yfinance
- Matplotlib, seaborn

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Open the notebook**
```bash
jupyter notebook NVIDIA_Prediction.ipynb
```

**3. Run all cells**

`Kernel → Restart & Run All`

Data is fetched automatically from Yahoo Finance — no CSV file needed.

---

## Project Structure

```
├── NVIDIA_Prediction.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

---

## Key Takeaways

- Technical indicators alone are insufficient for reliably predicting short-term price direction
- Proper time-series validation is essential to avoid look-ahead bias
- Ensemble methods improve stability but not necessarily accuracy when base models share the same signal limitations
- Honest evaluation (AUC, confusion matrix, backtest) matters more than headline accuracy
