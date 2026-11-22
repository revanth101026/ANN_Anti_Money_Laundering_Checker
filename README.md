# Intelligent Anti-Money Laundering Detection System Using Artificial Neural Networks

An artificial neural network (ANN) that flags potentially suspicious financial transactions, with a Streamlit web app where you enter transaction details and get an instant result.

## Overview

Money laundering is rare in real transaction data, so a simple accuracy-based model would just predict "normal" every time. This project handles that imbalance and evaluates the model with metrics that suit fraud detection: PR-AUC, precision and recall.

## Dataset

[Synthetic Transaction Monitoring Dataset](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset) from Kaggle.

- About 9.5 million transactions
- Only 9,873 are laundering cases (about 0.1%)
- Features used: amount, payment and received currency, sender and receiver bank location, payment type

## Methodology

1. **Data preparation:** dropped identifier, date and label-leaking columns (such as `Laundering_type`). Kept all laundering rows and sampled 200,000 normal rows.
2. **Train / validation / test split first:** stratified, so validation and test sets keep the real class ratio and no information leaks.
3. **Preprocessing:** one-hot encoding for categorical columns and standard scaling for numeric columns, fitted on training data only.
4. **Class imbalance:** SMOTE applied to the training set only.
5. **Hyperparameter tuning:** Optuna (30 trials) maximizing validation PR-AUC.
6. **Model:** feed-forward ANN with dense layers, dropout and a sigmoid output.
7. **Threshold selection:** chosen on the validation set to maximize F1, then evaluated once on the test set.

## Results

Test set (41,975 transactions, 1,975 laundering cases):

| Metric | Value |
|---|---|
| PR-AUC | 0.276 |
| ROC-AUC | 0.797 |
| Precision (laundering) | 0.290 |
| Recall (laundering) | 0.341 |

PR-AUC is about 6 times better than random chance (0.047 on this test set).

**Note:** the test set contains about 4.7% laundering cases, while real data has about 0.1%. Precision on live traffic would be lower than shown here.

## Limitations

- Features are limited to single-transaction details. Account-level behaviour (transaction counts, fan-in and fan-out patterns) would likely improve results.
- The model output is a prediction, not proof of money laundering. Flagged transactions need human review.

## Project Structure

```
├── Intelligent_Anti-Money_Laundering_Detection_System_Using_Artificial_Neural_Networks.ipynb
├── app.py               # Streamlit web app
├── preprocess.pkl       # fitted preprocessing pipeline
├── aml_model.keras      # trained ANN
├── threshold.json       # decision threshold
├── requirements.txt
└── README.md
```

## Installation

pip install -r requirements.txt
```

## Usage

1. Run the notebook to train the model. It saves `preprocess.pkl`, `aml_model.keras` and `threshold.json`.
2. Start the web app:

```bash
streamlit run app.py
```

3. Open `http://localhost:8501`, enter the transaction details and press **Check transaction**.

## Tech Stack

Python, TensorFlow / Keras, scikit-learn, imbalanced-learn, Optuna, Pandas, NumPy, Streamlit

## Future Work

- Account-level behavioural features
- Comparison with gradient boosting models (LightGBM, XGBoost)
- Graph-based detection of transaction networks
- Model explainability (SHAP)

