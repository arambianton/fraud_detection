# Credit Card Fraud Detection

Binary classification on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284 807 transactions, 0.17 % fraud.

## What's inside

- **EDA** — class balance, hour-of-day patterns, correlation heatmap, KDEs of the top features.
- **Feature engineering** — cyclic hour (sin/cos), day flag, `log(1 + Amount)`.
- **Split** — stratified train / val / test (60 / 20 / 20). Val is used for early stopping and threshold tuning; test is touched once.
- **Models** — Logistic Regression, Random Forest, XGBoost, CatBoost.
- **Tuning** — Optuna (TPE, 25 trials each) on 3-fold CV PR-AUC.
- **Imbalance** — `scale_pos_weight` and SMOTE compared.
- **Metrics** — PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix; threshold picked by F1 on val.
- **Interpretability** — TreeSHAP on XGBoost.

## Run it

```bash
pip install -r requirements.txt
# put creditcard.csv (from Kaggle) next to main.ipynb
jupyter notebook main.ipynb
```

End-to-end: ~20-30 min