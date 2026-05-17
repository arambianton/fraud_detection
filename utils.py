from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
SECONDS_PER_DAY = 24 * 3600

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Data loading and splitting

def load_creditcard(path = "creditcard.csv"):
    """Load the Credit Card Fraud dataset."""
    df = pd.read_csv(path)
    return df.dropna(how="all").reset_index(drop=True)


def stratified_three_way_split(X: pd.DataFrame, y: pd.Series, val_size=0.2, test_size=0.2,random_state=RANDOM_STATE):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=random_state, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def class_imbalance_ratio(y: pd.Series):
    """Return N_neg / N_pos."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        raise ValueError("No pos samples in y.")
    return neg / pos


# Feature engineering

def add_engineered_features(df: pd.DataFrame):
    """Add cyclic hour features, day flag and log-amount; drop raw Time/Amount."""
    out = df.copy()
    seconds_in_day = out["Time"].to_numpy() % SECONDS_PER_DAY
    hour_float = seconds_in_day / 3600
    out["Hour"] = hour_float.astype(int)
    out["HourSin"] = np.sin(2 * np.pi * hour_float / 24)
    out["HourCos"] = np.cos(2 * np.pi * hour_float / 24)
    out["Day"] = (out["Time"].to_numpy() // SECONDS_PER_DAY).astype(int)
    out["LogAmount"] = np.log1p(out["Amount"])
    return out.drop(columns=["Amount", "Time"])


def feature_target_split(df: pd.DataFrame, target: str = "Class"):
    """Split a frame into (X, y)."""
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


# Model creators

def make_logreg(random_state=RANDOM_STATE):
    """LogReg with `class_weight='balanced'` and z-scaling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(class_weight=None, random_state=RANDOM_STATE):
    """Random forest baseline."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight=class_weight or "balanced_subsample",
        random_state=random_state,
    )


# Optuna hyperparameter tuning

def _cv_score(model_factory: Callable, X, y, n_splits=3, random_state=RANDOM_STATE):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    scores = []
    for tr_idx, va_idx in cv.split(X_arr, y_arr):
        model = model_factory()
        model.fit(X_arr[tr_idx], y_arr[tr_idx])
        proba = model.predict_proba(X_arr[va_idx])[:, 1]
        scores.append(average_precision_score(y_arr[va_idx], proba))
    return float(np.mean(scores))


def tune_xgboost(X, y, scale_pos_weight, n_trials=25, random_state=RANDOM_STATE):
    """3-fold CV PR-AUC search for XGBoost. Returns (best_params, best_score, study)."""

    def objective(trial: optuna.Trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 400),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 15.0),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        )

        def factory():
            return XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )

        return _cv_score(factory, X, y, n_splits=3, random_state=random_state)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def tune_catboost(X, y, scale_pos_weight, n_trials=25, random_state=RANDOM_STATE):
    """3-fold CV PR-AUC search for CatBoost. Returns (best_params, best_score, study)."""

    def objective(trial: optuna.Trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 150, 400),
            depth=trial.suggest_int("depth", 4, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_int("border_count", 32, 128),
        )

        def factory():
            return CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="PRAUC",
                scale_pos_weight=scale_pos_weight,
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            )

        return _cv_score(factory, X, y, n_splits=3, random_state=random_state)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


# Metrics, threshold selection, business cost

@dataclass
class ModelReport:
    name: str
    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def best_f1_threshold(y_true, y_proba):
    """Threshold that maximises F1 on (y_true, y_proba)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
        f1 = np.nan_to_num(f1, nan=0.0)
    if len(thresholds) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def evaluate(name, y_true, y_proba, threshold: Optional[float] = None):
    """Full evaluation at a chosen threshold (default 0.5)."""
    if threshold is None:
        threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ModelReport(
        name=name,
        pr_auc=float(average_precision_score(y_true, y_proba)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def report_table(reports):
    df = pd.DataFrame([r.to_dict() for r in reports])
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


# Plotting helpers

def plot_pr_curves(probas: Dict[str, np.ndarray], y_true: np.ndarray, ax=None):
    """PR curves creator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probas.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return ax


def plot_roc_curves(probas: Dict[str, np.ndarray], y_true: np.ndarray, ax=None):
    """ROC curves creator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, ax=None):
    """Confusion-matrix heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, ax=None):
    """Correlation heatmap of all numeric columns."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Feature correlation matrix")
    return ax


def plot_top_kde(df: pd.DataFrame, feature_cols: Iterable[str], target="Class", n_cols=3, figsize_per_row=3.0):
    """KDE for fraud vs non-fraud across the given features."""
    cols = list(feature_cols)
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, figsize_per_row * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        sns.kdeplot(data=df[df[target] == 0], x=col, ax=ax, label="Non-Fraud", common_norm=False)
        sns.kdeplot(data=df[df[target] == 1], x=col, ax=ax, label="Fraud", common_norm=False)
        ax.set_title(col)
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
