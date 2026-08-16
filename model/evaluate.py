"""Shared scoring helpers so every model is measured by identical code."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Malignant (label 0) is the positive class: in screening, the disease is what
# you are trying to detect, so recall must mean "cancers caught".
POS_LABEL = 0


def evaluate_model(name, model, X_test, y_test):
    """Score a trained model and return all six metrics as a dict."""
    y_pred = model.predict(X_test)
    # AUC needs a ranking, so use the probability of the positive class.
    y_proba = model.predict_proba(X_test)[:, POS_LABEL]
    y_binary = (np.asarray(y_test) == POS_LABEL).astype(int)

    return {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_binary, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred, pos_label=POS_LABEL), 4),
        "Recall": round(recall_score(y_test, y_pred, pos_label=POS_LABEL), 4),
        "F1": round(f1_score(y_test, y_pred, pos_label=POS_LABEL), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }


def report(metrics):
    """Print one model's scores as a readable block."""
    print(f"\n{metrics['ML Model Name']}")
    print("-" * len(metrics["ML Model Name"]))
    for key, value in metrics.items():
        if key != "ML Model Name":
            print(f"  {key:<10} {value}")
