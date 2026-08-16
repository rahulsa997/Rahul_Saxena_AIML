"""Trains all five models on one shared split; prints the table and writes test_data.csv."""

import sys
from pathlib import Path

import pandas as pd

# Let the sibling imports resolve whether run from the repo root or from model/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import decision_tree
import knn
import logistic_regression
import naive_bayes
import random_forest
from data_prep import prepare_data

# Order matters: it sets the row order in the printed table and the README.
MODEL_MODULES = [
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
]


def main():
    print("Preparing data (shared split for all five models)...")
    data = prepare_data()

    print(f"  Train: {data['X_train'].shape}   Test: {data['X_test'].shape}")
    print(f"  Features: {len(data['feature_columns'])}")
    print(f"  Classes: {data['class_names']}")

    results = []
    for module in MODEL_MODULES:
        print(f"\nTraining {module.NAME}...")
        results.append(module.train(data))

    metrics_df = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print("Comparison table -- copy these into the README:")
    print("=" * 78)
    print(metrics_df.to_string(index=False))
    print("\ntest_data.csv regenerated. No model files are written -- app.py")
    print("trains these same five models at startup.")


if __name__ == "__main__":
    main()
