"""Shared load/split/scale pipeline so all five models train on identical data."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent


def prepare_data(save_test_csv=True):
    """Load, split and scale the dataset; return everything downstream needs."""
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_columns = list(X.columns)
    class_names = list(dataset.target_names)

    # Stratify to keep the same class balance in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit on train only; fitting on everything would leak test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_test_csv:
        # Unscaled, so the app applies the scaler itself.
        test_df = X_test.copy()
        test_df[TARGET_COL] = y_test.values
        test_df.to_csv(REPO_ROOT / "test_data.csv", index=False)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "class_names": class_names,
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"Train: {data['X_train'].shape}  Test: {data['X_test'].shape}")
    print(f"Features: {len(data['feature_columns'])}")
    print(f"Classes: {data['class_names']}")
    print(f"test_data.csv written to {REPO_ROOT}")
