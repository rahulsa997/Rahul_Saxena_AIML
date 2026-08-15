# Breast Cancer Classification — Model Comparison & Streamlit App

**Author:** Rahul Saxena
**BITS ID:** 2025ac05155
**Email:** 2025ac05155@wilp.bits-pilani.ac.in
**Course:** M.Tech (AIML/DSE) — Machine Learning
**Assignment:** ML Assignment 2 — Work Integrated Learning Programmes Division, BITS Pilani

---

## a. Problem Statement

Breast cancer diagnosis relies on correctly distinguishing malignant tumors
from benign ones based on measurements taken from digitized images of a
breast mass. This project builds and compares five classification models
that predict whether a tumor is **malignant** or **benign** from a set of
cell-nuclei measurements, and exposes the trained models through an
interactive Streamlit app so predictions and evaluation metrics can be
inspected live on new test data.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also bundled with
  scikit-learn as `sklearn.datasets.load_breast_cancer` for reproducibility)
  — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — computed from
  digitized images of fine needle aspirate (FNA) of breast masses, describing
  characteristics of cell nuclei such as radius, texture, perimeter, area,
  smoothness, compactness, concavity, symmetry, and fractal dimension (each
  reported as mean, standard error, and "worst"/largest value).
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign (~37% / 63%) — moderately
  imbalanced but not extreme.
- **Train/test split:** 80% train (455 rows) / 20% test (114 rows),
  stratified on the target, `random_state=42`.
- **`test_data.csv`** (in this repo) is the held-out 20% test split, used
  both for offline evaluation and as the sample file to upload into the
  Streamlit app.

## c. GitHub Repository Link

> **TODO:** Replace with your repository URL after pushing, e.g.
> `https://github.com/<your-username>/<your-repo-name>`

Repository contains: `app.py`, `requirements.txt`, `README.md`,
`test_data.csv`, and `model/` (training script, saved model files, scaler,
and metrics).

## d. Models Used

All 5 models are **implemented from scratch in NumPy** (see *Implementation*
below) and were trained on the **same dataset** and the **same train/test
split**, with features standardized using `StandardScaler` (fit on the training
set only).

### Comparison Table

**Malignant is the positive class.** In screening, the disease is what you are
trying to detect, so Precision, Recall and F1 all describe how well a model
finds *cancer* — not how well it clears healthy patients.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9211 | 0.8988 | 0.8837 | 0.9048 | 0.8941 | 0.8313 |
| kNN | 0.9737 | 0.9884 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9942 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |

*(Produced by `python model/train_models.py`, which prints this table to the console.)*

### Validation against scikit-learn

Since all five algorithms are hand-written (see *Implementation* below), each was
checked against the equivalent scikit-learn estimator on the identical split:

| Model | My accuracy | sklearn accuracy | Predictions agreeing |
|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9825 | 99.1% |
| Decision Tree | 0.9211 | 0.9211 | 91.2% |
| kNN | 0.9737 | 0.9737 | **100.0%** |
| Naive Bayes | 0.9298 | 0.9298 | **100.0%** |
| Random Forest (Ensemble) | 0.9561 | 0.9561 | 98.2% |

**kNN and Naive Bayes reproduce scikit-learn exactly** — every prediction, every
metric. Both are deterministic (kNN stores the training set; Gaussian NB has a
closed-form fit), so exact agreement is strong evidence the implementations are
correct rather than coincidentally close.

Logistic Regression also converges to sklearn's exact scores once L2
regularization is included and gradient descent is run to convergence. The
remaining differences are in the two tree-based models, and are expected:
**Decision Tree** searches a sampled set of candidate thresholds rather than
every midpoint, so it sometimes picks a different but equally good split;
**Random Forest** uses 50 trees against sklearn's 200, and its randomness comes
from a different generator. Both still land on identical accuracy.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the best overall performance (Accuracy 0.9825, MCC 0.9623). The dataset's classes are close to linearly separable in the standardized feature space, so a simple linear decision boundary generalizes very well and doesn't overfit. |
| Decision Tree | Weakest of the five models (Accuracy 0.9211, AUC 0.8988). A single depth-5 tree captures the coarse structure but loses the fine-grained separability that comes from combining many features smoothly. Its AUC is by far the lowest, and the reason is structural: a depth-5 tree can only ever output one of at most 32 leaf probabilities, so it produces a coarse, step-like ranking. AUC measures ranking quality, so it is penalised hardest here — note that its *accuracy* is competitive while its AUC is not. |
| kNN | The most interesting trade-off in the table: **perfect precision (1.0000)** — every case it called malignant really was — but recall of only 0.9286, meaning it **missed 3 of the 42 malignant cases**. It is the most cautious model, never raising a false alarm, but in screening that caution is the wrong way round: a missed cancer is far costlier than an unnecessary follow-up test. Distance-based similarity works well here because the features are smooth continuous measurements that respond to scaling. |
| Naive Bayes | Solid but not top-tier (Accuracy 0.9298). Its independence assumption between features is violated in this dataset (many nucleus measurements — e.g. radius, perimeter, area — are highly correlated), which slightly hurts precision/recall relative to the top models, even though the AUC (0.9868) shows it still ranks positive vs. negative cases well. |
| Random Forest (Ensemble) | Consistently strong and robust (Accuracy 0.9561, AUC 0.9942 — second-best AUC overall). It is the clearest demonstration of ensembling in this table: built from exactly the same tree algorithm as the Decision Tree row, it lifts AUC from 0.8988 to 0.9942. Averaging 50 bootstrapped trees both cancels the variance of any single tree and replaces that coarse step-like ranking with a smooth averaged probability. |
| **Overall Winner for your dataset?** | **Logistic Regression** — best on 5 of the 6 metrics (Accuracy 0.9825, AUC 0.9954, Recall 0.9762, F1 0.9762, MCC 0.9623). kNN takes Precision with a perfect 1.0000, but that is the less important of the pair here. The clinching argument is the error that actually matters: **Logistic Regression missed only 1 of the 42 malignant cases**, against 3 for kNN and Random Forest and 4 for Decision Tree and Naive Bayes. It is both the most accurate model and the safest one. |

## Project Structure

```
project-folder/
├── app.py                     # Streamlit app (upload data, select model, view metrics)
├── requirements.txt
├── README.md
├── test_data.csv              # held-out test split, used by the app
├── .streamlit/
│   └── config.toml            # app theme configuration
└── model/
    ├── data_prep.py            # shared load / split / scale pipeline
    ├── evaluate.py             # shared scoring helpers (the 6 metrics)
    ├── logistic_regression.py  # LogisticRegressionScratch  — algorithm + training
    ├── decision_tree.py        # DecisionTreeScratch        — algorithm + training
    ├── knn.py                  # KNNScratch                 — algorithm + training
    ├── naive_bayes.py          # GaussianNaiveBayesScratch  — algorithm + training
    ├── random_forest.py        # RandomForestScratch        — algorithm + training
    └── train_models.py         # runs all 5, prints the comparison table
```

`model/` contains source code only — no binary artifacts.

## Implementation

**All five classification algorithms are written from scratch in NumPy.** No
scikit-learn estimator is used to train any model. Each lives in its own file
and can be run on its own:

```bash
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py
```

| File | Class | What was implemented |
|---|---|---|
| `logistic_regression.py` | `LogisticRegressionScratch` | Sigmoid, binary cross-entropy loss, batch gradient descent, L2 regularization. Numerically stable sigmoid to avoid overflow on large negative inputs. |
| `decision_tree.py` | `DecisionTreeScratch` | CART: Gini impurity, exhaustive split search over candidate thresholds, recursive tree growth, depth/leaf-size stopping rules. |
| `knn.py` | `KNNScratch` | Euclidean distance via the `‖a-b‖² = ‖a‖² + ‖b‖² - 2ab` expansion (one matrix multiply, no Python loop), `argpartition` for O(n) k-selection. |
| `naive_bayes.py` | `GaussianNaiveBayesScratch` | Per-class means/variances, Gaussian log-likelihood, log-space arithmetic with the log-sum-exp trick to avoid underflow. |
| `random_forest.py` | `RandomForestScratch` | Bootstrap aggregating, random feature subsets (√30 ≈ 5 per split), soft voting. Built on top of `DecisionTreeScratch`. |

scikit-learn is used only for the parts the assignment specifies directly —
the six evaluation metrics, `train_test_split`, and `StandardScaler` — never for
the classifiers themselves.

### How the app gets its models

There are **no saved model files**. `app.py` imports the five modules above and
trains all of them at startup, inside `@st.cache_resource` — so training happens
once per app instance (about 2.7 seconds) and is then shared by every visitor,
not repeated per page load.

This was a deliberate choice over shipping `.pkl` files:

- **No version fragility.** A pickled model records the library versions that
  wrote it, and fails in confusing ways when the deployed environment differs.
  Training live removes that entire class of deployment failure.
- **Nothing can go stale.** There is no saved artifact that could drift out of
  sync with the algorithm code.
- **The repo is readable.** `model/` holds only source a reviewer can actually
  inspect, rather than opaque binaries.

The cost is a few seconds on first load, which the app covers with a spinner.

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # regenerates models/metrics/test_data.csv (optional, already included)
streamlit run app.py
```

## Streamlit App Features

### Required features

| # | Feature | Where it lives |
|---|---|---|
| a | **Dataset upload (CSV)** — accepts `test_data.csv`, or any CSV with the same 30 feature columns + `target` | Sidebar |
| b | **Model selection dropdown** — all 5 models, trained at startup from `model/` | Sidebar |
| c | **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC | 🎯 Live Evaluation tab |
| d | **Confusion matrix + classification report** — interactive heatmap and per-class table | 🎯 Live Evaluation tab |

Every metric is **computed live from the uploaded file**. No saved results file
exists for the app to fall back on — upload a different CSV and every number on
every tab changes accordingly.

### Additional features

The app is organised into five tabs, and is written to be understandable by a
reader with no machine-learning background — each chart and metric carries a
plain-language explanation.

- **🏠 Overview** — the medical problem, what the 30 measurements physically
  represent, and each model's decision rule in one sentence.
- **🔬 Data Explorer** — class-balance donut, per-feature histogram split by
  diagnosis, correlation heatmap, and a filterable data table.
- **🎯 Live Evaluation** — the six metrics explained in cancer-screening terms,
  plus a confusion matrix narrated in words with the run's actual numbers,
  identifying which error type is clinically dangerous.
- **👤 Single Prediction** — inspect one patient at a time: the verdict,
  a confidence bar, a correct/incorrect badge, and that case's measurements
  against the file average. Switching models shows where they disagree.
- **🏆 Model Comparison** — all five models scored live on the uploaded data,
  overlaid ROC curves, and permutation feature importance (which works for every
  model, including kNN and Naive Bayes, which expose no native importances).

**Accessibility & design:** a light/dark mode toggle in the sidebar; a fixed
colour per model held consistent across every chart; ROC lines distinguished by
dash pattern as well as hue; and status colours always paired with an icon and
text label so meaning is never carried by colour alone.

## Live App

> **TODO:** Replace with your deployed Streamlit Community Cloud URL after
> deployment, e.g. `https://<your-app-name>.streamlit.app`

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App** → select this repository → branch `main` → file `app.py`.
4. Click **Deploy**.
5. Once live, test by uploading `test_data.csv` and cycling through each model.
