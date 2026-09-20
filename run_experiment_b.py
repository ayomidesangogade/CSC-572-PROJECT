"""
run_experiment_b.py

Experiment B: UCI (2014) -> PhiUSIIL (2024) cross-dataset generalisation test.

Trains the same 5 algorithms used in Experiment A, but restricted to the
6-feature common lexical subset shared by both datasets, on the UCI
training split. Evaluates on:
  (1) the UCI hold-out test set, using only the 6 common features
      (a fair "reduced-feature" control condition), and
  (2) the entire PhiUSIIL dataset, with the same 6 features recomputed
      directly from PhiUSIIL's raw URL column.

Reuses the same RANDOM_STATE, split ratio, and model hyperparameters as
the original notebook so Experiment B is a controlled variant of
Experiment A, not a different experiment.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from common_lexical_features import compute_common_features, COMMON_FEATURES

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load UCI (2014) dataset and select the common 6-feature subset
# ----------------------------------------------------------------------
print("Loading UCI dataset...")
uci = pd.read_csv("phishing_dataset.csv")
X_uci = uci[COMMON_FEATURES].copy()
y_uci = uci["Result"].map({-1: 1, 1: 0})  # 1 = phishing, 0 = legitimate (same convention as Experiment A)

X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.2, random_state=RANDOM_STATE, stratify=y_uci
)
print(f"UCI train: {X_train.shape}, UCI test: {X_test.shape}")

# ----------------------------------------------------------------------
# 2. Load PhiUSIIL (2024) dataset and compute the same 6 features
#    from its raw URL column
# ----------------------------------------------------------------------
print("Loading PhiUSIIL dataset...")
phi = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"PhiUSIIL raw shape: {phi.shape}")
# PhiUSIIL label convention: 1 = legitimate, 0 = phishing (opposite of ours)
y_phi = 1 - phi["label"]  # -> 1 = phishing, 0 = legitimate, matching UCI convention

print("Computing common lexical features for all PhiUSIIL URLs "
      "(pure string parsing, no network calls)...")
t0 = time.time()
feature_dicts = phi["URL"].apply(compute_common_features)
X_phi = pd.DataFrame(list(feature_dicts))[COMMON_FEATURES]
print(f"Done in {time.time() - t0:.1f}s")

# Sanity check: distribution of each common feature should look similar
# in shape (not identical) between UCI and PhiUSIIL if features are sane
print("\nFeature value counts sanity check (UCI train vs PhiUSIIL):")
for col in COMMON_FEATURES:
    print(f"  {col}: UCI={dict(X_train[col].value_counts())}  "
          f"PhiUSIIL={dict(X_phi[col].value_counts())}")

# ----------------------------------------------------------------------
# 3. Train the 5 models on the 6-feature UCI training subset
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
}

trained = {}
rows = []

def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
        "F1-score": f1_score(y, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probs),
    }, preds

print("\nTraining models on the 6-feature common subset...")
for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model

    uci_metrics, _ = evaluate(model, X_test, y_test)
    phi_metrics, phi_preds = evaluate(model, X_phi, y_phi)

    row = {"Model": name}
    for k, v in uci_metrics.items():
        row[f"UCI_{k}"] = round(v, 4)
    for k, v in phi_metrics.items():
        row[f"PhiUSIIL_{k}"] = round(v, 4)
    row["F1_change"] = round(phi_metrics["F1-score"] - uci_metrics["F1-score"], 4)
    rows.append(row)
    print(f"  {name}: UCI(6ft) F1={uci_metrics['F1-score']:.4f}  "
          f"PhiUSIIL F1={phi_metrics['F1-score']:.4f}  "
          f"(change {row['F1_change']:+.4f})")

results_df = pd.DataFrame(rows).sort_values("PhiUSIIL_F1-score", ascending=False).reset_index(drop=True)
results_df.to_csv("experiment_b_results.csv", index=False)
print("\nSaved experiment_b_results.csv")
print(results_df.to_string(index=False))

# ----------------------------------------------------------------------
# 4. Detailed look at the best model on PhiUSIIL (by F1 on PhiUSIIL)
# ----------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = trained[best_name]
print(f"\nBest model on PhiUSIIL (external test): {best_name}")

_, best_preds = evaluate(best_model, X_phi, y_phi)
print(classification_report(y_phi, best_preds, target_names=["Legitimate", "Phishing"]))

cm = confusion_matrix(y_phi, best_preds)
plt.figure(figsize=(6.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.title(f"Confusion Matrix: {best_name}\n(UCI-trained, 6 common features, tested on PhiUSIIL)", fontsize=11)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("experiment_b_confusion_matrix.png", dpi=150)
print("Saved experiment_b_confusion_matrix.png")

# ----------------------------------------------------------------------
# 5. Comparison bar chart: F1 on UCI(9ft) vs PhiUSIIL, per model
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df["UCI_F1-score"], width, label="UCI hold-out (6 features)")
plt.bar(x + width/2, results_df["PhiUSIIL_F1-score"], width, label="PhiUSIIL (external)")
plt.xticks(x, results_df["Model"], rotation=15)
plt.ylabel("F1-score")
plt.title("Experiment B: F1-score, UCI (6-feature subset) vs PhiUSIIL")
plt.legend()
plt.tight_layout()
plt.savefig("experiment_b_f1_comparison.png", dpi=150)
print("Saved experiment_b_f1_comparison.png")

with open("experiment_b_summary.json", "w") as f:
    json.dump({
        "uci_train_shape": list(X_train.shape),
        "uci_test_shape": list(X_test.shape),
        "phiusiil_shape": list(X_phi.shape),
        "common_features": COMMON_FEATURES,
        "best_model_on_phiusiil": best_name,
        "results": rows,
    }, f, indent=2)
print("Saved experiment_b_summary.json")
