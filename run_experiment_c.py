"""
run_experiment_c.py

Experiment C: train AND test on PhiUSIIL natively (its own 50 usable
features), following the same pipeline structure as Experiment A but on
the 2024 dataset. This resolves Suggestion 7 (section 5.7): it isolates
how much of Experiment B's recall drop was due to the restricted 6-feature
common subset vs. genuine concept drift.

Adapted from the user's own Phishing_URL_Detection_PhiUSIIL.ipynb, with one
practical fix: SVM (SVC via CalibratedClassifierCV) is trained on a
stratified subsample (their own notebook flagged this as a likely
necessity and left a commented-out fallback) since the full ~189K-row
training set makes the kernel SVM intractable. All other models are
trained on the full training set. Evaluation is always on the full test
set for every model.
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load and prepare data (mirrors the user's notebook exactly)
# ----------------------------------------------------------------------
print("Loading PhiUSIIL dataset...")
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
print("Shape:", df.shape)

id_cols = ["FILENAME", "URL", "Domain", "TLD", "Title"]
X = df.drop(columns=id_cols + ["label"])
y = df["label"].map({0: 1, 1: 0})  # 1 = phishing, 0 = legitimate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 2. Train and compare 5 models
#    SVM trained on a stratified 20,000-row subsample (their own
#    notebook's suggested fallback) for tractable runtime; every model
#    is evaluated on the FULL test set regardless.
# ----------------------------------------------------------------------
X_train_svm, y_train_svm = resample(
    X_train_scaled, y_train, n_samples=20000, stratify=y_train, random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True, False),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False, False),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False, False),
    "SVM": (CalibratedClassifierCV(SVC(), ensemble=False), True, True),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE), False, False),
}
# (model, use_scaled_features, use_svm_subsample)

trained_models = {}
model_uses_scaled = {}
results = []

for name, (model, use_scaled, use_subsample) in models.items():
    t0 = time.time()
    if use_subsample:
        Xtr, ytr = X_train_svm, y_train_svm
    else:
        Xtr = X_train_scaled if use_scaled else X_train
        ytr = y_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, ytr)
    trained_models[name] = model
    model_uses_scaled[name] = use_scaled

    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-score": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    })
    print(f"{name} done in {time.time() - t0:.1f}s"
          + (" (trained on 20K stratified subsample)" if use_subsample else ""))

results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
results_df.to_csv("experiment_c_results.csv", index=False)
print("\n" + results_df.round(4).to_string(index=False))

# ----------------------------------------------------------------------
# 3. Model comparison chart
# ----------------------------------------------------------------------
ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(
    kind="bar", figsize=(11, 6)
)
plt.title("Experiment C: Model Comparison (Native PhiUSIIL Training)")
plt.ylabel("Score")
plt.ylim(0.99, 1.001)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("experiment_c_model_comparison.png", dpi=150)
plt.close()
print("Saved experiment_c_model_comparison.png")

# ----------------------------------------------------------------------
# 4. Detailed evaluation of best model
# ----------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_uses_scaled = model_uses_scaled[best_name]
X_test_best = X_test_scaled if best_uses_scaled else X_test
print(f"\nBest model by F1-score: {best_name}")

preds = best_model.predict(X_test_best)
probs = best_model.predict_proba(X_test_best)[:, 1]
report = classification_report(y_test, preds, target_names=["Legitimate", "Phishing"])
print(report)
with open("experiment_c_classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n{report}")

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.title(f"Confusion Matrix: {best_name} (PhiUSIIL, native features)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("experiment_c_confusion_matrix.png", dpi=150)
plt.close()
print("Saved experiment_c_confusion_matrix.png")

# ----------------------------------------------------------------------
# 5. ROC curves for all 5 models
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 5.5))
for name, model in trained_models.items():
    Xte_i = X_test_scaled if model_uses_scaled[name] else X_test
    probs_i = model.predict_proba(Xte_i)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs_i)
    auc = roc_auc_score(y_test, probs_i)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Experiment C: ROC Curves (Native PhiUSIIL Training)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("experiment_c_roc_curves.png", dpi=150)
plt.close()
print("Saved experiment_c_roc_curves.png")

# ----------------------------------------------------------------------
# 6. Feature importance
# ----------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
elif hasattr(best_model, "coef_"):
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=False)
else:
    importances = None

if importances is not None:
    plt.figure(figsize=(8, 9))
    importances.head(15).sort_values().plot(kind="barh", color="teal")
    plt.title(f"Experiment C: Top 15 Feature Importances ({best_name})")
    plt.tight_layout()
    plt.savefig("experiment_c_feature_importance.png", dpi=150)
    plt.close()
    print("Saved experiment_c_feature_importance.png")
    importances.head(15).to_csv("experiment_c_top15_importances.csv")

# ----------------------------------------------------------------------
# 7. Cross-validation check (5-fold, on the FULL dataset; if best model
#    needs scaling, wrap in a pipeline; if best model is SVM, subsample
#    consistently to keep this tractable)
# ----------------------------------------------------------------------
print("\nRunning 5-fold cross-validation on best model...")
if best_name == "SVM":
    X_cv, y_cv = resample(X, y, n_samples=20000, stratify=y, random_state=RANDOM_STATE)
    cv_estimator = make_pipeline(StandardScaler(), SVC(probability=True))
else:
    X_cv, y_cv = X, y
    cv_estimator = make_pipeline(StandardScaler(), best_model) if best_uses_scaled else best_model

cv_scores = cross_val_score(cv_estimator, X_cv, y_cv, cv=5, scoring="f1")
print(f"5-fold CV F1-scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

# ----------------------------------------------------------------------
# 8. URLSimilarityIndex leakage check (referenced in the user's own
#    notebook as a documented property of PhiUSIIL)
# ----------------------------------------------------------------------
leak_corr = df["URLSimilarityIndex"].corr(y)
print(f"\nURLSimilarityIndex correlation with phishing label: {leak_corr:.4f}")

# ----------------------------------------------------------------------
# 9. Save model + summary
# ----------------------------------------------------------------------
joblib.dump(best_model, "phishing_model_phiusiil.pkl")
joblib.dump(list(X.columns), "feature_columns_phiusiil.pkl")

summary = {
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "results": results,
    "best_model": best_name,
    "cv_f1_scores": list(cv_scores),
    "cv_f1_mean": float(cv_scores.mean()),
    "cv_f1_std": float(cv_scores.std()),
    "url_similarity_index_corr_with_phishing": float(leak_corr),
    "top15_importances": importances.head(15).to_dict() if importances is not None else None,
}
with open("experiment_c_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved experiment_c_summary.json. Done.")
