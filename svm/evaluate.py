"""
evaluate.py
Evaluación final del SVM sobre el eval set de ASVspoof 2019 LA.
Run: python evaluate.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
SVM_DIR        = os.path.join(os.path.dirname(__file__))

# ── Load eval set ─────────────────────────────────────────────────
print("Loading eval embeddings...")
X_eval = np.load(os.path.join(EMBEDDINGS_DIR, "eval_embeddings.npy"))
y_eval = np.load(os.path.join(EMBEDDINGS_DIR, "eval_labels.npy"))

print(f"  Eval: {X_eval.shape} — Real: {y_eval.sum()} Fake: {(y_eval==0).sum()}")

# ── Load model ────────────────────────────────────────────────────
print("\nLoading SVM...")
model  = joblib.load(os.path.join(SVM_DIR, "best_svm_v3.pkl"))
scaler = joblib.load(os.path.join(SVM_DIR, "scaler_v3.pkl"))

# ── Normalize ─────────────────────────────────────────────────────
X_eval = scaler.transform(X_eval)

# ── Predict ───────────────────────────────────────────────────────
print("Running inference...")
y_pred  = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)[:, 1]  # prob of real

# ── Metrics ───────────────────────────────────────────────────────
auc = roc_auc_score(y_eval, y_proba)
f1  = f1_score(y_eval, y_pred)
cm  = confusion_matrix(y_eval, y_pred)

tn, fp, fn, tp = cm.ravel()
far = fp / (fp + tn)  # False Acceptance Rate
frr = fn / (fn + tp)  # False Rejection Rate
eer = (far + frr) / 2 # Equal Error Rate (approximation)

print("\n═══ Final Evaluation — ASVspoof 2019 LA Eval Set ════════")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  EER      : {eer:.4f}  ({eer*100:.2f}%)")
print(f"  FAR      : {far:.4f}  (fake accepted as real)")
print(f"  FRR      : {frr:.4f}  (real rejected as fake)")

print("\n─── Classification Report ────────────────────────────────")
print(classification_report(y_eval, y_pred, target_names=["fake", "real"]))

print("─── Confusion Matrix ─────────────────────────────────────")
print(f"  True Fake  (TN): {tn:>6}")
print(f"  False Real (FP): {fp:>6}  ← fakes that slipped through")
print(f"  False Fake (FN): {fn:>6}  ← real voices flagged as fake")
print(f"  True Real  (TP): {tp:>6}")
print("══════════════════════════════════════════════════════════")

# Agrega esto al final de evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns

# ── Confusion Matrix Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AASIST + SVM — ASVspoof 2019 LA Eval Set", fontsize=14, fontweight="bold")

# Plot 1: Confusion Matrix
cm_display = np.array([[tn, fp], [fn, tp]])

# Normalizar por fila
cm_norm = cm_display.astype("float") / cm_display.sum(axis=1, keepdims=True)

# Crear etiquetas combinadas (número + %)
annot_labels = np.array([
    [f"{tn}\n({cm_norm[0,0]*100:.1f}%)", f"{fp}\n({cm_norm[0,1]*100:.1f}%)"],
    [f"{fn}\n({cm_norm[1,0]*100:.1f}%)", f"{tp}\n({cm_norm[1,1]*100:.1f}%)"],
])

sns.heatmap(
    cm_norm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Predicted Fake", "Predicted Real"],
    yticklabels=["Actual Fake", "Actual Real"],
    ax=axes[0],
    linewidths=0.5,
    linecolor="gray",
)

axes[0].set_title("Confusion Matrix (Count + %)", fontsize=12)
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# Plot 2: Metrics bar chart
metrics = {
    "AUC-ROC"  : auc,
    "F1 (fake)": f1_score(y_eval, y_pred, pos_label=0),
    "F1 (real)": f1_score(y_eval, y_pred, pos_label=1),
    "Accuracy" : (tn + tp) / (tn + fp + fn + tp),
    "1 - EER"  : 1 - eer,
}

colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = axes[1].barh(list(metrics.keys()), list(metrics.values()), color=colors)
axes[1].set_xlim(0, 1.1)
axes[1].set_title("Performance Metrics", fontsize=12)
axes[1].set_xlabel("Score")

for bar, val in zip(bars, metrics.values()):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SVM_DIR, "evaluation_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to svm/evaluation_results.png")