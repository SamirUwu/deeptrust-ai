# Codigo para entrenar en Colab

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# ── Paths ─────────────────────────────────────────────────────────
BASE = "/content/drive/MyDrive/"

# ── Load ──────────────────────────────────────────────────────────
print("Loading embeddings...")
X_train = np.load(BASE + "train_embeddings.npy")
y_train = np.load(BASE + "train_labels.npy")
X_dev   = np.load(BASE + "dev_embeddings.npy")
y_dev   = np.load(BASE + "dev_labels.npy")

print(f"Train: {X_train.shape} — Real: {y_train.sum()} Fake: {(y_train==0).sum()}")
print(f"Dev:   {X_dev.shape}   — Real: {y_dev.sum()} Fake: {(y_dev==0).sum()}")

# ── Normalize ─────────────────────────────────────────────────────
print("\nNormalizing...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev   = scaler.transform(X_dev)

# ── Grid Search ───────────────────────────────────────────────────
param_grid = {
    "C"      : [0.1, 1, 10],
    "kernel" : ["rbf", "linear"],
    "gamma"  : ["scale", "auto", 0.001],
}

print(f"\nRunning Grid Search — {3*2*3} combinations x 3-fold CV = {3*2*3*3} fits")

grid_search = GridSearchCV(
    SVC(probability=True, class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

# ── Results ───────────────────────────────────────────────────────
print("\n─── Grid Search Results ──────────────────────────")
print(f"  Best params : {grid_search.best_params_}")
print(f"  Best F1     : {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred     = best_model.predict(X_dev)

print("\n─── Dev Set Performance ──────────────────────────")
print(classification_report(y_dev, y_pred, target_names=["fake", "real"]))

# ── Save to Drive ─────────────────────────────────────────────────
joblib.dump(best_model, BASE + "best_svm.pkl")
joblib.dump(scaler,     BASE + "scaler.pkl")
print("\nDone! best_svm.pkl and scaler.pkl saved to Drive")