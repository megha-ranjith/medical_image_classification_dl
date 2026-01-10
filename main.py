import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.image_preprocessing import preprocess_image
from models.cnn_feature_extractor import load_feature_extractor, extract_features
from preprocessing.pca_reduction import apply_pca
from models.classical_models import train_models
from models.ensemble import ensemble_predict
from models.rejection import apply_rejection
from evaluation.metrics import compute_metrics
from evaluation.plots import entropy_histogram
from config import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

# -------------------- Setup --------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# -------------------- Load Dataset --------------------
X, y = [], []

for label, cls in enumerate(["NORMAL", "TB"]):
    folder = f"data/raw/{cls}"
    for file in os.listdir(folder):
        if file.lower().endswith(".png"):
            X.append(preprocess_image(os.path.join(folder, file)))
            y.append(label)

X = np.array(X)
y = np.array(y)

# -------------------- CNN Feature Extraction --------------------
cnn = load_feature_extractor()
features = extract_features(cnn, X)

# -------------------- PCA --------------------
features_pca, pca = apply_pca(features, PCA_COMPONENTS)

# Save processed features
processed_df = pd.DataFrame(features_pca)
processed_df["label"] = y
processed_df.to_csv("data/processed/pca_features.csv", index=False)

# -------------------- 5-Fold Cross Validation --------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

metrics_list = []
coverages = []
tprs = []
aucs = []

mean_fpr = np.linspace(0, 1, 100)

fold = 1

for train_idx, test_idx in skf.split(features_pca, y):
    print(f"\n Fold {fold}")

    X_train, X_test = features_pca[train_idx], features_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train ensemble
    models = train_models(X_train, y_train)

    # Predict
    probs = ensemble_predict(models, X_test)

    # Rejection mechanism
    accepted, entropy_vals = apply_rejection(probs, ENTROPY_THRESHOLD)

    if np.sum(accepted) == 0:
        print(" All samples rejected in this fold. Skipping.")
        continue

    # Metrics on accepted samples only
    fold_metrics = compute_metrics(
        y_test[accepted],
        probs[accepted].argmax(axis=1),
        probs[accepted]
    )

    coverage = np.sum(accepted) / len(y_test)

    fold_metrics["Coverage"] = coverage
    metrics_list.append(fold_metrics)
    coverages.append(coverage)

    # ROC (only accepted samples)
    fpr, tpr, _ = roc_curve(
        y_test[accepted],
        probs[accepted][:, 1]
    )

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # Save entropy histogram per fold
    entropy_histogram(
        entropy_vals,
        f"results/plots/entropy_histogram_fold_{fold}.png"
    )

    fold += 1

# -------------------- Aggregate Results --------------------
results_df = pd.DataFrame(metrics_list)

mean_results = results_df.mean()
std_results = results_df.std()

final_results = pd.DataFrame({
    "Mean": mean_results,
    "Std": std_results
})

final_results.to_csv(
    "results/tables/final_results_5fold.csv"
)

# -------------------- Mean ROC Curve --------------------
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.figure()
plt.plot(
    mean_fpr,
    mean_tpr,
    label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (5-Fold Cross-Validation)")
plt.legend(loc="lower right")
plt.savefig("results/plots/mean_roc_curve.png")
plt.close()

# -------------------- Console Output --------------------
print("\n FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print(final_results)

print("\n Mean AUC:", mean_auc)
print(" Std AUC :", std_auc)
print(" Mean Coverage:", np.mean(coverages))
