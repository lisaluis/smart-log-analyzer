import pickle, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)
from preprocess import preprocess

def evaluate():
    X_train, X_test, y_train, y_test, features = preprocess()

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model = pickle.load(open(os.path.join(models_dir, "model.pkl"), "rb"))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print("\n📊 Model Evaluation Report")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation", fontsize=16, fontweight="bold")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "evaluation.png"), dpi=150)
    plt.close()
    print(f"✅ Evaluation plots saved to {models_dir}/evaluation.png")

    with open(os.path.join(models_dir, "evaluation.json"), "w") as f:
        json.dump({**metrics, "roc_auc": round(roc_auc, 4)}, f, indent=2)

    return metrics

if __name__ == "__main__":
    evaluate()
