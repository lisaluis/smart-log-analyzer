import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append('src')
from preprocess import preprocess

# ── Point MLflow to your tracking server ──────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("smart-log-analyzer")

def train():
    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "random_forest":      RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000)
    }

    best_model, best_acc = None, 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred)

            # ── Log params & metrics to MLflow ────
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # ── Log model artifact ─────────────────
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{name}: accuracy={acc:.4f} f1={f1:.4f}")

            pickle.dump(model, open(f"models/{name}.pkl", "wb"))

            if acc > best_acc:
                best_acc  = acc
                best_model = model

    pickle.dump(best_model, open("models/model.pkl", "wb"))
    print(f"\nBest model saved — accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()