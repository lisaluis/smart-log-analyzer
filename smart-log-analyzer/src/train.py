import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess import preprocess

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

def train():
    X_train, X_test, y_train, y_test, features = preprocess()

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Try MLflow; fall back silently if server is unavailable
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("smart-log-analyzer")
        use_mlflow = True
        print(f"📊 MLflow tracking at {MLFLOW_URI}")
    except Exception:
        use_mlflow = False
        print("⚠️  MLflow unavailable — training without tracking")

    model_candidates = {
        "random_forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_model, best_acc, results = None, 0, {}

    for name, model in model_candidates.items():
        def run_training():
            nonlocal best_model, best_acc
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            }
            results[name] = metrics

            if use_mlflow:
                mlflow.log_param("model_type", name)
                mlflow.log_param("n_features", len(features))
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  {name}: acc={metrics['accuracy']} f1={metrics['f1_score']} "
                  f"prec={metrics['precision']} recall={metrics['recall']}")

            pickle.dump(model, open(os.path.join(models_dir, f"{name}.pkl"), "wb"))

            if metrics["accuracy"] > best_acc:
                best_acc  = metrics["accuracy"]
                best_model = model

        if use_mlflow:
            with mlflow.start_run(run_name=name):
                run_training()
        else:
            run_training()

    pickle.dump(best_model, open(os.path.join(models_dir, "model.pkl"), "wb"))

    # Save results summary
    import json
    with open(os.path.join(models_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best model saved — accuracy: {best_acc:.4f}")
    print(f"📁 Saved to {models_dir}")
    return best_model, results

if __name__ == "__main__":
    train()
