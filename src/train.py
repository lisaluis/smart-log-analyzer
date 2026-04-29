import pickle, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train():
    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000)
    }

    best_model, best_acc = None, 0
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        f1  = f1_score(y_test, model.predict(X_test))
        print(f"{name}: accuracy={acc:.4f} f1={f1:.4f}")
        pickle.dump(model, open(f"models/{name}.pkl", "wb"))
        if acc > best_acc:
            best_acc  = acc
            best_model = model

    pickle.dump(best_model, open("models/model.pkl", "wb"))
    print(f"Best model saved — accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()