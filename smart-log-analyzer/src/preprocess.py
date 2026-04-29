import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, os

def preprocess(path=None):
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "ai4i2020.csv")

    df = pd.read_csv(path)

    # Drop non-feature columns
    cols_to_drop = [c for c in ['UDI', 'Product ID', 'Type'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    FEATURES = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    TARGET   = 'Machine failure'

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    pickle.dump(scaler, open(os.path.join(models_dir, "scaler.pkl"), "wb"))

    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Failure rate (train): {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test, FEATURES

if __name__ == "__main__":
    preprocess()
