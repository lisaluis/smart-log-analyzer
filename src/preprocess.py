import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, os

def preprocess(path=None):
    if path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, "data", "ai4i2020.csv")

    df = pd.read_csv(path)
    df.drop(columns=['UDI', 'Product ID', 'Type'], inplace=True)

    X = df[['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    y = df['Machine failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess()