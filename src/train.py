from utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os

def main():
    print("Loading dataset...")
    X, y = load_data()

    print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)

    print("Model trained successfully.")

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Training completed. R2 Score: {r2:.4f}, MSE: {mse:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/linear_model.joblib")
    print("Model saved at artifacts/linear_model.joblib")

if __name__ == "__main__":
    main()
