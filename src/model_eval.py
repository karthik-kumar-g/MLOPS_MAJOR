import joblib
import numpy as np
import os
from utils import load_data
from sklearn.metrics import mean_squared_error, r2_score

def predict(X, coef, intercept):
    return np.dot(X, coef) + intercept

def load_params(path):
    params = joblib.load(path)
    return np.array(params["coef"]), params["intercept"]

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024

def main():
    print("Comparing Quantized vs Unquantized Models\n")


    X, y = load_data()


    unq_path = "../artifacts/params_unquantized.joblib"
    q_path = "../artifacts/params_quantized.joblib"

    # === Unquantized ===
    coef_unq, intercept_unq = load_params(unq_path)
    preds_unq = predict(X, coef_unq, intercept_unq)
    mse_unq = mean_squared_error(y, preds_unq)
    r2_unq = r2_score(y, preds_unq)
    size_unq = get_file_size_kb(unq_path)

    # === Quantized ===
    coef_q, intercept_q = load_params(q_path)
    preds_q = predict(X, coef_q, intercept_q)
    mse_q = mean_squared_error(y, preds_q)
    r2_q = r2_score(y, preds_q)
    size_q = get_file_size_kb(q_path)

    # === Output Results ===
    print("Full Dataset Evaluation")
    print("-" * 50)
    print(f"{'Metric':<20}{'Unquantized':<12}{'Quantized'}")
    print("-" * 50)
    print(f"{'File Size (KB)':<20}{size_unq:<12.2f}{size_q:.2f}")
    print(f"{'MSE':<20}{mse_unq:<12.4f}{mse_q:.4f}")
    print(f"{'R² Score':<20}{r2_unq:<12.4f}{r2_q:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
