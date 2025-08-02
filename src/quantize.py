import joblib
import numpy as np
import os
from utils import load_data

def quantize_params(params):
    return params.astype(np.float16)

def dequantize_params(quantized):
    return quantized.astype(np.float32)

def main():
    print("Loading model...")
    model = joblib.load("../artifacts/linear_model.joblib")
    coef = model.coef_
    intercept = model.intercept_

    print("Saving unquantized parameters...")
    os.makedirs("../artifacts", exist_ok=True)
    joblib.dump({
        'coef': coef,
        'intercept': intercept
    }, "../artifacts/params_unquantized.joblib")

    print("Quantizing parameters...")
    coef_q = quantize_params(coef)
    intercept_q = np.float16(intercept)

    joblib.dump({
        'coef': coef_q,
        'intercept': intercept_q
    }, "../artifacts/params_quantized.joblib")

    print("Quantized parameters saved.")

    print("Verifying prediction using dequantized parameters...")
    X, _ = load_data()
    X_sample = X[0]
    coef_deq = dequantize_params(coef_q)
    intercept_deq = float(intercept_q)

    pred = np.dot(X_sample, coef_deq) + intercept_deq
    print(f"Sample prediction (dequantized weights): {pred:.4f}")

if __name__ == "__main__":
    main()
