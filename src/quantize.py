import joblib
import numpy as np
import os
from utils import load_data
'''
def quantize_params(params):
    min_val = params.min()
    max_val = params.max()

    if max_val == min_val:
        scale = 1.0  # avoid divide-by-zero
        quantized = np.zeros_like(params, dtype=np.uint8)
    else:
        scale = 255 / (max_val - min_val)
        quantized = ((params - min_val) * scale).astype(np.uint8)

    return quantized, scale, min_val

def dequantize_params(quantized, scale, min_val):
    return quantized.astype(np.float32) / scale + min_val '''
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
    coef_q, coef_scale, coef_min = quantize_params(coef)
    intercept_q, intercept_scale, intercept_min = quantize_params(np.array([intercept]))

    joblib.dump({
        ''''coef': coef_q,
        'coef_scale': coef_scale,
        'coef_min': coef_min,
        'intercept': intercept_q,
        'intercept_scale': intercept_scale,
        'intercept_min': intercept_min'''
        'coef': quantize_params(coef),
        'intercept': np.float16(intercept)
    }, "../artifacts/params_quantized.joblib")

    print("Quantized parameters saved.")

    print("Verifying prediction using dequantized parameters...")
    X, _ = load_data()
    X_sample = X[0]
    coef_deq = dequantize_params(coef_q, coef_scale, coef_min)
    intercept_deq = dequantize_params(intercept_q, intercept_scale, intercept_min)[0]

    pred = np.dot(X_sample, coef_deq) + intercept_deq
    print(f"Sample prediction (dequantized weights): {pred:.4f}")

if __name__ == "__main__":
    main()
