import joblib
from utils import load_data

def main():
    print("Loading model...")
    model = joblib.load("../artifacts/linear_model.joblib")

    print("Fetching test sample...")
    X, _ = load_data()
    sample = X[5].reshape(1, -1)

    print("Making prediction...")
    prediction = model.predict(sample)

    print(f"Prediction for test sample: {prediction[0]:.4f}")

if __name__ == "__main__":
    main()
