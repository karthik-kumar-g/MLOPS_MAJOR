import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_model():
    data = fetch_california_housing()
    X, y = data.data, data.target
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y

def test_data_loading():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0
    assert data.target.shape[0] > 0
    print(" Dataset loaded successfully.")

def test_model_training():
    model, X, y = load_model()
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    print(" Model instance is LinearRegression and has been trained.")

def test_r2_score():
    model, X, y = load_model()
    r2 = r2_score(y, model.predict(X))
    assert r2 > 0.5
    print(f" R2 score is acceptable: {r2:.4f}")
