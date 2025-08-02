# MLOps Major Assignment — California Housing Regression

[[CI/CD Pipeline](https://github.com/karthik-kumar-g/MLOPS_MAJOR/actions)
[Docker Image](https://hub.docker.com/layers/karthikkumarg/mlops-major-app/latest/images/sha256-e9d1602f6dac6854c09f182cc7b846c09a3288bb87b0667d4a13ec3209734936)

## Objective

This project implements a **complete MLOps pipeline** for training, quantizing, evaluating, and deploying a Linear Regression model on the California Housing dataset using:

Unit Testing (`pytest`)
Dockerized Environment
Manual Quantization of Weights (using `float16`)
CI/CD via GitHub Actions Model Packaging and Evaluation

---

## Repository Structure

```bash
MLOPS_MAJOR/
├── artifacts/                 # Saved models and quantized parameters
│   ├── linear_model.joblib
│   ├── params_unquantized.joblib
│   └── params_quantized.joblib
├── data/
│   └── california.csv         # Full dataset
├── src/
│   ├── train.py               # Model training script
│   ├── predict.py             # Run prediction using saved model
│   ├── quantize.py            # Manual quantization (float16)
│   ├── model_eval.py          # Compare quantized vs unquantized
│   ├── utils.py               # Data loading + generation
│   └── save_dataset.py        # One-time dataset export script
├── tests/
│   └── test_train.py          # Unit tests for training pipeline
├── Dockerfile                 # Container for running training/inference
├── Makefile                   # Common command automation
├── requirements.txt           # Project dependencies
└── .github/workflows/ci.yml# GitHub Actions CI/CD pipeline
```
---

##Step:
##1.Clone & Setup
```bash
git clone https://github.com/karthik-kumar-g/MLOPS_MAJOR.git
cd MLOPS_MAJOR
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---

##2.Train the Model Locally
```bash
python src/train.py
```
---

##3.Quantize the models
```bash
python src/quantize.py
```

---

##4.Evaluate Quantized and Unquantized models
```bash
python src/model_eval.py
```

---

##Quantization Comparison (Latest CI/CD Run)
| Metric | Unquantized  | Quantized Model |
|------------------|------------------------|--------------------------|
| R² Score | *0.6062* | *0.6062* |
| Model Size (KB) | *0.40 KB* | *0.32 KB* |
| MSE | *0.5243*  | *0.5244*  |

---

##5.Run Inside Dockerfile
```bash
docker build -t mlops-major-app .
docker run --rm -v $(pwd):/app -w /app/src mlops-major-app python train.py
docker run --rm -v $(pwd):/app -w /app/src mlops-major-app python predict.py
```

---

##6.CI/CD via GitHub actions
Every push to main:
	•	Runs pytest test cases
	•	Trains & quantizes the model
	•	Evaluates performance
	•	Builds Docker image
	•	Runs predict.py and model_eval.py inside the Docker container

View CI results here → [Actions Tab](https://github.com/karthik-kumar-g/MLOPS_MAJOR/actions)

---

##7.Unit Testing
```bash
pytest tests/
```

---

##Scripts Breakdown
| Script  | Description |
|------------------------|--------------------------|
| train.py  | Trains and saves Linear Regression model  |
| predict.py  | Loads model and makes prediction  |
| quantize.py | Converts weights to float16 manually  |
| model_eval.py | Compares quantized vs unquantized |
| save_dataset.py | One-time fetch of California Housing data |
| utils.py  | Contains data loading, saving utilities |
| test_train.py | Unit tests for training script  |

---
