import pandas as pd
import os

def load_data():
    data_path = "../data/california.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop("MedHouseVal", axis=1).values
    y = df["MedHouseVal"].values
    return X, y
