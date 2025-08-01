import pandas as pd
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("data/california.csv", index=False)
print("Dataset saved to data/california.csv")
