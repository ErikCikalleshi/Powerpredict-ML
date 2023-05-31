import pandas as pd
import os

DATASET_PATH = "."

if os.path.exists("/data/mlproject22"):
    DATASET_PATH = "/data/mlproject22"

powerpredict = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv.zip"))
X = powerpredict.drop(columns=["power_consumption"])
y = powerpredict[["power_consumption"]]

print(X.shape)
print(y.shape)


def drop_object_columns(df):
    drop_cols = [c for t, c in zip([t != "object" for t in df.dtypes], df.columns) if not t]
    return df.drop(columns=drop_cols)


DOC = drop_object_columns


def predict_show_metrics(name, reg, metric):
    print(f"{name}", metric(y, reg.predict(DOC(X))))
