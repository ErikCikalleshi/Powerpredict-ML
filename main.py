import pandas as pd
import os
import sklearn
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from nn import neural_network


def get_encoded(powerpredict_df):
    powerpredict_df = powerpredict_df.dropna()
    categorical_cols = powerpredict_df.select_dtypes(include=['object']).columns
    encoded_df = pd.get_dummies(powerpredict_df, columns=categorical_cols)
    return encoded_df


def leader_board_predict_fn(values):
    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    encoded_values = get_encoded(powerpredict_df)
    print(encoded_values["power_consumption"])
    x = encoded_values.drop("power_consumption", axis=1)
    y = encoded_values["power_consumption"]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    values_encoded = get_encoded(values)
    values_encoded = values_encoded.reindex(columns=x_train.columns, fill_value=0)

    # Convert values_encoded to float type
    values_encoded = values_encoded.astype(float)

    values_tensor = torch.Tensor(values_encoded.values)

    model = neural_network(x_train.values.astype(float), y_train.values.astype(float))
    predictions = model(values_tensor).detach().numpy()

    return predictions


if __name__ == "__main__":
    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    X_test = powerpredict_df.drop(columns=["power_consumption"])

    y_predicted = leader_board_predict_fn(X_test)
    print(y_predicted)
