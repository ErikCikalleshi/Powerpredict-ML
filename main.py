import pandas as pd
import os
import sklearn
import numpy as np
import torch

from nn import neural_network


def get_encoded(powerpredict_df):
    powerpredict_df = powerpredict_df.dropna()  # drop rows with missing values

    categorical_cols = powerpredict_df.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict_df, columns=categorical_cols)
    return encoded_df


def leader_board_predict_fn(values):
    # encoded_values = get_encoded(values)  # Encode the input values
    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    encoded_values = get_encoded(powerpredict_df)
    model = neural_network(encoded_values)  # Train the neural network on the encoded data
    print(model)
    print(values)
    values = values.apply(pd.to_numeric, errors='coerce')
    values = values.astype(np.float32)
    values_tensor = torch.Tensor(values.values)

    # Reshape values_tensor if necessary
    if len(values_tensor.shape) == 1:
        values_tensor = values_tensor.unsqueeze(0)
    print(values_tensor.shape)
    # Make predictions using the trained model
    predictions = model(values_tensor).detach().numpy()

    return predictions


if __name__ == "__main__":
    DATASET_PATH = "."

    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    powerpredict_df = powerpredict_df.dropna()  # drop rows with missing values

    categorical_cols = powerpredict_df.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict_df, columns=categorical_cols)

    # model = neural_network(encoded_df)

    X_test = powerpredict_df.drop(columns=["power_consumption"])
    y_test = powerpredict_df[["power_consumption"]]
    y_predicted = leader_board_predict_fn(X_test)
