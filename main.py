import pandas as pd
import os
import sklearn
import numpy as np
from nn import neural_network

if __name__ == "__main__":
    DATASET_PATH = "."

    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    powerpredict_df = powerpredict_df.dropna()  # drop rows with missing values

    categorical_cols = powerpredict_df.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict_df, columns=categorical_cols)

    neural_network(encoded_df)


