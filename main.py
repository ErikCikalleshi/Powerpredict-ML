import pandas as pd
import os
import sklearn
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from nn import neural_network

# init model globally
model = None

def get_encoded(powerpredict):
    powerpredict = powerpredict.dropna()  # drop rows with missing values

    categorical_cols = powerpredict.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict, columns=categorical_cols)
    return encoded_df


def leader_board_predict_fn(values):
    values_encoded = get_encoded(values)
    #values_encoded = values_encoded.reindex(columns=x_train.columns, fill_value=0)

    # Convert values_encoded to float type
    values_encoded = values_encoded.astype(float)

    values_tensor = torch.Tensor(values_encoded.values)
    predictions = model(values_tensor).detach().numpy()

    return predictions


def best_epochs():
    num_epochs = 10
    epoch_losses = []
    for epoch in range(1, num_epochs+1):
        model = neural_network(x_train.values.astype(float), y_train.values.astype(float),
                               x_val.values.astype(float), y_val.values.astype(float), epoch)
        y_predict = leader_board_predict_fn(X_test)
        epoch_mae = mean_absolute_error(y_test, y_predict)
        print("Mean Absolute Error (MAE):", epoch_mae)
        epoch_losses.append(epoch_mae)

    #Create a chart
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance across Epochs')
    plt.show()


if __name__ == "__main__":
    print(torch.cuda.is_available())

    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    encoded_values = get_encoded(powerpredict_df)

    x = encoded_values.drop("power_consumption", axis=1)
    y = encoded_values["power_consumption"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
                           y_val.values.astype(float), 6)
    best_epochs()
    actual_values = encoded_values["power_consumption"]
    X_test = powerpredict_df.drop(columns=["power_consumption"])
    y_test = powerpredict_df["power_consumption"]
    # Get the predicted values using the model
    y_predict = leader_board_predict_fn(X_test)
    #
    # # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_predict)

    # print("Mean Absolute Error (MAE):", mae)


