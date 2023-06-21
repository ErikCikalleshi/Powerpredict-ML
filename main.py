import pandas as pd
import os
import sklearn
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import time

from nn import neural_network

# init model globally
model = None


def get_encodedV2(powerpredict):
    correlation_threshold = 0.05
    categorical_cols = powerpredict.select_dtypes(include=['object']).columns
    encoded_df = pd.get_dummies(powerpredict, columns=categorical_cols)
    encoded_df = encoded_df.fillna(encoded_df.mean())

    # Calculate correlation matrix
    correlation_matrix = encoded_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))

    # Find index of feature columns with correlation lower than correlation_threshold and do not include power_consumption
    to_drop = [column for column in upper.columns if
               any(upper[column] < correlation_threshold) and column != "power_consumption"]

    # Drop features with lower correlation
    encoded_df = encoded_df.drop(encoded_df[to_drop], axis=1)
    return encoded_df


def get_encoded(powerpredict):
    powerpredict = powerpredict.dropna()  # drop rows with missing values
    # remove colums which start with _main and _description
    #powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_main')]
    #powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_description')]

    categorical_cols = powerpredict.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict, columns=categorical_cols)
    encoded_df = encoded_df.fillna(encoded_df.mean())  # drop rows with missing values

    return encoded_df


def leader_board_predict_fn(values):
    global model
    values_encoded = get_encoded(values)
    # values_encoded = values_encoded.reindex(columns=x_train.columns, fill_value=0)

    # Convert values_encoded to float type
    values_encoded = values_encoded.astype(float)

    values_tensor = torch.Tensor(values_encoded.values)
    predictions = model(values_tensor).detach().numpy()

    return predictions


def best_epochs():
    global model
    num_epochs = 10
    epoch_losses = []
    for epoch in range(6, num_epochs + 1):
        model = neural_network(x_train.values.astype(float), y_train.values.astype(float),
                               x_val.values.astype(float), y_val.values.astype(float), epoch)
        y_predict = leader_board_predict_fn(X_test)
        epoch_mae = mean_absolute_error(y_test, y_predict)
        print(epoch, "Mean Absolute Error (MAE):", epoch_mae)
        epoch_losses.append(epoch_mae)

    # Create a chart
    plt.plot(range(6, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance across Epochs')
    plt.show()


def plot_predictions(y_test, y_predict):
    plt.scatter(y_test, y_predict)
    plt.xlabel('Actual Power Consumption')
    plt.ylabel('Predicted Power Consumption')
    plt.title('Actual vs. Predicted Power Consumption')
    plt.show()


def plot_residuals(y_test, y_predict):
    y_test = y_test.values.flatten()  # Flatten y_test to make it 1-dimensional
    y_predict = y_predict.flatten()  # Flatten y_predict to make it 1-dimensional

    residuals = y_test - y_predict
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Actual Power Consumption')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()


def train_evaluate_knn(x_train, y_train, x_val, y_val, x_test):
    # Train the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed
    knn_model.fit(x_train, y_train)

    # Evaluate the model on the validation set
    val_predictions = knn_model.predict(x_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print("Validation Mean Absolute Error (MAE) for KNN:", val_mae)

    # Evaluate the model on the test set
    test_predictions = knn_model.predict(x_test)

    return test_predictions


def train_evaluate_random_forest(x_train, y_train, x_val, y_val, x_test):
    # Train the Random Forest model
    random_forest_model = RandomForestRegressor(n_estimators=5, random_state=42)  # You can adjust the number of estimators as needed
    random_forest_model.fit(x_train, y_train)

    # Evaluate the model on the validation set
    val_predictions = random_forest_model.predict(x_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print("Validation Mean Absolute Error (MAE) for Random Forest:", val_mae)

    # Evaluate the model on the test set
    test_predictions = random_forest_model.predict(x_test)

    return test_predictions

def train_evaluate_decision_tree(x_train, y_train, x_val, y_val, x_test):
    # Train the decision tree model
    decision_tree_model = DecisionTreeRegressor(random_state=42)  # You can adjust the random_state as needed
    decision_tree_model.fit(x_train, y_train)

    # Evaluate the model on the validation set
    val_predictions = decision_tree_model.predict(x_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print("Validation Mean Absolute Error (MAE) for Decision Tree:", val_mae)

    # Evaluate the model on the test set
    test_predictions = decision_tree_model.predict(x_test)

    return test_predictions


if __name__ == "__main__":
    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    encoded_values = get_encoded(powerpredict_df)

    x = encoded_values.drop("power_consumption", axis=1)
    y = encoded_values["power_consumption"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    X_test = powerpredict_df.drop(columns=["power_consumption"])
    actual_values = encoded_values["power_consumption"]

    y_test = powerpredict_df["power_consumption"]

    # best_epochs()

    # model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
    #                        y_val.values.astype(float), 8)
   # y_predict = leader_board_predict_fn(X_test)
    # make now a prediction with k nearest neighbors
    # knn_predictions = train_evaluate_knn(x_train, y_train, x_val, y_val, x)
    # random_forest_predictions = train_evaluate_random_forest(x_train, y_train, x_val, y_val, x)
    # decision_tree_predictions = train_evaluate_decision_tree(x_train, y_train, x_val, y_val, x)
    # time neural network
    start = time.time()
    model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
                            y_val.values.astype(float), 8)
    end = time.time()
    print("Time to train neural network: ", end - start)
    y_predict = leader_board_predict_fn(X_test)

    #plot_predictions(y_test, y_predict)

    #plot_residuals(y_test, y_predict)