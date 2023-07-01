import pandas as pd
import os
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

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
    # powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_main')]
    # powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_description')]

    categorical_cols = powerpredict.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict, columns=categorical_cols)
    encoded_df = encoded_df.fillna(encoded_df.mean())  # drop rows with missing values

    return encoded_df


def leader_board_predict_fn(values):
    x_values_encoded = get_encoded(values)
    x_values_encoded = x_values_encoded.reindex(columns=x_train.columns, fill_value=0)

    # Convert values_encoded to float type
    x_values_encoded = x_values_encoded.astype(float)

    values_tensor = torch.Tensor(x_values_encoded.values)
    # Uncomment for Neural Network prediction
    # predictions = model(values_tensor).detach().numpy()
    # Uncomment for Random Forest prediction
    predictions = model.predict(x_values_encoded)

    return predictions


def best_epochs():
    global model
    num_epochs = 20
    epoch_losses = []
    for epoch in range(1, num_epochs):
        model = neural_network(x_train.values.astype(float), y_train.values.astype(float),
                               x_val.values.astype(float), y_val.values.astype(float), epoch)
        y_predict = leader_board_predict_fn(y_val)
        epoch_mae = mean_absolute_error(y_test, y_predict)
        print(epoch, "Mean Absolute Error (MAE):", epoch_mae)
        epoch_losses.append(epoch_mae)

    # Create a chart
    plt.plot(range(1, num_epochs), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance across Epochs')
    plt.show()


def best_estimators():
    num_estimators = [2, 4, 8, 10, 15, 20, 40, 50]  # Adjust the list of estimators as needed
    mae_scores = []

    for n_estimators in num_estimators:
        # Train the Random Forest model
        random_forest_model = train_evaluate_random_forest(x_train, y_train, n_estimators)

        # Evaluate the model on the validation set
        val_predictions = random_forest_model.predict(x_val)
        print("Validation Mean Absolute Error (MAE) for Random Forest:", mean_absolute_error(y_val, val_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        mae_scores.append(val_mae)

    # Create a chart
    plt.plot(num_estimators, mae_scores, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance across Different Number of Estimators')
    plt.show()


def train_evaluate_linear_regression(x, y):
    # Train the Random Forest model
    lr_model = LinearRegression()
    lr_model.fit(x, y)

    return lr_model


def train_evaluate_random_forest(x_train, y_train, n_estimators):
    # Train the Random Forest model
    random_forest_model = RandomForestRegressor(n_estimators=n_estimators,
                                                random_state=42)  # You can adjust the number of estimators as needed
    random_forest_model.fit(x_train, y_train)

    return random_forest_model


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
    plt.title('Residual Plot Neural Network')
    plt.show()


def plot_decision_tree(random_forest_model):
    # Extract a single decision tree from the Random Forest model
    decision_tree = random_forest_model.estimators_[0]

    # Plot the decision tree
    plt.figure(figsize=(10, 10))
    tree.plot_tree(decision_tree, filled=True)
    plt.show()


def plot_comparison(models, mae_scores):
    plt.bar(models, mae_scores)
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Comparison based on MAE')
    plt.show()


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

    # best_estimators()

    # best_epochs()

    # model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
    #                         y_val.values.astype(float), 20)
    model = train_evaluate_linear_regression(x_train, y_train)

    y_predict = leader_board_predict_fn(x_val)

    # print mae
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_val, y_predict))

    # make a random forest plot
    # plot_decision_tree(model)

    models = ['Linear Regression', 'Random Forest', 'Neural Network']
    mae_scores = [3094.9474070944225, 1681.2182666666665,
                  3530.0693807373045]  # Replace with actual MAE scores for each model

    # Plot the model comparison
    plot_comparison(models, mae_scores)

    # plot_residual(y_test, y_predict)

    ################################################## Testing purposes
    # time neural network
    # start = time.time()
    # model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
    #                        y_val.values.astype(float), 8)
    # end = time.time()
    # print("Time to train neural network: ", end - start)
