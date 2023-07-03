import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from nn import neural_network

# init model globally
model = [None, None, None]


def load_model(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    else:
        return None


def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


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
    powerpredict = powerpredict.dropna()
    # remove colums which start with _main and _description
    powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_main')]
    powerpredict = powerpredict.loc[:, ~powerpredict.columns.str.startswith('_description')]

    categorical_cols = powerpredict.select_dtypes(
        include=['object']).columns  # get columns where we have to encode the data

    encoded_df = pd.get_dummies(powerpredict, columns=categorical_cols)
    encoded_df = encoded_df.fillna(encoded_df.mean())  # drop rows with missing values

    return encoded_df


def leader_board_predict_fn(values, model_type):
    x_values_encoded = get_encoded(values)
    x_values_encoded = x_values_encoded.reindex(columns=x_train.columns, fill_value=0)

    x_values_encoded = x_values_encoded.astype(float)

    values_tensor = torch.Tensor(x_values_encoded.values)

    if model_type == 'Neural Network':
        predictions = model[2](values_tensor).detach().numpy()
    elif model_type == 'Random Forest':
        predictions = model[1].predict(x_values_encoded)
    elif model_type == 'Linear Regression':
        predictions = model[0].predict(x_values_encoded)
    else:
        raise ValueError("Invalid model type specified.")

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


def plot_residuals(y_test, y_predict_rf, y_predict_nn):
    y_test = y_test.values.flatten()  # Flatten y_test to make it 1-dimensional
    residuals_rf = y_test - y_predict_rf.flatten()  # Calculate residuals for Random Forest model
    residuals_nn = y_test - y_predict_nn.flatten()  # Calculate residuals for Neural Network model

    noise_rf = np.random.normal(0, 0.1, len(residuals_rf))  # Generate random noise for Random Forest
    noise_nn = np.random.normal(0, 0.1, len(residuals_nn))  # Generate random noise for Neural Network

    plt.scatter(y_test, residuals_rf + noise_rf, label='Random Forest')
    plt.scatter(y_test, residuals_nn + noise_nn, label='Neural Network')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Actual Power Consumption')
    plt.ylabel('Residuals')
    plt.title('Residuals on Neural Network and Random Forest')
    plt.legend()
    plt.show()


def plot_decision_tree(random_forest_model):
    # Extract a single decision tree from the Random Forest model
    decision_tree = random_forest_model.estimators_[0]

    # Plot the decision tree
    plt.figure(figsize=(10, 10))
    tree.plot_tree(decision_tree, filled=True)
    plt.show()


def plot_comparison():
    models = ['Linear Regression', 'Random Forest', 'Neural Network']
    mae_scores = [3094.9474070944225, 1681.2182666666665,
                  3530.0693807373045]  # Replace with actual MAE scores for each model
    plt.bar(models, mae_scores)
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Comparison based on MAE')
    plt.show()


def tune_hyperparameters(x_train, y_train, x_val, y_val, num_epochs):
    dropout_rates = [0.2, 0.4, 0.6]  # Adjust the dropout rates to be tested
    weight_decays = [0.0001, 0.001, 0.01]  # Adjust the weight decays to be tested

    best_model = None
    best_loss = float('inf')
    best_dropout_rate = None
    best_weight_decay = None

    for dropout_rate in dropout_rates:
        for weight_decay in weight_decays:
            # Train the model with the current hyperparameters
            model[2] = neural_network(x_train.values.astype(float), y_train.values.astype(float),
                                      x_val.values.astype(float),
                                      y_val.values.astype(float), num_epochs, dropout_rate, weight_decay)

            # Evaluate the model on the validation set
            val_predictions = leader_board_predict_fn(x_val, 'Neural Network')
            val_loss = mean_absolute_error(y_val, val_predictions)

            # Check if the current model performs better than the previous best model
            if val_loss < best_loss:
                best_model = model
                best_loss = val_loss
                best_dropout_rate = dropout_rate
                best_weight_decay = weight_decay

    return best_model, best_dropout_rate, best_weight_decay


if __name__ == "__main__":
    DATASET_PATH = "."
    powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
    encoded_values = get_encoded(powerpredict_df)

    x = encoded_values.drop("power_consumption", axis=1)
    y = encoded_values["power_consumption"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    # For the HYPERPARAMETER TUNING
    # num_epochs = 10
    # best_model, best_dropout_rate, best_weight_decay = tune_hyperparameters(x_train, y_train, x_val, y_val, num_epochs)
    # print("Best Dropout Rate:", best_dropout_rate)
    # print("Best Weight Decay:", best_weight_decay)

    # Train the final model with the best hyperparameters
    # final_model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
    #                         y_val.values.astype(float), num_epochs, best_dropout_rate, best_weight_decay)

    X_test = powerpredict_df.drop(columns=["power_consumption"])
    actual_values = encoded_values["power_consumption"]

    y_test = powerpredict_df["power_consumption"]

    # best_estimators()

    # best_epochs()

    linear_regression_model_path = "linear_regression_model.pkl"
    model[0] = load_model(linear_regression_model_path)
    if model[0] is None:
        model[0] = train_evaluate_linear_regression(x_train, y_train)
        save_model(model[0], linear_regression_model_path)

    # Load or create the Random Forest model
    random_forest_model_path = "random_forest_model.pkl"
    model[1] = load_model(random_forest_model_path)
    if model[1] is None:
        model[1] = train_evaluate_random_forest(x_train, y_train, 8)
        save_model(model[1], random_forest_model_path)

    # Load or create the Neural Network model
    neural_network_model_path = "neural_network_model.pkl"
    model[2] = load_model(neural_network_model_path)
    if model[2] is None:
        model[2] = neural_network(x_train.values.astype(float), y_train.values.astype(float),
                                  x_val.values.astype(float),
                                  y_val.values.astype(float), 10, 0.2, 0.001)
        save_model(model[2], neural_network_model_path)

    y_predict_lr = leader_board_predict_fn(x_val, 'Linear Regression')
    y_predict_rf = leader_board_predict_fn(x_val, 'Random Forest')
    y_predict_nn = leader_board_predict_fn(x_val, 'Neural Network')

    print("Linear Regression MAE: ", mean_absolute_error(y_val, y_predict_lr))
    print("Random Forest MAE: ", mean_absolute_error(y_val, y_predict_rf))
    print("Neural Network MAE: ", mean_absolute_error(y_val, y_predict_nn))

    # plot_residuals(y_val, y_predict_rf, y_predict_nn)

    # make a random forest plot
    # plot_decision_tree(model)

    # Plot the model comparison
    # plot_comparison()

    # plot_residual(y_test, y_predict)

    ################################################## Testing purposes
    # time neural network
    # start = time.time()
    # model = neural_network(x_train.values.astype(float), y_train.values.astype(float), x_val.values.astype(float),
    #                        y_val.values.astype(float), 8)
    # end = time.time()
    # print("Time to train neural network: ", end - start)
