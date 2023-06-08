import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def neural_network(x_train, y_train):
    # Create a PyTorch dataset
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))

    # Create data loaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the architecture of the neural network
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)  # reshape the targets to match the shape of outputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model


