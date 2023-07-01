import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def neural_network(x_train, y_train, x_val, y_val, num_epochs):
    # Create a PyTorch dataset for training data
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))

    # Create data loaders for batch processing for training data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create a PyTorch dataset for validation data
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

    # Create data loaders for batch processing for validation data
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                targets = targets.view(-1, 1)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Training Loss={loss.item():.4f}, Validation Loss={val_loss:.4f}")

    return model



