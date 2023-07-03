import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def neural_network(x_train, y_train, x_val, y_val, num_epochs, best_dropout_rate, best_weight_decay):
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the architecture of the neural network
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(best_dropout_rate),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(best_dropout_rate),
        nn.Linear(32, 1)
    )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=best_weight_decay)

    # Lists to store losses
    train_losses = []
    val_losses = []

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
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        #print(f"Epoch {epoch+1}: Training Loss={loss.item():.4f}, Validation Loss={val_loss:.4f}")

    # Plotting
    epochs = range(1, num_epochs + 1)
    # plt.plot(epochs, train_losses, label='Training Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()

    return model
