import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

def train(model, X_train, y_train, num_epochs=50, lr=1e-4, loss_fn=None, optimizer=None):
    """
    Trains a neural network model, evaluating on test data periodically to save the best version.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        num_epochs (int): Total number of training epochs. Default is 500.
        lr (float): Learning rate for the optimizer. Default is 1e-2.
        loss_fn (nn.Module, optional): Loss function. Defaults to MSELoss if None.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam if None.

    Returns:
        tuple: (final_model, train_losses)
            - final_model (nn.Module): The model state after the last epoch.
            - train_losses (list): History of training losses per epoch.
    """
    # Set default loss function to MSELoss (better for image-to-image/regression)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize tracking variables
    train_losses = []

    # Data Type Safety Checks
    # For MSELoss, both input and target must be Float tensors
    if X_train is not None and not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if y_train is not None and not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set to training mode

        if X_train is not None:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_train)
            # Simple shape check to prevent silent broadcasting bugs with MSE
            if outputs.shape != y_train.shape:
                print(f"Warning: Output shape {outputs.shape} mismatch with Target {y_train.shape}")
                # We attempt to proceed, as some broadcasting might be intentional, but usually it's not.

            loss = loss_fn(outputs, y_train)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        else:
            break
    
    return model, train_losses