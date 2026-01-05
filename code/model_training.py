import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

def train(model, X_train, y_train, X_test, y_test, num_epochs=500, lr=1e-2, loss_fn=None, optimizer=None):
    """
    Trains a neural network model, evaluating on test data periodically to save the best version.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        X_test (torch.Tensor): Testing input data.
        y_test (torch.Tensor): Testing target data.
        num_epochs (int): Total number of training epochs. Default is 500.
        lr (float): Learning rate for the optimizer. Default is 1e-2.
        loss_fn (nn.Module, optional): Loss function. Defaults to MSELoss if None.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam if None.

    Returns:
        tuple: (final_model, best_model, train_losses)
            - final_model (nn.Module): The model state after the last epoch.
            - best_model (nn.Module): The model state with the lowest test loss.
            - train_losses (list): History of training losses per epoch.
    """
    # Set default loss function to MSELoss (better for image-to-image/regression)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize tracking variables
    train_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    min_test_loss = float('inf')

    # Data Type Safety Checks
    # For MSELoss, both input and target must be Float tensors
    if X_train is not None and not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if y_train is not None and not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    if X_test is not None and not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if y_test is not None and not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    # Training Loop
    for epoch in tqdm(range(num_epochs)):
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

        # Periodic Evaluation every 100 epochs
        if (epoch + 1) % 100 == 0 and X_test is not None and y_test is not None:
            model.eval()  # Set to evaluation mode
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = loss_fn(test_outputs, y_test).item()

                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

                # Check for improvement
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"  -> New best model found! (Loss: {min_test_loss:.4f})")

            model.train()  # Return to training mode

    # Create a separate instance for the best model
    best_model = copy.deepcopy(model)
    if min_test_loss != float('inf'):
        best_model.load_state_dict(best_model_wts)
        print(f"Training complete. Best Test Loss: {min_test_loss:.4f}")
    else:
        print("Training complete. No test phase or improvement recorded.")

    return model, best_model, train_losses