#from pydantic.experimental.pipeline import transform

import model_architecture as ma
from model_training import train
import plotting_images as plot
from torchvision import transforms
import  transform_data
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

def get_all_files_in_directories(directory_list):
    """
        Collects full paths of all files within the given list of directories.

        Args:
            directory_list (list): List of directory paths (strings).

        Returns:
            list: A list containing the full paths of all files found.
        """
    all_files = []

    for directory in directory_list:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    return all_files

def get_games_directories(games_list):
    tagges_list, generated_list = [], []
    for game in games_list:
        tagges_list.append(r"data/game$_per_frame/tagged_images".replace('$', str(game)))
        generated_list.append(r"data/game$_per_frame/generated_images".replace('$', str(game)))
    return tagges_list, generated_list


def load_model_from_file(model, version):
    """
    Loads a PyTorch model's state dictionary from the specified file.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filepath (str): The full path including filename to load the model from.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    filepath = "models/model_" + version
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return model

    try:
        model.load_state_dict(torch.load(filepath))
        print(f"Model loaded successfully from: {filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")

    return model
def save_model_to_file(model, version):
    """
    Saves the PyTorch model's state dictionary to the specified file.

    Args:
        model (torch.nn.Module): The model to save.
        version (str): The version of the model (an id).
    """

    # Create directory if it doesn't exist
    filepath = "models/model_" + version
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        torch.save(model.state_dict(), filepath)
        print(f"Model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("build model")
    model = ma.model1()
    model.to(device)
    print("load loss function")
    loss_fn = nn.MSELoss()
    print("load optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print("load file list")
    train_games_x, train_games_y = get_games_directories([2])
    test_games_x, test_games_y = get_games_directories([])
    print("start load training data")
    X_train, y_train = transform_data.get_data_ready(
        get_all_files_in_directories(train_games_x),
        get_all_files_in_directories(train_games_y), device)
    X_test, y_test = None, None
    try:
        print("start load test data")
        X_test, y_test = transform_data.get_data_ready(
            get_all_files_in_directories(test_games_x),
            get_all_files_in_directories(test_games_y), device)
    except Exception as e:
        print("Error: ", e)
    print(torch.cuda.memory_allocated(0))
    print("start train model")
    model, best_model, train_losses = train(model=model, X_train=X_train,
                                            y_train=y_train, X_test=X_test,
                                            y_test=y_test,loss_fn=loss_fn,
                                            optimizer=optimizer, num_epochs=1000)

    plot.plot_loss(train_losses)


    #plot.plot_image_pairs_with_text()
    save_model_to_file(best_model, 'v1.01')
    save_model_to_file(model, 'v1.0')