#to run remotly use: tmux

import torch
import torch.cuda
from torchvision import transforms
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import model_architecture as ma
from model_training import train
import plotting_images as plot
import transform_data


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
            directory = "../" + directory
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

def get_model_by_number(model_number):
    method = getattr(ma, f"model{model_number}")
    model = method()
    return model

def load_model_from_file(model_number, version):
    """
    Loads a PyTorch model's state dictionary from the specified file.

    Args:
        model_number (int):the model type to load him
        filepath (str): The full path including filename to load the model from.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    model = get_model_by_number(model_number)
    filepath = "models/model_" + str(model_number) + '/' + version
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return model

    try:
        model.load_state_dict(torch.load(filepath))
        print(f"Model loaded successfully from: {filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")

    return model
def save_model_to_file(model_number, model, version):
    """
    Saves the PyTorch model's state dictionary to the specified file.

    Args:
        model_number (int): the model type to save
        model (torch.nn.Module): The model to save.
        version (str): The version of the model (an id).
    """

    # Create directory if it doesn't exist
    filepath = "models/model_" + str(model_number) + '/v' + version
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        torch.save(model.state_dict(), filepath)
        print(f"Model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    #current best lr=1e-4, epoches=70

    model = get_model_by_number(2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    model.to(device)
    lr=1e-4
    optimizer =optim.SGD(model.parameters(), lr=lr)
    loss_fn=nn.L1Loss()

    train_games = [2,4,5]

    train_games_x, train_games_y = get_games_directories(train_games)

    print("start load training data")
    X_train, y_train = transform_data.get_data_ready(
                       get_all_files_in_directories(train_games_x),
                       get_all_files_in_directories(train_games_y), 
                       device)

    print("lr=1e-4")
    for i in range(20):
        print("Training for ", (i+1)*5, " epochs")
        model, train_losses = train(model, X_train, y_train, num_epochs=5, lr=1e-4,)
        #ma.get_model_summary(model)
        #model = load_model_from_file(1, '1.1')
        plot.visualize_model_output(model, #load_model_from_file(1, 'v1.0'),
                                        "../data/game2_per_frame/tagged_images/frame_000200.jpg", device=device)

