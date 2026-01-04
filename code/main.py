from pydantic.experimental.pipeline import transform

import model_architecture as ma
from model_training import train
import plotting_images as plot
import  transform_data as transform
import torch
import torch.nn as nn
import torch.optim as optim
import os

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
        tagges_list.append("data/game$_per_frame/tagged_images".replace('$', str(game)))
        generated_list.append("data/game$_per_frame/generated_images".replace('$', str(game)))
    return tagges_list, generated_list


def save_model_to_file(model, version):
    """
    Saves the PyTorch model's state dictionary to the specified file.

    Args:
        model (torch.nn.Module): The model to save.
        version (str): The version of the model (an id).
    """

    # Create directory if it doesn't exist
    filepath = "model/model_" + version
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        torch.save(model.state_dict(), filepath)
        print(f"Model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    model = ma.model1()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train_games_x, train_games_y = get_games_directories([2,4,5,6])
    test_games_x, test_games_y = get_games_directories([7])
    X_train, y_train = transform.get_data_ready(
        get_all_files_in_directories(train_games_x),
        get_all_files_in_directories(train_games_y))
    X_test, y_test = transform.get_data_ready(
        get_all_files_in_directories(test_games_x),
        get_all_files_in_directories(test_games_y))


    model, best_model, train_losses = train(model=model, X_train=X_train,
                                            y_train=y_train, X_test=X_test,
                                            y_test=y_test,loss_fn=loss_fn,
                                            optimizer=optimizer, num_epochs=10000)

    plot.plot_loss(train_losses)
    save_model_to_file(best_model, 'v1')