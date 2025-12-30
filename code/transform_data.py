import torch
from torchvision import transforms
from PIL import Image


def get_data_ready(input_image_paths, target_image_paths):
    """
    Takes lists of input and target image paths and returns processed tensors.

    Args:
        input_image_paths (list): List of file paths to the input JPG images.
        target_image_paths (list): List of file paths to the target JPG images.

    Returns:
        tuple: (input_tensors, target_tensors)
            - input_tensors (torch.Tensor): Batch of processed input images.
            - target_tensors (torch.Tensor): Batch of processed target images.
    """
    # Basic check to ensure lists correspond
    assert len(input_image_paths) == len(target_image_paths), "Lists must have the same length"

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        # Normalize: (mean_R, mean_G, mean_B), (std_R, std_G, std_B)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensors = []
    target_tensors = []

    for in_path, tgt_path in zip(input_image_paths, target_image_paths):
        # Load images
        input_img = Image.open(in_path).convert("RGB")
        target_img = Image.open(tgt_path).convert("RGB")

        # Apply transforms
        input_tensors.append(transform(input_img))
        target_tensors.append(transform(target_img))

    # Stack list of tensors into a single batch tensor
    return torch.stack(input_tensors), torch.stack(target_tensors)