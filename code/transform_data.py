import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def get_mean():
    return [0.485, 0.456, 0.406]
def get_std():
    return [0.229, 0.224, 0.225]

def transform(size=(480, 480), mean=None, std=None):
    if mean is None:
        mean = get_mean()
    if std is None:
        std = get_std()

    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_data_ready(input_image_paths, device='cpu', target_size=(160, 160)):
    """
    Takes lists of input and target image paths and returns processed tensors.

    Args:
        input_image_paths (list): List of file paths to the input JPG images.
        device (torch.device): the device that the model will work on
        target_size (tuple): Desired spatial size (H, W) for the target tensors.

    Returns:
        tuple: (input_tensors, target_tensors)
            - input_tensors (torch.Tensor): Batch of processed input images.
    """

    # Define the transformation pipeline
    if(input_image_paths is [] or input_image_paths is None):
        return None


    input_transform = transform(size=target_size)


    input_tensors = []

    for in_path in input_image_paths:
        # Load images
        input_img = Image.open(in_path).convert("RGB")

        # Apply transforms
        input_tensors.append(input_transform(input_img))

    # Stack list of tensors into a single batch tensor
    return torch.stack(input_tensors).to(device)