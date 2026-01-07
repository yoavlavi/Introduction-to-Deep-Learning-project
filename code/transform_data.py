import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def transform(size=(480, 480), mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_data_ready(input_image_paths, target_image_paths, device, target_size=(160, 160)):
    """
    Takes lists of input and target image paths and returns processed tensors.

    Args:
        input_image_paths (list): List of file paths to the input JPG images.
        target_image_paths (list): List of file paths to the target JPG images.
        device (torch.device): the device that the model will work on
        target_size (tuple): Desired spatial size (H, W) for the target tensors.

    Returns:
        tuple: (input_tensors, target_tensors)
            - input_tensors (torch.Tensor): Batch of processed input images.
            - target_tensors (torch.Tensor): Batch of processed target images.
    """
    # Basic check to ensure lists correspond
    assert len(input_image_paths) == len(target_image_paths), "Lists must have the same length"

    # Define the transformation pipeline
    if(input_image_paths is [] or target_image_paths is []):
        return None, None
    print("input images: ")
    print(input_image_paths)


    input_transform = transform(size=(480,480))

    target_transform = transform(target_size)

    input_tensors = []
    target_tensors = []

    for in_path, tgt_path in tqdm(zip(input_image_paths, target_image_paths)):
        # Load images
        input_img = Image.open(in_path).convert("RGB")
        target_img = Image.open(tgt_path).convert("RGB")

        # Apply transforms
        input_tensors.append(input_transform(input_img))
        target_tensors.append(target_transform(target_img))

    # Stack list of tensors into a single batch tensor
    return torch.stack(input_tensors).to(device), torch.stack(target_tensors).to(device)