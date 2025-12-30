import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


def plot_image_pairs_with_text(image_pairs, texts, denormalize=None):
    """
    Plots a list of image pairs side-by-side with a title above each pair.

    Args:
        image_pairs (list of tuple): List where each element is a tuple (img1, img2).
                                     Images can be file paths (str), PIL Images, or PyTorch Tensors.
        texts (list of str): List of titles corresponding to each pair.
        denormalize (callable, optional): Function to denormalize tensor images before plotting.
                                          Useful if images were normalized (e.g. ImageNet stats).
    """
    if len(image_pairs) != len(texts):
        raise ValueError("Length of image_pairs and texts must match.")

    n = len(image_pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    # Handle the case of a single pair (axes is 1D array)
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img1, img2 = image_pairs[i]
        title = texts[i]

        # Helper to process image for display
        def prepare_image(img):
            if isinstance(img, str):
                return Image.open(img).convert("RGB")
            elif isinstance(img, torch.Tensor):
                if denormalize:
                    img = denormalize(img)
                # Clone to avoid modifying original, detach from graph, move to cpu
                img = img.clone().detach().cpu()
                # Convert (C, H, W) -> (H, W, C) for matplotlib
                if img.dim() == 3 and img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0)
                # Squeeze single channel dims (e.g. grayscale 1x28x28 -> 28x28)
                return img.squeeze().numpy()
            elif isinstance(img, np.ndarray):
                return img
            else:
                return img  # Assume PIL image or compatible

        disp_img1 = prepare_image(img1)
        disp_img2 = prepare_image(img2)

        # Plot first image
        axes[i, 0].imshow(disp_img1)
        axes[i, 0].axis('off')

        # Plot second image
        axes[i, 1].imshow(disp_img2)
        axes[i, 1].axis('off')

        # Set title for the pair centered over both columns
        # We use the left axis to set the title but position it to cover both
        axes[i, 0].set_title(title, x=1.1, y=1.05, ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()