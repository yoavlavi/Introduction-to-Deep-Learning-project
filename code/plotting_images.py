import matplotlib.pyplot as plt
import torch
from PIL import Image
from transform_data import transform

def plot_loss(losses, title="Training loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

def visualize_model_output(model, image_path, device='cpu'):
    """
    Loads an image, runs it through the model, and plots the original vs output.

    Args:
        model (torch.nn.Module): Trained model.
        image_path (str): Relative path to the input image.
        device (str or torch.device): Device to run inference on.
    """
    model.eval()
#    model.to(device)
    # Load and Preprocess
    img = Image.open(image_path).convert("RGB")
    
    # Same stats as used in training
    mean = [0.5,0.5,0.5]#[0.485, 0.456, 0.406]
    std = [0.5,0.5,0.5] #[0.229, 0.224, 0.225]

    preprocess = transform(size=(480, 480))


    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Denormalize for visualization
    def denormalize(tensor, std, mean):
        # clone to avoid modifying original tensor in-place
        t = tensor.clone().detach().cpu().squeeze(0)
        for i in range(3):
            t[i] = t[i] * std[i] + mean[i]
        return t.permute(1, 2, 0).clamp(0, 1).numpy()

    original_disp = denormalize(input_tensor, std, mean)
    output_disp = denormalize(output_tensor, [1,1,1], [0,0,0])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_disp)
    axes[0].set_title("Original Input")
    axes[0].axis("off")

    axes[1].imshow(output_disp)
    axes[1].set_title("Model Output")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()