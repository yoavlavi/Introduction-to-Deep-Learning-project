import torch
import torch.nn as nn
from torchinfo import summary


class Reshape(nn.Module):
    """
    Helper module to reshape tensors inside nn.Sequential.
    """

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


def get_model_summary(model, input_size=(3, 480, 480)):
    """
    Prints a summary of the model layers and output shapes.
    """
    print("-" * 50)
    print(f"{'Layer Type':<20} | {'Output Shape':<25}")
    print("-" * 50)

    x = torch.zeros(1, *input_size)  # Create dummy input

    for layer in model:
        x = layer(x)
        layer_name = layer.__class__.__name__
        print(f"{layer_name:<20} | {str(list(x.shape)):<25}")

    print("-" * 50)


def build_model(input_size=(3, 480, 480), config=None, activation_fn=nn.ReLU):
    """
    Generates a CNN model based on a configuration list.

    Args:
        input_size (tuple): (channels, height, width). Default: (3, 480, 480).
        config (list): List of tuples defining the architecture.
        activation_fn (class): The activation function class to use (default: nn.ReLU).

    Returns:
        nn.Sequential: The constructed model.
    """
    if config is None:
        # Default simple classification config if none provided
        config = [
            ("conv", 32, 3, 1, 1), ("maxpool", 2, 2),
            ("conv", 64, 3, 1, 1), ("maxpool", 2, 2),
            ("avgpool", (6, 6)),
            ("flatten",),
            ("fc", 512),
            ("fc_final", 13)
        ]

    layers = []
    
    # Track current dimensions
    current_c = input_size[0]
    current_h = input_size[1]
    current_w = input_size[2]
    
    # Track flattened size
    current_flat_features = 0

    for layer_def in config:
        layer_type = layer_def[0]

        if layer_type == "conv":
            # format: ("conv", out_c, k, s, p)
            out_c, k, s, p = layer_def[1:]
            layers.append(nn.Conv2d(current_c, out_c, k, s, p))
            layers.append(activation_fn(inplace=True))
            
            # Update dims: floor((h + 2*p - k) / s + 1)
            current_h = (current_h + 2*p - k) // s + 1
            current_w = (current_w + 2*p - k) // s + 1
            current_c = out_c
            
        elif layer_type == "maxpool":
            k, s = layer_def[1:]
            layers.append(nn.MaxPool2d(k, s))
            # Update dims
            current_h = (current_h - k) // s + 1
            current_w = (current_w - k) // s + 1
            
        elif layer_type == "avgpool":
            out_h, out_w = layer_def[1]
            layers.append(nn.AdaptiveAvgPool2d((out_h, out_w)))
            current_h = out_h
            current_w = out_w
            
        elif layer_type == "flatten":
            layers.append(nn.Flatten())
            current_flat_features = current_c * current_h * current_w
            
        elif layer_type == "fc":
            out_f = layer_def[1]
            # Use the calculated flat features
            layers.append(nn.Linear(current_flat_features, out_f))
            layers.append(activation_fn(inplace=True))
            layers.append(nn.Dropout(0.5))
            current_flat_features = out_f
            
        elif layer_type == "fc_final":
            out_f = layer_def[1]
            layers.append(nn.Linear(current_flat_features, out_f))
            current_flat_features = out_f

        elif layer_type == "upsample":
            scale, mode, align = layer_def[1:]
            layers.append(nn.Upsample(scale_factor=scale, mode=mode, align_corners=align))
            current_h = int(current_h * scale)
            current_w = int(current_w * scale)
            
        elif layer_type == "resize":
            h, w, mode, align = layer_def[1:]
            layers.append(nn.Upsample(size=(h, w), mode=mode, align_corners=align))
            current_h = h
            current_w = w
            
        elif layer_type == "unflatten":
            # (channels, height, width)
            shape = layer_def[1]
            layers.append(Reshape(shape))
            current_c, current_h, current_w = shape

    return nn.Sequential(*layers)

def model2():
    config = []
    config.append(("conv", 16, 3, 3, 1))
    config.append(("conv", 32, 3, 1, 1))
    config.append(("conv", 3, 3, 1, 1))
    model = build_model(input_size=(3, 480, 480), config=config)

    return model

def model4():
    """
    Builds an improved Autoencoder architecture:
    - Input: (3, 480, 480)
    - Encoder: Deep Conv layers increasing channels (3 -> 256)
    - Bottleneck: Compresses features to a latent vector
    - Decoder: Expands and gradually reduces channels back to image
    - Output: (3, 160, 160)
    """
    
    # Feature map size at bottleneck (before flatten)
    # 30x30 spatial dims * 256 channels
    enc_channels = 256
    flat_dim = enc_channels * 30 * 30 
    latent_dim = 1024 # Bottleneck size

    config = [
        # --- Encoder ---
        # Increase channels to capture features, downsample spatial dims
        # 1. (3, 480, 480) -> (32, 240, 240)
        ("conv", 32, 3, 2, 1), 
        
        # 2. (32, 240, 240) -> (64, 120, 120)
        ("conv", 64, 3, 2, 1),
        
        # 3. (64, 120, 120) -> (128, 60, 60)
        ("conv", 128, 3, 2, 1),
        
        # 4. (128, 60, 60) -> (256, 30, 30)
        ("conv", enc_channels, 3, 2, 1),
        
        # --- Bottleneck ---
        ("flatten",),
        
        # FC 1: Compression (Feature Map -> Latent Vector)
        ("fc", latent_dim),
        
        # FC 2: Expansion (Latent Vector -> Feature Map size)
        ("fc", flat_dim),
        
        # --- Decoder ---
        # Unflatten back to spatial tensor: (256, 30, 30)
        ("unflatten", (enc_channels, 30, 30)),
        
        # Upsample straight to target resolution (160x160)
        ("resize", 160, 160, 'bilinear', False),
        
        # Refine features after resizing (Gradually reduce channels)
        ("conv", 128, 3, 1, 1), # (256 -> 128)
        ("conv", 64, 3, 1, 1),  # (128 -> 64)
        
        # Final Output Layer (Map to 3 RGB channels)
        ("conv", 3, 3, 1, 1)    # (64 -> 3)
    ]

    model = build_model(input_size=(3, 480, 480), config=config)
    return model


def model3():
    """
    Builds the specific Autoencoder architecture requested:
    - Input: (3, 480, 480)
    - Encoder: 4 Conv layers (halving dimensions each time: 240, 120, 60, 30)
    - Bottleneck: Flatten -> FC -> FC -> Unflatten
    - Decoder: Upsample to 160 -> Conv
    """
    
    # Calculate flat size: 3 channels * 30 * 30
    config = [
        # --- Encoder ---
        # 1. (3,480,480) -> (3, 240, 240)
        # Using stride 2 to downsample
        ("conv", 3, 3, 2, 1), 
        
        # 2. (3, 240, 240) -> (3, 120, 120)
        ("conv", 3, 3, 2, 1),
        
        # 3. (3, 120, 120) -> (3, 60, 60)
        ("conv", 3, 3, 2, 1),
        
        # 4. (3, 60, 60) -> (3, 30, 30)
        ("conv", 3, 3, 2, 1),
        
        # --- Bottleneck ---
        ("flatten",),
        
        # FC 1: 2700 -> 2700
        ("fc", 2700),
        
        # FC 2: 2700 -> 2700
        ("fc", 2700),
        
        # --- Decoder ---
        # Unflatten: 2700 -> (3, 30, 30)
        ("unflatten", (3, 30, 30)),
        
        # Target: (3, 160, 160)
        # We need to upsample from 30 to 160 (factor of 5.333...)
        # Since 'resize' is more precise than 'upsample' for odd factors:
        ("resize", 160, 160, 'bilinear', False),
        
        # Final Conv to smooth features/set channels
        ("conv", 3, 3, 1, 1) 
    ]

    model = build_model(input_size=(3, 480, 480), config=config)
    return model

def model1():
    """
    Builds a model with:
    - Input: 1x3x480x480
    - 25 Convolutional Layers
    - Output: 1x3x160x160 (Fully Convolutional)
    """

    config = []

    # --- 25 Convolutional Layers ---

    # Layer 1: Downsample 480 -> 160 using stride 3
    # (480 + 2*1 - 3) / 3 + 1 = 160
    config.append(("conv", 16, 3, 3, 1))

    # Layers 2-24: Standard convolutions preserving spatial dims (160x160)
    for _ in range(23):
        config.append(("conv", 32, 3, 1, 1))

        # Layer 25: Final convolution mapping to 3 output channels
    # Still preserves 160x160 size
    config.append(("conv", 3, 3, 1, 1))

    # Build the model
    # Note: No 'fc' or 'flatten' layers used.
    model = build_model(input_size=(3, 480, 480), config=config)

    return model

# --- Example Usage ---
def model_exsample():
    print("--- All-In-One Config Model ---")

    # Input: (3, 480, 480)
    # Target Output: (3, 480, 480)
    # Using: conv, maxpool, avgpool, flatten, fc, fc_final, unflatten, upsample, resize

    all_in_one_config = [
        # 1. Feature Extraction (Encoder)
        ("conv", 16, 3, 1, 1),  # [3, 480, 480] -> [16, 480, 480]
        ("maxpool", 2, 2),  # -> [16, 240, 240]
        ("conv", 32, 3, 1, 1),  # -> [32, 240, 240]
        ("avgpool", (4, 4)),  # -> [32, 4, 4] (Adaptive Pool)

        # 2. Bottleneck (MLP)
        ("flatten",),  # -> [32*4*4] = [512]
        ("fc", 256),  # -> [256]
        ("fc_final", 512),  # -> [512] (Using fc_final as part of hidden bottleneck)

        # 3. Reconstruction (Decoder)
        # We need to manually reshape back to spatial dims: [32, 4, 4]
        ("unflatten", (32, 4, 4)),

        ("upsample", 4, 'nearest', None),  # -> [32, 16, 16] (Scale factor 4)
        ("resize", 240, 240, 'bilinear', False),  # -> [32, 240, 240]
        ("conv", 16, 3, 1, 1),  # -> [16, 240, 240]

        ("upsample", 2, 'bilinear', False),  # -> [16, 480, 480]
        ("conv", 3, 1, 1, 0)  # -> [3, 480, 480]
    ]
    model = build_model(config=all_in_one_config)

    x = torch.randn(1, 3, 480, 480)
    y = model(x)

    print("model description: \n")
    get_model_summary(model, x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print("\nConfiguration includes:")
    print("- conv, maxpool, avgpool (Encoder)")
    print("- flatten, fc, fc_final (Bottleneck)")
    print("- unflatten, upsample, resize (Decoder)")