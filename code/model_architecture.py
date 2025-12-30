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

    # We need to track the current number of channels/features to build layers
    current_channels = input_size[0]

    # Let's track 'current_flat_features' for FC layers
    current_flat_features = 0

    for layer_def in config:
        layer_type = layer_def[0]

        if layer_type == "conv":
            out_c, k, s, p = layer_def[1:]
            layers.append(nn.Conv2d(current_channels, out_c, k, s, p))
            layers.append(activation_fn(inplace=True))
            current_channels = out_c

        elif layer_type == "maxpool":
            k, s = layer_def[1:]
            layers.append(nn.MaxPool2d(k, s))

        elif layer_type == "avgpool":
            out_h, out_w = layer_def[1]
            layers.append(nn.AdaptiveAvgPool2d((out_h, out_w)))
            # After pooling, the spatial dim is fixed.
            current_flat_features = current_channels * out_h * out_w

        elif layer_type == "flatten":
            layers.append(nn.Flatten())

        elif layer_type == "fc":
            out_f = layer_def[1]
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

        elif layer_type == "resize":
            h, w, mode, align = layer_def[1:]
            layers.append(nn.Upsample(size=(h, w), mode=mode, align_corners=align))

        elif layer_type == "unflatten":
            # (channels, height, width)
            shape = layer_def[1]
            layers.append(Reshape(shape))
            current_channels = shape[0]  # Update channels for subsequent convs

    return nn.Sequential(*layers)


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