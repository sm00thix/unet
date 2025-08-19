"""
PyTorch Hub configuration for U-Net implementation.
Based on U-Net: Convolutional Networks for Biomedical Image Segmentation
by Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597

Author: Ole-Christian Galbo Engstrøm
Email: ocge@foss.dk
"""

dependencies = ["torch"]

# Import the U-Net implementation
from unet import UNet as _UNet


def unet(
    pretrained=False,
    in_channels=3,
    out_channels=1,
    pad=True,
    bilinear=True,
    normalization=None,
    **kwargs,
):
    """
    U-Net model for semantic segmentation

    This implementation follows the original U-Net architecture with options for
    different normalization techniques, upsampling methods, and padding strategies.

    Args:
        pretrained (bool): If True, returns a model pre-trained on a dataset (not yet available)
        in_channels (int): Number of input channels (default: 3 for RGB images)
        out_channels (int): Number of output channels/classes (default: 1 for binary segmentation)
        pad (bool): If True, the input size is preserved by zero-padding convolutions and, if necessary, the results of the upsampling operations.
                    If False, output size will be reduced compared to input size (default: True)
        bilinear (bool): If True, use bilinear upsampling. If False, use transposed convolution (default: True)
        normalization (None | str): Normalization type. Options:
                                   - None: No normalization
                                   - 'bn': Batch normalization
                                   - 'ln': Layer normalization
                                   (default: None)
        **kwargs: Additional arguments (currently unused but available for future extensions)

    Returns:
        torch.nn.Module: U-Net model with intermediate channels [64, 128, 256, 512, 1024]

    Example:
        >>> import torch
        >>>
        >>> # Basic U-Net for binary segmentation (e.g., medical imaging)
        >>> model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False)
        >>>
        >>> # Multi-class segmentation (e.g., 21 classes for PASCAL VOC)
        >>> model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, out_channels=21)
        >>>
        >>> # U-Net with batch normalization
        >>> model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, normalization='bn')
        >>>
        >>> # U-Net with transposed convolution upsampling instead of bilinear interpolation
        >>> model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, bilinear=False)
        >>>
        >>> # Grayscale input (e.g., medical images, satellite imagery)
        >>> model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, in_channels=1)
        >>>
        >>> # Forward pass
        >>> x = torch.randn(1, 3, 256, 256)  # (batch, channels, height, width)
        >>> with torch.no_grad():
        ...     output = model(x)
        >>> print(f"Input shape: {x.shape}")
        >>> print(f"Output shape: {output.shape}")  # (1, out_channels, 256, 256) if pad=True

    Note:
        - The model uses intermediate channels [64, 128, 256, 512, 1024] following the original paper
        - When pad=True, output spatial dimensions are identical to input spatial dimensions
        - When pad=False, output will be smaller than input due to valid convolutions and potential dropping of rows/columns in the strided pooling layers
        - Bilinear upsampling uses fewer parameters than transposed convolution and avoids checkerboard artifacts
        - Normalization can be set to 'bn' for batch normalization or 'ln' for layer normalization
    """

    # Create model with specified parameters
    model = _UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        pad=pad,
        bilinear=bilinear,
        normalization=normalization,
    )

    if pretrained:
        raise NotImplementedError(
            "Pretrained weights are not yet available. "
            "The model will be initialized with random weights using Kaiming normal initialization. "
            "Please train the model on your specific dataset for optimal performance."
        )

    return model


def unet_bn(pretrained=False, in_channels=3, out_channels=1, **kwargs):
    """
    U-Net model with Batch Normalization

    Batch Normalization can be beneficial for training stability when using larger batch sizes.

    Args:
        pretrained (bool): If True, returns a model pre-trained on a dataset (not yet available)
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        **kwargs: Additional arguments passed to the base unet function

    Returns:
        torch.nn.Module: U-Net model with batch normalization

    Example:
        >>> model = torch.hub.load('sm00thix/unet', 'unet_bn', pretrained=False)
        >>> # Equivalent to: unet(normalization='bn')
    """
    return unet(
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        normalization="bn",
        **kwargs,
    )


def unet_ln(pretrained=False, in_channels=3, out_channels=1, **kwargs):
    """
    U-Net model with Layer Normalization

    Layer normalization can be beneficial when batch sizes are small.

    Args:
        pretrained (bool): If True, returns a model pre-trained on a dataset (not yet available)
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        **kwargs: Additional arguments passed to the base unet function

    Returns:
        torch.nn.Module: U-Net model with layer normalization

    Example:
        >>> model = torch.hub.load('sm00thix/unet', 'unet_ln', pretrained=False)
        >>> # Equivalent to: unet(normalization='ln')
    """
    return unet(
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        normalization="ln",
        **kwargs,
    )


def unet_medical(pretrained=False, **kwargs):
    """
    U-Net model configured for medical image segmentation

    Configured with grayscale input (typical for medical images) and binary output (e.g., organ/background segmentation).

    Args:
        pretrained (bool): If True, returns a model pre-trained on a dataset (not yet available)
        **kwargs: Additional arguments passed to the base unet function

    Returns:
        torch.nn.Module: U-Net model optimized for medical imaging

    Example:
        >>> model = torch.hub.load('sm00thix/unet', 'unet_medical', pretrained=False)
        >>> # Single channel input, batch normalization
        >>> x = torch.randn(1, 1, 512, 512)  # Typical medical image size
        >>> output = model(x)
    """
    return unet(pretrained=pretrained, in_channels=1, out_channels=1, **kwargs)


def unet_transconv(pretrained=False, in_channels=3, out_channels=1, **kwargs):
    """
    U-Net model using transposed convolution for upsampling

    Uses transposed convolution instead of bilinear upsampling.

    Args:
        pretrained (bool): If True, returns a model pre-trained on a dataset (not yet available)
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        **kwargs: Additional arguments passed to the base unet function

    Returns:
        torch.nn.Module: U-Net model with transposed convolution upsampling

    Example:
        >>> model = torch.hub.load('sm00thix/unet', 'unet_transconv', pretrained=False)
        >>> # Equivalent to: unet(bilinear=False)
    """
    return unet(
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        bilinear=False,  # Use transposed convolution
        **kwargs,
    )


def model_info(model_name="unet"):
    """
    Get detailed information about a specific model variant

    Args:
        model_name (str): Name of the model variant

    Returns:
        dict: Detailed information about the model

    Example:
        >>> info = torch.hub.load('sm00thix/unet', 'model_info', model_name='unet_bn')
        >>> print(info['description'])
    """
    info_dict = {
        "unet": {
            "description": "Standard U-Net with configurable parameters",
            "intermediate_channels": [64, 128, 256, 512, 1024],
            "default_params": {
                "in_channels": 3,
                "out_channels": 1,
                "pad": True,
                "bilinear": True,
                "normalization": None,
            },
            "use_cases": ["General semantic segmentation", "Custom configuration"],
        },
        "unet_bn": {
            "description": "U-Net with Batch Normalization",
            "intermediate_channels": [64, 128, 256, 512, 1024],
            "default_params": {"normalization": "bn"},
            "use_cases": ["May stabilize training with large batch sizes"],
        },
        "unet_ln": {
            "description": "U-Net with Layer Normalization",
            "intermediate_channels": [64, 128, 256, 512, 1024],
            "default_params": {"normalization": "ln"},
            "use_cases": ["May stabilize training with small batch sizes"],
        },
        "unet_medical": {
            "description": "U-Net optimized for medical image segmentation",
            "intermediate_channels": [64, 128, 256, 512, 1024],
            "default_params": {
                "in_channels": 1,
                "out_channels": 1,
            },
            "use_cases": ["Medical image segmentation", "Grayscale images"],
        },
        "unet_transconv": {
            "description": "U-Net with transposed convolution upsampling",
            "intermediate_channels": [64, 128, 256, 512, 1024],
            "default_params": {"bilinear": False},
            "use_cases": [
                "Implemented for consistency with the original U-Net paper",
                "Typically not recommended as it may introduce checkerboard artifacts",
            ],
        },
    }

    return info_dict.get(
        model_name,
        f"Model '{model_name}' not found. Available models: {list(info_dict.keys())}",
    )


# Example usage for documentation
_EXAMPLE_USAGE = """
# Load and use U-Net models
import torch

# Basic usage
model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False)
print(f"Model loaded: {model.__class__.__name__}")

# Multi-class segmentation example
model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, out_channels=21)  # PASCAL VOC classes

# Medical imaging example
model = torch.hub.load('sm00thix/unet', 'unet_medical', pretrained=False)

# Original U-Net with transposed convolution upsampling and no padding
model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False, in_channels=1, out_channels=1, pad=False, bilinear=False, normalization=None)

# Example forward pass
model = torch.hub.load('sm00thix/unet', 'unet', pretrained=False)
x = torch.randn(1, 3, 256, 256)  # RGB image
with torch.no_grad():
    output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")

# List all available models
available_models = torch.hub.list('sm00thix/unet')
print(f"Available models: {available_models}")

# Get model information
info = torch.hub.load('sm00thix/unet', 'model_info', model_name='unet_bn')
print(f"Model info: {info}")
"""
