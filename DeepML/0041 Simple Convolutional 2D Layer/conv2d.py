import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)

    # kH, kW = kernel.shape()
    # H, W = input_matrix.shape()
    # H_out = (H + 2*padding - kH) / stride + 1
    # W_out = (W + 2*padding - kW) / stride + 1

    input = input_matrix.unsqueeze(0).unsqueeze(0)
    # weight = 2
    weight = kernel.unsqueeze(0).unsqueeze(0)

    rst = torch.conv2d(input, weight, stride=stride, padding=padding)
    rst = rst.squeeze(0).squeeze(0)

    return rst