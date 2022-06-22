import torch
import torch.nn as nn
from .modules import CausalCNN, Softplus, SqueezeChannels


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels manipulated in the causal CNN.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the output time series of the
           causal CNN is reduced.
        out_channels (int): Number of output classes.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        dropout (float): The dropout probability between 0 and 1.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, dropout):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        spatial = Spatial(reduced_size, dropout)
        reduce_size = torch.nn.AdaptiveAvgPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear1 = torch.nn.Linear(reduced_size, 26)
        linear2 = torch.nn.Linear(26, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, spatial, reduce_size, squeeze, linear1,
            nn.BatchNorm1d(num_features=26), nn.ReLU(), nn.Dropout(dropout), linear2,
        )

    def forward(self, x):
        return self.network(x)


class CausalCNNVEncoder(torch.nn.Module):
    """
    Variational encoder. Difference is that we need two outputs: mean and
    standard deviation.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels manipulated in the causal CNN.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the output time series of the
           causal CNN is reduced.
        out_channels (int): Number of output classes.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        softplus_eps (float): Small number to add for stability of the Softplus activation.
        dropout (float): The dropout probability between 0 and 1.
        sd_output (bool): Put to true when using this class inside a VAE, as
            an additional output for the SD is added.
    """
    def __init__(self, in_channels: int, channels: int, depth: int, reduced_size: int,
                 out_channels: int, kernel_size: int, softplus_eps: float, dropout: float, 
                 sd_output: bool = True):
        super(CausalCNNVEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze,
        )
        self.linear_mean = torch.nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        if self.sd_output:
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(reduced_size, out_channels),
                Softplus(softplus_eps),
            )

    def forward(self, x):
        out = self.network(x)
        if self.sd_output:
            return self.linear_mean(out), self.linear_sd(out)
        return self.linear_mean(out).squeeze()
