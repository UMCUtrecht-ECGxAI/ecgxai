import torch
import torch.nn as nn


class Softplus(nn.Module):
    """
    Applies Softplus to the output and adds a small number.

    Attributes:
        eps (int): Small number to add for stability.
    """
    def __init__(self, eps: float):
        super(Softplus, self).__init__()
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) + self.eps


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    Attributes:
        chomp_size (int): Number of elements to remove.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size of the applied non-residual convolutions.
        padding: Zero-padding applied to the left of the input of the
           non-residual convolutions.
        final (bool) Disables, if True, the last activation function.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, final=False, forward=True):
        super(CausalConvolutionBlock, self).__init__()

        Conv1d = torch.nn.Conv1d if forward else torch.nn.ConvTranspose1d
        
        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

        in_channels (int): Number of input channels.
        channels (int): Number of channels processed in the network and of output
           channels.
        depth (int): Depth of the network.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, forward=True):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        # double the dilation size if forward, if backward
        # we start at the final dilation and work backwards
        dilation_size = 1 if forward else 2**depth
        
        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size,
                forward,
            )]
            # double the dilation at each step if forward, otherwise
            # halve the dilation
            dilation_size = dilation_size * 2 if forward else dilation_size // 2

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Spatial(nn.Module):
    def __init__(self, channels, dropout, forward=True):
        super(Spatial, self).__init__()
        Conv1d = nn.Conv1d if forward else nn.ConvTranspose1d
        self.network = nn.Sequential(
            Conv1d(channels, channels, 1),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)