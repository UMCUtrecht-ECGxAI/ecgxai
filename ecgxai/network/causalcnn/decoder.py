import torch
import torch.nn as nn
from .modules import CausalCNN, Softplus

class CausalCNNVDecoder(torch.nn.Module):
    """
    Variational decoder.
    """
    def __init__(self, k, width, in_channels, channels, depth, out_channels,
                 kernel_size, gaussian_out, softplus_eps, dropout):
        super(CausalCNNVDecoder, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.gaussian_out = gaussian_out
        self.linear1 = torch.nn.Linear(k, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, in_channels * width)
        self.causal_cnn = CausalCNN(
            in_channels, channels, depth, out_channels, kernel_size,
            forward=False,
        )
        if self.gaussian_out:
            self.linear_mean = nn.Linear(out_channels * width, 
                                         out_channels * width)
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(out_channels * width, out_channels * width),
                Softplus(softplus_eps),
            )
        
    def forward(self, x):
        """
        Returns a reconstruction of the original 8x600 ECG, by decoding
        the given compression.
        """
        B, _ = x.shape
        # from (BxK) to (BxC)
        out = self.linear1(x)
        # from (BxC) to (Bx(C*600))
        out = self.linear2(out)
        # from (Bx(C*600)) to (BxCx600)
        out = out.view(B, self.in_channels, self.width)
        # deconvolve through the causal CNN
        out = self.causal_cnn(out)
        if self.gaussian_out:
            nflat_shape = out.shape
            # flatten the output to shape (Bx(8*600))
            out = torch.flatten(out, start_dim=1)
            return self.linear_mean(out).reshape(nflat_shape), self.linear_sd(out).reshape(nflat_shape)
        return out