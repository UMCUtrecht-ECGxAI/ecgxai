
import torch
import torch.nn as nn
import pytorch_lightning as pl


class CNNDoubleResidual(pl.LightningModule):
    """
        Residual 'double' CNN as described by  Hannun et al. converting a multi channel input to some flat output dimension
    """
    def __init__(
            self,
            num_layers: int = 15,
            in_sample_dim: int = 600,
            in_channels: int = 32,
            kernel_size: int = 16,
            dropout_rate: float = 0.2,
            sub_sample_every: int = 4,
            double_channel_every: int = 4,
            act_func : nn.Module = nn.ReLU(),
            batchnorm : bool = True
    ):
        super(CNNDoubleResidual, self).__init__()
        
        self.num_layers = num_layers
        self.in_sample_dim = in_sample_dim
        self.in_channels = in_channels
        self.sub_sample_every = sub_sample_every
        self.double_channel_every = double_channel_every

        self.layers = []

        layer_sample_dim = in_sample_dim
        for layer_idx in range(num_layers):
            # Double output channels every 4th layer
            output_channels = self.in_channels * (2**((layer_idx + 1) // self.double_channel_every))
            # Subsample input every few layers
            sub_sample_rate = 2 if layer_idx > 0 and layer_idx % sub_sample_every == 0 else 1

            if int(layer_sample_dim + 1) % 2 == 0 and bool(kernel_size % 2):
                sub_sample_extra_pad = -1
            elif int(layer_sample_dim + 1) % 2 == 0 or bool(kernel_size % 2):
                sub_sample_extra_pad = 0
            else:
                sub_sample_extra_pad = 1

            self.layers.append(
                ResidualMaxPoolDoubleConvBlockForward(
                    in_channels,
                    output_channels,
                    sub_sample_rate=sub_sample_rate,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    batchnorm=batchnorm,
                    act_func=act_func,
                    sub_sample_extra_pad=sub_sample_extra_pad
                )
            )
            layer_sample_dim /= sub_sample_rate
            in_channels = output_channels

        # print(self.calculate_output_dim())

        self.layers.append(nn.Flatten())
        self.layers = nn.Sequential(*self.layers)

    def calculate_output_dim(self):
        cnn_output_channels = self.in_channels * (2**((self.num_layers) // self.double_channel_every))
        cnn_output_samples = self.in_sample_dim // (2**((self.num_layers - 1) // self.sub_sample_every))
        return int(cnn_output_channels * cnn_output_samples), cnn_output_channels, cnn_output_samples

    def forward(self, x):
        return self.layers(x)


class ResidualMaxPoolDoubleConvBlockForward(nn.Module):
    """
        Residual convblock adding a max-pool over the input to the out, performs 2 convolutions 
        with (optionally) a batchnorm an activation function and dropout.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        sub_sample_rate: int = 2,
        kernel_size: int = 16,
        dropout_rate : float = 0.2,
        act_func : nn.Module = nn.ReLU(),
        batchnorm : bool = True,
        sub_sample_extra_pad : int = 1
    ):
        super(ResidualMaxPoolDoubleConvBlockForward, self).__init__()

        self.sub_sample_rate = sub_sample_rate
        self.channel_padding = out_dim - in_dim
        
        self.pool = nn.MaxPool1d(self.sub_sample_rate)

        # assuming keras style 'same' padding -- padding is applied through nn.ConstantPad1d
        P = int((kernel_size - 1) // 2)
        padding = (P, P + 1 if kernel_size % 2 == 0 else P)

        pre_modules = []
        if batchnorm:
            pre_modules.append(nn.BatchNorm1d(in_dim))

        if act_func:
            pre_modules.append(act_func)

        self.f = nn.Sequential(
            *pre_modules,
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(in_dim, out_dim, kernel_size, stride=self.sub_sample_rate),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.ConstantPad1d((P, P + sub_sample_extra_pad) if self.sub_sample_rate % 2 == 0 else padding, 0),
            nn.Conv1d(out_dim, out_dim, kernel_size, stride=1),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.pool(torch.functional.F.pad(x, (0, 0, 0, self.channel_padding))) + self.f.forward(x)


class Reshape(nn.Module): 
    """ Reshapes an input tensor into the given shape """
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
