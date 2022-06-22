"""
Autoencoder systems.

DB van de Pavert, RR van de Leur and MB Vessies
"""

import torch.nn as nn

from ecgxai.utils.exceptions import ModelOutputShapeMismatchException
from ecgxai.systems.base_system import BaseSystem


class AE(BaseSystem):
    def __init__(
        self,
        encoder_class: nn.Module,
        decoder_class: nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.encoder = encoder_class
        self.decoder = decoder_class

    def forward(self, batch):
        x = batch['waveform']
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        if x.shape != reconstruction.shape:
            raise ModelOutputShapeMismatchException(reconstruction.shape, x.shape)

        return reconstruction, z

    @property
    def step_args(self):
        return ['x', 'reconstruction', 'z']

    def _step(self, batch):
        x = batch['waveform']
        reconstruction, z = self.forward(batch)

        if x.shape != reconstruction.shape:
            raise ModelOutputShapeMismatchException(reconstruction.shape, x.shape)

        return {
            'x': x,
            'reconstruction': reconstruction,
            'z': z
        }

