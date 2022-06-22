"""
Variational autoencoder system

MB Vessies and RR van de Leur
"""

import torch
from torch import nn

from ecgxai.utils.exceptions import ModelOutputShapeMismatchException
from ecgxai.systems.base_system import BaseSystem


class VAE(BaseSystem):
    """ Basic VAE system """

    def __init__(
        self,
        encoder_class: nn.Module,
        decoder_class: nn.Module,
        std_is_log : bool = True,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        
        self.save_hyperparameters()

        self.std_is_log = std_is_log

        self.encoder = encoder_class
        self.decoder = decoder_class

    def reparameterize(self, mu : torch.Tensor, std : torch.Tensor):
        """
            The 'reparameterization trick', allowing for differentiable sampling
            Arguments:
                mu : torch.Tensor       -- Mean of the normal distribution to sample from
                std : torch.Tensor      -- Standard deviation of the distribution to sample from, if input is log_std, this should
                    be specified through the 'std_is_log' parameter of the system
            returns:
                z : torch.Tensor        -- Sample from the distribution
        """
        if self.std_is_log:
            std = std.exp()
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(std))
        z = mu + (eps * std)
        return z

    def forward(self, x : torch.Tensor, deterministic : bool = False):
        """
            Takes a batch dictionary of ECGs as input and reconstucts it through the VAE model

            Arguments:
                x : dict                -- (batch of) ECG(s) used as input for the model
                deterministic : bool    -- Wether or not apply the reperameterization trick and use a sampling operation in the latent space, 
                    might be preferable to disable this during inference
            Returns:
                reconstruction : torch.Tensor -- Reconstruction of the orginal input
        """
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
            
        mu, std = self.encoder(x)
        z = mu if deterministic else self.reparameterize(mu, std)

        reconstruction = self.decoder(z)

        if x.shape != reconstruction.shape:
            raise ModelOutputShapeMismatchException(reconstruction.shape, x.shape)

        return {
            'x': x,
            'reconstruction': reconstruction,
            'z': z,
            'mu': mu,
            'std': std,
        }

    @property
    def step_args(self):
        return ['x', 'reconstruction', 'mu', 'std']
    
    def _step(self, batch : dict):
        """
            Takes an ECG as input and reconstucts it through the VAE model, then calculates the loss for training

            Arguments:
                batch : dict     -- (batch dictionary of) ECG(s) used as input for the model, ECG waveform is expected under key ['waveform'] in the dict
            Returns:
                loss : float     --  Loss
                step_dict : dict --  Dictionary of intermediate calculation results as well as final model outputs
        """
        x = batch['waveform']

        return self.forward(x)


class GaussianVAE(VAE):
    """ VAE system with mean and standard deviation as final output """

    def __init__(
        self,
        **kwargs
    ):
        super(GaussianVAE, self).__init__(**kwargs)

    def forward(self, x : torch.Tensor, deterministic : bool = False):
        """
            Takes a batch of ECGs as input and reconstucts it through the VAE model

            Arguments:
                x : tensor              -- (batch of) ECG(s) used as input for the model
                deterministic : bool    -- Wether or not apply the reperameterization trick and use a sampling operation in the latent space,
                    might be preferable to disable this during inference
            Returns:
                results : dict          -- Dictionary containing all outputs of the VAE
        """
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)

        mu, std = self.encoder(x)
        z = mu if deterministic else self.reparameterize(mu, std)

        reconstruction_mean, reconstruction_std = self.decoder(z)

        if x.shape != reconstruction_mean.shape:
            raise ModelOutputShapeMismatchException(message=f'Model output (mean) shape {x.shape} does not match expected output shape {reconstruction_mean.shape}.')

        if x.shape != reconstruction_std.shape:
            raise ModelOutputShapeMismatchException(message=f'Model output (std) shape {x.shape} does not match expected output shape {reconstruction_std.shape}.')

        return {
            'x': x,
            'reconstruction': reconstruction_mean,
            'reconstruction_mean': reconstruction_mean,
            'reconstruction_std': reconstruction_std,
            'z': z,
            'mu': mu,
            'std': std,
        }

    @property
    def step_args(self):
        return ['x', 'reconstruction', 'reconstruction_mean', 'reconstruction_std', 'mu', 'std']  

    def _step(self, batch : dict):
        """
            Takes an ECG as input and reconstucts it through the VAE model, then calculates the loss for training

            Arguments:
                batch : dict     -- (batch dictionary of) ECG(s) used as input for the model, ECG waveform is expected under key ['waveform'] in the dict
            Returns:
                step_dict : dict --  Dictionary of intermediate calculation results as well as final model outputs
        """
        x = batch['waveform']

        return self.forward(x)
