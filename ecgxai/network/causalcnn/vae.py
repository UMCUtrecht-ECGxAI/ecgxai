import torch
import torch.nn as nn
from .encoder import CausalCNNVEncoder
from .decoder import CausalCNNVDecoder


class VAE(nn.Module):
    """
    The VAE method is able to encode a given input into
    mean and log. variance parameters for Gaussian
    distributions that make up a compressed space that represents
    the input. Then, a sample from that space can be decoded into
    an attempted, probably less detailed, reconstruction of the
    original input.
    """
    
    def __init__(self, encoder_params, decoder_params):
        super().__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.encoder = CausalCNNVEncoder(**encoder_params)
        self.decoder = CausalCNNVDecoder(**decoder_params)
    
    def reparameterize(self, mu, sd):
        """Sample from a Gaussian distribution with given mean and s.d."""
        eps = torch.randn_like(sd)
        return mu + eps * sd
    
    def forward(self, x):
        """Return reconstructed input after compressing it"""
        enc_mu, enc_sd = self.encoder(x)
        z = self.reparameterize(enc_mu, enc_sd)
        if self.decoder_params['gaussian_out']:
            dec_mu, dec_sd = self.decoder(z)
            # reconstruction is equal to the mean value for the gauss. distr. of each point
            # reshape the mean vector to be of same size as input (Bx8x600 for median ECGs)
            recon_x = dec_mu.view(x.shape)
            return recon_x, z, [(enc_mu, enc_sd), (dec_mu, dec_sd)]
        recon_x = self.decoder(z)
        return recon_x, z, [(enc_mu, enc_sd)]
