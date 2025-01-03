import torch.nn as nn
from .sparse_nn import SparseEncoder, Decoder

class Autoencoder(nn.Module):
    def __init__(self, encoder, imputation_decoder, subsurface_decoder):
        super().__init__()
        self.encoder = encoder
        self.imputation_decoder = imputation_decoder
        self.subsurface_decoder = subsurface_decoder

    def forward(self, x, mask):
        features = self.encoder(x * mask)
        sample_output = self.imputation_decoder(features[::-1])
        parameter_output = self.subsurface_decoder(features[::-1])
        return sample_output, parameter_output
