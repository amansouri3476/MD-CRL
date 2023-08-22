import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import numpy as np


class Encoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.encoder_layers.items()]
        )        

    def forward(self, x):
        
        # input `x` has shape: [batch_size, x_dim]
        return self.layers(x)

class Decoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # note that encoder layers are instantiated recursively by hydra, so we only need to connect them
        # by nn.Sequential
        self.layers = torch.nn.Sequential(
            *[layer_config for _, layer_config in self.hparams.decoder_layers.items()]
        )
                
    def forward(self, z):
        
        # `z` has shape: [batch_size, z_dim].
        
        # [batch_size, x_dim]
        return self.layers(z)
        

class FCAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_fc = hydra.utils.instantiate(self.hparams.encoder_fc)
        self.decoder_fc = hydra.utils.instantiate(self.hparams.decoder_fc)
        
    def forward(self, x):
        # `x` has shape: [batch_size, x_dim]
        z = self.encoder_fc(x)
        x_hat = self.decoder_fc(z)

        return z, x_hat
