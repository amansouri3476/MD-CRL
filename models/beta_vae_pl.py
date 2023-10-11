import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
import os
import numpy as np
import utils.general as utils
log = utils.get_logger(__name__)
from models.modules.beta_vae import BetaVAE

class BetaVAEPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.z_dim = self.hparams.get("z_dim", 128)
        self.model = BetaVAE(z_dim=self.z_dim)
        self.beta = self.hparams.get("beta", 1)
        self.save_encoded_data = self.hparams.get("save_encoded_data", False)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def loss(self, images, recons, z_hat, mu, logvar):

        # images, recons: [batch_size, num_channels, width, height]


        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        reconstruction_loss = F.mse_loss(recons.permute(0, 2, 3, 1), images.permute(0, 2, 3, 1), reduction="mean")

        return reconstruction_loss, total_kld

    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images = train_batch["image"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, recons, mu, logvar = self(images)
        recon_loss, total_kld = self.loss(images, recons, z_hat, mu, logvar)
        beta_vae_loss = recon_loss + self.beta*total_kld
        loss = beta_vae_loss
        self.log(f"train_loss", loss.item(), prog_bar=True)

        log.info(f"images.max(): {images.max()}, recons.max(): {recons.max()}, images.min(): {images.min()}, recons.min(): {recons.min()}\n images.mean(): {images.mean()}, recons.mean(): {recons.mean()}")
        log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}, z_hat.std(): {z_hat.std()}")

        if self.save_encoded_data:
            self.training_step_outputs.append({"z_hat":z_hat})

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images = valid_batch["image"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, recons, mu, logvar = self(images)
        recon_loss, total_kld = self.loss(images, recons, z_hat, mu, logvar)
        beta_vae_loss = recon_loss + self.beta*total_kld
        loss = beta_vae_loss
        self.log(f"val_loss", loss.item(), prog_bar=True)
        
        # fit a linear regression from z_hat on z
        z = valid_batch["z"] # [batch_size, n_balls * z_dim_ball]
        r2, mse_loss = self.compute_r2(z, z_hat)
        self.log(f"r2", r2, prog_bar=True)
        r2, mse_loss = self.compute_r2(z_hat, z)
        self.log(f"~r2", r2, prog_bar=True)

        # select half of z_dim dimensions randomly
        inv_indices = np.random.choice(self.z_dim, self.z_dim//2, replace=False)
        spu_indices = np.setdiff1d(np.arange(self.z_dim), inv_indices)
        # fit a linear regression from z_hat on z_invariant dimensions
        z_invariant = valid_batch["z_invariant"] # [batch_size, n_balls_invariant * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_invariant, z_hat[:, inv_indices])
        self.log(f"hz_z_r2", r2, prog_bar=True)
        r2, mse_loss = self.compute_r2(z_hat[:, inv_indices], z_invariant)
        self.log(f"hz_z_~r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat on z_spurious dimensions
        z_spurious = valid_batch["z_spurious"] # [batch_size, n_balls_spurious * z_dim_ball]
        r2, mse_loss = self.compute_r2(z_spurious, z_hat[:, spu_indices])
        self.log(f"hz_~z_r2", r2, prog_bar=True)
        r2, mse_loss = self.compute_r2(z_hat[:, spu_indices], z_spurious)
        self.log(f"hz_~z_~r2", r2, prog_bar=True)

        if self.save_encoded_data:
            self.validation_step_outputs.append({"z_hat":z_hat})

        return loss

    def on_train_epoch_end(self):

        self.training_step_outputs.clear()
        return
    
    def on_validation_epoch_end(self):

        self.validation_step_outputs.clear()
        return



def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld