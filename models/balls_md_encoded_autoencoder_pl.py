import torch
from torch.nn import functional as F
from .balls_autoencoder_pl import BallsAutoencoderPL
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss


class BallsMDEncodedAutoencoderPL(BallsAutoencoderPL):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        
        self.num_domains = self.hparams.num_domains
        self.z_dim_invariant_model = self.hparams.z_dim_invariant
        self.penalty_weight = self.hparams.penalty_weight
        self.wait_steps = self.hparams.wait_steps
        self.linear_steps = self.hparams.linear_steps
        self.penalty_criterion = self.hparams.penalty_criterion
        if self.penalty_criterion == "minmax":
            self.penalty_loss = penalty_loss_minmax
            self.loss_transform = self.hparams.loss_transform
        elif self.penalty_criterion == "stddev":
            self.penalty_loss = penalty_loss_stddev
        else:
            raise ValueError(f"penalty_criterion {self.penalty_criterion} not supported")
        self.stddev_threshold = self.hparams.stddev_threshold
        self.stddev_eps = self.hparams.stddev_eps
        self.hinge_loss_weight = self.hparams.hinge_loss_weight
        # assert that the z_dim of this model is less than that of its encoder
        # assert self.hparams.z_dim <= self.model.encoder_fc.hparams.latent_dim, f"z_dim of this model ({self.hparams.z_dim}) is greater than that of its encoder ({self.model.encoder_fc.hparams.latent_dim})"

    def loss(self, x, x_hat, z_hat, domains):

        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        if self.penalty_criterion == "minmax":
            penalty_loss_args = [self.hparams.top_k, self.loss_transform, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        else:
            penalty_loss_args = [self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        penalty_loss_value, hinge_loss_value = self.penalty_loss(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
        loss = reconstruction_loss + penalty_loss_value * self.penalty_weight

        return loss, reconstruction_loss, penalty_loss_value, hinge_loss_value

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"valid_penalty_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return

    def training_step(self, train_batch, batch_idx):
        # x: [batch_size, z_dim]
        x, domains = train_batch["x"], train_batch["domain"]

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, z_dim]
        z_hat, x_hat = self(x)
        if batch_idx % 50 == 0:
            print(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}")
            print(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}")
        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_hinge_loss", hinge_loss_value.item())
        self.log(f"train_loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        # images: [batch_size, num_channels, width, height]
        x, z_invariant, z_spurious, domain, color = valid_batch["x"], valid_batch["z_invariant"], valid_batch["z_spurious"], valid_batch["domain"], valid_batch["color"]

        # z_hat: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        if batch_idx % 50 == 0:
            if self.penalty_criterion == "minmax":
                # print all z_hat mins of all domains
                print(f"============== z_hat min all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].min().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                # print all z_hat maxs of all domains
                print(f"============== z_hat max all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].max().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                print(f"============== ============== ============== ==============\n")
            elif self.penalty_criterion == "stddev":
                # print all z_hat stds of all domains for each of z_dim_invariant dimensions
                for dim in range(self.z_dim_invariant_model):
                    print(f"============== z_hat std all domains dim {dim} ==============\n{[z_hat[(domain == i).squeeze(), dim].std().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                print(f"============== ============== ============== ==============\n")


        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score

        # fit a linear regression from z_hat to z
        z = valid_batch["z"] # [batch_size, n_balls * z_dim_ball]
        clf = LinearRegression().fit(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        pred_z = clf.predict(z_hat.detach().cpu().numpy())
        r2 = r2_score(z.detach().cpu().numpy(), pred_z)
        self.log(f"z_hat_z_r2", r2, prog_bar=True)

        # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        clf = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z_invariant.detach().cpu().numpy())
        pred_z_invariant = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2 = r2_score(z_invariant.detach().cpu().numpy(), pred_z_invariant)
        self.log(f"z_hat_inv_z_inv_r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        clf = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), z_spurious.detach().cpu().numpy())
        pred_z_spurious = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2 = r2_score(z_spurious.detach().cpu().numpy(), pred_z_spurious)
        self.log(f"z_hat_inv_z_spur_r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat spurious dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        clf = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z_invariant.detach().cpu().numpy())
        pred_z_invariant = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2 = r2_score(z_invariant.detach().cpu().numpy(), pred_z_invariant)
        self.log(f"z_hat_spur_z_inv_r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat spurious dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        clf = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), z_spurious.detach().cpu().numpy())
        pred_z_spurious = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2 = r2_score(z_spurious.detach().cpu().numpy(), pred_z_spurious)
        self.log(f"z_hat_spur_z_spur_r2", r2, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)


        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_hinge_loss", hinge_loss_value.item())
        self.log(f"val_loss", loss.item())
        return {"loss": loss, "pred_z": z_hat}

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
