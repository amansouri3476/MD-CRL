import torch
from torch.nn import functional as F
from .mixing_autoencoder_pl import MixingAutoencoderPL
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)
from models.utils import penalty_loss_minmax, penalty_loss_stddev, penalty_domain_classification, hinge_loss


class MixingMDEncodedAutoencoderPL(BallsAutoencoderPL):
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
        if self.penalty_criterion["minmax"]:
            # self.penalty_loss = penalty_loss_minmax
            self.loss_transform = self.hparams.loss_transform
        # elif self.penalty_criterion:
        #     # self.penalty_loss = penalty_loss_stddev
        if self.penalty_criterion["domain_classification"]:
            # self.penalty_loss = penalty_domain_classification
            from models.modules.multinomial_logreg import LogisticRegressionModel
            from torch import nn
            self.multinomial_logistic_regression = LogisticRegressionModel(self.z_dim_invariant_model, self.num_domains)
            self.multinomial_logistic_regression = self.multinomial_logistic_regression.to(self.device)
            self.domain_classification_loss = nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(f"penalty_criterion {self.penalty_criterion} not supported")
        self.stddev_threshold = self.hparams.stddev_threshold
        self.stddev_eps = self.hparams.stddev_eps
        self.hinge_loss_weight = self.hparams.hinge_loss_weight
        # assert that the z_dim of this model is less than that of its encoder
        # assert self.hparams.z_dim <= self.model.encoder_fc.hparams.latent_dim, f"z_dim of this model ({self.hparams.z_dim}) is greater than that of its encoder ({self.model.encoder_fc.hparams.latent_dim})"

    def loss(self, x, x_hat, z_hat, domains):

        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        penalty_loss_value = 0.0
        hinge_loss_value = hinge_loss(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight)

        if self.penalty_criterion["minmax"]:
            penalty_loss_args = [self.hparams.top_k, self.loss_transform]
            penalty_loss_value_ = penalty_loss_minmax(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion["stddev"]:
            penalty_loss_args = []
            penalty_loss_value_ = penalty_loss_stddev(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion["domain_classification"]:
            penalty_loss_args = [self.multinomial_logistic_regression, self.domain_classification_loss]
            penalty_loss_value_ = penalty_domain_classification(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        # if self.penalty_criterion == "minmax":
        #     penalty_loss_args = [self.hparams.top_k, self.loss_transform, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        # elif self.penalty_criterion == "stddev":
        #     penalty_loss_args = [self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        # else: # domain_classification
        #     penalty_loss_args = [self.multinomial_logistic_regression, self.domain_classification_loss]
        # penalty_loss_value, hinge_loss_value = self.penalty_loss(z_hat, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
        penalty_loss_value = penalty_loss_value * self.penalty_weight
        hinge_loss_value = hinge_loss_value * self.hinge_loss_weight
        loss = reconstruction_loss + penalty_loss_value + hinge_loss_value

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
        if batch_idx % 20 == 0:
            log.info(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}")
            log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}")
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

        if batch_idx % 20 == 0:
            if self.penalty_criterion["minmax"] == 1.:
                # print all z_hat mins of all domains
                log.info(f"============== z_hat min all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].min().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                # print all z_hat maxs of all domains
                log.info(f"============== z_hat max all domains ==============\n{[z_hat[(domain == i).squeeze(), :self.z_dim_invariant_model].max().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion["stddev"] == 1.:
                # print all z_hat stds of all domains for each of z_dim_invariant dimensions
                for dim in range(self.z_dim_invariant_model):
                    log.info(f"============== z_hat std all domains dim {dim} ==============\n{[z_hat[(domain == i).squeeze(), dim].std().detach().cpu().numpy().item() for i in range(self.num_domains)]}\n")
                log.info(f"============== ============== ============== ==============\n")
            if self.penalty_criterion["domain_classification"] == 1.:
                # log the weigth matrix of the multinomial logistic regression model
                log.info(f"============== multinomial logistic regression model weight matrix ==============\n{self.multinomial_logistic_regression.linear.weight}\n")


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
