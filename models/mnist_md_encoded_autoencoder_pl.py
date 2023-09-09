import torch
from torch.nn import functional as F
from .autoencoder_pl import AutoencoderPL
import hydra
from omegaconf import OmegaConf
import wandb
import os
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
import utils.general as utils
log = utils.get_logger(__name__)
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss


class MNISTMDEncodedAutoencoderPL(AutoencoderPL):
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
            penalty_loss_args = [self.hparams.top_k, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
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
        x, labels, domains, colors = train_batch["x"], train_batch["label"], train_batch["domain"], train_batch["color"]

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, z_dim]
        z_hat, x_hat = self(x)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_hinge_loss", hinge_loss_value.item())
        self.log(f"train_loss", loss.item())

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        x, labels, domains, colors = valid_batch["x"], valid_batch["label"], valid_batch["domain"], valid_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, x_hat = self(x)

        # we have the set of labels and latents. We want to train a classifier to predict the labels from latents
        # using multinomial logistic regression using sklearn
        # import sklearn
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        # fit a multinomial logistic regression from z to labels and 
        # multinomial/linear regression to colors (based on color representations)

        # 1. predicting labels from z_hat[:z_dim_invariant_model]
        try:
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
            pred_labels = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
            self.log(f"digits_accuracy_z", accuracy, prog_bar=True)
        except:
            # if there is only one class, we can't fit a logistic regression
            pass

        # 2. predicting colors from z_hat[:z_dim_invariant_model]
        if self.trainer.datamodule.train_dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_z", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            clf = LinearRegression().fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_r2_z", r2, prog_bar=True)

        # 3. predicting labels from z_hat[z_dim_invariant_model:]
        try:
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
            pred_labels = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
            self.log(f"digits_accuracy_~z", accuracy, prog_bar=True)
        except:
            # if there is only one class, we can't fit a logistic regression
            pass

        # 4. predicting colors from z_hat[z_dim_invariant_model:]
        if self.trainer.datamodule.train_dataset.generation_strategy == "manual":
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_~z", accuracy, prog_bar=True)
        else:
            clf = LinearRegression().fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_r2_~z", r2, prog_bar=True)

        # overall accuracy with all z
        try:
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # predict the labels from z
            pred_labels = clf.predict(z_hat.detach().cpu().numpy())
            # compute the accuracy
            accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
            self.log(f"val_digits_accuracy", accuracy, prog_bar=True)
        except:
            # if there is only one class, we can't fit a logistic regression
            pass

        # fit a linear regression from z to colours
        clf = LinearRegression().fit(z_hat.detach().cpu().numpy(), colors.detach().cpu().numpy())
        pred_colors = clf.predict(z_hat.detach().cpu().numpy())
        r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
        self.log(f"val_r2_colors", r2, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)


        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_hinge_loss", hinge_loss_value.item())
        self.log(f"val_loss", loss.item())
        return {"loss": loss, "pred_z": z_hat}

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
