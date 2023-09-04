import torch
from torch.nn import functional as F
from .autoencoder_pl import AutoencoderPL
import hydra
from omegaconf import OmegaConf
import wandb
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss


class MNISTMDAutoencoderPL(AutoencoderPL):
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

    def loss(self, images, recons, z, domains):

        reconstruction_loss = F.mse_loss(recons.permute(0, 2, 3, 1), images.permute(0, 2, 3, 1), reduction="mean")
        if self.penalty_criterion == "minmax":
            penalty_loss_args = [self.hparams.top_k, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        else:
            penalty_loss_args = [self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight]
        penalty_loss_value, hinge_loss_value = self.penalty_loss(z, domains, self.hparams.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
        loss = reconstruction_loss + penalty_loss_value * self.penalty_weight

        return loss, reconstruction_loss, penalty_loss_value, hinge_loss_value

    def on_training_start(self, *args, **kwargs):
        self.log(f"val_reconstruction_loss", 0.0)
        self.log(f"valid_penalty_loss", 0.0)
        self.log(f"val_loss", 0.0)
        return 
    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels, domains, colors = train_batch["image"], train_batch["label"], train_batch["domain"], train_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(images, recons, z, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_loss", loss.item())

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels, domains, colors = valid_batch["image"], valid_batch["label"], valid_batch["domain"], valid_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        # we have the set of labels and latents. We want to train a classifier to predict the labels from latents
        # using multinomial logistic regression using sklearn
        # import sklearn
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        # fit a multinomial logistic regression from z to labels and 
        # multinomial/linear regression to colors (based on color representations)

        # 1. predicting labels from z[:z_dim_invariant_model]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), labels.detach().cpu().numpy())
        pred_labels = clf.predict(z[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"digits_accuracy_z", accuracy, prog_bar=True)

        # 2. predicting colors from z[:z_dim_invariant_model]
        if self.trainer.datamodule.train_dataset.dataset.generation_strategy == "manual": # colors are indexed
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_z", accuracy, prog_bar=True)
        else: # colors are triplets of rgb, there we need to measure r2 score of linear regression
            clf = LinearRegression().fit(z[:, :self.z_dim_invariant_model].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, :self.z_dim_invariant_model].detach().cpu().numpy())
            r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_r2_z", r2, prog_bar=True)

        # 3. predicting labels from z[z_dim_invariant_model:]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        pred_labels = clf.predict(z[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"digits_accuracy_~z", accuracy, prog_bar=True)

        # 4. predicting colors from z[z_dim_invariant_model:]
        if self.trainer.datamodule.train_dataset.dataset.generation_strategy == "manual":
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_accuracy_~z", accuracy, prog_bar=True)
        else:
            clf = LinearRegression().fit(z[:, self.z_dim_invariant_model:].detach().cpu().numpy(), colors.detach().cpu().numpy())
            pred_colors = clf.predict(z[:, self.z_dim_invariant_model:].detach().cpu().numpy())
            r2 = r2_score(colors.detach().cpu().numpy(), pred_colors)
            self.log(f"colors_r2_~z", r2, prog_bar=True)

        # overall accuracy with all z
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # predict the labels from z
        pred_labels = clf.predict(z.detach().cpu().numpy())
        # compute the accuracy
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"val_accuracy", accuracy, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)


        loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(images, recons, z, domains)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_loss", loss.item())
        return {"loss": loss, "pred_z": z}

    # def validation_epoch_end(self, validation_step_outputs):
    #     z_disentanglement = [v["true_z"] for v in validation_step_outputs]
    #     h_z_disentanglement = [v["pred_z"] for v in validation_step_outputs]
    #     z_disentanglement = torch.cat(z_disentanglement, 0)
    #     h_z_disentanglement = torch.cat(h_z_disentanglement, 0)
        
    #     (linear_disentanglement_score, _), _ = linear_disentanglement(
    #         z_disentanglement, h_z_disentanglement, mode="r2", train_test_split=True
    #     )

    #     (permutation_disentanglement_score, _), _ = permutation_disentanglement(
    #         z_disentanglement,
    #         h_z_disentanglement,
    #         mode="pearson",
    #         solver="munkres",
    #         rescaling=True,
    #     )
    #     mse = F.mse_loss(z_disentanglement, h_z_disentanglement).mean(0)
    #     self.log("Linear_Disentanglement", linear_disentanglement_score, prog_bar=True)
    #     self.log(
    #         "Permutation_Disentanglement",
    #         permutation_disentanglement_score,
    #         prog_bar=True,
    #     )
    #     self.log("MSE", mse, prog_bar=True)
    #     wandb.log(
    #         {
    #             "mse": mse,
    #             "Permutation Disentanglement": permutation_disentanglement_score,
    #             "Linear Disentanglement": linear_disentanglement_score,
    #         }
    #     )
        
    # def test_epoch_end(self, test_step_outputs):
    #     z_disentanglement = [v["true_z"] for v in test_step_outputs]
    #     h_z_disentanglement = [v["pred_z"] for v in test_step_outputs]
    #     z_disentanglement = torch.cat(z_disentanglement, 0)
    #     h_z_disentanglement = torch.cat(h_z_disentanglement, 0)
    #     (linear_disentanglement_score, _), _ = linear_disentanglement(
    #         z_disentanglement, h_z_disentanglement, mode="r2", train_test_split=True
    #     )

    #     (permutation_disentanglement_score, _), _ = permutation_disentanglement(
    #         z_disentanglement,
    #         h_z_disentanglement,
    #         mode="pearson",
    #         solver="munkres",
    #         rescaling=True,
    #     )
    #     mse = F.mse_loss(z_disentanglement, h_z_disentanglement).mean(0)
    #     self.log("Linear_Disentanglement", linear_disentanglement_score, prog_bar=True)
    #     self.log(
    #         "Permutation Disentanglement",
    #         permutation_disentanglement_score,
    #         prog_bar=True,
    #     )
    #     self.log("MSE", mse, prog_bar=True)
    #     wandb.log(
    #         {
    #             "mse": mse,
    #             "Permutation Disentanglement": permutation_disentanglement_score,
    #             "Linear_Disentanglement": linear_disentanglement_score,
    #         }
    #     )