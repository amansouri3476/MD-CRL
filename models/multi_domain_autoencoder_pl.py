import torch
from torch.nn import functional as F
from .autoencoder_pl import AutoencoderPL
from omegaconf import OmegaConf
import wandb
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement


class MultiDomainAutoencoderPL(AutoencoderPL):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.z_dim = self.hparams.z_dim
        # assert that the z_dim of this model is less than that of its encoder
        # assert self.hparams.z_dim <= self.model.encoder_fc.hparams.latent_dim, f"z_dim of this model ({self.hparams.z_dim}) is greater than that of its encoder ({self.model.encoder_fc.hparams.latent_dim})"

    def loss(self, images, recons, z, domains):

        reconstruction_loss = F.mse_loss(recons.permute(0, 2, 3, 1), images.permute(0, 2, 3, 1), reduction="mean")
        penalty_loss = 0.

        # domain_z_mins is a torch tensor of shape [num_domains, d, top_k] containing the top_k smallest
        # values of the first d dimensions of z for each domain
        # domain_z_maxs is a torch tensor of shape [num_domains, d, top_k] containing the top_k largest
        # values of the first d dimensions of z for each domain
        domain_z_mins = torch.zeros((self.hparams.num_domains, self.hparams.z_dim, self.hparams.top_k))
        domain_z_maxs = torch.zeros((self.hparams.num_domains, self.hparams.z_dim, self.hparams.top_k))

        # z is [batch_size, latent_dim], so is domains. For the first d dimensions
        # of z, find the top_k smallest values of that dimension in each domain
        # find the mask of z's for each domain
        # for each domain, and for each of the first d dimensions, 
        # find the top_k smallest values of that z dimension in that domain
        for domain_idx in range(self.hparams.num_domains):
            domain_mask = (domains == domain_idx).squeeze()
            domain_z = z[domain_mask]
            # for each dimension i among the first d dimensions of z, find the top_k
            # smallest values of dimension i in domain_z
            for i in range(self.hparams.z_dim):
                domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0)
                domain_z_sorted = domain_z_sorted.squeeze()
                domain_z_sorted = domain_z_sorted[:self.hparams.top_k]
                domain_z_mins[domain_idx, i, :] = domain_z_sorted
                # find the top_k largest values of dimension i in domain_z
                domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0, descending=True)
                domain_z_sorted = domain_z_sorted.squeeze()
                domain_z_sorted = domain_z_sorted[:self.hparams.top_k]
                domain_z_maxs[domain_idx, i, :] = domain_z_sorted

        mse_mins = F.mse_loss(domain_z_mins[0], domain_z_mins[1], reduction="mean")
        mse_maxs = F.mse_loss(domain_z_maxs[0], domain_z_maxs[1], reduction="mean")
        
        penalty_loss = (mse_mins + mse_maxs) * self.hparams.penalty_weight

        loss = reconstruction_loss + penalty_loss
        return loss, reconstruction_loss, penalty_loss

    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels, domains, colors = train_batch["image"], train_batch["label"], train_batch["domain"], train_batch["color"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z, domains)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"penalty_loss", penalty_loss.item())
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
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        # fit a multinomial logistic regression from z to labels and colors
        # we have 4 classification tasks: 
        # 1. predicting labels from z[:z_dim]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim].detach().cpu().numpy(), labels.detach().cpu().numpy())
        pred_labels = clf.predict(z[:, :self.z_dim].detach().cpu().numpy())
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"digits_accuracy_z", accuracy, prog_bar=True)

        # 2. predicting colors from z[:z_dim]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, :self.z_dim].detach().cpu().numpy(), colors.detach().cpu().numpy())
        pred_colors = clf.predict(z[:, :self.z_dim].detach().cpu().numpy())
        accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
        self.log(f"colors_accuracy_z", accuracy, prog_bar=True)

        # 3. predicting labels from z[z_dim:]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim:].detach().cpu().numpy(), labels.detach().cpu().numpy())
        pred_labels = clf.predict(z[:, self.z_dim:].detach().cpu().numpy())
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"digits_accuracy_~z", accuracy, prog_bar=True)

        # 4. predicting colors from z[z_dim:]
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z[:, self.z_dim:].detach().cpu().numpy(), colors.detach().cpu().numpy())
        pred_colors = clf.predict(z[:, self.z_dim:].detach().cpu().numpy())
        accuracy = accuracy_score(colors.detach().cpu().numpy(), pred_colors)
        self.log(f"colors_accuracy_~z", accuracy, prog_bar=True)

        # overall accuracy with all z
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # predict the labels from z
        pred_labels = clf.predict(z.detach().cpu().numpy())
        # compute the accuracy
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"val_accuracy", accuracy, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z[:, :self.z_dim], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z[:, self.z_dim:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)


        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z, domains)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"valid_penalty_loss", penalty_loss.item())
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