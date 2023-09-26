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
from models.utils import penalty_loss_minmax, penalty_loss_stddev, penalty_domain_classification, hinge_loss


class BallsMDEncodedAutoencoderPL(BallsAutoencoderPL):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.penalty_weight = self.hparams.get("penalty_weight", 1.0)
        self.wait_steps = self.hparams.get("wait_steps", 0)
        self.linear_steps = self.hparams.get("linear_steps", 1)

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
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        # fit a linear regression from z_hat to z
        z = valid_batch["z"] # [batch_size, n_balls * z_dim_ball]
        r2 = r2_score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"r2_linreg", r2, prog_bar=True)

        # fit a linear regression from z_hat invariant dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        r2 = r2_score(z_invariant.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_z_r2_linreg", r2, prog_bar=True)
        
        # fit a linear regression from z_hat invariant dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        r2 = r2_score(z_spurious.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_~z_r2_linreg", r2, prog_bar=True)
        
        # fit a linear regression from z_hat spurious dims to z_invariant dimensions
        # z_invariant: [batch_size, n_balls_invariant * z_dim_ball]
        r2 = r2_score(z_invariant.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_z_r2_linreg", r2, prog_bar=False)
        
        # fit a linear regression from z_hat spurious dims to z_spurious dimensions
        # z_spurious: [batch_size, n_balls_spurious * z_dim_ball]
        r2 = r2_score(z_spurious.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_~z_r2_linereg", r2, prog_bar=False)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant_model], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=False)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant_model:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=False)

        # compute the regression scores with MLPRegressor
        hidden_layer_size = z_hat.shape[1]

        # fit a MLP regression from z_hat to z
        reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        r2_score = reg.score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"r2_mlpreg", r2_score, prog_bar=True)

        # fit a MLP regression from z_hat invariant dims to z_invariant dimensions
        reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_invariant.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2_score = reg.score(z_invariant.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_z_r2_mlpreg", r2_score, prog_bar=True)

        # fit a MLP regression from z_hat invariant dims to z_spurious dimensions
        reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_spurious.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        r2_score = reg.score(z_spurious.detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        self.log(f"hz_~z_r2_mlpreg", r2_score, prog_bar=True)

        # fit a MLP regression from z_hat spurious dims to z_invariant dimensions
        reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_invariant.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2_score = reg.score(z_invariant.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_z_r2_mlpreg", r2_score, prog_bar=False)

        # fit a MLP regression from z_hat spurious dims to z_spurious dimensions
        reg = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_spurious.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        r2_score = reg.score(z_spurious.detach().cpu().numpy(), z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        self.log(f"~hz_~z_r2_mlpreg", r2_score, prog_bar=False)

        # compute domain classification accuracy with multinoimal logistic regression for z_hat, z_invariant, z_spurious
        clf = LogisticRegression(random_state=0, max_iter=500).fit(z_hat.detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat.detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"domain_acc_logreg", acc, prog_bar=True)

        # domain prediction from zhat z_dim_invariant dimensions
        clf = LogisticRegression(random_state=0, max_iter=500).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"hz_domain_acc_logreg", acc, prog_bar=True)

        # domain prediction from zhat z_dim_spurious dimensions
        clf = LogisticRegression(random_state=0, max_iter=500).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"~hz_domain_acc_logreg", acc, prog_bar=True)

        # compute domain classification accuracy with MLPClassifier for z_hat, z_invariant, z_spurious
        clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_hat.detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat.detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"domain_acc_mlp", acc, prog_bar=True)

        # domain prediction from zhat z_dim_invariant dimensions
        clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat[:, :self.z_dim_invariant_model].detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"hz_domain_acc_mlp", acc, prog_bar=True)

        # domain prediction from zhat z_dim_spurious dimensions
        clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=(hidden_layer_size, hidden_layer_size)).fit(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy(), domain.detach().cpu().numpy())
        pred_domain = clf.predict(z_hat[:, self.z_dim_invariant_model:].detach().cpu().numpy())
        acc = accuracy_score(domain.detach().cpu().numpy(), pred_domain)
        self.log(f"~hz_domain_acc_mlp", acc, prog_bar=True)


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
