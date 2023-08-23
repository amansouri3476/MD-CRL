import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement
from models.utils import penalty_loss


class MixingAutoencoderPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.z_dim_invariant = self.hparams.z_dim_invariant
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False)
        if self.hparams.get("autoencoder_ckpt_path", None) is not None:    
            ckpt_path = self.hparams["autoencoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))
            self.hparams.pop("autoencoder_ckpt_path") # we don't want this to be save with the ckpt, sicne it will raise key errors when we further train the model
                                                  # and load it for evaluation.

        # freeze the parameters of encoder if needed
        if self.hparams.autoencoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        else: # if the flag is set to true we should correct the requires_grad flags, i.e. we might
              # initially freeze it for some time, but then decide to let it finetune.
            for param in self.model.parameters():
                param.requires_grad = True


    def forward(self, x):
        return self.model(x)

    def loss(self, x, x_hat, z_hat, domains):

        # x, x_hat: [batch_size, x_dim]
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        
        penalty_loss_value = penalty_loss(z_hat, domains, self.hparams.num_domains, self.hparams.top_k, self.z_dim_invariant)
        loss = reconstruction_loss + penalty_loss_value * self.hparams.penalty_weight
        return loss, reconstruction_loss, penalty_loss_value

    def training_step(self, train_batch, batch_idx):

        # x: [batch_size, x_dim]
        x, z, domain = train_batch["x"], train_batch["z"], train_batch["domain"]

        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        loss, reconstruction_loss, penalty_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        self.log(f"train_penalty_loss", penalty_loss_value.item())
        self.log(f"train_loss", loss.item())

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # x: [batch_size, x_dim]
        x, z, domain = valid_batch["x"], valid_batch["z"], valid_batch["domain"]


        # z: [batch_size, latent_dim]
        # x_hat: [batch_size, x_dim]
        z_hat, x_hat = self(x)

        print(f"============== z_hat min domain 1 ==============\n{z_hat[(domain == 0).squeeze(), :2].min()}\n")
        print(f"============== z_hat min domain 2 ==============\n{z_hat[(domain == 1).squeeze(), :2].min()}\n")
        print(f"============== z_hat max domain 1 ==============\n{z_hat[(domain == 0).squeeze(), :2].max()}\n")
        print(f"============== z_hat max domain 2 ==============\n{z_hat[(domain == 1).squeeze(), :2].max()}\n")

        # we have the set of z and z_hat. We want to train a linear regression to predict the
        # z from z_hat using sklearn, and report regression scores
        # import linear regression from sklearn
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics

        # noise = torch.randn_like(z) * 0.00
        # noise.to(z.device)
        # create a linear regression object
        reg = LinearRegression()
        # fit the linear regression object to the data
        reg.fit(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        # get the r2 score of the linear regression
        # r2_score = metrics.r2_score(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        # create a noise tensor of size [batch_size, z_dim]
        # TODO: log [5x5] matrix of regression | regression from 
        # TOOD: use multiple domains, flip the signs. [-1,2] [2,3]
        # z_hat[:5] from z (whole), or two regression
        # z_hat[:5] from both parts of z
        # r2_score = reg.score(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy())
        r2_score = reg.score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        # log the score
        self.log(f"val_r2", r2_score, prog_bar=True)
        # print(f"======linear regression weights:\n{reg.coef_}")

        # we have 4 regression tasks: 
        # 1. predicting z[:z_dim_invariant] from z_hat[:z_dim_invariant]
        reg = LinearRegression()
        # reg.fit(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        reg.fit(z[:, :self.z_dim_invariant].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant].detach().cpu().numpy())
        # r2_score = metrics.r2_score(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        r2_score = reg.score(z[:, :self.z_dim_invariant].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant].detach().cpu().numpy())
        self.log(f"val_r2_hz_z", r2_score, prog_bar=True)

        # 2. predicting z[:z_dim_invariant] from z_hat[z_dim_invariant:]
        reg = LinearRegression()
        # reg.fit(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        reg.fit(z[:, self.z_dim_invariant:].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant].detach().cpu().numpy())
        # r2_score = metrics.r2_score(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy())
        r2_score = reg.score(z[:, self.z_dim_invariant:].detach().cpu().numpy(), z_hat[:, :self.z_dim_invariant].detach().cpu().numpy())
        self.log(f"val_r2_~hz_z", r2_score, prog_bar=True)

        # 3. predicting z[:z_dim_invariant] from z_hat[:z_dim_invariant]
        reg = LinearRegression()
        # reg.fit(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        reg.fit(z[:, :self.z_dim_invariant].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant:].detach().cpu().numpy())
        # r2_score = metrics.r2_score(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        r2_score = reg.score(z[:, :self.z_dim_invariant].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant:].detach().cpu().numpy())
        self.log(f"val_r2_hz_~z", r2_score, prog_bar=True)

        # 4. predicting z[z_dim_invariant:] from z_hat[z_dim_invariant:]
        reg = LinearRegression()
        # reg.fit(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        reg.fit(z[:, self.z_dim_invariant:].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant:].detach().cpu().numpy())
        # r2_score = metrics.r2_score(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        # r2_score = reg.score(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy())
        r2_score = reg.score(z[:, self.z_dim_invariant:].detach().cpu().numpy(), z_hat[:, self.z_dim_invariant:].detach().cpu().numpy())
        self.log(f"val_r2_~hz_~z", r2_score, prog_bar=True)

        # comptue the average norm of first z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, :self.z_dim_invariant], dim=1).mean()
        self.log(f"z_norm", z_norm, prog_bar=True)
        # comptue the average norm of the last n-z_dim dimensions of z
        z_norm = torch.norm(z_hat[:, self.z_dim_invariant:], dim=1).mean()
        self.log(f"~z_norm", z_norm, prog_bar=True)

        loss, reconstruction_loss, penalty_loss_value = self.loss(x, x_hat, z_hat, domain)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        self.log(f"val_penalty_loss", penalty_loss_value.item())
        self.log(f"val_loss", loss.item(), prog_bar=True)

        # print the weights of the decoder
        # print(f"=============== Model Weights ===============:\n{self.model.decoder_fc.layers[0].weight}")
        # print(f"=============== Mixing G ===============:\n{self.trainer.datamodule.train_dataset.dataset.G.t()}")

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 1, figsize=(10, 4))
        # ax[0].matshow(self.model.decoder_fc.layers[0].weight.detach().cpu().numpy())
        # ax[0].set_title('Decoder Weight')
        # ax[1].matshow(self.trainer.datamodule.train_dataset.dataset.G.t())
        # ax[1].set_title('Mixing G')
        # import wandb
        # wandb.log({f"Decoder W vs. G": fig})
            
        # ax[0,0].plot(z_hat.detach().cpu().numpy(), z.detach().cpu().numpy(), ".")
        # ax[0,0].set_title('all z_hat vs. all z')
        # ax[0,0].set_title('all z_hat vs. all z')
        # ax[0,1].plot(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy(), ".")
        # ax[0,1].set_title('z_hat vs. z')
        # ax[0,2].plot(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, :self.z_dim_invariant].detach().cpu().numpy(), ".")
        # ax[0,2].set_title('~z_hat vs. z')
        # ax[0,3].plot(z_hat[:, :self.z_dim_invariant].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy(), ".")
        # ax[0,3].set_title('z_hat vs. ~z')
        # ax[0,4].plot(z_hat[:, self.z_dim_invariant:].detach().cpu().numpy(), z[:, self.z_dim_invariant:].detach().cpu().numpy(), ".")
        # ax[0,4].set_title('~z_hat vs. ~z')

        # for idx in range(5):            
        #     ax[0,idx].grid(False)
        #     ax[0,idx].axis('off')
        #     ax[1,idx].grid(False)
        #     ax[1,idx].axis('off')

        # self.log({f"Regressions": fig})


        return {"loss": loss, "pred_z": z}
