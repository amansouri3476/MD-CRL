import torch
from torch.nn import functional as F
from .base_pl import BasePl
import hydra
from omegaconf import OmegaConf
import wandb
from utils.disentanglement_utils import linear_disentanglement, permutation_disentanglement


class AutoencoderPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.model = hydra.utils.instantiate(self.hparams.autoencoder, _recursive_=False)
        if self.hparams.get("autoencoder_ckpt_path", None) is not None:    
            ckpt_path = self.hparams["autoencoder_ckpt_path"]
            # only load the weights, i.e. HPs should be overwritten from the passed config
            # b/c maybe the ckpt has num_slots=7, but we want to test it w/ num_slots=12
            # NOTE: NEVER DO self.model = self.model.load_state_dict(...), raises _IncompatibleKey error
            self.model.load_state_dict(torch.load(ckpt_path))
            self.hparams.pop("autoencoder_ckpt_path") # we don't want this to be save with the ckpt, sicne it will raise key errors when we further train the model
                                                  # and load it for evaluation.

        # remove the state_dict_randomstring.ckpt to avoid cluttering the space
        import os
        import glob
        state_dicts_list = glob.glob('./state_dict_*.pth')
        # for state_dict_ckpt in state_dicts_list:
        #     try:
        #         os.remove(state_dict_ckpt)
        #     except:
        #         print("Error while deleting file: ", state_dict_ckpt)

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

    def loss(self, images, recons, z):

        reconstruction_loss = F.mse_loss(recons, images, reduction="mean")
        penalty_loss = 0.
        loss = reconstruction_loss + penalty_loss
        return loss, reconstruction_loss, penalty_loss

    def configure_optimizers(self):

        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                   , params
                                                                  )
        
        if self.hparams.get("scheduler_config"):
            # for pytorch scheduler objects, we should use utils.instantiate()
            if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
                scheduler = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer)

            # for transformer function calls, we should use utils.call()
            elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
                scheduler = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer)
            
            else:
                raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                
            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict["scheduler"] = scheduler

            return [optimizer], [scheduler_dict]
        else:
            # no scheduling
            return [optimizer]


    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        z, recons = self(images)
        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"penalty_loss", penalty_loss.item())
        self.log(f"train_loss", loss.item())

        return loss

    def validation_step(self, valid_batch, batch_idx):
        images, labels = valid_batch
        z, recons = self(images)
        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"valid_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"valid_penalty_loss", penalty_loss.item())
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