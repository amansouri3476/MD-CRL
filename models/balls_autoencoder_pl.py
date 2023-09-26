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


class BallsAutoencoderPL(BasePl):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # self.model = ResNet18Autoencoder()
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
        
        self.training_step_outputs = []
        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)

    def loss(self, images, recons, z):

        # images, recons: [batch_size, num_channels, width, height]
        reconstruction_loss = F.mse_loss(recons.permute(0, 2, 3, 1), images.permute(0, 2, 3, 1), reduction="mean")
        penalty_loss = 0.
        loss = reconstruction_loss + penalty_loss
        return loss, reconstruction_loss, penalty_loss

    def training_step(self, train_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images = train_batch["image"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, recons = self(images)
        log.info(f"images.max(): {images.max()}, recons.max(): {recons.max()}, images.min(): {images.min()}, recons.min(): {recons.min()}\n images.mean(): {images.mean()}, recons.mean(): {recons.mean()}")
        log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}, z_hat.std(): {z_hat.std()}")
        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z_hat)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"penalty_loss", penalty_loss.item())
        self.log(f"train_loss", loss.item(), prog_bar=True)

        self.training_step_outputs.append({"z_hat":z_hat})

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images = valid_batch["image"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z_hat, recons = self(images)

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z_hat)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"valid_penalty_loss", penalty_loss.item())
        self.log(f"val_loss", loss.item(), prog_bar=True)
        
        # fit a linear regression from z_hat to z
        z = valid_batch["z"] # [batch_size, n_balls * z_dim_ball]
        r2 = r2_score(z.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"hz_z_r2", r2, prog_bar=True)

        # fit a linear regression from z_hat to z_invariant dimensions
        z_invariant = valid_batch["z_invariant"] # [batch_size, n_balls_invariant * z_dim_ball]
        r2 = r2_score(z_invariant.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"hz_z_r2", r2, prog_bar=True)
        
        # fit a linear regression from z_hat to z_spurious dimensions
        z_spurious = valid_batch["z_spurious"] # [batch_size, n_balls_spurious * z_dim_ball]
        r2 = r2_score(z_spurious.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"hz_~z_r2", r2, prog_bar=True)

        # fit a linear regression from z to colours
        colors_ = valid_batch["color"] # valid_batch["color"]: [batch_size, n_balls_invariant + n_balls_spurious, 1]
        colors_ = colors_.reshape(colors_.shape[0], -1)
        r2 = r2_score(colors_.detach().cpu().numpy(), z_hat.detach().cpu().numpy())
        self.log(f"hz_colors_r2", r2, prog_bar=True)

        self.validation_step_outputs.append({"z_hat":z_hat})

        return {"loss": loss, "pred_z": z_hat}

    def on_train_epoch_end(self):

        # load the train dataset, and replace its "image" key with the new_data["z_hat"] key
        # and save it as a pt file
        train_dataset = torch.load(os.path.join(self.trainer.datamodule.path_to_files, f"train_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['train']-self.trainer.datamodule.num_samples['valid']}.pt"))
        new_data = dict.fromkeys(train_dataset.data[0].keys())

        # stack the values of each key in the new_data dict
        # new_data is a list of dicts
        for key in new_data.keys():
            new_data[key] = torch.stack([torch.tensor(train_dataset.data[i][key]) for i in range(len(train_dataset))], dim=0)
        new_data.pop("image", None)
        for key in self.training_step_outputs[0].keys():
            new_data[key] = torch.zeros((len(train_dataset), self.training_step_outputs[0][key].shape[-1]))

        for batch_idx, training_step_output in enumerate(self.training_step_outputs):
            # save the outputs of the encoder as a new dataset
            training_step_output_batch_size = list(training_step_output.values())[0].shape[0]
            start = batch_idx * self.trainer.datamodule.train_dataloader().batch_size
            end = start + min(self.trainer.datamodule.train_dataloader().batch_size, training_step_output_batch_size)
            for key, val in zip(training_step_output.keys(), training_step_output.values()):
                try:
                    new_data[key][start:end] = val.detach().cpu()
                except:
                    new_data[key][start:end] = val.unsqueeze(-1).detach().cpu()

        # save the new dataset as a pt file in hydra run dir or working directory
        log.info(f"Saving the encoded training dataset of length {len(new_data[key])} at: {os.getcwd()}")
        torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_train.pt"))
        self.training_step_outputs.clear()

        return
    
    def on_validation_epoch_end(self):

        # load the valid dataset, and replace its "image" key with the new_data["z_hat"] key
        # and save it as a pt file
        try:
            path = os.path.join(self.trainer.datamodule.path_to_files, f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
            valid_dataset = torch.load(path).dataset
            log.info(f"Loaded the validation dataset of length {len(valid_dataset)} from: {path}")
        except:
            path = os.path.join(self.trainer.datamodule.path_to_files, f"valid_dataset_{self.trainer.datamodule.datamodule_name}_{self.trainer.datamodule.num_samples['valid']}.pt")
            valid_dataset = torch.load(path)
            log.info(f"Loaded the validation dataset of length {len(valid_dataset)} from: {path}")
        new_data = dict.fromkeys(valid_dataset.data[0].keys())

        # stack the values of each key in the new_data dict
        # new_data is a list of dicts
        for key in new_data.keys():
            new_data[key] = torch.stack([torch.tensor(valid_dataset.data[i][key]) for i in range(len(valid_dataset))], dim=0)
        new_data.pop("image", None)
        for key in self.validation_step_outputs[0].keys():
            new_data[key] = torch.zeros((len(valid_dataset), self.validation_step_outputs[0][key].shape[-1]))

        for batch_idx, validation_step_output in enumerate(self.validation_step_outputs):
            # save the outputs of the encoder as a new dataset
            validation_step_output_batch_size = list(validation_step_output.values())[0].shape[0]
            start = batch_idx * self.trainer.datamodule.val_dataloader().batch_size
            end = start + min(self.trainer.datamodule.val_dataloader().batch_size, validation_step_output_batch_size)
            for key, val in zip(validation_step_output.keys(), validation_step_output.values()):
                try:
                    new_data[key][start:end] = val.detach().cpu()
                except:
                    new_data[key][start:end] = val.unsqueeze(-1).detach().cpu()
        
        # save the new dataset as a pt file in hydra run dir or working directory
        log.info(f"Saving the encoded validation dataset of length {len(new_data[key])} at: {os.getcwd()}")
        torch.save(new_data, os.path.join(os.getcwd(), f"encoded_img_{self.trainer.datamodule.datamodule_name}_valid.pt"))
        self.validation_step_outputs.clear()

        return
