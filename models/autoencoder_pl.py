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
        images, labels = train_batch["image"], train_batch["label"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"train_reconstruction_loss", reconstruction_loss.item())
        # self.log(f"penalty_loss", penalty_loss.item())
        self.log(f"train_loss", loss.item())

        return loss

    def validation_step(self, valid_batch, batch_idx):

        # images: [batch_size, num_channels, width, height]
        images, labels = valid_batch["image"], valid_batch["label"]

        # z: [batch_size, latent_dim]
        # recons: [batch_size, num_channels, width, height]
        z, recons = self(images)

        # we have the set of labels and latents. We want to train a classifier to predict the labels from latents
        # using multinomial logistic regression using sklearn
        # import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        # fit a multinomial logistic regression from z to labels
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(z.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # predict the labels from z
        pred_labels = clf.predict(z.detach().cpu().numpy())
        # compute the accuracy
        accuracy = accuracy_score(labels.detach().cpu().numpy(), pred_labels)
        self.log(f"val_accuracy", accuracy, prog_bar=True)
        loss, reconstruction_loss, penalty_loss = self.loss(images, recons, z)
        self.log(f"val_reconstruction_loss", reconstruction_loss.item())
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


import torch.nn as nn
import torchvision.models as models

# Autoencoder with ResNet18 Encoder
class ResNet18Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet18Autoencoder, self).__init__()
        
        num_channels = 3
        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=True)
        # Modify the last fully connected layer to output 64 features
        z_dim = kwargs.get("z_dim", 64)
        resnet18.fc = nn.Linear(512, z_dim)
        self.encoder = resnet18 # nn.Sequential(*list(resnet18.children())[:-2])  # Exclude the last two layers

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # # nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Output range [0, 1] for images
        )


    def forward(self, x):

        # x: [batch_size, num_channels, width, height]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.view(encoded.size(0), 64, 1, 1))
        return encoded, decoded
