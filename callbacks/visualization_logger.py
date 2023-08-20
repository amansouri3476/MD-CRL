from pytorch_lightning.callbacks import Callback
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance


class VisualizationLoggerCallback(Callback):
    

    def __init__(self, visualize_every=100, n_samples=3, **kwargs):
        
        self.visualize_every = visualize_every
        self.n_samples = n_samples
    
    def on_fit_start(self, trainer, pl_module):
        
        self.datamodule = trainer.datamodule
        self.model = pl_module.model
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % (self.visualize_every+1) == 0:
            
            pl_module.eval()
            with torch.no_grad():
            
                images = batch[0][:self.n_samples]
                z, recons = self.model(images)

                # renormalize image and recons?

                fig, ax = plt.subplots(2, self.n_samples, figsize=(10, 4))
                for idx in range(self.n_samples):
                    image = images[idx].squeeze().cpu().numpy()
                    recon_ = recons[idx].squeeze().cpu().numpy()

                    # Visualize.
                    ax[0,idx].imshow(image, cmap="gray")
                    ax[0,idx].set_title('Image')
                    ax[1,idx].imshow((recon_ * 255).astype(np.uint8), vmin=0, vmax=255, cmap="gray")
                    ax[1,idx].set_title('Recon.')
                    ax[0,idx].grid(False)
                    ax[0,idx].axis('off')
                    ax[1,idx].grid(False)
                    ax[1,idx].axis('off')

                wandb.log({f"Reconstructions": fig})

            
