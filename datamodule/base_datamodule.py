import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os.path as path
import hydra
import numpy as np
from datamodule.augmentations import ALL_AUGMENTATIONS

AUGMENTATIONS = {k: lambda x: v(x, order=1) for k, v in ALL_AUGMENTATIONS.items()}

import utils.general as utils
log = utils.get_logger(__name__)
    

class BaseDataModule(LightningDataModule):
    def __init__(self
                 , seed: int = 1234
                 , batch_size: int= 128
                 , **kwargs):
        
        super().__init__()
        
        # So all passed parameters are accessible through self.hparams
        self.save_hyperparameters(logger=False)
        
        self.seed = seed
        self.dataset_parameters = self.hparams.dataset["dataset_parameters"]
        
        self.num_samples = {}
        for split in self.dataset_parameters.keys(): # splits: ['train','valid','test']
            self.num_samples[split] = self.dataset_parameters[split]["dataset"]["num_samples"]
        
        self.dirname = os.path.dirname(__file__)
        self.dataset_name = self.hparams.dataset_name
        self.transforms = self.hparams.transforms
        self.augs = self.hparams.transform.get("augs", None)        

    def prepare_data(self):
        """
        Docs: Use this method to do things that might write to disk or that need to be done only from 
        a single process in distributed settings.
        - download
        - tokenize
        - etc.
        """
    
        if self.augs is not None:
            augmentations = [v for (k, v) in AUGMENTATIONS.items() if k in self.augs]
        else:
            augmentations = []

        if self.transforms is not None:
            transform = transforms.Compose([hydra.utils.instantiate(t) for _, t in self.transforms.items()])
        else:
            transform = []
        
        if self.num_samples["train"] < 1:
            lengths = [self.num_samples["train"], self.num_samples["valid"]]
        else:
            lengths = [self.num_samples["train"]-self.num_samples["valid"], self.num_samples["valid"]]

        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                            hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"]
                                                            , transform=transform
                                                            , augmentations=augmentations
                                                            )
                            , lengths=lengths
                            , generator=torch.Generator().manual_seed(self.seed)
        )

        self.test_dataset = hydra.utils.instantiate(self.dataset_parameters["test"]["dataset"]
                                                            , transform=transform
                                                            , augmentations=augmentations
                                                            )



    def setup(self, stage=None):
        """
        Docs: There are also data operations you might want to perform on every GPU. Use setup to do 
        things like:
        - count number of classes
        - build vocabulary
        - perform train/val/test splits
        - apply transforms (defined explicitly in your datamodule)
        - etc.
        """
        pass
        

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.train_dataset,
            **self.dataset_parameters['train']['dataloader'],
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset, **self.dataset_parameters['valid']['dataloader']
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, **self.dataset_parameters['test']['dataloader']
        )
    
    def renormalize(self):
        for _, t in self.transforms.items():
            if "Standardize" in t["_target_"]:
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
            
            # TODO: add more options if required
    
    
    
    
