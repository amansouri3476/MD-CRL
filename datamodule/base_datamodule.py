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
        self.path_to_files = self.hparams["data_dir"]
        self.save_dataset = self.hparams.save_dataset
        self.load_dataset = self.hparams.load_dataset
        self.dataset_name = self.hparams.dataset_name
        self.datamodule_name = self.hparams.datamodule_name
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

        if not self.load_dataset:
            log.info(f"Generating the data from scratch.")
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                                hydra.utils.instantiate(self.dataset_parameters["train"]["dataset"]
                                                                , transform=transform
                                                                , augmentations=augmentations
                                                                )
                                , lengths=lengths
                                # , generator=torch.Generator().manual_seed(self.seed)
            )
            log.info(f"\n---------------------------\n---------------------------\ntrain_dataset size: {len(self.train_dataset)}\nvalid_dataset size: {len(self.valid_dataset)}\n---------------------------\n---------------------------")
            # self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(100))
            # self.valid_dataset = torch.utils.data.Subset(self.valid_dataset, range(100))

            # TODO: This way, G is different among splits
            self.test_dataset = hydra.utils.instantiate(self.dataset_parameters["test"]["dataset"]
                                                                , transform=transform
                                                                , augmentations=augmentations
                                                                )
        else:
            # log the tnime it takes to load the dataset
            import time
            start_time = time.perf_counter()
            log.info(f"Loading the whole dataset files from {self.path_to_files}")
            self.train_dataset = torch.load(os.path.join(self.path_to_files, f"train_dataset_{self.datamodule_name}_{self.num_samples['train']}.pt"))
            log.info(f"Loading the train dataset files took {time.perf_counter() - start_time} seconds.")
            start_time = time.perf_counter()
            self.valid_dataset = torch.load(os.path.join(self.path_to_files, f"valid_dataset_{self.datamodule_name}_{self.num_samples['valid']}.pt"))
            log.info(f"Loading the valid dataset files took {time.perf_counter() - start_time} seconds.")
            # self.test_dataset = torch.load(os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))

        if self.save_dataset:
            if not os.path.exists(self.path_to_files):
                os.makedirs(self.path_to_files)
            log.info(f"Saving the whole dataset files to {self.path_to_files}")
            # torch.save(self.train_dataset, os.path.join(self.path_to_files, f"train_dataset_{self.dataset_name}_{self.num_samples['train']}.pt"))
            # torch.save(self.valid_dataset, os.path.join(self.path_to_files, f"valid_dataset_{self.dataset_name}_{self.num_samples['valid']}.pt"))
            # torch.save(self.test_dataset, os.path.join(self.path_to_files, f"test_dataset_{self.dataset_name}_{self.num_samples['test']}.pt"))
            torch.save(self.train_dataset, os.path.join(self.path_to_files, f"train_dataset_{self.datamodule_name}_{len(self.train_dataset)}.pt"))
            torch.save(self.valid_dataset, os.path.join(self.path_to_files, f"valid_dataset_{self.datamodule_name}_{len(self.valid_dataset)}.pt"))
            # torch.save(self.test_dataset, os.path.join(self.path_to_files, f"test_dataset_{self.datamodule_name}_{len(self.test_dataset)}.pt"))



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
