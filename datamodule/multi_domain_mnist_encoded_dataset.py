import torch
import torchvision
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
import os


class MNISTMultiDomainEncodedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        num_domains: int = 2,
        **kwargs,
    ):
        super(MNISTMultiDomainEncodedDataset, self).__init__()

        self.split = kwargs.get("split", "train")
        self.num_domains = num_domains
        self.generation_strategy = kwargs.get("generation_strategy", "auto")
        self.path_to_files = kwargs.get("data_dir", None)
        self.data = torch.load(os.path.join(self.path_to_files, f"encoded_img_multi_domain_mnist_{self.num_domains}_{self.split}.pt"))

    def __getitem__(self, idx):
        return {"x": self.data["z"][idx], "label": self.data["label"][idx], "domain": self.data["domain"][idx], "color": self.data["color"][idx]}

    def __len__(self):
        return len(self.data["z"])
