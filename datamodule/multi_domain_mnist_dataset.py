from .abstract_mnist_dataset import MNISTBase
import torch
import torchvision
from torchvision import transforms
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

COLORS = [
    [1., 0., 0.], # red
    [0., 1., 0.], # green
    [0., 0., 1.], # blue
    [1., 1., 0.], # yellow
    [1., 0., 1.], # magenta
    [0., 1., 1.], # cyan
    [1., 1., 1.], # white
    [0., 0., 0.], # black
]
class MNISTMultiDomainDataset(MNISTBase):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        num_samples: int = 20000,
        path: str = "/network/datasets/torchvision",
        num_domains: int = 2,
        domain_lengths: Optional[list] = None,
        # list of list color indices for each domain
        domain_color_list: Optional[list] = None,
        domain_color_probs: Optional[list] = None,
        **kwargs,
    ):
        super(MNISTMultiDomainDataset, self).__init__(transform
                                            , num_samples
                                            , path
                                            ,**kwargs
                                            )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.path = path
        self.split = kwargs.get("split", "train")
        self.num_domains = num_domains
        self.domain_lengths = domain_lengths
        self.domain_color_list = domain_color_list
        self.domain_color_probs = domain_color_probs
        self.data = self._generate_data()

    def _generate_data(self):
        new_data = {}
        if self.split == "train":
            data = torchvision.datasets.MNIST(self.path, True, transform=self.transform)
        else:
            data = torchvision.datasets.MNIST(self.path, False, transform=self.transform)
        print(f"------------------ split: {self.split} ------------------")
        min_value_channel = data[0][0].min()
        # create the new data by extending each image with two more color channels
        # example: domain_indices[0] = [0, 2, 5, 6, 8, 9, 10], domain_indices[1] = [1, 3, 4, 7]
        domains_indices, mapping = random_split(list(range(len(data))), [len(data) * x for x in self.domain_lengths])

        # for each split, create the splits of colors
        for domain_idx in range(self.num_domains):
            # example: domain_indices[0] = domain_indices_ = [0, 2, 5, 6, 8, 9, 10]
            domain_indices_ = domains_indices[domain_idx] # some indices of the range(len(data))

            # splits of the range(len(domain_indices_)), i.e., domain_color_indices[0] is a list of indices of domain domain_idx that have the first color, and so on
            # example: domain_color_indices[0] = [0, 1, 4, 5], domain_color_indices[1] = [2, 3, 6]
            # , with the mapping to domain_indices_ being: domain_color_indices[0] = [0, 1, 4, 5] 
            # corresponding to domain_indices_[domain_color_indices[0]] = domain_indices_[[0, 1, 4, 5]] = [0, 2, 8, 9]
            domain_color_indices, _ = random_split(list(range(len(domain_indices_))), [len(domain_indices_) * x for x in self.domain_color_probs[domain_idx]])
            for color_idx in range(len(domain_color_indices)): # number of colors per each domain
                color_indices = domain_color_indices[color_idx] # indices of domain domain_idx that have the color color_idx, e.g., [0, 1, 4, 5]
                print(f"color_indices: {color_indices}")
                color = self.domain_color_list[domain_idx][color_idx] # selected from the list of colors for domain domain_idx
                for idx in color_indices: # for idx in [0, 1, 4, 5]
                    img, label = data[domain_indices_[idx]] # domain_indices_[idx] = [0, 2, 8, 9]
                    img = img.permute(1, 2, 0).repeat(1, 1, 3)
                    # change the color of bright pixels to color
                    digit_pixels = img[:, :, 0] > 0.3
                    bg_pixels = img[:, :, 0] <= 0.3
                    img[digit_pixels, :] = img[digit_pixels, :] * torch.tensor(COLORS[color]).float()
                    img[bg_pixels, :] = torch.tensor([min_value_channel, min_value_channel, min_value_channel]).float()
                    new_data[domain_indices_[idx]] = (img, label, mapping[domain_indices_[idx]], color)
                
        # new_data.append((img, label, mapping[domain_indices_[idx]], color))

        return new_data

    def __getitem__(self, idx):
        return {"image": self.data[idx][0], "label": self.data[idx][1], "domain": self.data[idx][2], "color": self.data[idx][3]}

import random

def random_split(array, sizes):
    if sum(sizes) != len(array):
        raise ValueError("Sum of sizes must equal the length of the array")

    indices = list(range(len(array)))
    random.shuffle(indices)

    splits = []
    start = 0
    split_id = 0
    mapping = {}
    for size in sizes:
        end = int(start + size)
        split_indices = indices[start:end]
        split = [array[i] for i in indices[start:end]]
        splits.append(split)
        for idx in split_indices:
            mapping[array[idx]] = split_id
        start = end
        split_id += 1

    return splits, mapping

