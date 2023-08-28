import torch
import torchvision
from torchvision import transforms
import numpy as np
import random
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = False

class SyntheticMixingDataset(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        num_samples: int = 20000,
        num_domains: int = 2,
        z_dim: int = 10,
        z_dim_invariant: int = 5,
        x_dim: int = 20,
        domain_lengths: Optional[list] = None,
        domain_dist_ranges: Optional[list] = None,
        invariant_dist_params: Optional[list] = None,
        linear: bool = True,
        **kwargs,
    ):
        super(SyntheticMixingDataset, self).__init__()

        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.z_dim = z_dim
        self.z_dim_invariant = z_dim_invariant
        self.z_dim_spurious = int(z_dim - z_dim_invariant)
        self.x_dim = x_dim
        self.num_samples = num_samples
        self.num_domains = num_domains
        self.domain_lengths = domain_lengths
        self.domain_dist_ranges = domain_dist_ranges
        self.invariant_dist_params = invariant_dist_params
        self.mixing_G = self._generate_mixing_G(linear, z_dim, x_dim)
        self.data = self._generate_data()

    def __len__(self) -> int:
        return self.num_samples

    def _generate_data(self):

        # data is a tensor of size [num_samples, z_dim] where the first z_dim_invariant dimensions are sampled from uniform [0,1]
        z_data = torch.zeros(self.num_samples, self.z_dim)
        # the first z_dim_invariant dimensions are sampled from uniform [0,1]
        z_data_invar = torch.rand(self.num_samples, self.z_dim_invariant) * (self.invariant_dist_params[1] - self.invariant_dist_params[0]) + self.invariant_dist_params[0]
        z_data[:, :self.z_dim_invariant] = z_data_invar

        # for each domain, create its data, i.e., a tensor of size [num_samples, z_dim-z_dim_invariant] 
        # where each dimension is sampled from uniform [domain_dist_ranges[domain_idx][0], domain_dist_ranges[domain_idx][0]]
        domain_mask = torch.zeros(self.num_samples, 1)
        start = 0
        for domain_idx in range(self.num_domains):
            domain_size = int(self.domain_lengths[domain_idx] * self.num_samples)
            end = domain_size + start
            domain_data = torch.rand(domain_size, self.z_dim_spurious) * (self.domain_dist_ranges[domain_idx][1] - self.domain_dist_ranges[domain_idx][0]) + self.domain_dist_ranges[domain_idx][0]
            z_data[start:end, self.z_dim_invariant:] = domain_data
            domain_mask[start:end] = domain_idx
            start = end

        # now the data is a tensor of size [num_samples, z_dim] where the first z_dim_invariant
        # dimensions are sampled from uniform [0,1], and the rest are sampled according to domain distributions
        # now shuffle the data and the domain mask similarly and create the final data
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        z_data = z_data[indices]
        domain_mask = domain_mask[indices]
        x_data = self.mixing_G(z_data)

        return x_data, z_data, domain_mask

    def __getitem__(self, idx):
        return {"x": self.data[0][idx], "z": self.data[1][idx], "domain": self.data[2][idx]}

    def _generate_mixing_G(self, linear, z_dim, x_dim):
        if linear:
            # create an invertible matrix mapping from z_dim to x_dim for which
            # the entries are sampled from uniform [0,1]
            # this matrix should be operating on batches of size [num_samples, z_dim]

            # TODO: This G should be used by all splits of the dataset!
            G = torch.rand(z_dim, x_dim)
            # make sure the above G is full rank

            while np.linalg.matrix_rank(G) < min(z_dim, x_dim):
                G = torch.rand(z_dim, x_dim)
            self.G = G
            # return a lambda function that takes a batch of z and returns Gz
            return lambda z: torch.matmul(z, G)

        else:
            pass