import torch
import torchvision
from torchvision import transforms
import numpy as np
import random
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
log_ = True

class SyntheticMixingDataset(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        generation_strategy: str = "auto",
        num_samples: int = 20000,
        num_domains: int = 2,
        z_dim: int = 10,
        z_dim_invariant: int = 5,
        x_dim: int = 20,
        domain_lengths: Optional[list] = None,
        domain_dist_ranges: Optional[list] = None,
        domain_dist_ranges_pos: Optional[list] = None,
        domain_dist_ranges_neg: Optional[list] = None,
        invariant_dist_params: Optional[list] = None,
        linear: bool = True,
        **kwargs,
    ):
        super(SyntheticMixingDataset, self).__init__()

        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.generation_strategy = generation_strategy
        self.z_dim = z_dim
        self.z_dim_invariant = z_dim_invariant
        self.z_dim_spurious = int(z_dim - z_dim_invariant)
        self.x_dim = x_dim
        self.num_samples = num_samples
        self.num_domains = num_domains
        self.domain_lengths = domain_lengths if generation_strategy else [1 / num_domains] * num_domains
        self.domain_dist_ranges = domain_dist_ranges
        self.invariant_dist_params = invariant_dist_params
        self.mixing_architecture_config = kwargs["mixing_architecture"]
        self.non_linearity = kwargs["non_linearity"]
        self.polynomial_degree = kwargs["polynomial_degree"]
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

        if self.generation_strategy == "manual":
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
        else:
            # for each dimension of the spurious part of z, for each domain, first decide whether it is positive or negative
            # by tossing a fair coin. Then sample the low and high of this domain from the corresponding domain_dist_ranges_pos
            # or domain_dist_ranges_neg. Then sample from uniform [low, high] for each domain. 
            # Repeat this procedure for all spurious dimensions of z.
            domain_mask = torch.zeros(self.num_samples, 1)
            for dim_idx in range(self.z_dim_spurious):
                for domain_idx in range(self.num_domains):
                    # toss a fair coin to decide whether this dimension is positive or negative
                    coin = random.randint(0, 1)
                    if coin == 0:
                        # sample low and high of the domain distribution from the range
                        # specified in negative domain distribution. Make sure low < high
                        low = random.rand() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                        high = random.rand() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                        while low > high:
                            low = random.rand() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                            high = random.rand() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                    else:
                        # sample low and high of the domain distribution from the range
                        # specified in positive domain distribution. Make sure low < high
                        low = random.rand() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                        high = random.rand() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                        while low > high:
                            low = random.rand() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                            high = random.rand() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                    
                    # sample from uniform [low, high] for each domain
                    domain_size = int(self.domain_lengths[domain_idx] * self.num_samples)
                    start = int(domain_size * domain_idx)
                    end = start + domain_size
                    domain_data = torch.rand(domain_size, 1) * (high - low) + low
                    z_data[start:end, self.z_dim_invariant + dim_idx] = domain_data
                    domain_mask[start:end] = domain_idx

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
            if self.non_linearity == "mlp":
                # instantiate the non-linear G with hydra
                self.G = torch.nn.Sequential(
                *[layer_config for _, layer_config in self.mixing_architecture_config.items()]
                )

                # print all of the parameters of self.G
                if log_:
                    log.info("G params:")
                    for name, param in self.G.named_parameters():
                        log.info(f"{name}: {param}")
                # return a lambda function that takes a batch of z and returns Gz
                # make sure the output does not require any grads, and is simply a torch tensor
                return lambda z: self.G(z).detach()
            elif self.non_linearity == "polynomial":
                # return a function that takes a batch of z and returns a polynomial of z
                # Generate random coefficients for the polynomial
                coefficients = torch.randn(self.polynomial_degree + 1, z_dim)
                def polynomial_function(z):
                    return torch.sum(coefficients.unsqueeze(0) * z.unsqueeze(1).pow(torch.arange(self.polynomial_degree + 1, dtype=z.dtype, device=z.device)), dim=2)
                self.G = polynomial_function
                
                # Define the polynomial function using lambda
                return lambda z: self.G(z).detach()

