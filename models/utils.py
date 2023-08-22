import numpy as np
import torch
from torch import nn
import collections.abc

def set_seed(args):
    np.random.seed(args.seed)

    # This makes sure that the seed is used for random initialization of nn modules provided by nn init
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
# Shouldn't we use this last one?
#     torch.backends.cudnn.deterministic = True


# Don't worry about randomization and seed here. It's taken care of by set_seed above, and pl seed_everything
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


from torch.nn import functional as F
def penalty_loss(z, domains, num_domains, top_k, z_dim_invar):

    # domain_z_mins is a torch tensor of shape [num_domains, d, top_k] containing the top_k smallest
    # values of the first d dimensions of z for each domain
    # domain_z_maxs is a torch tensor of shape [num_domains, d, top_k] containing the top_k largest
    # values of the first d dimensions of z for each domain
    domain_z_mins = torch.zeros((num_domains, z_dim_invar, top_k))
    domain_z_maxs = torch.zeros((num_domains, z_dim_invar, top_k))

    # z is [batch_size, latent_dim], so is domains. For the first d dimensions
    # of z, find the top_k smallest values of that dimension in each domain
    # find the mask of z's for each domain
    # for each domain, and for each of the first d dimensions, 
    # find the top_k smallest values of that z dimension in that domain
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]
        # for each dimension i among the first d dimensions of z, find the top_k
        # smallest values of dimension i in domain_z
        for i in range(z_dim_invar):
            domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0)
            domain_z_sorted = domain_z_sorted.squeeze()
            domain_z_sorted = domain_z_sorted[:top_k]
            domain_z_mins[domain_idx, i, :] = domain_z_sorted
            # find the top_k largest values of dimension i in domain_z
            domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0, descending=True)
            domain_z_sorted = domain_z_sorted.squeeze()
            domain_z_sorted = domain_z_sorted[:top_k]
            domain_z_maxs[domain_idx, i, :] = domain_z_sorted

    mse_mins = F.mse_loss(domain_z_mins[0], domain_z_mins[1], reduction="mean")
    mse_maxs = F.mse_loss(domain_z_maxs[0], domain_z_maxs[1], reduction="mean")
    
    return (mse_mins + mse_maxs)
