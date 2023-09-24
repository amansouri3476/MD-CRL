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
def penalty_loss_minmax(z, domains, num_domains, z_dim_invariant, *args):

    top_k = args[0]
    loss_transform = args[1]
    # domain_z_mins is a torch tensor of shape [num_domains, d, top_k] containing the top_k smallest
    # values of the first d dimensions of z for each domain
    # domain_z_maxs is a torch tensor of shape [num_domains, d, top_k] containing the top_k largest
    # values of the first d dimensions of z for each domain
    domain_z_mins = torch.zeros((num_domains, z_dim_invariant, top_k))
    domain_z_maxs = torch.zeros((num_domains, z_dim_invariant, top_k))

    # z is [batch_size, latent_dim], so is domains. For the first d dimensions
    # of z, find the top_k smallest values of that dimension in each domain
    # find the mask of z's for each domain
    # for each domain, and for each of the first d dimensions, 
    # find the top_k smallest values of that z dimension in that domain
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]

        if domain_z.shape[0] < top_k:
            print(f"WARNING: domain_z.shape[0] < top_k for domain {domain_idx}")
            domain_z_mins[domain_idx, :, domain_z.shape[0]:] = 0.
            domain_z_maxs[domain_idx, :, domain_z.shape[0]:] = 0.
        for i in range(z_dim_invariant):
            # for each dimension i among the first d dimensions of z, find the top_k
            # smallest values of dimension i in domain_z
            domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0)
            domain_z_sorted = domain_z_sorted.squeeze()
            domain_z_sorted = domain_z_sorted[:top_k]
            if domain_z.shape[0] < top_k:
                domain_z_mins[domain_idx, i, :domain_z.shape[0]] = domain_z_sorted
            else:
                domain_z_mins[domain_idx, i, :] = domain_z_sorted

            # find the top_k largest values of dimension i in domain_z
            domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0, descending=True)
            domain_z_sorted = domain_z_sorted.squeeze()
            if domain_z.shape[0] < top_k:
                domain_z_maxs[domain_idx, i, :domain_z.shape[0]] = domain_z_sorted
            else:
                domain_z_sorted = domain_z_sorted[:top_k]
                domain_z_maxs[domain_idx, i, :] = domain_z_sorted

    # compute the pairwise mse of domain_z_mins and add them all together. Same for domain_z_maxs
    mse_mins = 0
    mse_maxs = 0
    if loss_transform == "mse":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                mse_mins += F.mse_loss(domain_z_mins[i], domain_z_mins[j], reduction="mean")
                mse_maxs += F.mse_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="mean")
    elif loss_transform == "l1":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                mse_mins += F.l1_loss(domain_z_mins[i], domain_z_mins[j], reduction="mean")
                mse_maxs += F.l1_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="mean")
    elif loss_transform == "log_mse":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                # mse_mins += F.mse_loss(torch.log(domain_z_mins[i]), torch.log(domain_z_mins[j]), reduction="mean")
                # mse_maxs += F.mse_loss(torch.log(domain_z_maxs[i]), torch.log(domain_z_maxs[j]), reduction="mean")
                # mse_mins += torch.log(F.mse_loss(domain_z_mins[i], domain_z_mins[j], reduction="none")).mean()
                # mse_maxs += torch.log(F.mse_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="none")).mean()
                mse_mins += torch.log((domain_z_mins[i] - domain_z_mins[j])**2).mean()
                mse_maxs += torch.log((domain_z_maxs[i] - domain_z_maxs[j])**2).mean()
    elif loss_transform == "log_l1":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                # mse_mins += F.l1_loss(torch.log(domain_z_mins[i]), torch.log(domain_z_mins[j]), reduction="mean")
                # mse_maxs += F.l1_loss(torch.log(domain_z_maxs[i]), torch.log(domain_z_maxs[j]), reduction="mean")
                mse_mins += torch.log(F.l1_loss(domain_z_mins[i], domain_z_mins[j], reduction="none")).mean()
                mse_maxs += torch.log(F.l1_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="none")).mean()

    # hinge_loss_value = hinge_loss(z, domains, num_domains, z_dim_invariant, *args[2:])
    # return (mse_mins + mse_maxs) + hinge_loss_value, hinge_loss_value
    return (mse_mins + mse_maxs)

def penalty_loss_stddev(z, domains, num_domains, z_dim_invariant, *args):
    
    # domain_z_invariant_stddev is a torch tensor of shape [num_domains, d] containing the standard deviation
    # of the first d dimensions of z for each domain
    domain_z_invariant_stddev = torch.zeros((num_domains, z_dim_invariant)).to(z.device)

    # z is [batch_size, latent_dim], so is domains. For the first d dimensions
    # of z, find the standard deviation of that dimension in each domain
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]
        # for each dimension i among the first d dimensions of z, find the standard deviation
        # of dimension i in domain_z
        for i in range(z_dim_invariant):
            domain_z_stddev = torch.std(domain_z[:, i], dim=0)
            domain_z_invariant_stddev[domain_idx, i] = domain_z_stddev
    
    # for each of the d dimensions, compute the pairwise mse of its stddev across domains
    # and add them all together in mse_stddev. mse_stddev is a tensor of size [d]
    mse_stddev = torch.zeros(z_dim_invariant).to(z.device)
    for i in range(z_dim_invariant):
        for j in range(num_domains):
            for k in range(j+1, num_domains):
                mse_stddev[i] += F.mse_loss(domain_z_invariant_stddev[j, i], domain_z_invariant_stddev[k, i], reduction="mean")

    # hinge_loss_value = hinge_loss(z, domains, num_domains, z_dim_invariant, *args)
    # return mse_stddev.sum() + hinge_loss_value, hinge_loss_value
    return mse_stddev.sum()


def hinge_loss(z, domains, num_domains, z_dim_invariant, *args):
    
    gamma = args[0]
    epsilon = args[1]
    hinge_loss_weight = args[2]

    # compute the variance regularization term using the hinge loss along each dimension of z.
    # The hinge loss is 0 if the variance is greater than gamma, and gamma - sqrt(variance + epsilon)
    # otherwise. The variance regularization term is the sum of the hinge losses along each dimension
    # of z
    variance_reg = torch.zeros(num_domains).to(z.device)
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]
        # for each dimension i among the first d dimensions of z, find the variance
        # of dimension i in domain_z
        for i in range(z_dim_invariant):
            variance_reg[domain_idx] += F.relu(gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon))
            # print(f"-----1-----:{torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon)}\n-----2-----:{gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon)}-----3-----:{F.relu(gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon))}\n-----4-----:{variance_reg[domain_idx]}\n-----5-----:{hinge_loss_weight}")
        # take its mean over z_dim_invariant dimensions
        variance_reg[domain_idx] = variance_reg[domain_idx] / z_dim_invariant

    return variance_reg.sum() # * hinge_loss_weight

def penalty_domain_classification(z, domains, num_domains, z_dim_invariant, *args):

    multinomial_logistic_regression_model = args[0]
    classification_loss = args[1]
    pred_domains_log_proba = multinomial_logistic_regression_model(z[:, :z_dim_invariant])
    domain_classification_loss = classification_loss(pred_domains_log_proba, domains.squeeze().long()).to(z.device)

    # we want to return a penalty, i.e., a positive number. So we return the negative cross entropy loss
    # so that the model tries to increase the cross entropy loss as much as possible, resulting in z_inv
    # not being predictive of domain as much as possible
    return -domain_classification_loss
    # return torch.tensor(-domain_classification_loss, device=z.device).unsqueeze(0), 0.
