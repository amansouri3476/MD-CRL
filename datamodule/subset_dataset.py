import torch
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    def __init__(self, subset, indices, transform):
        self.data = []
        for i in indices:
            self.data.append(subset[i])
        self.length = len(self.data)
        self.transform = transform
        self.mean_ = subset.mean_
        self.std_ = subset.std_

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

    def renormalize(self):
        for t in self.transform.transforms:
            if t.__class__.__name__ == "Standardize":
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
        #     else:
        #         return lambda x: x
            
        # return lambda x: x * self.std_ + self.mean_
