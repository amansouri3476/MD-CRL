import torch
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    def __init__(self, subset, indices, transform):
        self.data = []
        for i in indices:
            self.data.append(subset[i])
        self.length = len(self.data)
        self.transform = transform
        self.mean_ = subset.mean_ if hasattr(subset, "mean_") else 0.0
        self.std_ = subset.std_ if hasattr(subset, "std_") else 1.0
        self.min_ = subset.min_ if hasattr(subset, "min_") else 0.0
        self.max_ = subset.max_ if hasattr(subset, "max_") else 1.0
        # extract all attributes of subset.dataset as the attributes of this new dataset
        for attr in dir(subset):
            if not attr.startswith("__") and not attr == "data":
                setattr(self, attr, getattr(subset, attr))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

    def renormalize(self):
        for t in self.transform.transforms:
            if t.__class__.__name__ == "Standardize":
                """Renormalize from [-1, 1] to [0, 1]."""
                return lambda x: x / 2.0 + 0.5
        # for t in self.transform.transforms:
        #     if t.__class__.__name__ == "Standardize":
        #         """Renormalize from [-1, 1] to [0, 1]."""
        #         return lambda x: (x * self.std_ + self.mean_) / 2.0 + 0.5
        # else:
        #     return lambda x: x
        
        # return lambda x: x * self.std_ + self.mean_
        # return lambda x: x * (self.max_ - self.min_) + self.min_
