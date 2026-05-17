# src/dataset_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset

class ChromDataset(Dataset):
    def __init__(self, X_path, y_path, meta_path=None):
        self.X = np.load(X_path)          # (N, 8, 240)
        self.y = np.load(y_path)          # (N,)
        if meta_path:
            self.meta = np.load(meta_path)  # (N, 5)
        else:
            self.meta = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if self.meta is not None:
            meta = torch.tensor(self.meta[idx], dtype=torch.float32)
            return x, y, meta
        return x, y