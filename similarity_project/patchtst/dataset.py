import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self, series, seq_len=30):
        self.X, self.y = [], []

        for i in range(len(series) - seq_len):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

        self.y = self.y.unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.y[idx]