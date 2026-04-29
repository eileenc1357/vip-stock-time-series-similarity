from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from dataset import StockDataset

def make_loader(stock_list, stock_data, seq_len=30, batch_size=32):
    datasets = []

    for s in stock_list:
        if s not in stock_data:
            continue

        series = stock_data[s].reshape(-1, 1)

        scaler = StandardScaler()
        series = scaler.fit_transform(series).flatten()

        datasets.append(StockDataset(series, seq_len))

    # merge datasets properly (not raw concat)
    X = []
    y = []

    for d in datasets:
        for i in range(len(d)):
            xi, yi = d[i]
            X.append(xi)
            y.append(yi)

    import torch
    X = torch.stack(X)
    y = torch.stack(y)

    dataset = torch.utils.data.TensorDataset(X, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)