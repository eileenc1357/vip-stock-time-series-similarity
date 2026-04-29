import os
import kagglehub
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.patchtst import PatchTST
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_sp500_data(path):
    file_path = os.path.join(path, "sp500_stocks.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    df = pd.read_csv(file_path)

    # Basic cleaning
    df = df.sort_values(["Symbol", "Date"])
    df = df.dropna()

    return df


# -----------------------------
# 2. Dataset Class
# -----------------------------
class StockDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.X = []
        self.y = []

        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 3. Build DataLoader
# -----------------------------
def build_dataloader(df, symbol="AAPL", seq_len=30):
    df_symbol = df[df["Symbol"] == symbol]

    if len(df_symbol) == 0:
        raise ValueError(f"No data found for symbol {symbol}")

    # prices = df_symbol["Close"].values.reshape(-1, 1)
    prices = df_symbol["Close"].pct_change().dropna().values.reshape(-1, 1)

    # scaler = StandardScaler()
    # prices = scaler.fit_transform(prices)
    prices = prices.astype(np.float32)

    dataset = StockDataset(prices, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader, None


# -----------------------------
# 4. Main Pipeline
# -----------------------------
if __name__ == "__main__":
    print("Downloading dataset...")

    path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

    print("Downloaded path:", path)
    print("Files in dataset:", os.listdir(path))

    print("\nLoading data...")
    df = load_sp500_data(path)

    print("Data loaded. Shape:", df.shape)

    print("\nBuilding dataloader...")

    loader, scaler = build_dataloader(df, symbol="ABBV")  # ABBV exists in your printout

    print("Dataloader ready!")

    # DEBUG CHECK (important)
    print("Loader type:", type(loader))

    # Test batch
    for X, y in loader:
        print("\nBatch X shape:", X.shape)
        print("Batch y shape:", y.shape)
        break

# -----------------------------
# 5. Model + Training
# -----------------------------
model = PatchTST()

# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss() #MSE is too sensitive for noisy financial returns, smooth l1 is better 
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining started...")

model.train()

for epoch in range(3):  # keep small for now
    total_loss = 0

    for X, y in loader:
        pred = model(X)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    for X, y in loader:
        pred = model(X)

        print("Sample prediction:", pred[:5].squeeze())
        print("Actual:", y[:5].squeeze())
        break   