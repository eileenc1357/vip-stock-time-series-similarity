import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# ===============================
# Reproducibility (VERY IMPORTANT)
# ===============================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ===============================
# 1. LOAD DATA
# ===============================

def load_stock_prices(folder):
    series_list = []

    for file in os.listdir(folder):
        if not file.endswith(".jsonl"):
            continue

        ticker = file.replace(".jsonl", "")
        path = os.path.join(folder, file)

        dates, prices = [], []

        with open(path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue

                if "close" in data and "timestamp" in data:
                    prices.append(data["close"])
                    dates.append(pd.to_datetime(data["timestamp"], unit="ms"))

        if len(prices) == 0:
            continue

        s = pd.Series(prices, index=dates, name=ticker)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).sort_index().ffill()
    return prices


def compute_returns(prices):
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    return returns


# ===============================
# 2. DATASET BUILDER
# ===============================

def build_dataset(data, selected, target, window=20):
    X_raw = data[selected].values
    y_raw = data[target].values

    X, y = [], []

    for t in range(window, len(data) - 1):
        X.append(X_raw[t-window:t])
        y.append(y_raw[t+1])

    return np.array(X), np.array(y)


# ===============================
# 3. SELECTION METHODS
# ===============================

def select_similarity(sim_matrix, tickers, target, k):
    idx = tickers.index(target)
    scores = sim_matrix[idx]
    ranked = np.argsort(scores)[::-1]

    selected = []
    for i in ranked:
        if tickers[i] != target:
            selected.append(tickers[i])
        if len(selected) == k:
            break

    return selected + [target]


def select_random(tickers, target, k):
    others = [t for t in tickers if t != target]
    return random.sample(others, k) + [target]


def select_univariate(target):
    return [target]


# ===============================
# 4. SIMPLE PATCHTST MODEL
# ===============================

class SimplePatchTST(nn.Module):
    def __init__(self, seq_len, n_features, d_model=64):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(seq_len * n_features, d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze()


def train_patchtst(X_train, y_train, X_test, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = SimplePatchTST(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # training
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # inference
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()

    return preds


# ===============================
# 5. EXPERIMENT
# ===============================

def run_experiment(data, sim_matrix, tickers, target, k_values=[1,3,5,10], window=20):

    results = []

    for k in k_values:

        print(f"\nRunning k={k}")

        sim_sel = select_similarity(sim_matrix, tickers, target, k)
        rand_sel = select_random(tickers, target, k)
        uni_sel = select_univariate(target)

        X_sim, y_sim = build_dataset(data, sim_sel, target, window)
        X_rand, y_rand = build_dataset(data, rand_sel, target, window)
        X_uni, y_uni = build_dataset(data, uni_sel, target, window)

        # align lengths
        min_len = min(len(X_sim), len(X_rand), len(X_uni))
        X_sim, y_sim = X_sim[:min_len], y_sim[:min_len]
        X_rand, y_rand = X_rand[:min_len], y_rand[:min_len]
        X_uni, y_uni = X_uni[:min_len], y_uni[:min_len]

        split = int(0.8 * min_len)

        def split_data(X, y):
            return X[:split], X[split:], y[:split], y[split:]

        X_sim_tr, X_sim_te, y_sim_tr, y_sim_te = split_data(X_sim, y_sim)
        X_rand_tr, X_rand_te, y_rand_tr, y_rand_te = split_data(X_rand, y_rand)
        X_uni_tr, X_uni_te, y_uni_tr, y_uni_te = split_data(X_uni, y_uni)

        preds_sim = train_patchtst(X_sim_tr, y_sim_tr, X_sim_te)
        preds_rand = train_patchtst(X_rand_tr, y_rand_tr, X_rand_te)
        preds_uni = train_patchtst(X_uni_tr, y_uni_tr, X_uni_te)

        mse_sim = mean_squared_error(y_sim_te, preds_sim)
        mse_rand = mean_squared_error(y_rand_te, preds_rand)
        mse_uni = mean_squared_error(y_uni_te, preds_uni)

        print(f"Similarity MSE: {mse_sim:.6f}")
        print(f"Random MSE: {mse_rand:.6f}")
        print(f"Univariate MSE: {mse_uni:.6f}")

        results.append({
            "k": k,
            "similarity": mse_sim,
            "random": mse_rand,
            "univariate": mse_uni
        })

    return pd.DataFrame(results)


# ===============================
# 6. MAIN
# ===============================

if __name__ == "__main__":

    # folder = "./data/new_directory"
    folder = "./similarity_project/data/new_directory"
    print("Loading data...")
    prices = load_stock_prices(folder)
    returns = compute_returns(prices).iloc[-250:]

    tickers = list(returns.columns)

    print("Computing similarity...")
    X = StandardScaler().fit_transform(returns.T)
    sim_matrix = cosine_similarity(X)

    target = tickers[0]
    print("Target stock:", target)

    df = run_experiment(returns, sim_matrix, tickers, target)

    print("\nFinal Results:")
    print(df)

    # plot
    plt.plot(df["k"], df["similarity"], label="Similarity")
    plt.plot(df["k"], df["random"], label="Random")
    plt.plot(df["k"], df["univariate"], label="Univariate")

    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("PatchTST Stock Selection Comparison")
    output_dir = os.path.join("./similarity_project/outputs", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "patchtst_results.png"))
    plt.show()