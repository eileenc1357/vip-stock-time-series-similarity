import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import kagglehub

from models.patchtst import PatchTST
from data_builder import load_data, build_stock_universe
from similarity import get_similar_stocks
from make_loader import make_loader

# ---------------- CONFIG ----------------
SEQ_LEN = 30
TOP_K = 5
EPOCHS = 3

SAMPLE_SIZES = [5, 10, 20, 50]
TRAIN_SPLIT = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DATA ----------------
def split_data(stock_data):
    train_data, test_data = {}, {}

    for k, v in stock_data.items():
        split = int(len(v) * TRAIN_SPLIT)
        train_data[k] = v[:split]
        test_data[k] = v[split:]

    return train_data, test_data


# ---------------- MODEL TRAIN ----------------
def train_model(loader):
    model = PatchTST().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    model.train()

    for ep in range(EPOCHS):
        total_loss = 0

        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1}: {total_loss:.4f}")

    return model


# ---------------- EVAL ----------------
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.MSELoss()

    losses = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            losses.append(loss_fn(pred, y).item())

    return np.mean(losses)


# ---------------- RUN SINGLE EXPERIMENT ----------------
def run_for_target(target, symbols, df, train_data, test_data):

    try:
        topk = get_similar_stocks(df, target, TOP_K)
    except:
        return None

    randk = np.random.choice(symbols, TOP_K, replace=False)

    loaders_train = {
        "topk": make_loader(topk, train_data),
        "rand": make_loader(randk, train_data),
        "single": make_loader([target], train_data)
    }

    loaders_test = {
        "topk": make_loader(topk, test_data),
        "rand": make_loader(randk, test_data),
        "single": make_loader([target], test_data)
    }

    results = {}

    for key in loaders_train:
        model = train_model(loaders_train[key])
        results[key] = evaluate(model, loaders_test[key])

    return {
        "Target": target,
        "Top-K": results["topk"],
        "Random": results["rand"],
        "Single": results["single"]
    }


# ---------------- MAIN ----------------
print("Downloading dataset...")
path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

df = load_data(path + "/sp500_stocks.csv")
stock_data = build_stock_universe(df)

symbols = list(stock_data.keys())
train_data, test_data = split_data(stock_data)

print(f"Loaded {len(symbols)} stocks")


all_results = []

# ---------------- SCALING LOOP ----------------
for size in SAMPLE_SIZES + [len(symbols)]:

    print(f"\n===== SIZE {size} =====")

    targets = np.random.choice(symbols, size, replace=False)

    for target in targets:

        print(f"Target: {target}")

        res = run_for_target(
            target,
            symbols,
            df,
            train_data,
            test_data
        )

        if res is not None:
            res["NumTargets"] = size
            all_results.append(res)


# ---------------- RESULTS ----------------
results_df = pd.DataFrame(all_results)

print("\n===== SAMPLE RESULTS =====")
print(results_df.head())

results_df.to_csv("results_full.csv", index=False)

summary = results_df.groupby("NumTargets")[["Top-K", "Random", "Single"]].mean()

print("\n===== SUMMARY =====")
print(summary)

summary.to_csv("summary.csv")


# ---------------- PLOTS ----------------
plt.figure()

x = np.arange(len(summary))
w = 0.25

plt.bar(x - w, summary["Top-K"], width=w, label="Top-K")
plt.bar(x, summary["Random"], width=w, label="Random")
plt.bar(x + w, summary["Single"], width=w, label="Single")

plt.xticks(x, summary.index)
plt.xlabel("Number of Target Stocks")
plt.ylabel("Avg Test Loss")
plt.title("Performance vs Scale")
plt.legend()

plt.tight_layout()
plt.savefig("bar_plot.png")
plt.show()


plt.figure()

plt.plot(summary.index, summary["Top-K"], marker="o", label="Top-K")
plt.plot(summary.index, summary["Random"], marker="o", label="Random")
plt.plot(summary.index, summary["Single"], marker="o", label="Single")

plt.xlabel("Number of Target Stocks")
plt.ylabel("Avg Test Loss")
plt.title("Scaling Behavior")
plt.legend()
plt.grid()

plt.savefig("scaling_plot.png")
plt.show()