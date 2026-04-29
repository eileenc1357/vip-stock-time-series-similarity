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
TOP_K = 5
EPOCHS = 2
SAMPLE_SIZES = [5, 10, 20, 50]
TRAIN_SPLIT = 0.8
SEEDS = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DATA SPLIT ----------------
def split_data(stock_data):
    train_data, test_data = {}, {}
    for k, v in stock_data.items():
        split = int(len(v) * TRAIN_SPLIT)
        train_data[k] = v[:split]
        test_data[k] = v[split:]
    return train_data, test_data


# ---------------- TRAIN ----------------
def train_model(loader):
    model = PatchTST().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    model.train()

    for _ in range(EPOCHS):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# ---------------- EVAL ----------------
def evaluate_model(model, loader):
    model.eval()
    loss_fn = nn.MSELoss()

    losses = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            losses.append(loss_fn(pred, y).item())

    return np.mean(losses)


# ---------------- MAIN ----------------
print("Loading dataset...")
path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

df = load_data(path + "/sp500_stocks.csv")
stock_data = build_stock_universe(df)

symbols = list(stock_data.keys())
train_data, test_data = split_data(stock_data)

print(f"Stocks loaded: {len(symbols)}")

results = []

# ---------------- EXPERIMENT LOOP ----------------
for seed in SEEDS:
    print(f"\n\n========== SEED {seed} ==========")
    np.random.seed(seed)

    for size in SAMPLE_SIZES + [len(symbols)]:

        print(f"\n===== SIZE {size} =====")

        targets = np.random.choice(symbols, min(size, len(symbols)), replace=False)

        topk_losses = []
        rand_losses = []
        single_losses = []

        for t in targets:

            print(f"Target: {t}")

            try:
                topk_group = get_similar_stocks(df, t, TOP_K)
            except:
                continue

            rand_group = np.random.choice(symbols, TOP_K, replace=False)
            single_group = [t]

            # loaders
            topk_loader = make_loader(topk_group, test_data)
            rand_loader = make_loader(rand_group, test_data)
            single_loader = make_loader(single_group, test_data)

            # train models
            model_topk = train_model(make_loader(topk_group, train_data))
            model_rand = train_model(make_loader(rand_group, train_data))
            model_single = train_model(make_loader(single_group, train_data))

            # eval
            topk_losses.append(evaluate_model(model_topk, topk_loader))
            rand_losses.append(evaluate_model(model_rand, rand_loader))
            single_losses.append(evaluate_model(model_single, single_loader))

        # aggregate per size + seed
        results.append({
            "Seed": seed,
            "NumTargets": size,

            "Top-K_mean": np.mean(topk_losses) if topk_losses else np.nan,
            "Top-K_std": np.std(topk_losses) if topk_losses else np.nan,

            "Random_mean": np.mean(rand_losses) if rand_losses else np.nan,
            "Random_std": np.std(rand_losses) if rand_losses else np.nan,

            "Single_mean": np.mean(single_losses) if single_losses else np.nan,
            "Single_std": np.std(single_losses) if single_losses else np.nan,
        })


# ---------------- RESULTS ----------------
df_results = pd.DataFrame(results)

print("\n===== RAW RESULTS =====")
print(df_results.head())

# average across seeds
summary = df_results.groupby("NumTargets").mean(numeric_only=True)

print("\n===== AVERAGED RESULTS =====")
print(summary)

# signal metric
summary["TopK_minus_Random"] = summary["Top-K_mean"] - summary["Random_mean"]

print("\n===== SIGNAL (TopK - Random) =====")
print(summary["TopK_minus_Random"])

# save
df_results.to_csv("raw_results.csv", index=False)
summary.to_csv("summary_results.csv")


# ---------------- PLOTS ----------------
plt.figure()

plt.plot(summary.index, summary["Top-K_mean"], marker="o", label="Top-K")
plt.plot(summary.index, summary["Random_mean"], marker="o", label="Random")
plt.plot(summary.index, summary["Single_mean"], marker="o", label="Single")

plt.xlabel("Number of Target Stocks")
plt.ylabel("Avg Test Loss")
plt.title("Scaling Experiment (Averaged over Seeds)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("scaling_plot.png")
plt.show()