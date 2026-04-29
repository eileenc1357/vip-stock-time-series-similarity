import numpy as np
import torch
import torch.nn as nn
from models.patchtst import PatchTST
from data_builder import load_data, build_stock_universe
from similarity import get_similar_stocks
from make_loader import make_loader
import kagglehub

SEQ_LEN = 30
TOP_K = 5
EPOCHS = 3
SPLIT = 0.8


# ---------------- TRAIN ----------------
def train(loader):
    model = PatchTST()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    model.train()

    for ep in range(EPOCHS):
        total = 0
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {ep+1}: {total:.4f}")

    return model


# ---------------- EVALUATE ----------------
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            losses.append(loss_fn(pred, y).item())

    return np.mean(losses)


# ---------------- SPLIT DATA ----------------
def split_stock_data(stock_data):
    train_data = {}
    test_data = {}

    for symbol, series in stock_data.items():
        split_idx = int(len(series) * SPLIT)

        if split_idx <= SEQ_LEN:
            continue

        train_data[symbol] = series[:split_idx]
        test_data[symbol] = series[split_idx:]

    return train_data, test_data


# ---------------- MAIN ----------------
path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
df = load_data(path + "/sp500_stocks.csv")

stock_data = build_stock_universe(df)
symbols = list(stock_data.keys())

train_data, test_data = split_stock_data(stock_data)

targets = ["ABBV", "AMZN", "MSFT", "XOM", "JPM"]

results = []

for target in targets:
    if target not in train_data or target not in test_data:
        continue

    print(f"\n######## TARGET: {target} ########")

    # --- SELECT STOCK GROUPS ---
    topk = get_similar_stocks(df, target, TOP_K)
    randk = np.random.choice(symbols, TOP_K, replace=False)
    single = [target]

    # --- BUILD LOADERS (TRAIN) ---
    loader_topk = make_loader(topk, train_data)
    loader_rand = make_loader(randk, train_data)
    loader_single = make_loader(single, train_data)

    # --- TRAIN ---
    print("\n=== TOP-K ===")
    m1 = train(loader_topk)

    print("\n=== RANDOM ===")
    m2 = train(loader_rand)

    print("\n=== SINGLE ===")
    m3 = train(loader_single)

    # --- TEST LOADER (ONLY TARGET) ---
    test_loader = make_loader([target], test_data)

    # --- EVALUATE ---
    loss_topk = evaluate(m1, test_loader)
    loss_rand = evaluate(m2, test_loader)
    loss_single = evaluate(m3, test_loader)

    print("\n--- TEST RESULTS ---")
    print("Top-K:", loss_topk)
    print("Random:", loss_rand)
    print("Single:", loss_single)

    results.append((target, loss_topk, loss_rand, loss_single))


# ---------------- SUMMARY ----------------
print("\n======= FINAL SUMMARY =======")

for r in results:
    print(f"{r[0]} | Top-K: {r[1]:.5f} | Random: {r[2]:.5f} | Single: {r[3]:.5f}")