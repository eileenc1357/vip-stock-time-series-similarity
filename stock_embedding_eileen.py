import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt


# =========================================================
# 1. Load Stock Data from JSONL Files
# =========================================================

directory = "new_directory"

series_dict = {}

for file in os.listdir(directory):

    if file.endswith(".jsonl"):

        ticker = file.replace(".jsonl", "")
        path = os.path.join(directory, file)

        df = pd.read_json(path, lines=True)

        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        df = df.sort_values("date")

        df = df.set_index("date")

        series_dict[ticker] = df["close"]


# Align by timestamp
data = pd.concat(series_dict, axis=1)

# Forward fill missing prices
data = data.ffill()

# Drop rows still missing many values
data = data.dropna(thresh=int(0.8 * data.shape[1]))

print("Loaded data shape:", data.shape)
# =========================================================
# 2. Convert Prices → Returns
# =========================================================

returns = data.pct_change().dropna()

print("Returns shape:", returns.shape)


# =========================================================
# 3. Standardize Time Series
# =========================================================

scaler = StandardScaler()

scaled = scaler.fit_transform(returns.T)

tickers = list(returns.columns)


# =========================================================
# 4. Unified Similarity Framework
# =========================================================

def compute_similarity(method, data, tickers):

    embedding = method.fit_transform(data)

    similarity = cosine_similarity(embedding)

    similarity_df = pd.DataFrame(
        similarity,
        index=tickers,
        columns=tickers
    )

    return similarity_df, embedding


# =========================================================
# 5. Top-K Similarity Interface
# =========================================================

def get_top_k(similarity_df, ticker, k=10):

    sims = similarity_df.loc[ticker].drop(ticker)

    k = min(k, len(sims))

    return sims.sort_values(ascending=False).head(k)


# =========================================================
# 6. Define Methods
# =========================================================

n_neighbors = min(10, len(tickers) - 1)

n_components = min(5, scaled.shape[1], scaled.shape[0])

methods = {

    "PCA": PCA(n_components=n_components),

    "KPCA": KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=0.1
    ),

    "Isomap": Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors
    ),

    "LLE": LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors
    )
}

results = {}
embeddings = {}


# =========================================================
# 7. Run Embedding Methods
# =========================================================

for name, method in methods.items():

    print("\nRunning", name)

    sim_df, embed = compute_similarity(method, scaled, tickers)

    embeddings[name] = embed

    results[name] = {
        "similarity_matrix": sim_df
    }


# =========================================================
# 8. Select 5 Random Companies
# =========================================================

import random

test_companies = random.sample(tickers, min(5, len(tickers)))

print("\nSelected companies:", test_companies)


# =========================================================
# 9. Top-K Similar Stocks
# =========================================================

for method in results:

    sim_df = results[method]["similarity_matrix"]

    print("\n==============================")
    print(f"{method} Similarity Results")
    print("==============================")

    for company in test_companies:

        print(f"\nTop 10 similar to {company}")

        print(get_top_k(sim_df, company, 10))


# =========================================================
# 10. Heatmap Visualization
# =========================================================

example_method = list(results.keys())[0]

sim_matrix = results[example_method]["similarity_matrix"]

plt.figure(figsize=(10,8))

sns.heatmap(
    sim_matrix,
    cmap="coolwarm",
    annot=False
)

plt.title(f"{example_method} Stock Similarity Matrix")

plt.tight_layout()

plt.savefig("similarity_heatmap.png")

print("\nSaved heatmap to similarity_heatmap.png")


# =========================================================
# 11. Embedding Visualization
# =========================================================

embed = embeddings[example_method]

plt.figure(figsize=(7,6))

plt.scatter(
    embed[:,0],
    embed[:,1]
)

for i, ticker in enumerate(tickers):

    plt.text(
        embed[i,0],
        embed[i,1],
        ticker,
        fontsize=8
    )

plt.title(f"{example_method} Stock Embedding")

plt.tight_layout()

plt.savefig("embedding_plot.png")

print("Saved embedding plot to embedding_plot.png")