"""
visibility_graph.py
Structural & Topological Embedding — Natural Visibility Graph (NVG)

Based on:
Lacasa et al. (2008): "From time series to complex networks: The visibility graph"

Run standalone:
    python visibility_graph.py
"""

import os
import json
import random
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 1. Visibility graph construction
# ---------------------------------------------------------------------------

def build_nvg(series: np.ndarray) -> nx.Graph:
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)

        xi = series[i]

        for j in range(i + 2, n):
            xj = series[j]

            ks = np.arange(i + 1, j)
            line_heights = xi + (xj - xi) * (ks - i) / (j - i)

            if np.all(series[i + 1:j] < line_heights):
                G.add_edge(i, j)

    return G


# ---------------------------------------------------------------------------
# 2. Feature extraction (23 dimensions)
# ---------------------------------------------------------------------------

def graph_features(G: nx.Graph, n_bins: int = 10) -> np.ndarray:

    n = G.number_of_nodes()

    degrees = np.array([d for _, d in G.degree()], dtype=float)

    # Degree statistics
    deg_mean = degrees.mean()
    deg_std  = degrees.std()
    deg_min  = degrees.min()
    deg_max  = degrees.max()

    if deg_std > 1e-12:
        deg_skew = np.mean(((degrees - deg_mean) / deg_std) ** 3)
        deg_kurt = np.mean(((degrees - deg_mean) / deg_std) ** 4) - 3
    else:
        deg_skew = 0
        deg_kurt = 0

    degree_stats = np.array([
        deg_mean, deg_std, deg_min, deg_max, deg_skew, deg_kurt
    ])

    # Graph topology
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    if nx.is_connected(G):
        lcc = G
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    avg_path = nx.average_shortest_path_length(lcc) if lcc.number_of_nodes() > 1 else 0
    diameter = nx.diameter(lcc) if lcc.number_of_nodes() > 1 else 0

    topology = np.array([
        density, avg_clustering, transitivity, avg_path, float(diameter)
    ])

    # Degree histogram
    hist, _ = np.histogram(
        degrees,
        bins=n_bins,
        range=(degrees.min(), degrees.max() + 1)
    )

    hist = hist.astype(float)

    if hist.sum() > 0:
        hist /= hist.sum()

    # Spectral features
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals = np.sort(np.linalg.eigvalsh(L))

    largest_eigval = eigvals[-1]

    if nx.is_connected(G):
        fiedler = eigvals[1] if n > 1 else 0
    else:
        L_lcc = nx.normalized_laplacian_matrix(lcc).toarray()
        ev = np.sort(np.linalg.eigvalsh(L_lcc))
        fiedler = ev[1] if lcc.number_of_nodes() > 1 else 0

    spectral = np.array([largest_eigval, fiedler])

    return np.concatenate([degree_stats, topology, hist, spectral])


# ---------------------------------------------------------------------------
# 3. Embedder
# ---------------------------------------------------------------------------

class VisibilityGraphEmbedder:

    def __init__(self, n_bins: int = 10, verbose: bool = False):

        self.n_bins = n_bins
        self.verbose = verbose
        self.feature_dim_ = None


    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        X = np.asarray(X, dtype=float)

        n_samples = X.shape[0]

        embeddings = []

        for idx in range(n_samples):

            if self.verbose and idx % 50 == 0:
                print(f"  NVG embedding: {idx}/{n_samples}")

            G = build_nvg(X[idx])

            vec = graph_features(G, n_bins=self.n_bins)

            embeddings.append(vec)

        embeddings = np.vstack(embeddings)

        self.feature_dim_ = embeddings.shape[1]

        return embeddings


# ---------------------------------------------------------------------------
# 4. Load dataset
# ---------------------------------------------------------------------------

def _load_data(data_dir):

    series_list = []

    for file in os.listdir(data_dir):

        if not file.endswith(".jsonl"):
            continue

        ticker = file.replace(".jsonl", "")

        dates = []
        prices = []

        with open(os.path.join(data_dir, file)) as f:

            for line in f:

                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "close" in d and "timestamp" in d:

                    prices.append(d["close"])
                    dates.append(pd.to_datetime(d["timestamp"], unit="ms"))

        if prices:

            series_list.append(pd.Series(prices, index=dates, name=ticker))

    prices_df = pd.concat(series_list, axis=1).sort_index().ffill()

    returns = np.log(prices_df / prices_df.shift(1))

    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

    return returns, list(returns.columns)


# ---------------------------------------------------------------------------
# 5. Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    HERE = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(HERE, "..", "data", "new_directory")

    print("\nLoading data...")

    returns, tickers = _load_data(DATA_DIR)

    returns = returns.iloc[-252:]

    print(f"{len(tickers)} tickers")

    # scale time series

    scaler = StandardScaler()

    X = scaler.fit_transform(returns.T)

    print("\nBuilding NVG embeddings")

    embedder = VisibilityGraphEmbedder(verbose=True)

    embeddings = embedder.fit_transform(X)

    print("Embedding shape:", embeddings.shape)

    # scale embedding features

    scaler = StandardScaler()

    embeddings = scaler.fit_transform(embeddings)

    # PCA visualization

    pca = PCA(n_components=2)

    X2 = pca.fit_transform(embeddings)

    plt.figure(figsize=(6,6))
    plt.scatter(X2[:,0], X2[:,1])
    plt.title("NVG Embedding Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    print("\nFeature variances:")
    print(np.var(embeddings, axis=0))

    # show one embedding

    sample = embeddings[0]

    print("\nExample embedding vector:")

    print(sample)

    # similarity search using distance

    dist_matrix = euclidean_distances(embeddings)

    dist_df = pd.DataFrame(dist_matrix, index=tickers, columns=tickers)

    test_companies = random.sample(tickers, min(3, len(tickers)))

    print("\nTop 5 similar tickers (Euclidean distance):")

    for company in test_companies:

        top5 = dist_df.loc[company].drop(company).sort_values().head(5)

        print(f"\n{company}")

        for ticker, score in top5.items():

            print(f"{ticker:<10} {score:.4f}")