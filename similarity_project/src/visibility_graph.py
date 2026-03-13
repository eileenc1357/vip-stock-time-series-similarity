"""
visibility_graph.py
Structural & Topological Embedding — Natural Visibility Graph (NVG)
Based on Lacasa et al. (2008): "From time series to complex networks: The visibility graph"
and the taxonomy in Ghahremani & Metsis (2025).

Run standalone:
    python visibility_graph.py
    -> prints the 23-dim embedding for one ticker
    -> prints top 5 similar tickers for 3 random companies

Import into main.py:
    from visibility_graph import VisibilityGraphEmbedder
"""

import os
import json
import random
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Visibility graph construction
# ---------------------------------------------------------------------------

def build_nvg(series: np.ndarray) -> nx.Graph:
    """
    Build a Natural Visibility Graph (NVG) from a 1-D time series.

    Nodes i and j are connected if every intermediate point k lies
    strictly below the straight line from (i, x_i) to (j, x_j):
        x_k < x_i + (x_j - x_i) * (k - i) / (j - i)   for all i < k < j

    Adjacent nodes are always connected.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)          # adjacent always visible
        xi = series[i]
        for j in range(i + 2, n):
            xj = series[j]
            ks = np.arange(i + 1, j, dtype=float)
            line_heights = xi + (xj - xi) * (ks - i) / (j - i)
            if np.all(series[i + 1: j] < line_heights):
                G.add_edge(i, j)
    return G


# ---------------------------------------------------------------------------
# 2. Feature extraction  (23-dim vector per series)
# ---------------------------------------------------------------------------

def graph_features(G: nx.Graph, n_bins: int = 10) -> np.ndarray:
    """
    Extract a fixed-length 23-dim feature vector from a visibility graph.

    Breakdown:
      6  degree statistics      (mean, std, min, max, skewness, kurtosis)
      5  graph topology         (density, avg clustering, transitivity,
                                 avg path length, diameter — on LCC)
      10 degree-distribution    (normalized histogram over n_bins bins)
      2  spectral               (largest Laplacian eigenvalue, Fiedler value)
    """
    n       = G.number_of_nodes()
    degrees = np.array([d for _, d in G.degree()], dtype=float)

    # -- Degree statistics (6) -----------------------------------------------
    deg_mean = degrees.mean()
    deg_std  = degrees.std()
    deg_min  = degrees.min()
    deg_max  = degrees.max()
    if deg_std > 1e-12:
        deg_skew = float(np.mean(((degrees - deg_mean) / deg_std) ** 3))
        deg_kurt = float(np.mean(((degrees - deg_mean) / deg_std) ** 4)) - 3.0
    else:
        deg_skew = deg_kurt = 0.0
    degree_stats = np.array([deg_mean, deg_std, deg_min, deg_max, deg_skew, deg_kurt])

    # -- Graph topology (5) --------------------------------------------------
    density        = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    transitivity   = nx.transitivity(G)

    if nx.is_connected(G):
        lcc = G
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    avg_path = nx.average_shortest_path_length(lcc) if lcc.number_of_nodes() > 1 else 0.0
    diameter = nx.diameter(lcc)                      if lcc.number_of_nodes() > 1 else 0
    topology = np.array([density, avg_clustering, transitivity, avg_path, float(diameter)])

    # -- Degree distribution histogram (10) ----------------------------------
    hist, _ = np.histogram(degrees, bins=n_bins, range=(degrees.min(), degrees.max() + 1))
    hist     = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()

    # -- Spectral features (2) -----------------------------------------------
    L              = nx.normalized_laplacian_matrix(G).toarray()
    eigvals        = np.sort(np.linalg.eigvalsh(L))
    largest_eigval = eigvals[-1]

    if nx.is_connected(G):
        algebraic_conn = float(eigvals[1]) if n > 1 else 0.0
    else:
        ev_lcc         = np.sort(np.linalg.eigvalsh(
                             nx.normalized_laplacian_matrix(lcc).toarray()))
        algebraic_conn = float(ev_lcc[1]) if lcc.number_of_nodes() > 1 else 0.0
    spectral = np.array([largest_eigval, algebraic_conn])

    return np.concatenate([degree_stats, topology, hist, spectral])


# ---------------------------------------------------------------------------
# 3. Embedder  (drop-in for SimilarityModel)
# ---------------------------------------------------------------------------

class VisibilityGraphEmbedder:
    """
    Transforms a batch of time series into 23-dim NVG embedding vectors.

    fit_transform(X)  ->  np.ndarray shape (n_samples, 23)
      X : (n_samples, n_timesteps), one row per ticker
    """

    def __init__(self, n_bins: int = 10, normalize_emb: bool = True, verbose: bool = False):
        self.n_bins        = n_bins
        self.normalize_emb = normalize_emb
        self.verbose       = verbose
        self.feature_dim_  = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X         = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        embeddings = []

        for idx in range(n_samples):
            if self.verbose and idx % 50 == 0:
                print(f"  NVG embedding: {idx}/{n_samples}")
            G   = build_nvg(X[idx])
            vec = graph_features(G, n_bins=self.n_bins)
            embeddings.append(vec)

        embeddings = np.vstack(embeddings)
        if self.normalize_emb:
            embeddings = normalize(embeddings)
        self.feature_dim_ = embeddings.shape[1]
        return embeddings

    def fit(self, X: np.ndarray) -> "VisibilityGraphEmbedder":
        self.fit_transform(X)
        return self


# ---------------------------------------------------------------------------
# 4. Standalone demo  (only runs when you do: python visibility_graph.py)
# ---------------------------------------------------------------------------

def _load_data(data_dir: str):
    """Load JSONL stock files and return (returns matrix, tickers list)."""
    series_list = []
    for file in os.listdir(data_dir):
        if not file.endswith(".jsonl"):
            continue
        ticker = file.replace(".jsonl", "")
        dates, prices = [], []
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
    returns   = np.log(prices_df / prices_df.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return returns, list(returns.columns)


if __name__ == "__main__":
    # ── paths ────────────────────────────────────────────────────────────────
    HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(HERE, "..", "data", "new_directory")

    print("=" * 55)
    print("  Visibility Graph Embedding — standalone demo")
    print("=" * 55)

    # ── load & scale ─────────────────────────────────────────────────────────
    print(f"\nLoading data from: {DATA_DIR}")
    returns, tickers = _load_data(DATA_DIR)
    print(f"  {len(tickers)} tickers, {returns.shape[0]} trading days")
    returns = returns.iloc[-252:]
    scaler = StandardScaler()
    X      = scaler.fit_transform(returns.T)   # shape: (n_tickers, n_days)

    # ── build embeddings ─────────────────────────────────────────────────────
    print("\nBuilding NVG embeddings (23-dim per ticker)...")
    embedder   = VisibilityGraphEmbedder(n_bins=10, verbose=True)
    embeddings = embedder.fit_transform(X)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # ── show one embedding vector ─────────────────────────────────────────────
    sample_ticker = tickers[0]
    sample_vec    = embeddings[0]
    labels = (
        ["deg_mean", "deg_std", "deg_min", "deg_max", "deg_skew", "deg_kurt"]
        + ["density", "avg_clustering", "transitivity", "avg_path", "diameter"]
        + [f"hist_{i}" for i in range(10)]
        + ["largest_eigval", "fiedler"]
    )
    print(f"\n23-dim embedding vector for '{sample_ticker}':")
    print(f"  {'feature':<20}  value")
    print(f"  {'-'*35}")
    for label, val in zip(labels, sample_vec):
        print(f"  {label:<20}  {val: .6f}")

    # ── top 5 similar for 3 random tickers ───────────────────────────────────
    sim_matrix = cosine_similarity(embeddings)
    sim_df     = pd.DataFrame(sim_matrix, index=tickers, columns=tickers)

    test_companies = random.sample(tickers, min(3, len(tickers)))
    print(f"\nTop 5 similar tickers (cosine similarity on NVG embeddings):")
    for company in test_companies:
        top5 = sim_df.loc[company].drop(company).sort_values(ascending=False).head(5)
        print(f"\n  '{company}':")
        for ticker, score in top5.items():
            print(f"    {ticker:<12}  {score:.4f}")

    print("\nDone. Import VisibilityGraphEmbedder into main.py to use in the full pipeline.")