import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import wasserstein_distance
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SimilarityModel:

    def __init__(self, model=None, is_autoencoder=False, metric="cosine", dtw_z_normalize=True):

        self.model = model
        self.is_autoencoder = is_autoencoder
        self.metric = metric.lower()
        self.dtw_z_normalize = dtw_z_normalize

        self.tickers = None
        self.embedding = None
        self.similarity_df = None

        # for DTW
        self._dtw_series_matrix = None
        self._dtw_index = None

    def fit(self, X, tickers):

        self.tickers = tickers

        if self.metric == "cosine":
            if self.is_autoencoder:
    
                self.model.fit(X)
                embedding = self.model.embed(X)
    
            else:
    
                embedding = self.model.fit_transform(X)

            # Normalize embeddings before computing cosine similarity
            embedding = normalize(embedding)
    
            self.embedding = embedding
    
            sim = cosine_similarity(embedding)
    
            self.similarity_df = pd.DataFrame(
                sim,
                index=tickers,
                columns=tickers
            )
            return

        if self.metric == "dtw":
            series_matrix = np.asarray(X, dtype=float)
            if series_matrix.ndim != 2:
                raise ValueError(f"DTW expects X shaped (n_tickers, series_length). Got {series_matrix.shape}.")
            if series_matrix.shape[0] != len(self.tickers):
                raise ValueError("DTW expects one row per ticker in X.")

            # Store for efficient computation
            self._dtw_series_matrix = series_matrix
            self._dtw_index = {t: i for i, t in enumerate(self.tickers)}

            self.embedding = None
            self.similarity_df = None
            return

        if self.metric == "wasserstein":
            series_matrix = np.asarray(X, dtype=float)
            n_tickers = len(self.tickers)
            
            distances = np.zeros((n_tickers, n_tickers))
            for i in range(n_tickers):
                for j in range(i + 1, n_tickers):
                    d = wasserstein_distance(series_matrix[i], series_matrix[j])
                    distances[i, j] = d
                    distances[j, i] = d

            sim = 1.0 / (1.0 + distances)
            np.fill_diagonal(sim, 1.0)
            
            self.similarity_df = pd.DataFrame(
                sim,
                index=tickers,
                columns=tickers
            )
            return

        raise ValueError(f"Unknown metric: {self.metric}")


    def top_k(self, ticker, k=10):
        if self.metric == "dtw":
            if self._dtw_series_matrix is None:
                raise RuntimeError("DTW model not fit yet.")

            i = self._dtw_index[ticker]
            x = self._dtw_series_matrix[i]

            sims = {}
            for t, j in self._dtw_index.items():
                if j == i:
                    continue
                d = self._dtw_distance(x, self._dtw_series_matrix[j])
                sims[t] = self._distance_to_similarity(d)

            return pd.Series(sims).sort_values(ascending=False).head(k)

        
        sims = self.similarity_df.loc[ticker].drop(ticker)

        return sims.sort_values(ascending=False).head(k)

    # -----------------------
    # DTW helpers
    # -----------------------
    def _z_norm(self, x, eps=1e-12):
        x = np.asarray(x, dtype=float).reshape(-1)
        mu = x.mean()
        sigma = x.std()
        if sigma < eps:
            return x - mu
        return (x - mu) / sigma

    def _dtw_distance(self, x, y):
        from dtw import dtw  # from dtw-python package

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.dtw_z_normalize:
            x = self._z_norm(x)
            y = self._z_norm(y)

        alignment = dtw(x, y)

        # use normalizedDistance if available, else distance
        d = getattr(alignment, "normalizedDistance", None)
        if d is None:
            d = alignment.distance

        return float(d)

    def _distance_to_similarity(self, d):
        # inverse transform => (0, 1] to fit top-k methodology
        return 1.0 / (1.0 + d)

    # No longer using due to computational limits, but kept for future use
    def _dtw_similarity_matrix(self, series_matrix):
        n = series_matrix.shape[0]
        sim = np.zeros((n, n), dtype=float)

        np.fill_diagonal(sim, 1.0)

        for i in range(n):
            xi = series_matrix[i]
            for j in range(i + 1, n):
                d = self._dtw_distance(xi, series_matrix[j])
                s = self._distance_to_similarity(d)
                sim[i, j] = s
                sim[j, i] = s

        return sim

def plot_similarity_graph(similarity_df, sector_map, threshold=0.8):
    """
    Plots a force-directed graph of stock similarities to cluster them by sector.
    """
    G = nx.Graph()
    
    for ticker in similarity_df.index:
        G.add_node(ticker, sector=sector_map.get(ticker, "Unknown"))
        
    for i in range(len(similarity_df.index)):
        for j in range(i + 1, len(similarity_df.columns)):
            sim = similarity_df.iloc[i, j]
            if sim > threshold:
                G.add_edge(similarity_df.index[i], similarity_df.columns[j], weight=1.0/sim)
                
    pos = nx.kamada_kawai_layout(G, weight='weight')
    
    plt.figure(figsize=(12, 8))
    unique_sectors = list(set(nx.get_node_attributes(G, 'sector').values()))
    colors = plt.cm.get_cmap('tab10', len(unique_sectors))
    
    for idx, sector in enumerate(unique_sectors):
        nodelist = [n for n, attr in G.nodes(data=True) if attr['sector'] == sector]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=[colors(idx)], 
                                label=sector, node_size=300, alpha=0.8)
        
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=9)
    
    handles = [mpatches.Patch(color=colors(i), label=sec) for i, sec in enumerate(unique_sectors)]
    plt.legend(handles=handles, title="Sectors")
    plt.title("Stock Similarity Network")
    plt.tight_layout()

    save_path = "./outputs/similarity_network.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    print(f"Saved network graph to {save_path}")

    plt.show()
    plt.close()

    
    
    
