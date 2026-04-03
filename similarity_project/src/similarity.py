import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from scipy.stats import wasserstein_distance
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SimilarityModel:

    def __init__(self, model=None, is_autoencoder=False, metric="cosine", dtw_z_normalize=True):
        self.model = model
        self.is_autoencoder = is_autoencoder
        self.metric = metric.lower().strip()
        self.dtw_z_normalize = dtw_z_normalize

        self.tickers = None
        self.embedding = None
        self.similarity_df = None

    def fit(self, X, tickers):
        self.tickers = tickers

        # ===============================
        # COSINE / EMBEDDING MODELS
        # ===============================
        if self.metric == "cosine":

            if self.model is not None:
                if self.is_autoencoder:
                    self.model.fit(X)
                    embedding = self.model.embed(X)
                else:
                    embedding = self.model.fit_transform(X)

                embedding = normalize(embedding)
            else:
                embedding = normalize(X)

            self.embedding = embedding
            sim = cosine_similarity(embedding)

            self.similarity_df = pd.DataFrame(sim, index=tickers, columns=tickers)
            return

        # ===============================
        # EUCLIDEAN
        # ===============================
        if self.metric == "euclidean":
            dist = euclidean_distances(X)
            sim = 1.0 / (1.0 + dist)

        elif self.metric == "wasserstein":
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

            self.similarity_df = pd.DataFrame(sim, index=tickers, columns=tickers)
            return

        # ===============================
        # CORRELATION
        # ===============================
        if self.metric == "correlation":
            sim = np.corrcoef(X)

            self.similarity_df = pd.DataFrame(sim, index=tickers, columns=tickers)
            return

        # ===============================
        # DTW
        # ===============================
        # if self.metric == "dtw":

        #     series_matrix = np.asarray(X, dtype=float)

        #     if self.dtw_z_normalize:
        #         series_matrix = np.array([self._z_norm(s) for s in series_matrix])
        # return sim
    
    def top_k(self, ticker, k=5):
        """
        Return top-k most similar tickers (excluding itself).
        """

        if self.similarity_df is None:
            raise ValueError("Model not fitted yet.")

        if ticker not in self.similarity_df.index:
            raise ValueError(f"{ticker} not found in similarity matrix.")

        sims = self.similarity_df.loc[ticker].copy()

        # remove self
        sims = sims.drop(ticker)

        # sort by similarity (descending)
        sims = sims.sort_values(ascending=False)

        return sims.head(k)


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
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodelist,
            node_color=[colors(idx)],
            label=sector,
            node_size=300,
            alpha=0.8
        )
        
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

    
    
    
