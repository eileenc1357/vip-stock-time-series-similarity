import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize


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