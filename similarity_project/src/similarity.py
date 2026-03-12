import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


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
