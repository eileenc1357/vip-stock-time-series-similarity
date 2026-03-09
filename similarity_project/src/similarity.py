import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class SimilarityModel:

    def __init__(self, model, is_autoencoder=False):

        self.model = model
        self.is_autoencoder = is_autoencoder

        self.tickers = None
        self.embedding = None
        self.similarity_df = None


    def fit(self, X, tickers):

        self.tickers = tickers

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


    def top_k(self, ticker, k=10):

        sims = self.similarity_df.loc[ticker].drop(ticker)

        return sims.sort_values(ascending=False).head(k)