# --- evaluation.py ---
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr, kendalltau

def evaluate_embeddings_logreg(X_train_emb, y_train, X_test_emb, y_test, name="Model"):
    """
    Trains a logistic regression on embeddings and evaluates performance.
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_emb, y_train)
    
    y_pred = clf.predict(X_test_emb)
    y_proba = clf.predict_proba(X_test_emb)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba)
    }
    return metrics

def get_ticker_embeddings(window_embeddings, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Averages window embeddings by ticker to get a single vector per ticker.
    """
    df = pd.DataFrame(window_embeddings)
    df["ticker"] = meta["ticker"].values
    return df.groupby("ticker").mean()

def get_top_k_similar(ticker_emb_df: pd.DataFrame, query_ticker: str, k: int=10):
    """
    Returns top-k most similar tickers based on cosine distance.
    """
    if query_ticker not in ticker_emb_df.index:
        raise ValueError(f"{query_ticker} not found in embeddings.")
        
    X = ticker_emb_df.values
    labels = ticker_emb_df.index.tolist()
    q_idx = labels.index(query_ticker)
    
    dists = cosine_distances(X[[q_idx]], X).ravel()
    
    # Sort by distance (ascending)
    sorted_indices = np.argsort(dists)
    
    # Filter out self (distance 0)
    results = []
    for idx in sorted_indices:
        if labels[idx] == query_ticker:
            continue
        results.append((labels[idx], dists[idx]))
        if len(results) >= k:
            break
            
    return pd.DataFrame(results, columns=["ticker", "cosine_dist"])

def compare_rankings(df_a, df_b):
    """
    Compare two Top-K similarity rankings.

    df_a, df_b:
        DataFrames returned from get_top_k_similar()
        Must contain 'ticker' column.
    """

    # Align tickers
    tickers = list(set(df_a['ticker']).intersection(set(df_b['ticker'])))

    if len(tickers) < 2:
        return {"spearman": None, "kendall": None}

    # Rank dictionaries
    rank_a = {t:i for i,t in enumerate(df_a['ticker'])}
    rank_b = {t:i for i,t in enumerate(df_b['ticker'])}

    ranks_a = [rank_a[t] for t in tickers]
    ranks_b = [rank_b[t] for t in tickers]

    spearman = spearmanr(ranks_a, ranks_b).correlation
    kendall = kendalltau(ranks_a, ranks_b).correlation

    return {
        "spearman": spearman,
        "kendall": kendall
    }
