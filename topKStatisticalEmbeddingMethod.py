import os
import json
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


# =====================================================
# DATA LOADING
# =====================================================

def load_stock_prices(folder):

    print("Loading stock data from:", folder)

    files = os.listdir(folder)
    print("Files found:", files[:10], "...")

    price_series = []

    for file in files:

        if not file.endswith(".jsonl"):
            continue

        ticker = file.replace(".jsonl", "")
        path = os.path.join(folder, file)

        dates = []
        prices = []

        with open(path) as f:
            for line in f:

                data = json.loads(line)

                if "close" in data and "timestamp" in data:

                    prices.append(data["close"])

                    dates.append(
                        pd.to_datetime(data["timestamp"], unit="ms")
                    )

        if len(prices) == 0:
            continue

        s = pd.Series(prices, index=dates, name=ticker)

        price_series.append(s)

    prices = pd.concat(price_series, axis=1)

    print("Price matrix shape:", prices.shape)

    return prices


# =====================================================
# RETURNS + CLEANING
# =====================================================

def compute_returns(prices):

    returns = prices.pct_change()

    returns = returns.replace([np.inf, -np.inf], np.nan)

    returns = returns.ffill()

    returns = returns.dropna(axis=1, thresh=int(0.7 * len(returns)))

    returns = returns.dropna()

    print("Returns shape:", returns.shape)

    return returns


# =====================================================
# STANDARDIZATION
# =====================================================

def standardize_returns(returns):

    scaler = StandardScaler()

    X = scaler.fit_transform(returns)

    return X


# =====================================================
# PCA METHOD
# =====================================================

def run_pca(X, stock_names, k=10):

    pca = PCA(n_components=k)

    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=stock_names,
        columns=[f"PCA_{i}" for i in range(k)]
    )

    scores = loadings.abs().sum(axis=1)

    top = scores.sort_values(ascending=False).head(k)

    return top, scores


# =====================================================
# ICA METHOD
# =====================================================

def run_ica(X, stock_names, k=10):

    ica = FastICA(n_components=k, random_state=42)

    ica.fit(X)

    loadings = pd.DataFrame(
        ica.mixing_,
        index=stock_names,
        columns=[f"ICA_{i}" for i in range(k)]
    )

    scores = loadings.abs().sum(axis=1)

    top = scores.sort_values(ascending=False).head(k)

    return top, scores


# =====================================================
# CCA METHOD
# =====================================================

def run_cca(X, stock_names, k=10):

    half = X.shape[1] // 2

    X1 = X[:, :half]
    X2 = X[:, half:]

    cca = CCA(n_components=k)

    cca.fit(X1, X2)

    weights = pd.DataFrame(
        cca.x_weights_,
        index=stock_names[:half],
        columns=[f"CCA_{i}" for i in range(k)]
    )

    scores = weights.abs().sum(axis=1)

    top = scores.sort_values(ascending=False).head(k)

    return top, scores


# =====================================================
# COMBINED RANKING
# =====================================================

def combine_rankings(pca_scores, ica_scores):

    combined = pd.DataFrame({
        "PCA": pca_scores,
        "ICA": ica_scores
    }).fillna(0)

    combined["score"] = combined.sum(axis=1)

    top = combined.sort_values("score", ascending=False).head(10)

    return top


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_factor_pipeline(folder):

    prices = load_stock_prices(folder)

    returns = compute_returns(prices)

    X = standardize_returns(returns)

    stock_names = returns.columns


    # PCA
    top_pca, pca_scores = run_pca(X, stock_names)

    print("\nTop PCA Stocks")
    print(top_pca)


    # ICA
    top_ica, ica_scores = run_ica(X, stock_names)

    print("\nTop ICA Stocks")
    print(top_ica)


    # CCA
    top_cca, cca_scores = run_cca(X, stock_names)

    print("\nTop CCA Stocks")
    print(top_cca)


    # Combined
    top_combined = combine_rankings(pca_scores, ica_scores)

    print("\nTop Combined Stocks")
    print(top_combined)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    folder = "sampled_stocks/new_directory"

    run_factor_pipeline(folder)
