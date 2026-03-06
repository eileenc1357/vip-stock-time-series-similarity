import os
import json
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# 1. LOAD STOCK JSONL FILES
# -----------------------------------

folder = "sampled_stocks/new_directory"

print("Looking in folder:", folder)

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
        print(f"No prices in {file}")
        continue

    s = pd.Series(prices, index=dates, name=ticker)

    price_series.append(s)


# -----------------------------------
# 2. BUILD PRICE MATRIX
# -----------------------------------

prices = pd.concat(price_series, axis=1)

print("\nPrice matrix shape:", prices.shape)
print(prices.head())


# -----------------------------------
# 3. CONVERT PRICES → RETURNS
# -----------------------------------

returns = prices.pct_change()

# replace infinite values
returns = returns.replace([np.inf, -np.inf], np.nan)

# forward fill gaps
returns = returns.ffill()

# drop stocks missing too much data
returns = returns.dropna(axis=1, thresh=int(0.7 * len(returns)))

# drop remaining NaN rows
returns = returns.dropna()

print("\nReturns shape:", returns.shape)

stock_names = returns.columns


# -----------------------------------
# 4. STANDARDIZE DATA
# -----------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(returns)


# -----------------------------------
# 5. PCA (Principal Components)
# -----------------------------------

k = 10

pca = PCA(n_components=k)

pca.fit(X)

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=stock_names,
    columns=[f"PCA_{i}" for i in range(k)]
)

pca_scores = pca_loadings.abs().sum(axis=1)

top_pca = pca_scores.sort_values(ascending=False).head(k)

print("\nTop PCA Stocks:")
print(top_pca)


# -----------------------------------
# 6. ICA (Independent Components)
# -----------------------------------

ica = FastICA(n_components=k, random_state=42)

ica.fit(X)

ica_loadings = pd.DataFrame(
    ica.mixing_,
    index=stock_names,
    columns=[f"ICA_{i}" for i in range(k)]
)

ica_scores = ica_loadings.abs().sum(axis=1)

top_ica = ica_scores.sort_values(ascending=False).head(k)

print("\nTop ICA Stocks:")
print(top_ica)


# -----------------------------------
# 7. CCA (Canonical Correlation)
# -----------------------------------

half = X.shape[1] // 2

X1 = X[:, :half]
X2 = X[:, half:]

cca = CCA(n_components=k)

cca.fit(X1, X2)

cca_weights = pd.DataFrame(
    cca.x_weights_,
    index=stock_names[:half],
    columns=[f"CCA_{i}" for i in range(k)]
)

cca_scores = cca_weights.abs().sum(axis=1)

top_cca = cca_scores.sort_values(ascending=False).head(k)

print("\nTop CCA Stocks:")
print(top_cca)

# -----------------------------------
# 8. COMBINE FACTOR RANKINGS
# -----------------------------------

combined = pd.DataFrame({
    "PCA": pca_scores,
    "ICA": ica_scores
}).fillna(0)

combined["score"] = combined.sum(axis=1)

top_combined = combined.sort_values(
    "score",
    ascending=False
).head(k)

print("\nTop Combined Stocks:")
print(top_combined)