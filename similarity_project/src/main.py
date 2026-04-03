import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from data_loader import load_stock_prices, compute_returns
from similarity import SimilarityModel, plot_similarity_graph
from models import MAEModel, VAEModel
from sklearn.decomposition import FastICA

from visibility_graph import VisibilityGraphEmbedder

from forecasting import (
    evaluate_similarity_method,
    evaluate_random_baseline,
    evaluate_univariate,
    symmetry_score,
)

def rank_correlation(list_a, list_b):
    """
    Compute Spearman and Kendall correlation between two rankings.

    """

    # Only compare shared tickers
    common = [x for x in list_a if x in list_b]

    if len(common) < 2:
        return None, None

    rank_a = [list_a.index(x) for x in common]
    rank_b = [list_b.index(x) for x in common]

    spearman = spearmanr(rank_a, rank_b).correlation
    kendall = kendalltau(rank_a, rank_b).correlation

    return spearman, kendall


# ===============================
# Load Data
# ===============================

prices = load_stock_prices("./data/new_directory")

returns = compute_returns(prices)

# speed fix, reduce time series to past 250 days, for dtw
returns = returns.iloc[-250:]

tickers = list(returns.columns)


# ===============================
# Standardize
# ===============================

scaler = StandardScaler()

X = scaler.fit_transform(returns.T)

# Raw per-ticker sequences for DTW only
X_dtw = returns.T.to_numpy(dtype=float)


# ===============================
# Define Models
# ===============================

input_dim = X.shape[1]

methods = {

    "PCA": SimilarityModel(
        PCA(n_components=5)
    ),

    "KernelPCA": SimilarityModel(
        KernelPCA(
            n_components=5,
            kernel="rbf",
            gamma=10
        )
    ),

    "Isomap": SimilarityModel(
        Isomap(
            n_components=5,
            n_neighbors=10
        )
    ),

    "LLE": SimilarityModel(
        LocallyLinearEmbedding(
            n_components=5,
            n_neighbors=10
        )
    ),

    "MAE": SimilarityModel(
        MAEModel(input_dim, latent_dim=16),
        is_autoencoder=True
    ),

    "VAE": SimilarityModel(
        VAEModel(input_dim, latent_dim=16),
        is_autoencoder=True
    ),

    "ICA": SimilarityModel(
      
        FastICA(
          n_components=5,
          random_state=0
          )
    ),
    # ===============================
    # BASELINES
    # ===============================
    "CosineRaw": SimilarityModel(metric="cosine"),
    "Euclidean": SimilarityModel(metric="euclidean"),
    "Correlation": SimilarityModel(metric="correlation"),

    # "DTW": SimilarityModel(
    #     model=None,
    #     metric="dtw",
    #     dtw_z_normalize=True
    # ),

    "VisibilityGraph": SimilarityModel(
        VisibilityGraphEmbedder(
            n_bins=10, 
            verbose=True
        )
    ),

    "Wasserstein": SimilarityModel(
        model=None,
        metric="wasserstein"
    )
}


# ===============================
# Run Models
# ===============================

for name, model in methods.items():

    print("\nRunning", name)

    if name == "DTW":
        model.fit(X_dtw, tickers)

    elif name == "Correlation":
        # 🔥 use raw returns (not standardized)
        model.fit(returns.T.to_numpy(), tickers)

    else:
        model.fit(X, tickers)


# ===============================
# Select Companies
# ===============================

test_companies = tickers

print("\nSelected companies:", test_companies)


# ===============================
# Similarity comparisons
# ===============================
topk_results = {}

example_company = tickers[0]

for method_name, model in methods.items():

    print("\n==============================")
    print(method_name)
    print("==============================")

    topk_results[method_name] = {}

    for company in test_companies:

        result = model.top_k(company, 10)

        # only show one example per model
        if company == example_company:
            print("\nExample Top 10 similar to", company)
            print(result)

        topk_results[method_name][company] = list(result.index)

print("\n\n==============================")
print("Average Rank Correlation Between Models")
print("==============================")

method_names = list(methods.keys())

for i in range(len(method_names)):
    for j in range(i+1, len(method_names)):

        m1 = method_names[i]
        m2 = method_names[j]

        spearman_vals = []
        kendall_vals = []

        for company in test_companies:

            r1 = topk_results[m1][company]
            r2 = topk_results[m2][company]

            spearman, kendall = rank_correlation(r1, r2)

            if spearman is not None:
                spearman_vals.append(spearman)

            if kendall is not None:
                kendall_vals.append(kendall)

        print(
            f"{m1} vs {m2} -> "
            f"Spearman: {np.mean(spearman_vals):.3f}, "
            f"Kendall: {np.mean(kendall_vals):.3f}"
        )
# ===============================
# FORECASTING EVALUATION
# ===============================

print("\n\n==============================")
print("Forecasting Performance")
print("==============================")
print("\n\n==============================")
print("Methods being compared")
print("==============================")

for name in methods:
    print(name)

results = {}

k = 10  # number of similar stocks

for method_name, model in methods.items():

    print(f"\nEvaluating {method_name}")

    mses = []

    for idx, target in enumerate(tickers):

        mse = evaluate_similarity_method(
            model,
            target,
            returns,
            k=k
        )
        if idx < 2:
            neighbors = model.top_k(target, k).index.tolist()
            print(f"{method_name} | {target} neighbors: {neighbors[:3]}")

        mses.append(mse)

    results[method_name] = np.mean(mses)


# ===============================
# BASELINES
# ===============================

print("\nEvaluating Random Baseline")

random_mses = []
for target in tickers:
    mse = evaluate_random_baseline(target, returns, tickers, k=k)
    random_mses.append(mse)

results["Random"] = np.mean(random_mses)


print("\nEvaluating Univariate Baseline")

uni_mses = []
for target in tickers:
    mse = evaluate_univariate(target, returns)
    uni_mses.append(mse)

results["Univariate"] = np.mean(uni_mses)


# ===============================
# PRINT RESULTS
# ===============================

print("\n\n==============================")
print("Final Results (MSE ↓ better)")
print("==============================")

print("\n--- Embedding Methods ---")
for method in ["PCA", "KernelPCA", "Isomap", "LLE", "ICA", "MAE", "VAE"]:
    if method in results:
        print(f"{method}: {results[method]:.6f}")

print("\n--- Similarity Baselines ---")
for method in ["CosineRaw", "Euclidean", "Correlation"]:
    print(f"{method}: {results[method]:.6f}")

print("\n--- True Baselines ---")
print(f"Random: {results['Random']:.6f}")
print(f"Univariate: {results['Univariate']:.6f}")

# ===============================
# Symmetry Scores
# ===============================

print("\n\n==============================")
print("Symmetry Scores (↓ better)")
print("==============================")

for method_name, model in methods.items():
    if model.similarity_df is None:
        print(f"{method_name}: skipped")
        continue

    score = symmetry_score(model.similarity_df)
    print(f"{method_name}: {score:.6f}")



# ===============================
# Heatmap
# ===============================

example_model = list(methods.values())[0]

sim_matrix = example_model.similarity_df

plt.figure(figsize=(10,8))

sns.heatmap(
    sim_matrix,
    cmap="coolwarm",
    annot=False
)

plt.title("Stock Similarity Matrix")

plt.tight_layout()

plt.savefig("./outputs/similarity_heatmap.png")

print("\nSaved heatmap")




# ===============================
# Network Graph
# ===============================

print("\nGenerating Similarity Network Graph...")

# Dummy sector data assigned to each ticker for now.
sector_map = { ticker: "Tech" for ticker in tickers[:len(tickers)//2] }
sector_map.update({ ticker: "Finance" for ticker in tickers[len(tickers)//2:] })

wasserstein_model = methods.get("Wasserstein")
if wasserstein_model and wasserstein_model.similarity_df is not None:
    plot_similarity_graph(wasserstein_model.similarity_df, sector_map, threshold=0.8)
else:
    print("Network Graph model did not run successfully, network graph will not be generated/updated.")