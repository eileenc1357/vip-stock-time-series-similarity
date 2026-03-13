import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from data_loader import load_stock_prices, compute_returns
from similarity import SimilarityModel
from models import MAEModel, VAEModel
from sklearn.decomposition import FastICA
from scipy.stats import spearmanr, kendalltau

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

tickers = list(returns.columns)


# ===============================
# Standardize
# ===============================

scaler = StandardScaler()

X = scaler.fit_transform(returns.T)


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
}


# ===============================
# Run Models
# ===============================

for name, model in methods.items():

    print("\nRunning", name)

    model.fit(X, tickers)


# ===============================
# Select 5 Companies
# ===============================

test_companies = tickers

print("\nSelected companies:", test_companies)


# ===============================
# Top 10 Similar
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