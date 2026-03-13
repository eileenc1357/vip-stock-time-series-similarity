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
            gamma=0.1
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

    

    "Wasserstein": SimilarityModel(
        model=None,
        metric="wasserstein"
    ),
}


# ===============================
# Run Models
# ===============================

for name, model in methods.items():

    print("\nRunning", name)

    if name == "DTW":
        model.fit(X_dtw, tickers)   # raw returns per ticker, shape (n_tickers, n_time)
    else:
        model.fit(X, tickers)       # standardized for embeddings


# ===============================
# Select 5 Companies
# ===============================

test_companies = random.sample(tickers, 5)

print("\nSelected companies:", test_companies)


# ===============================
# Top 10 Similar
# ===============================

for method_name, model in methods.items():

    print("\n==============================")
    print(method_name)
    print("==============================")

    for company in test_companies:

        print("\nTop 10 similar to", company)

        print(model.top_k(company, 10))


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