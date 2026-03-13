# Stock Similarity via Representation Learning

This project explores how different dimensionality reduction and representation learning methods capture similarity between stocks based on their historical return patterns.

Multiple approaches are compared in a unified pipeline, including:

### Linear Methods

* PCA
* ICA

### Nonlinear Manifold Methods

* Kernel PCA
* Isomap
* Locally Linear Embedding (LLE)

### Neural Representation Learning

* Masked Autoencoder (MAE)
* Variational Autoencoder (VAE)

The goal is to evaluate how different representations affect nearest-neighbor similarity relationships between securities.

---

# Project Structure

```
similarity_project
│
├── data/                    # Folder containing stock price JSONL files
│
├── outputs/
│   └── similarity_heatmap.png
│
└── src
    ├── main.py              # Main pipeline that runs all algorithms
    ├── data_loader.py       # Loads price data and computes returns
    ├── similarity.py        # Unified similarity model wrapper
    └── models.py            # Neural models (MAE and VAE)
```

---

# How It Works

The pipeline follows these steps:

1. Load stock price data
2. Compute log returns
3. Standardize the data
4. Generate embeddings using different dimensionality reduction methods
5. Compute cosine similarity between stock embeddings
6. Retrieve the Top-K most similar stocks
7. Compare model agreement** using rank correlation metrics
8. Visualize similarity** via a heatmap

All models are wrapped inside a common `SimilarityModel` interface so they can be run through the same pipeline.

# Running the Project

From the repository root:

```
cd similarity_project
python src/main.py
```

This will:

* load the dataset
* train all models
* compute stock similarity matrices
* print example Top-10 similar stocks
* compute rank correlations between models
* save a similarity heatmap to outputs/similarity_heatmap.png

# Dependencies

Main libraries used:

```
numpy
pandas
scikit-learn
tensorflow
seaborn
matplotlib
```

Install with:

```
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib
```
