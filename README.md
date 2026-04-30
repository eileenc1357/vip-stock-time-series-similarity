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
* PatchTST (Patch Time Series Transformer)

The goal is to evaluate how different representations affect nearest-neighbor similarity relationships between securities.

PatchTST is a Transformer-based model for time series that operates on local patches of the input sequence. In this project, it is used to generate embeddings of stock return series for similarity comparison.

## PatchTST Experiments

PatchTST is included in the main benchmark pipeline for fair comparison against other representation learning methods.

For more extensive or faster experimentation with PatchTST, additional scripts are provided in the `experiments/` directory (e.g., `fast_patchtst.py`), which allow:

- faster iteration on model configurations  
- isolated testing of PatchTST performance  
- more granular analysis of embedding quality  

These scripts are intended for deeper analysis beyond the unified comparison framework in `main.py`.

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
