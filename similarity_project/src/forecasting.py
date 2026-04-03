import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def compute_embeddings(X, n_components=10):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def compute_similarity(embeddings):
    return cosine_similarity(embeddings)


def get_neighbors(similarity_matrix, k=5):
    return np.argsort(-similarity_matrix, axis=1)[:, 1:k+1]


def neighbor_forecast(y, neighbors, alpha=0.0):
    """
    y: shape (num_stocks, time)
    neighbors: shape (num_stocks, k)

    alpha:
        0.0 → only neighbors
        0.5 → mix of self + neighbors
    """
    num_stocks, T = y.shape
    preds = np.zeros((num_stocks, T-1))

    for i in range(num_stocks):
        for t in range(T-1):
            neighbor_vals = y[neighbors[i], t+1]  # FUTURE of neighbors
            neighbor_mean = np.mean(neighbor_vals)

            self_val = y[i, t+1]

            preds[i, t] = alpha * self_val + (1 - alpha) * neighbor_mean

    return preds


def evaluate_mse(preds, y):
    true = y[:, 1:]
    return np.mean((preds - true) ** 2)


def run_pipeline(X, y, k=5):
    """
    X: features for embedding (num_stocks, features)
    y: time series (num_stocks, time)
    """

    # Step 1: embeddings
    embeddings = compute_embeddings(X)

    # Step 2: similarity
    sim = compute_similarity(embeddings)

    # Step 3: neighbors
    neighbors = get_neighbors(sim, k=k)

    # Step 4: forecast (THIS is the important change)
    preds = neighbor_forecast(y, neighbors, alpha=0.0)

    # Step 5: evaluate
    mse = evaluate_mse(preds, y)

    return mse, preds

# ===============================
# NEW: Evaluation Functions
# ===============================

def evaluate_similarity_method(model, target, returns_df, k=5):
    """
    Use top-k similar stocks to forecast target stock.
    """

    tickers = list(returns_df.columns)
    target_idx = tickers.index(target)

    # get neighbors
    neighbors = model.top_k(target, k).index.tolist()
    neighbor_indices = [tickers.index(n) for n in neighbors]

    y = returns_df.to_numpy().T  # shape (num_stocks, time)

    T = y.shape[1]
    preds = []
    true = []

    for t in range(T - 1):
        neighbor_vals = y[neighbor_indices, t+1]
        pred = np.mean(neighbor_vals)

        preds.append(pred)
        true.append(y[target_idx, t+1])

    preds = np.array(preds)
    true = np.array(true)

    return np.mean((preds - true) ** 2)


def evaluate_random_baseline(target, returns_df, tickers, k=5):
    """
    Randomly pick k stocks instead of similar ones.
    """

    target_idx = tickers.index(target)
    y = returns_df.to_numpy().T

    other_indices = [i for i in range(len(tickers)) if tickers[i] != target]
    random_neighbors = np.random.choice(other_indices, size=k, replace=False)

    T = y.shape[1]
    preds = []
    true = []

    for t in range(T - 1):
        neighbor_vals = y[random_neighbors, t+1]
        pred = np.mean(neighbor_vals)

        preds.append(pred)
        true.append(y[target_idx, t+1])

    return np.mean((np.array(preds) - np.array(true)) ** 2)


def evaluate_univariate(target, returns_df):
    """
    Predict using the stock itself (baseline).
    """

    tickers = list(returns_df.columns)
    target_idx = tickers.index(target)

    y = returns_df.to_numpy().T

    T = y.shape[1]
    preds = []
    true = []

    for t in range(T - 1):
        pred = y[target_idx, t]   # previous value
        actual = y[target_idx, t+1]

        preds.append(pred)
        true.append(actual)

    return np.mean((np.array(preds) - np.array(true)) ** 2)


# ===============================
# symmetry score
# ===============================

def symmetry_score(sim_df):
    return np.mean(np.abs(sim_df.values - sim_df.values.T))

