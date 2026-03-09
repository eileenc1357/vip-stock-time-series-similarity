# --- preprocessing.py ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def _generate_ol_steps(n: int, window: int, overlap: int) -> List[Tuple[int, int]]:
    stride = window - overlap
    return [(i, i + window) for i in range(0, n - window + 1, stride)]

def create_windows(rets: pd.DataFrame, window: int=60, overlap: int=30, threshold: float=0.03):
    """
    Generates sliding windows from returns, ensuring contiguity (no gaps).
    Returns X (windows), y (labels), and meta (ticker/time info).
    """
    X_parts, y_parts, meta_parts = [], [], []

    for tkr in rets.columns:
        s = rets[tkr]
        ok = s.notna().to_numpy()
        if ok.sum() < window:
            continue

        # Identify contiguous blocks
        change = np.where(np.diff(ok.astype(int)) != 0)[0] + 1
        boundaries = np.r_[0, change, len(s)]

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if not ok[a:b].all():
                continue
            block = s.iloc[a:b]
            if len(block) < window:
                continue

            # Slide window over contiguous block
            for start, end in _generate_ol_steps(len(block), window, overlap):
                w = block.iloc[start:end].to_numpy(dtype=np.float32)
                X_parts.append(w)
                y_parts.append(int(np.max(w) > threshold))
                meta_parts.append({"ticker": tkr, "t_end": block.index[end - 1]})

    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    X = pd.DataFrame(X_parts)
    y = pd.Series(y_parts, dtype=int, name="label")
    meta = pd.DataFrame(meta_parts)
    return X, y, meta

def temporal_train_val_test_split(X, y, meta, train_frac=0.6, val_frac=0.2):
    """
    Splits data chronologically PER TICKER to avoid lookahead bias.
    Returns tuples of (X_train, X_val, X_test), (y_train, y_val, y_test), (meta_train, meta_val, meta_test).
    """
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    meta_train, meta_val, meta_test = [], [], []

    for t in meta["ticker"].unique():
        # Get indices for this ticker sorted by time
        idx = meta.index[meta["ticker"] == t]
        idx = meta.loc[idx].sort_values("t_end").index.to_numpy()

        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        # Split indices
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]

        # Append data
        if len(train_idx) > 0: 
            X_train.append(X.loc[train_idx])
            y_train.append(y.loc[train_idx])
            meta_train.append(meta.loc[train_idx])
        if len(val_idx) > 0:   
            X_val.append(X.loc[val_idx])
            y_val.append(y.loc[val_idx])
            meta_val.append(meta.loc[val_idx])
        if len(test_idx) > 0:  
            X_test.append(X.loc[test_idx])
            y_test.append(y.loc[test_idx])
            meta_test.append(meta.loc[test_idx])

    # Concatenate
    X_out = (pd.concat(X_train, ignore_index=True), pd.concat(X_val, ignore_index=True), pd.concat(X_test, ignore_index=True))
    y_out = (pd.concat(y_train, ignore_index=True), pd.concat(y_val, ignore_index=True), pd.concat(y_test, ignore_index=True))
    meta_out = (pd.concat(meta_train, ignore_index=True), pd.concat(meta_val, ignore_index=True), pd.concat(meta_test, ignore_index=True))
    
    return X_out, y_out, meta_out

def standardize_features(X_train, X_val, X_test):
    """
    Fits scaler on training data, transforms all sets.
    """
    scaler = StandardScaler()
    
    # Convert to numpy float32 for consistency
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)

    scaler.fit(X_train_np)

    return (
        scaler.transform(X_train_np),
        scaler.transform(X_val_np),
        scaler.transform(X_test_np),
        scaler
    )
