import pandas as pd
import numpy as np
from data_loader import load_prices_from_zip, get_log_returns
from preprocessing import create_windows, temporal_train_val_test_split, standardize_features
from models import MAEModel, VAEModel
from evaluation import evaluate_embeddings_logreg, get_ticker_embeddings, get_top_k_similar

def main():
    # 1. Load Data
    # Update these paths if running locally
    zip_path = "/content/sampled_stocks.zip"
    extract_root = "/content/sampled_stocks_extracted"
    
    print("Loading data...")
    try:
        prices = load_prices_from_zip(zip_path, extract_root)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    if prices.empty:
        print("No data found inside zip.")
        return

    # 2. Preprocess
    print("Calculating returns...")
    rets = get_log_returns(prices)
    
    print("Creating windows...")
    # Adjust window/overlap params as needed
    X, y, meta = create_windows(rets, window=60, overlap=30, threshold=0.03)
    
    if X.empty:
        print("No valid windows created.")
        return

    print(f"Dataset shape: {X.shape}")
    
    print("Splitting data...")
    # The refactored function returns flattened tuples: X_train, X_val, X_test, y_train, y_val, y_test
    # We unpack the 3 nested tuples returned by the function
    (X_train, X_val, X_test), (y_train, y_val, y_test), (meta_train, meta_val, meta_test) = \
        temporal_train_val_test_split(X, y, meta, train_frac=0.6, val_frac=0.2)
        
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    print("Standardizing...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(X_train, X_val, X_test)
    
    # 3. Train Models
    input_dim = X_train_scaled.shape[1]
    
    # --- MAE ---
    print("\nTraining MAE Model...")
    mae = MAEModel(input_dim=input_dim, latent_dim=16)
    mae.fit(X_train_scaled, validation_data=X_val_scaled, epochs=5, verbose=1) # Low epochs for demo
    
    mae_train_emb = mae.embed(X_train_scaled)
    mae_test_emb = mae.embed(X_test_scaled)
    
    # --- VAE ---
    print("\nTraining VAE Model...")
    vae = VAEModel(input_dim=input_dim, latent_dim=16)
    vae.fit(X_train_scaled, epochs=5, verbose=1)
    
    vae_train_emb = vae.embed(X_train_scaled)
    vae_test_emb = vae.embed(X_test_scaled)
    
    # 4. Evaluation
    print("\n--- Evaluation Results ---")
    mae_metrics = evaluate_embeddings_logreg(mae_train_emb, y_train, mae_test_emb, y_test, name="MAE")
    print("MAE Metrics:", mae_metrics)
    
    vae_metrics = evaluate_embeddings_logreg(vae_train_emb, y_train, vae_test_emb, y_test, name="VAE")
    print("VAE Metrics:", vae_metrics)

    # 5. Top-K Analysis
    print("\n--- Top-K Similarity Analysis ---")
    # Reconstruct full dataset for comprehensive ticker embeddings
    X_full = np.concatenate([X_train_scaled, X_val_scaled, X_test_scaled], axis=0)
    meta_full = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    
    # Calculate embeddings for all windows
    mae_full_emb = mae.embed(X_full)
    vae_full_emb = vae.embed(X_full)
    
    # Average embeddings per ticker
    mae_ticker_emb = get_ticker_embeddings(mae_full_emb, meta_full)
    vae_ticker_emb = get_ticker_embeddings(vae_full_emb, meta_full)
    
    # --- EDIT THIS QUERY TICKER ---
    query_ticker = "GOOGL"
    k = 10
    
    print(f"\nTop {k} similar to {query_ticker} (MAE):")
    try:
        df_mae_top = get_top_k_similar(mae_ticker_emb, query_ticker, k)
        print(df_mae_top.to_string(index=False))
    except ValueError as e:
        print(e)

    print(f"\nTop {k} similar to {query_ticker} (VAE):")
    try:
        df_vae_top = get_top_k_similar(vae_ticker_emb, query_ticker, k)
        print(df_vae_top.to_string(index=False))
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
