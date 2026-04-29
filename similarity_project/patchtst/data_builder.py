import numpy as np
import pandas as pd
def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(["Symbol", "Date"]).dropna()
    return df
def build_stock_universe(df, price_col="Close", min_len=50):
    """
    Output:
    {
        "AAPL": np.array([...]),
        "MSFT": np.array([...]),
    }
    """
    stock_data = {}
    for symbol in df["Symbol"].unique():
        series = df[df["Symbol"] == symbol][price_col].values
        series = series[~np.isnan(series)]
        if len(series) < min_len:
            continue
        stock_data[symbol] = series.astype(np.float32)
    return stock_data