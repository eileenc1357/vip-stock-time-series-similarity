# --- data_loader.py ---
import zipfile
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_prices_from_zip(zip_path: str, extract_root: str) -> pd.DataFrame:
    """
    Unzips the file, reads JSONL files, and returns a pivot table of Close prices.
    """
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

    # Find JSONL files
    jsonl_files = sorted(extract_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {extract_root}")

    # Read all files
    all_rows = []
    for fp in jsonl_files:
        ticker = fp.stem.upper()
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["ticker"] = ticker
                    all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    
    # Convert timestamps (ms) to datetime
    raw["dt"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True).dt.tz_convert(None)

    # Pivot to wide format (Index=Date, Columns=Ticker)
    prices = (
        raw.pivot_table(index="dt", columns="ticker", values="close", aggfunc="last")
           .sort_index()
    )
    return prices

def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates log returns from price data.
    """
    return np.log(prices).diff()
