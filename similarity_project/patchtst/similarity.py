import pandas as pd
import numpy as np


def build_returns(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()

    pivot = df.pivot(index="Date", columns="Symbol", values="Close")

    # forward + backward fill (pandas-safe)
    returns = pivot.pct_change().ffill().bfill()

    return returns


def get_similar_stocks(df, target="AAPL", k=5, method="correlation"):
    returns = build_returns(df)

    target = target.upper()

    print("DEBUG symbols sample:", list(returns.columns[:15]))

    if target not in returns.columns:
        raise ValueError(f"{target} not found. Sample: {list(returns.columns[:15])}")

    target_series = returns[target]

    similarities = {}

    for col in returns.columns:
        if col == target:
            continue

        aligned = pd.concat([target_series, returns[col]], axis=1).dropna()

        if len(aligned) < 20:
            continue

        similarities[col] = aligned.corr().iloc[0, 1]

    ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:k]]