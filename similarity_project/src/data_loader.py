import os
import json
import pandas as pd
import numpy as np


def load_stock_prices(folder):

    series_list = []

    for file in os.listdir(folder):

        if not file.endswith(".jsonl"):
            continue

        ticker = file.replace(".jsonl", "")
        path = os.path.join(folder, file)

        dates = []
        prices = []

        with open(path) as f:
            for line in f:

                try:
                    data = json.loads(line)
                except:
                    continue

                if "close" in data and "timestamp" in data:

                    prices.append(data["close"])

                    dates.append(
                        pd.to_datetime(data["timestamp"], unit="ms")
                    )

        if len(prices) == 0:
            continue

        s = pd.Series(prices, index=dates, name=ticker)

        series_list.append(s)

    prices = pd.concat(series_list, axis=1)

    prices = prices.sort_index()

    prices = prices.ffill()

    print("Loaded price matrix:", prices.shape)

    return prices


def compute_returns(prices):

    returns = np.log(prices / prices.shift(1))

    returns = returns.replace([np.inf, -np.inf], np.nan)

    returns = returns.fillna(0)

    print("Returns shape:", returns.shape)

    return returns