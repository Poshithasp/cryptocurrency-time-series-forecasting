import pandas as pd
from pathlib import Path

def load_processed(symbol="BTC-USD"):
    path = Path("../data/raw") / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return df
