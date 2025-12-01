# src/preprocessing.py
import pandas as pd
from pathlib import Path

def clean_data(input_path, output_path):
    """
    Reads raw CSV, cleans numeric columns, fills missing dates, 
    and saves cleaned CSV to output_path.
    """
    # Read raw CSV
    df_raw = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")
    
    # Ensure numeric columns are numbers
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    
    # Fill missing dates & forward fill
    df_clean = df_raw.asfreq("D").ffill()
    
    # Save cleaned CSV
    df_clean.to_csv(output_path)
    
    return df_clean
