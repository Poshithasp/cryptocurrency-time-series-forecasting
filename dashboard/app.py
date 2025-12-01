import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_collection import load_processed
from src.models import load_arima, load_prophet, load_lstm
from pathlib import Path

st.set_page_config(layout="wide", page_title="Crypto Forecast")
st.title("Cryptocurrency Forecast Dashboard")

symbol = st.sidebar.selectbox("Symbol", ["BTC-USD"])
df = load_processed(symbol)

# --- Historical Price Plot ---
st.subheader("Historical Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
st.plotly_chart(fig, use_container_width=True)

# --- Forecast Comparison ---
reports_path = Path("../reports/test_forecasts.csv")
if reports_path.exists():
    fc = pd.read_csv(reports_path, parse_dates=["date"]).set_index("date")
    
    st.subheader("Model Comparison (Test Period)")
    fig2 = go.Figure()
    
    # Plot only historical Actual (last 60 days)
    fig2.add_trace(go.Scatter(
        x=fc.index[:60],
        y=fc['actual'].iloc[:60],
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Forecast models (last 60 + 7 future)
    fig2.add_trace(go.Scatter(x=fc.index, y=fc['arima'], name='ARIMA'))
    fig2.add_trace(go.Scatter(x=fc.index, y=fc['prophet'], name='Prophet'))
    fig2.add_trace(go.Scatter(x=fc.index, y=fc['lstm'], name='LSTM'))
    
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Run the notebook to generate forecasts first.")
