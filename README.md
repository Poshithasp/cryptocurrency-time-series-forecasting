## Time-Series Cryptocurrency Forecasting Project

This project applies **data analytics, machine learning, and time-series forecasting** to predict cryptocurrency price movements. It includes complete workflows—from data collection to model deployment—using Python, statistical modeling, and deep learning.

---

## Project Overview
The goal of this project is to analyze cryptocurrency price trends and forecast future values using historical data. It implements multiple forecasting models to compare performance and improve prediction accuracy.

The system includes:
- Data preprocessing
- Exploratory data analysis
- ARIMA, LSTM, and Prophet forecasting models
- Volatility insights
- Interactive dashboard for visualization

---

## Technologies & Tools Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- TensorFlow / Keras (LSTM model)
- Facebook Prophet
- Streamlit Dashboard
- APIs for data collection (Yahoo Finance / CoinGecko)

---

## Project Features

### 1. Cryptocurrency Data Collection
- Fetches real-time & historical data
- Integrates APIs such as CoinGecko, Binance, Yahoo Finance

### 2. Data Preprocessing & Exploration
- Handles missing values
- Resampling, trend detection
- Visualization of close prices, moving averages, volatility

### 3. Time-Series Forecasting Models
- **ARIMA**: Statistical forecasting
- **LSTM Neural Network**: Deep learning sequential model
- **Prophet**: Trend + seasonality forecasting

### 4. Volatility & Sentiment Analysis
- Measures price fluctuations
- (Optional) Sentiment data from crypto news & social media

### 5. Interactive Dashboard
- Built with Streamlit or Plotly
- Shows:
  - Historical trends
  - Real-time price updates
  - Forecasted values
  - Model comparisons

---

## Folder Structure
project/
│── data/
│── src/
│── notebooks/
│── models/
│── dashboard/
│── reports/
│── images/
│── requirements.txt
│── README.md

---

##  How to Run the Project

1️.Clone the repository
```bash
git clone https://github.com/your-username/TimeSeries_Forecasting_Project.git
cd TimeSeries_Forecasting_Project

2️.Install dependencies
pip install -r requirements.txt

3️. Run analysis notebook

Open Jupyter Notebook and run the .ipynb files.

4️. Run the Streamlit dashboard
streamlit run app.py
portfolios

## Author

Poshitha S P
Data Analytics Enthusiast
