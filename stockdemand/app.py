import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet 
from scipy.stats import norm
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---

# Cache data loading to avoid re-reading the file on every interaction
@st.cache_data
def load_data(url):
    """Loads and preprocesses the retail data."""
    df = pd.read_excel(url)
    df.columns = [c.strip() for c in df.columns]
    df = df[['StockCode', 'Quantity', 'InvoiceDate']].dropna()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0] # Filter out returns
    return df

# Cache the aggregation of daily sales data
@st.cache_data
def get_daily_sales(_df):
    """Aggregates sales data to a daily level per StockCode."""
    daily = _df.copy()
    daily['date'] = daily['InvoiceDate'].dt.floor('d')
    daily_sales = daily.groupby(['date', 'StockCode']).agg(units_sold=('Quantity', 'sum')).reset_index()
    return daily_sales

# Cache Prophet model fitting to avoid retraining on every run for the same SKU
@st.cache_resource
def get_prophet_forecast(_daily_sales, sku):
    """Fits a Prophet model and returns the forecast for a given SKU."""
    series = _daily_sales[_daily_sales['StockCode'] == sku][['date', 'units_sold']].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )
    
    if len(series) < 30: # Minimum data points for a meaningful forecast
        return None, None

    m = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    m.fit(series)
    
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    return m, forecast

# --- Calculation Function ---
def compute_inventory_stats(forecast_series, order_cost=50, holding_cost=5, lead_time_days=7, service_level=0.95):
    """Computes EOQ, Safety Stock, and ROP from forecast data."""
    # Use forecast for the next 30 days for calculations
    future_demand = forecast_series[forecast_series['ds'] > forecast_series['ds'].max() - pd.Timedelta(days=30)]['yhat']
    
    avg_daily_demand = future_demand.mean()
    if avg_daily_demand < 0:
        avg_daily_demand = 0 # Demand cannot be negative

    std_daily_demand = future_demand.std()
    
    # Economic Order Quantity (EOQ)
    annual_demand = avg_daily_demand * 365
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost) if holding_cost > 0 else 0
    
    # Safety Stock (SS)
    z_score = norm.ppf(service_level)
    safety_stock = z_score * std_daily_demand * np.sqrt(lead_time_days)
    
    # Reorder Point (ROP)
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    
    return {
        "avg_daily_forecast": avg_daily_demand,
        "EOQ": eoq,
        "SafetyStock": safety_stock if safety_stock > 0 else 0,
        "ROP": rop if rop > 0 else 0
    }

# --- Plotting Functions ---
def plot_forecast(forecast, series):
    """Plots the historical data and the Prophet forecast using Plotly."""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=series['ds'], y=series['y'], mode='lines', name='Historical Sales', line=dict(color='royalblue')))
    
    # Add forecast line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast (yhat)', line=dict(color='darkorange', dash='dash')))
    
    # Add uncertainty interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        name='Upper Bound',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        name='Uncertainty',
        showlegend=False
    ))

    fig.update_layout(
        title="Demand Forecast vs. Historical Sales",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend=dict(x=0.01, y=0.99)
    )
    return fig

# --- Main App Logic ---
def main():
    st.title("📈 Stock Demand Forecasting & Inventory Optimization")
    st.markdown("This dashboard analyzes historical sales data to forecast future demand and calculate optimal inventory levels.")

    # --- Data Loading and Preparation ---
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = load_data(data_url)
    daily_sales = get_daily_sales(df)
    
    # Get top N SKUs by total quantity sold for the dropdown
    top_skus = df.groupby('StockCode')['Quantity'].sum().nlargest(100).index.tolist()

    # --- Sidebar for User Inputs ---
    st.sidebar.header("⚙️ User Inputs")
    selected_sku = st.sidebar.selectbox("Select a Product (StockCode):", top_skus, index=0)
    
    st.sidebar.subheader("Inventory Parameters")
    order_cost = st.sidebar.number_input("Cost per Order ($)", min_value=1, value=50)
    holding_cost = st.sidebar.number_input("Holding Cost per Unit per Year ($)", min_value=0.1, value=5.0, step=0.1)
    lead_time = st.sidebar.slider("Lead Time (Days)", 1, 30, 7)
    service_level = st.sidebar.slider("Service Level (%)", 80, 99, 95, format="%d%%") / 100.0

    # --- Main Panel Display ---
    st.header(f"Analysis for SKU: `{selected_sku}`")
    
    # Filter data for the selected SKU
    sku_daily_sales = daily_sales[daily_sales['StockCode'] == selected_sku].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )
    
    if sku_daily_sales.empty:
        st.warning("No data available for the selected SKU.")
        return

    # --- Forecasting ---
    with st.spinner("Fitting the Prophet model and generating forecast..."):
        model, forecast = get_prophet_forecast(daily_sales, selected_sku)

    if forecast is None:
        st.error("Not enough data to generate a forecast for this SKU (minimum 30 data points required).")
        return

    # --- Display Results ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Demand Forecast")
        forecast_plot = plot_forecast(forecast, sku_daily_sales)
        st.plotly_chart(forecast_plot, use_container_width=True)

    with col2:
        st.subheader("📦 Inventory Metrics")
        stats = compute_inventory_stats(forecast, order_cost, holding_cost, lead_time, service_level)
        
        st.metric(label="Average Daily Demand (Next 30 Days)", value=f"{stats['avg_daily_forecast']:.2f} units")
        st.metric(label="Economic Order Quantity (EOQ)", value=f"{stats['EOQ']:.0f} units", help="The optimal order size that minimizes total inventory costs.")
        st.metric(label="Safety Stock (SS)", value=f"{stats['SafetyStock']:.0f} units", help=f"Buffer stock to reduce risk of stockouts (at {service_level:.0%} service level).")
        st.metric(label="Reorder Point (ROP)", value=f"{stats['ROP']:.0f} units", help="The inventory level at which a new order should be placed.")

    # --- DataFrames Display ---
    st.subheader("🔍 Raw Data & Forecast Details")
    
    tab1, tab2 = st.tabs(["Forecast Data", "Historical Daily Sales"])
    
    with tab1:
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly', 'yearly']].tail(30).round(2))
    
    with tab2:
        st.dataframe(sku_daily_sales.sort_values('ds', ascending=False))


if __name__ == "__main__":

    main()
