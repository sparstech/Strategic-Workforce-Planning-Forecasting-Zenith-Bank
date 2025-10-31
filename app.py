import streamlit as st
import pandas as pd
import numpy as np
import os
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import plotly.express as px

st.set_page_config(layout="wide", page_title="Zenith Bank Workforce Forecasting")

BASE = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(BASE, "workforce_data.csv")
FORECAST_OUT = os.path.join(os.path.dirname(__file__), "forecast.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.title("Zenith Bank — Strategic Workforce Planning & Forecasting")
st.markdown("Interactive multi-department headcount forecasting (Prophet with ARIMA fallback).")

# Filters
departments = sorted(df['Department'].unique())
regions = sorted(df['Region'].unique())

col1, col2 = st.columns([2,1])
with col1:
    sel_dept = st.selectbox("Select Department", departments, index=0)
with col2:
    sel_region = st.selectbox("Select Region", regions, index=0)

hist = df[(df['Department']==sel_dept)&(df['Region']==sel_region)].sort_values('Year')

st.subheader(f"Historical headcount — {sel_dept} ({sel_region})")
fig = px.line(hist, x='Year', y='Headcount', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Forecasting
horizon = st.number_input("Forecast horizon (years)", min_value=1, max_value=5, value=3)
if st.button("Run Forecast"):
    # prepare time series for Prophet
    ts = hist[['Year','Headcount']].rename(columns={'Year':'ds','Headcount':'y'})
    # convert year to date (use Jan 1 of each year)
    ts['ds'] = pd.to_datetime(ts['ds'].astype(int).astype(str) + '-01-01')

    try:
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        m.fit(ts)
        future = m.make_future_dataframe(periods=horizon, freq='YS')
        fcst = m.predict(future)
        res = fcst[['ds','yhat','yhat_lower','yhat_upper']].copy()
        res['Year'] = res['ds'].dt.year
        out = res[['Year','yhat','yhat_lower','yhat_upper']].rename(columns={'yhat':'Forecast_Headcount','yhat_lower':'Lower','yhat_upper':'Upper'})
    except Exception as e:
        st.warning("Prophet failed, falling back to ARIMA. Error: " + str(e))
        # fallback ARIMA on the 'y' series
        y = ts.set_index('ds')['y'].astype(float)
        model = ARIMA(y, order=(1,1,0))
        model_fit = model.fit()
        future_years = pd.date_range(start=y.index[-1], periods=horizon+1, freq='YS')[1:]
        preds = model_fit.get_forecast(steps=horizon)
        mean = preds.predicted_mean
        ci = preds.conf_int(alpha=0.05)
        out_rows = []
        for i, d in enumerate(future_years):
            out_rows.append({
                'Year': d.year,
                'Forecast_Headcount': float(mean.iloc[i]),
                'Lower': float(ci.iloc[i,0]),
                'Upper': float(ci.iloc[i,1])
            })
        out = pd.DataFrame(out_rows)

    # show forecast table
    st.subheader("Forecast (next {} years)".format(horizon))
    st.dataframe(out)

    # combine historical + forecast for plotting
    hist_plot = hist.copy()
    hist_plot['Type'] = 'Historical'
    plot_fore = out.copy()
    plot_fore['Type'] = 'Forecast'
    plot_fore['Headcount'] = plot_fore['Forecast_Headcount']
    plot_hist = pd.concat([hist_plot[['Year','Headcount','Type']], plot_fore[['Year','Headcount','Type']]], ignore_index=True)
    fig2 = px.line(plot_hist, x='Year', y='Headcount', color='Type', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Export forecast to CSV (appends or overwrites per run)
    # We'll write Department, Region, Year, Forecast_Headcount, Lower, Upper
    out_export = out.copy()
    out_export['Department'] = sel_dept
    out_export['Region'] = sel_region
    out_export = out_export[['Department','Region','Year','Forecast_Headcount','Lower','Upper']]
    # load existing forecast file
    if os.path.exists(FORECAST_OUT):
        existing = pd.read_csv(FORECAST_OUT)
        # remove any existing rows for this dept-region-year combinations
        merged = pd.concat([existing[~((existing['Department']==sel_dept)&(existing['Region']==sel_region))], out_export], ignore_index=True)
        merged.to_csv(FORECAST_OUT, index=False)
    else:
        out_export.to_csv(FORECAST_OUT, index=False)
    st.success(f"Forecast saved to {FORECAST_OUT}")
