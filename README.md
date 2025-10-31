# Zenith Bank — Strategic Workforce Planning & Forecasting

**Case study:** Multi-department workforce forecasting for Zenith Bank.

## Contents
- `data/workforce_data.csv` — sample historical workforce dataset (2016–2024)
- `notebooks/workforce_forecast.ipynb` — Jupyter notebook with forecasting workflow (Prophet + ARIMA fallback)
- `app.py` — Streamlit dashboard to explore historical and forecasted headcounts
- `forecast.csv` — model output (auto-updated by notebook/app when you run them)
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Quick start (Windows Command Prompt)
1. Open Command Prompt.
2. Change to project folder:
   ```
   cd %USERPROFILE%\Downloads\zenith_workforce_forecasting
   ```
3. Create & activate virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Project overview
This project forecasts headcount at department-region level for a 3-year horizon. The notebook demonstrates how to:
- aggregate historical data
- train Prophet models per department-region
- fallback to ARIMA if Prophet fails
- export `forecast.csv` for HR planning

## Quick start for first time Run (Windows PowerShell):

```Windows powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
jupyter notebook \workforce_forecast.ipynb
streamlit run app.py
```

## Re Run
venv\Scripts\Activate.ps1
streamlit run app.py