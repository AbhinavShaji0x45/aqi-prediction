# Chennai AQI Prediction

This project trains an academic prototype model for Chennai AQI forecasting from daily historical AQI values.

## Rebuild Data

```powershell
.\.venv\Scripts\python.exe prepare_data.py
```

This reads the yearly Excel files in the repo with relative paths, writes `clean_2021.csv` through `clean_2025.csv`, and rebuilds `clean_aqi.csv`.

## Train And Forecast

```powershell
.\.venv\Scripts\python.exe aqi_model.py
```

This trains the 19-feature champion `VotingRegressor`, writes `aqi_champion_model.joblib`, updates `model_metadata.json`, and regenerates `aqi_forecast_30days.csv`.

## Generate Analysis Report

```powershell
.\.venv\Scripts\python.exe analysis_report.py
```

This writes report-ready metrics, graphs, feature importance, and a markdown summary into the `reports` folder.
The model comparison includes Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and the champion voting ensemble.

## See How To Use The Model Code

```powershell
.\.venv\Scripts\python.exe use_model_example.py
```

This demonstrates every main function in `aqi_model.py`: loading data, creating features, building and training the model, saving/loading the bundle, creating a next-day feature row, forecasting a chosen date, and regenerating the standard 30-day forecast.

## Model Status

The current model is suitable for academic demonstration and trend exploration. It should not be treated as production-ready without external validation, monitoring, and additional predictors such as weather, traffic, or event data.
