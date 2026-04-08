# Chennai AQI Analysis Report

## Dataset

- Rows: 1548
- Date range: 2021-01-01 to 2025-03-31
- Mean AQI: 72.56
- Minimum AQI: 24.00
- Maximum AQI: 245.00

AQI category counts:

| category | days |
| --- | ---: |
| Good | 237 |
| Satisfactory | 1125 |
| Moderate | 179 |
| Poor | 7 |
| Very Poor | 0 |
| Severe | 0 |

## Model Results

Best model on the chronological 80/20 test split: **Linear Regression**

- R2 score: 0.5611
- RMSE: 17.8793
- MAE: 12.6626

Average walk-forward validation for the champion model:

- R2 score: 0.4520
- RMSE: 19.2594
- MAE: 13.3076

## Forecast Summary

- Forecast period: 2025-04-01 to 2025-04-30
- Mean forecast AQI: 64.54
- Minimum forecast AQI: 61.24
- Maximum forecast AQI: 69.46
- Forecast standard deviation: 2.28

## Generated Figures

- `reports/figures/01_aqi_trend.png`
- `reports/figures/02_monthly_distribution.png`
- `reports/figures/03_yearly_average.png`
- `reports/figures/04_aqi_category_counts.png`
- `reports/figures/05_autocorrelation.png`
- `reports/figures/06_model_metric_comparison.png`
- `reports/figures/07_actual_vs_predicted.png`
- `reports/figures/08_actual_vs_predicted_scatter.png`
- `reports/figures/09_prediction_error_over_time.png`
- `reports/figures/10_residuals.png`
- `reports/figures/11_feature_importance.png`
- `reports/figures/12_walk_forward_metrics.png`
- `reports/figures/13_forecast.png`
