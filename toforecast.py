import pandas as pd
import aqi_model
target=pd.Timestamp('2025-04-02')
df=aqi_model.load_aqi_data()
bundle=aqi_model.load_model_bundle()
days=(target - df['Date'].max()).days
forecast=aqi_model.forecast_next_days(bundle['model'], df, days=days)
print(forecast[forecast['Date'] == target.date().isoformat()].to_string(index=False))