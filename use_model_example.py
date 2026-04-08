from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import aqi_model


def main() -> None:
    """Demonstrate the normal order for using the helpers in aqi_model.py."""

    print("1. Load AQI data")
    df = aqi_model.load_aqi_data()
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("\n2. Create model features")
    featured = aqi_model.create_features(df)
    print(f"Feature rows: {len(featured)}")
    print(f"Feature columns: {aqi_model.FEATURE_COLUMNS}")

    print("\n3. Build an untrained champion model")
    untrained_model = aqi_model.build_champion_model()
    print(untrained_model)

    print("\n4. Train the champion model")
    train_result = aqi_model.train_champion(df)
    print(json.dumps(train_result.metrics, indent=2))
    print(f"Training rows: {train_result.training_size}")
    print(f"Test rows: {train_result.test_size}")

    print("\n5. Save model bundle and metadata")
    aqi_model.save_model_bundle(train_result)
    print(f"Saved: {aqi_model.MODEL_FILE}")
    print(f"Saved: {aqi_model.METADATA_FILE}")

    print("\n6. Load the saved model bundle")
    bundle = aqi_model.load_model_bundle()
    model = bundle["model"]
    print(f"Feature version: {bundle['feature_version']}")
    print(f"Loaded feature count: {len(bundle['feature_columns'])}")

    print("\n7. Create the next day's feature row")
    next_date = df["Date"].max() + pd.Timedelta(days=1)
    # The model cannot predict from just a date. It needs the same 19 features
    # used during training, so we build those features from the AQI history.
    next_features = aqi_model.make_next_feature_row(df, next_date)
    print(f"Next date: {next_date.date()}")
    print(next_features.to_string(index=False))

    print("\n8. Predict the next day's AQI")
    next_prediction = model.predict(next_features)[0]
    print(f"Forecast for {next_date.date()}: {next_prediction:.2f}")

    print("\n9. Forecast a chosen future date")
    target_date = pd.Timestamp("2025-04-15")
    # Forecast recursively up to the target date, then select that row.
    days_needed = (target_date - df["Date"].max()).days
    forecast_to_target = aqi_model.forecast_next_days(model, df, days=days_needed)
    target_row = forecast_to_target[forecast_to_target["Date"] == target_date.date().isoformat()]
    print(target_row.to_string(index=False))

    print("\n10. Create the standard 30-day forecast file")
    forecast_30_days = aqi_model.forecast_next_days(model, df, days=30)
    forecast_30_days.to_csv(aqi_model.FORECAST_FILE, index=False)
    print(f"Saved: {aqi_model.FORECAST_FILE}")
    print(forecast_30_days.head().to_string(index=False))

    print("\n11. Full train-and-forecast workflow")
    print("This calls load, train, save, and forecast in one step.")
    # This is the shortest command to refresh the saved model and forecast CSV.
    aqi_model.train_and_forecast(days=30)

    print("\n12. Optional: inspect the saved metadata file")
    metadata = json.loads(Path(aqi_model.METADATA_FILE).read_text(encoding="utf-8"))
    print(f"Readiness: {metadata['readiness']}")


if __name__ == "__main__":
    main()
