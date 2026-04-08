from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# This file is the main reusable model code for the project.
# It does not depend on the notebooks. Running this file will:
# 1. read clean_aqi.csv
# 2. create time-series features
# 3. train the final ensemble model
# 4. save the model and metadata
# 5. create a 30-day AQI forecast CSV
DATA_FILE = Path("clean_aqi.csv")
MODEL_FILE = Path("aqi_champion_model.joblib")
METADATA_FILE = Path("model_metadata.json")
FORECAST_FILE = Path("aqi_forecast_30days.csv")
ENSEMBLE_WEIGHT_RATIONALE = (
    "Weights are manually chosen from notebook experimentation, not optimized by a search. "
    "XGBoost, Gradient Boosting, and Random Forest receive the largest weights because they "
    "can model non-linear AQI patterns, while Linear Regression keeps a smaller trend-following "
    "component. LightGBM is evaluated in the report but not included in the saved ensemble "
    "because it performs worse in the current test results. Current report metrics show Linear "
    "Regression is still slightly stronger on the 80/20 test split, so these weights should be "
    "treated as experimental for academic demonstration."
)

# Keep the feature contract in one place so training, saving, and inference
# always use the same column order.
FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "lag5",
    "lag7",
    "ma7",
    "ma14",
    "ma30",
    "std7",
    "std14",
    "day_of_week",
    "month",
    "quarter",
    "day_of_year",
    "min7",
    "max7",
    "range7",
    "ema7",
]


@dataclass(frozen=True)
class TrainResult:
    """Container for the trained model and the values needed for metadata."""

    model: VotingRegressor
    metrics: dict[str, float]
    training_size: int
    test_size: int


def load_aqi_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the cleaned AQI dataset and ensure chronological order."""

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe time-series features from historical AQI values."""

    featured = df.copy()
    featured["Date"] = pd.to_datetime(featured["Date"])
    featured = featured.sort_values("Date").reset_index(drop=True)

    # Lag features expose AQI values from previous days to the model.
    # Example: for 2025-04-01, lag1 means the AQI from 2025-03-31.
    featured["lag1"] = featured["AQI"].shift(1)
    featured["lag2"] = featured["AQI"].shift(2)
    featured["lag3"] = featured["AQI"].shift(3)
    featured["lag4"] = featured["AQI"].shift(4)
    featured["lag5"] = featured["AQI"].shift(5)
    featured["lag7"] = featured["AQI"].shift(7)

    # Rolling statistics are shifted by one day so the current day's AQI is
    # never used to predict itself.
    # This is important because using today's AQI to predict today's AQI would
    # make the model look better in testing but fail in real forecasting.
    past_aqi = featured["AQI"].shift(1)
    featured["ma7"] = past_aqi.rolling(7).mean()
    featured["ma14"] = past_aqi.rolling(14).mean()
    featured["ma30"] = past_aqi.rolling(30).mean()
    featured["std7"] = past_aqi.rolling(7).std()
    featured["std14"] = past_aqi.rolling(14).std()
    featured["min7"] = past_aqi.rolling(7).min()
    featured["max7"] = past_aqi.rolling(7).max()
    featured["range7"] = featured["max7"] - featured["min7"]
    featured["ema7"] = past_aqi.ewm(span=7, adjust=False).mean()

    # Calendar features help the model learn simple weekly/monthly patterns.
    featured["day_of_week"] = featured["Date"].dt.dayofweek
    featured["month"] = featured["Date"].dt.month
    featured["quarter"] = featured["Date"].dt.quarter
    featured["day_of_year"] = featured["Date"].dt.dayofyear

    return featured.dropna().reset_index(drop=True)


def build_champion_model() -> VotingRegressor:
    """Build the final weighted ensemble used by the project."""

    # Weight rationale:
    # - Random Forest: 25%, adds robustness by averaging many decision trees.
    # - Gradient Boosting: 30%, captures non-linear changes.
    # - XGBoost: 30%, another strong boosted-tree model.
    # - Linear Regression: 15%, keeps a simple trend-following baseline.
    #
    # These weights are manually selected from notebook experimentation. They
    # are not the result of an automatic grid search, and the report currently
    # shows Linear Regression performs slightly better on the 80/20 test split.
    return VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)),
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            ("lr", LinearRegression()),
        ],
        weights=[0.25, 0.30, 0.30, 0.15],
    )


def train_champion(df: pd.DataFrame) -> TrainResult:
    """Train the champion model with a chronological 80/20 train/test split."""

    # Create the exact same input columns that will later be used for forecasts.
    featured = create_features(df)
    x = featured[FEATURE_COLUMNS]
    y = featured["AQI"]

    # Do not shuffle time-series data: the test set must stay in the future
    # relative to the training set.
    split = int(len(featured) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = build_champion_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2_score": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    return TrainResult(
        model=model,
        metrics=metrics,
        training_size=len(x_train),
        test_size=len(x_test),
    )


def save_model_bundle(result: TrainResult) -> None:
    """Save the model together with its required feature contract."""

    # Saving the feature list with the model prevents a common mistake:
    # loading the model later but passing columns in the wrong order.
    joblib.dump(
        {
            "model": result.model,
            "feature_columns": FEATURE_COLUMNS,
            "feature_version": "19-feature-lag-rolling-calendar-v1",
        },
        MODEL_FILE,
    )

    # Metadata is human-readable, while the joblib bundle is what code loads.
    metadata = {
        "champion_model": {
            "file": str(MODEL_FILE),
            "type": "VotingRegressor",
            "components": {
                "random_forest": {"weight": 0.25, "n_estimators": 150, "max_depth": 12},
                "gradient_boosting": {
                    "weight": 0.30,
                    "n_estimators": 150,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                },
                "xgboost": {
                    "weight": 0.30,
                    "n_estimators": 150,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                },
                "linear_regression": {"weight": 0.15},
            },
            "metrics": result.metrics,
            "weight_rationale": ENSEMBLE_WEIGHT_RATIONALE,
        },
        "features": FEATURE_COLUMNS,
        "feature_version": "19-feature-lag-rolling-calendar-v1",
        "training_size": result.training_size,
        "test_size": result.test_size,
        "readiness": "academic prototype; not production-ready without external validation and monitoring",
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model_bundle(path: Path = MODEL_FILE) -> dict:
    """Load the saved model bundle created by save_model_bundle."""

    return joblib.load(path)


def make_next_feature_row(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    """Build one future feature row from the latest available history."""

    # Append a blank future row so pandas shift/rolling operations calculate
    # the next day's features from the rows before it.
    # The blank AQI is not used directly; the features for that row come from
    # earlier actual or forecasted AQI values.
    placeholder = pd.DataFrame({"Date": [next_date], "AQI": [np.nan]})
    candidate = pd.concat([history[["Date", "AQI"]], placeholder], ignore_index=True)
    featured = candidate.copy()
    featured["lag1"] = featured["AQI"].shift(1)
    featured["lag2"] = featured["AQI"].shift(2)
    featured["lag3"] = featured["AQI"].shift(3)
    featured["lag4"] = featured["AQI"].shift(4)
    featured["lag5"] = featured["AQI"].shift(5)
    featured["lag7"] = featured["AQI"].shift(7)

    past_aqi = featured["AQI"].shift(1)
    featured["ma7"] = past_aqi.rolling(7).mean()
    featured["ma14"] = past_aqi.rolling(14).mean()
    featured["ma30"] = past_aqi.rolling(30).mean()
    featured["std7"] = past_aqi.rolling(7).std()
    featured["std14"] = past_aqi.rolling(14).std()
    featured["min7"] = past_aqi.rolling(7).min()
    featured["max7"] = past_aqi.rolling(7).max()
    featured["range7"] = featured["max7"] - featured["min7"]
    featured["ema7"] = past_aqi.ewm(span=7, adjust=False).mean()

    featured["Date"] = pd.to_datetime(featured["Date"])
    featured["day_of_week"] = featured["Date"].dt.dayofweek
    featured["month"] = featured["Date"].dt.month
    featured["quarter"] = featured["Date"].dt.quarter
    featured["day_of_year"] = featured["Date"].dt.dayofyear
    return featured[FEATURE_COLUMNS].iloc[[-1]]


def forecast_next_days(model: VotingRegressor, history: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Forecast future AQI values recursively for the requested number of days."""

    # For multi-day forecasting, day 2 depends on the prediction made for day 1,
    # day 3 depends on days 1 and 2, and so on. This is called recursive
    # forecasting.
    forecast_history = history.copy()
    forecast_history["Date"] = pd.to_datetime(forecast_history["Date"])
    forecast_rows = []

    for _ in range(days):
        next_date = forecast_history["Date"].max() + timedelta(days=1)
        next_features = make_next_feature_row(forecast_history, next_date)
        prediction = float(model.predict(next_features)[0])
        forecast_rows.append({"Date": next_date.date().isoformat(), "AQI_Forecast": prediction})
        # Feed each prediction back into history so the following forecast day
        # can use it as lag1/lag2/etc.
        forecast_history = pd.concat(
            [
                forecast_history,
                pd.DataFrame({"Date": [next_date], "AQI": [prediction]}),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(forecast_rows)


def train_and_forecast(days: int = 30) -> None:
    """Run the standard workflow: train, save, and write a forecast CSV."""

    df = load_aqi_data()
    result = train_champion(df)
    save_model_bundle(result)
    forecast = forecast_next_days(result.model, df, days=days)
    forecast.to_csv(FORECAST_FILE, index=False)
    print(f"Saved {MODEL_FILE}")
    print(f"Saved {METADATA_FILE}")
    print(f"Saved {FORECAST_FILE}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    train_and_forecast()
