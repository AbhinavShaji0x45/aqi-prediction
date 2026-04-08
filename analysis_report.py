from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

import aqi_model


REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


def ensure_report_dirs() -> None:
    """Create the output folders used by the report."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(name: str) -> str:
    """Save the active Matplotlib figure and return its report-relative path."""

    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path.as_posix()


def split_features(featured: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features chronologically so future rows are not used for training."""

    split = int(len(featured) * 0.8)
    x = featured[aqi_model.FEATURE_COLUMNS]
    y = featured["AQI"]
    return x.iloc[:split], x.iloc[split:], y.iloc[:split], y.iloc[split:]


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Train candidate models and collect metrics, predictions, and fitted estimators."""

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        ),
        # Extra boosting models for report comparison. They are not part of
        # the saved champion model unless their metrics justify changing it.
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "Champion Voting Ensemble": aqi_model.build_champion_model(),
    }

    rows = []
    predictions = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        fitted_models[name] = model
        rows.append(
            {
                "model": name,
                "r2_score": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("r2_score", ascending=False)
    return metrics, predictions, fitted_models


def walk_forward_scores(featured: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the champion model across multiple forward-moving time splits."""

    x = featured[aqi_model.FEATURE_COLUMNS]
    y = featured["AQI"]
    splitter = TimeSeriesSplit(n_splits=5)
    rows = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        model = aqi_model.build_champion_model()
        model.fit(x.iloc[train_index], y.iloc[train_index])
        y_pred = model.predict(x.iloc[test_index])
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "r2_score": r2_score(y.iloc[test_index], y_pred),
                "rmse": np.sqrt(mean_squared_error(y.iloc[test_index], y_pred)),
                "mae": mean_absolute_error(y.iloc[test_index], y_pred),
            }
        )

    return pd.DataFrame(rows)


def plot_aqi_trend(df: pd.DataFrame) -> str:
    """Plot daily AQI with 7-day and 30-day moving averages."""

    plot_df = df.copy()
    plot_df["ma7"] = plot_df["AQI"].rolling(7).mean()
    plot_df["ma30"] = plot_df["AQI"].rolling(30).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["Date"], plot_df["AQI"], label="Daily AQI", alpha=0.45, linewidth=1)
    plt.plot(plot_df["Date"], plot_df["ma7"], label="7-day average", linewidth=1.5)
    plt.plot(plot_df["Date"], plot_df["ma30"], label="30-day average", linewidth=2)
    plt.title("Chennai AQI Over Time")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(alpha=0.25)
    return save_current_figure("01_aqi_trend.png")


def plot_monthly_distribution(df: pd.DataFrame) -> str:
    """Show how AQI values vary across calendar months."""

    plot_df = df.copy()
    plot_df["month"] = plot_df["Date"].dt.month

    month_data = [plot_df.loc[plot_df["month"] == month, "AQI"] for month in range(1, 13)]
    plt.figure(figsize=(12, 5))
    plt.boxplot(
        month_data,
        tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    plt.title("AQI Distribution By Month")
    plt.xlabel("Month")
    plt.ylabel("AQI")
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure("02_monthly_distribution.png")


def plot_yearly_average(df: pd.DataFrame) -> str:
    """Show the average AQI for each year in the dataset."""

    yearly = df.assign(year=df["Date"].dt.year).groupby("year", as_index=False)["AQI"].mean()

    plt.figure(figsize=(9, 5))
    bars = plt.bar(yearly["year"].astype(str), yearly["AQI"])
    plt.title("Average AQI By Year")
    plt.xlabel("Year")
    plt.ylabel("Average AQI")
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha="center", va="bottom")
    return save_current_figure("03_yearly_average.png")


def classify_aqi(value: float) -> str:
    """Map an AQI value to the standard Indian AQI category."""

    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def plot_aqi_categories(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Count days in each AQI category and save a bar chart."""

    category_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    categories = df["AQI"].apply(classify_aqi)
    counts = categories.value_counts().reindex(category_order, fill_value=0).reset_index()
    counts.columns = ["category", "days"]

    plt.figure(figsize=(10, 5))
    plt.bar(counts["category"], counts["days"])
    plt.title("AQI Category Counts")
    plt.xlabel("AQI Category")
    plt.ylabel("Number of Days")
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure("04_aqi_category_counts.png"), counts


def plot_autocorrelation(df: pd.DataFrame, max_lag: int = 30) -> str:
    """Plot how strongly AQI relates to previous days' AQI values."""

    correlations = [df["AQI"].autocorr(lag=lag) for lag in range(1, max_lag + 1)]

    plt.figure(figsize=(11, 5))
    plt.bar(range(1, max_lag + 1), correlations)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("AQI Autocorrelation By Lag")
    plt.xlabel("Lag in days")
    plt.ylabel("Correlation")
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure("05_autocorrelation.png")


def plot_model_metric_comparison(metrics: pd.DataFrame) -> str:
    """Compare candidate models using R2, RMSE, and MAE."""

    ordered = metrics.sort_values("r2_score", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].barh(ordered["model"], ordered["r2_score"])
    axes[0].set_title("R2 Score")
    axes[0].set_xlabel("Higher is better")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(ordered["model"], ordered["rmse"])
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Lower is better")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(ordered["model"], ordered["mae"])
    axes[2].set_title("MAE")
    axes[2].set_xlabel("Lower is better")
    axes[2].grid(axis="x", alpha=0.25)

    return save_current_figure("06_model_metric_comparison.png")


def plot_model_predictions(test_dates: pd.Series, y_test: pd.Series, predictions: dict[str, np.ndarray]) -> str:
    """Plot actual AQI and champion predictions over the test period."""

    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_test.values, label="Actual", linewidth=2)
    plt.plot(test_dates, predictions["Champion Voting Ensemble"], label="Champion prediction", linestyle="--")
    plt.title("Actual vs Predicted AQI On Test Set")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(alpha=0.25)
    return save_current_figure("07_actual_vs_predicted.png")


def plot_actual_vs_predicted_scatter(y_test: pd.Series, champion_pred: np.ndarray) -> str:
    """Show whether predictions track the actual values along the ideal diagonal."""

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, champion_pred, alpha=0.65)
    min_value = min(y_test.min(), champion_pred.min())
    max_value = max(y_test.max(), champion_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1)
    plt.title("Actual vs Predicted AQI Scatter")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.grid(alpha=0.25)
    return save_current_figure("08_actual_vs_predicted_scatter.png")


def plot_prediction_error_over_time(test_dates: pd.Series, y_test: pd.Series, champion_pred: np.ndarray) -> str:
    """Plot forecast errors over time to spot drift or bad periods."""

    errors = y_test.values - champion_pred
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, errors, linewidth=1.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Prediction Error Over Time")
    plt.xlabel("Date")
    plt.ylabel("Actual AQI - Predicted AQI")
    plt.grid(alpha=0.25)
    return save_current_figure("09_prediction_error_over_time.png")


def plot_residuals(y_test: pd.Series, champion_pred: np.ndarray) -> str:
    """Create residual diagnostic plots for the champion model."""

    residuals = y_test.values - champion_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(champion_pred, residuals, alpha=0.65)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals vs Predicted AQI")
    axes[0].set_xlabel("Predicted AQI")
    axes[0].set_ylabel("Residual")
    axes[0].grid(alpha=0.25)

    axes[1].hist(residuals, bins=25, edgecolor="black")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)
    return save_current_figure("10_residuals.png")


def plot_feature_importance(fitted_models: dict[str, object]) -> tuple[str, pd.DataFrame]:
    """Use the random-forest component to estimate feature importance."""

    champion = fitted_models["Champion Voting Ensemble"]
    random_forest = champion.named_estimators_["rf"]
    importance = pd.DataFrame(
        {
            "feature": aqi_model.FEATURE_COLUMNS,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance.head(12).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top Feature Importances From Champion Random Forest Component")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.25)
    return save_current_figure("11_feature_importance.png"), importance


def plot_walk_forward_metrics(walk_forward: pd.DataFrame) -> str:
    """Plot validation metrics across the walk-forward folds."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(walk_forward["fold"], walk_forward["r2_score"], marker="o")
    axes[0].set_title("Walk-Forward R2")
    axes[0].set_xlabel("Fold")
    axes[0].grid(alpha=0.25)

    axes[1].plot(walk_forward["fold"], walk_forward["rmse"], marker="o")
    axes[1].set_title("Walk-Forward RMSE")
    axes[1].set_xlabel("Fold")
    axes[1].grid(alpha=0.25)

    axes[2].plot(walk_forward["fold"], walk_forward["mae"], marker="o")
    axes[2].set_title("Walk-Forward MAE")
    axes[2].set_xlabel("Fold")
    axes[2].grid(alpha=0.25)

    return save_current_figure("12_walk_forward_metrics.png")


def plot_forecast(df: pd.DataFrame, fitted_models: dict[str, object]) -> tuple[str, pd.DataFrame]:
    """Generate and plot the next 30 forecasted AQI values."""

    forecast = aqi_model.forecast_next_days(fitted_models["Champion Voting Ensemble"], df, days=30)
    recent = df.tail(90)
    forecast_dates = pd.to_datetime(forecast["Date"])

    plt.figure(figsize=(12, 5))
    plt.plot(recent["Date"], recent["AQI"], label="Recent actual AQI", linewidth=2)
    plt.plot(forecast_dates, forecast["AQI_Forecast"], label="30-day forecast", linestyle="--", marker="o", markersize=3)
    plt.axvline(df["Date"].max(), color="black", linestyle=":", label="Forecast start")
    plt.title("30-Day AQI Forecast")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(alpha=0.25)
    return save_current_figure("13_forecast.png"), forecast


def write_markdown_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    walk_forward: pd.DataFrame,
    category_counts: pd.DataFrame,
    forecast: pd.DataFrame,
    figures: list[str],
) -> None:
    """Write the markdown summary that points to all generated report artifacts."""

    best = metrics.iloc[0]
    average_walk_forward = walk_forward[["r2_score", "rmse", "mae"]].mean()
    forecast_summary = forecast["AQI_Forecast"].agg(["mean", "min", "max", "std"])
    category_table = "\n".join(
        ["| category | days |", "| --- | ---: |"]
        + [f"| {row.category} | {row.days} |" for row in category_counts.itertuples(index=False)]
    )

    report = f"""# Chennai AQI Analysis Report

## Dataset

- Rows: {len(df)}
- Date range: {df["Date"].min().date()} to {df["Date"].max().date()}
- Mean AQI: {df["AQI"].mean():.2f}
- Minimum AQI: {df["AQI"].min():.2f}
- Maximum AQI: {df["AQI"].max():.2f}

AQI category counts:

{category_table}

## Model Results

Best model on the chronological 80/20 test split: **{best["model"]}**

- R2 score: {best["r2_score"]:.4f}
- RMSE: {best["rmse"]:.4f}
- MAE: {best["mae"]:.4f}

Average walk-forward validation for the champion model:

- R2 score: {average_walk_forward["r2_score"]:.4f}
- RMSE: {average_walk_forward["rmse"]:.4f}
- MAE: {average_walk_forward["mae"]:.4f}

## Forecast Summary

- Forecast period: {forecast["Date"].iloc[0]} to {forecast["Date"].iloc[-1]}
- Mean forecast AQI: {forecast_summary["mean"]:.2f}
- Minimum forecast AQI: {forecast_summary["min"]:.2f}
- Maximum forecast AQI: {forecast_summary["max"]:.2f}
- Forecast standard deviation: {forecast_summary["std"]:.2f}

## Generated Figures

{chr(10).join(f"- `{figure}`" for figure in figures)}
"""
    (REPORT_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Build all report tables, graphs, and markdown output."""

    ensure_report_dirs()

    df = aqi_model.load_aqi_data()
    featured = aqi_model.create_features(df)
    x_train, x_test, y_train, y_test = split_features(featured)
    test_dates = featured["Date"].iloc[-len(y_test) :]

    metrics, predictions, fitted_models = evaluate_models(x_train, x_test, y_train, y_test)
    walk_forward = walk_forward_scores(featured)

    figures = [
        plot_aqi_trend(df),
        plot_monthly_distribution(df),
        plot_yearly_average(df),
        plot_model_predictions(test_dates, y_test, predictions),
        plot_model_metric_comparison(metrics),
        plot_actual_vs_predicted_scatter(y_test, predictions["Champion Voting Ensemble"]),
        plot_prediction_error_over_time(test_dates, y_test, predictions["Champion Voting Ensemble"]),
        plot_residuals(y_test, predictions["Champion Voting Ensemble"]),
    ]
    category_figure, category_counts = plot_aqi_categories(df)
    figures.append(category_figure)
    figures.append(plot_autocorrelation(df))
    figures.append(plot_walk_forward_metrics(walk_forward))
    feature_figure, feature_importance = plot_feature_importance(fitted_models)
    forecast_figure, forecast = plot_forecast(df, fitted_models)
    figures.extend([feature_figure, forecast_figure])

    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    walk_forward.to_csv(REPORT_DIR / "walk_forward_metrics.csv", index=False)
    category_counts.to_csv(REPORT_DIR / "aqi_category_counts.csv", index=False)
    feature_importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    forecast.to_csv(REPORT_DIR / "report_forecast_30days.csv", index=False)
    write_markdown_report(df, metrics, walk_forward, category_counts, forecast, sorted(figures))

    print(f"Saved report files in {REPORT_DIR}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
