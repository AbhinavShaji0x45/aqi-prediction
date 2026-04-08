from __future__ import annotations

from pathlib import Path

import pandas as pd


YEARS = [2021, 2022, 2023, 2024, 2025]


def clean_year(year: int, base_dir: Path = Path(".")) -> pd.DataFrame:
    """Convert one yearly Excel sheet from day/month layout into Date/AQI rows."""

    source = base_dir / f"AQI_daily_city_level_chennai_{year}_chennai_{year}.xlsx"
    raw = pd.read_excel(source)

    # The source files contain non-day rows; keep only rows where Day is numeric.
    raw = raw[pd.to_numeric(raw["Day"], errors="coerce").notnull()].copy()
    raw["Day"] = raw["Day"].astype(int)

    # Convert month columns into one row per calendar date.
    cleaned = raw.melt(id_vars=["Day"], var_name="Month", value_name="AQI")
    cleaned["AQI"] = pd.to_numeric(cleaned["AQI"], errors="coerce")
    cleaned = cleaned.dropna()
    cleaned["Date"] = pd.to_datetime(
        cleaned["Day"].astype(str) + "-" + cleaned["Month"] + "-" + str(year),
        format="%d-%B-%Y",
    )

    return cleaned[["Date", "AQI"]].sort_values("Date").reset_index(drop=True)


def prepare_all_years(base_dir: Path = Path(".")) -> pd.DataFrame:
    """Clean all source spreadsheets and rebuild the combined AQI CSV."""

    yearly_frames = []
    for year in YEARS:
        cleaned = clean_year(year, base_dir)
        cleaned.to_csv(base_dir / f"clean_{year}.csv", index=False)
        yearly_frames.append(cleaned)

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined.to_csv(base_dir / "clean_aqi.csv", index=False)
    return combined


if __name__ == "__main__":
    combined_df = prepare_all_years()
    print(f"Prepared {len(combined_df)} rows in clean_aqi.csv")
