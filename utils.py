import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def _merge_external_data(X):
    file_path = "data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date", direction="nearest"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def _column_rename(X):
    column_name_mapping = {
        'numer_sta': 'Station Number',
        'date': 'Date and Time',
        'pmer': 'Sea Level Pressure (hPa)',
        'tend': 'Pressure Tendency (hPa/3h)',
        'cod_tend': 'Pressure Tendency Code',
        'dd': 'Wind Direction (°)',
        'ff': 'Wind Speed (m/s)',
        't': 'Air Temperature (°C)',
        'td': 'Dew Point Temperature (°C)',
        'u': 'Relative Humidity (%)',
        'vv': 'Visibility (m)',
        'ww': 'Present Weather Code',
        'w1': 'Past Weather Code 1',
        'w2': 'Past Weather Code 2',
        'n': 'Total Cloud Cover (oktas)',
        'nbas': 'Cloud Base Height (m)',
        'hbas': 'Lowest Cloud Base Height (m)',
        'cl': 'Low Cloud Type',
        'cm': 'Medium Cloud Type',
        'ch': 'High Cloud Type',
        'pres': 'Station Level Pressure (hPa)',
        'niv_bar': 'Barometer Altitude (m)',
        'geop': 'Geopotential Height (m)',
        'tend24': '24h Pressure Tendency (hPa)',
        'tn12': '12h Minimum Temperature (°C)',
        'tn24': '24h Minimum Temperature (°C)',
        'tx12': '12h Maximum Temperature (°C)',
        'tx24': '24h Maximum Temperature (°C)',
        'tminsol': 'Minimum Soil Temperature (°C)',
        'sw': 'Sunshine Duration (hours)',
        'tw': 'Wet Bulb Temperature (°C)',
        'raf10': '10min Max Wind Gust (m/s)',
        'rafper': 'Max Wind Gust (m/s)',
        'per': 'Measurement Period Duration',
        'etat_sol': 'Ground State',
        'ht_neige': 'Snow Height (cm)',
        'ssfrai': 'New Snow Depth (cm)',
        'perssfrai': 'New Snowfall Duration (hours)',
        'rr1': 'Rainfall (1h, mm)',
        'rr3': 'Rainfall (3h, mm)',
        'rr6': 'Rainfall (6h, mm)',
        'rr12': 'Rainfall (12h, mm)',
        'rr24': 'Rainfall (24h, mm)',
        'phenspe1': 'Special Weather Phenomenon 1',
        'phenspe2': 'Special Weather Phenomenon 2',
        'phenspe3': 'Special Weather Phenomenon 3',
        'phenspe4': 'Special Weather Phenomenon 4',
        'nnuage1': 'Layer 1 Cloud Cover (oktas)',
        'ctype1': 'Layer 1 Cloud Type',
        'hnuage1': 'Layer 1 Cloud Base Height (m)',
        'nnuage2': 'Layer 2 Cloud Cover (oktas)',
        'ctype2': 'Layer 2 Cloud Type',
        'hnuage2': 'Layer 2 Cloud Base Height (m)',
        'nnuage3': 'Layer 3 Cloud Cover (oktas)',
        'ctype3': 'Layer 3 Cloud Type',
        'hnuage3': 'Layer 3 Cloud Base Height (m)',
        'nnuage4': 'Layer 4 Cloud Cover (oktas)',
        'ctype4': 'Layer 4 Cloud Type',
        'hnuage4': 'Layer 4 Cloud Base Height (m)',
    }
    external_conditions = X.rename(columns=column_name_mapping)
    return external_conditions


def _process_datetime_features(df):
    # Ensure "Date and Time" is in datetime format
    df["Date and Time"] = pd.to_datetime(df["Date and Time"], errors="coerce")

    # Check for missing or invalid datetime entries
    if df["Date and Time"].isnull().any():
        print("Warning: Missing or invalid datetime entries found.")
        # Handle missing values if needed
        df = df.dropna(subset=["Date and Time"])

    # Extract date and time features
    df["measurement_date"] = df["Date and Time"].dt.date
    df["measurement_year"] = df["Date and Time"].dt.year
    df["measurement_month"] = df["Date and Time"].dt.month
    df["measurement_day_of_week"] = df["Date and Time"].dt.dayofweek
    df["measurement_day"] = df["Date and Time"].dt.day
    df["measurement_hour"] = df["Date and Time"].dt.hour

    # Determine if the day is a weekend
    df["measurement_is_weekend"] = np.where(
        df["measurement_day_of_week"] >= 5, 1, 0
    )

    # Handle school holidays
    unique_dates = df["measurement_date"].unique()

    # Example holiday mapping function
    d = SchoolHolidayDates()
    try:
        dict_school_holidays = {date: d.is_holiday_for_zone(date, "C") for date in unique_dates}
        df["is_school_holiday"] = df["measurement_date"].map(
            dict_school_holidays
        )
    except Exception as e:
        print(f"Error with school holidays mapping: {e}")
        df["is_school_holiday"] = 0  # Fallback to default value

    # Handle public holidays
    f = JoursFeries()
    try:
        dict_public_holidays = {
            date: f.is_bank_holiday(date, zone="Métropole") for date in unique_dates
        }
        df["is_public_holiday"] = df["measurement_date"].map(
            dict_public_holidays
        )
    except Exception as e:
        print(f"Error with public holidays mapping: {e}")
        df["is_public_holiday"] = 0  # Fallback to default value

    # Extract additional date and time features for the counter
    df["counter_year"] = df["Date and Time"].dt.year
    df["counter_month"] = df["Date and Time"].dt.month
    df["counter_day"] = df["Date and Time"].dt.day
    df["counter_hour"] = df["Date and Time"].dt.hour

    return df

