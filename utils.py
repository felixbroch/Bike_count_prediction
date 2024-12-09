import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from vacances_scolaires_france import SchoolHolidayDates
from datetime import date
from jours_feries_france import JoursFeries
from pathlib import Path

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
columns_to_drop = [
    "counter_id",
    "site_id",
    "site_name",
    "coordinates",
    "counter_technical_id",
    "Station Number",
]


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_test_data(path="data/final_test.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    return data


def _merge_external_data(X, is_train=True, columns_to_drop_store=None):
    file_path = "data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext = _column_rename(df_ext)

    # Ensure both X['date'] and df_ext['date'] are in datetime64[ns] format
    X["date"] = pd.to_datetime(X["date"], errors="coerce").astype("datetime64[ns]")
    df_ext["date"] = pd.to_datetime(df_ext["date"], errors="coerce").astype(
        "datetime64[ns]"
    )

    # Drop rows with invalid dates
    X = X.dropna(subset=["date"])
    df_ext = df_ext.dropna(subset=["date"])

    # Sort DataFrames for merge_asof
    X = X.sort_values("date").reset_index(drop=True)
    df_ext = df_ext.sort_values("date").reset_index(drop=True)

    # Add original index to preserve order
    X["orig_index"] = X.index

    # Perform merge_asof with handling for NaNs
    X = pd.merge_asof(X, df_ext, on="date", direction="nearest")

    # Restore original order and drop temp column
    X = X.sort_values("orig_index").drop(columns=["orig_index"])

    # Remove columns with more than 70% of NaN values
    threshold = 0.4
    X = X.loc[:, X.isna().mean() < threshold]

    if is_train:
        # Initialize columns_to_drop_store if it's None
        if columns_to_drop_store is None:
            columns_to_drop_store = []

        # Drop columns and store dropped columns
        dropped_cols = [col for col in columns_to_drop if col in X.columns]
        X = X.drop(columns=dropped_cols, errors="ignore")
        columns_to_drop_store.extend(dropped_cols)  # Store dropped columns
    else:
        # Align columns with training data by dropping the stored columns
        if columns_to_drop_store:
            X = X.drop(columns=columns_to_drop_store, errors="ignore")

    # Forward fill (ffill) or backward fill (bfill) the remaining columns
    X = X.ffill().bfill()

    return X


def _process_datetime_features(df):
    # Ensure "date" is in datetime format
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid datetime entries
    df = df.dropna(subset=["date"])

    # Extract date and time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Handle school and public holidays
    unique_dates = df["date"].dt.date.unique()
    d = SchoolHolidayDates()
    f = JoursFeries()

    try:
        dict_school_holidays = {
            date: d.is_holiday_for_zone(date, "C") for date in unique_dates
        }
        df["is_school_holiday"] = (
            df["date"].dt.date.map(dict_school_holidays).fillna(0).astype(int)
        )
    except Exception as e:
        print(f"Error with school holidays mapping: {e}")
        df["is_school_holiday"] = 0

    try:
        dict_public_holidays = {
            date: f.is_bank_holiday(date, zone="Métropole") for date in unique_dates
        }
        df["is_public_holiday"] = (
            df["date"].dt.date.map(dict_public_holidays).fillna(0).astype(int)
        )
    except Exception as e:
        print(f"Error with public holidays mapping: {e}")
        df["is_public_holiday"] = 0

    return df


dropped_columns_store = []  # Shared list for storing dropped columns


def _get_function_transformers(is_train=True):
    """Create the pipeline for function transformers."""
    return Pipeline(
        [
            (
                "external_data",
                FunctionTransformer(
                    lambda X: _merge_external_data(
                        X,
                        is_train=is_train,
                        columns_to_drop_store=dropped_columns_store,
                    ),
                    validate=False,
                ),
            ),
            (
                "date_features",
                FunctionTransformer(_process_datetime_features, validate=False),
            ),
        ]
    )


def _get_column_transformers():
    """Create the column transformer for preprocessing."""
    date_cols = [
        "year",
        "month",
        "weekday",
        "day",
        "hour",
        "is_weekend",
        "is_school_holiday",
        "is_public_holiday",
    ]
    categorical_cols = ["counter_name"]
    numerical_cols = [
        "latitude",
        "longitude",
        "Sea Level Pressure (hPa)",
        "Pressure Tendency (hPa/3h)",
        "Pressure Tendency Code",
        "Wind Direction (°)",
        "Wind Speed (m/s)",
        "Air Temperature (°C)",
        "Dew Point Temperature (°C)",
        "Relative Humidity (%)",
        "Visibility (m)",
        "Present Weather Code",
        "Past Weather Code 1",
        "Past Weather Code 2",
        "Total Cloud Cover (oktas)",
        "Cloud Base Height (m)",
        "Lowest Cloud Base Height (m)",
        "Low Cloud Type",
        "Station Level Pressure (hPa)",
        "24h Pressure Tendency (hPa)",
        "10min Max Wind Gust (m/s)",
        "Max Wind Gust (m/s)",
        "Measurement Period Duration",
        "Ground State",
        "Snow Height (cm)",
        "New Snow Depth (cm)",
        "New Snowfall Duration (hours)",
        "Rainfall (1h, mm)",
        "Rainfall (3h, mm)",
        "Rainfall (6h, mm)",
        "Rainfall (12h, mm)",
        "Rainfall (24h, mm)",
        "Layer 1 Cloud Cover (oktas)",
        "Layer 1 Cloud Type",
        "Layer 1 Cloud Base Height (m)",
    ]

    # Use an instance of StandardScaler
    return ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )


def _get_model():
    """Create the XGBoost model with hyperparameters."""
    return XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.8189577147756041,
        learning_rate=0.11986932069472364,
        max_depth=8,
        n_estimators=374,
        subsample=0.6854480891474511
    )


def _get_estimator():
    """Combine the function transformers, column transformers, and model into a single pipeline."""
    function_transformers = _get_function_transformers()
    column_transformers = _get_column_transformers()
    model = _get_model()

    # Combine all components into a final pipeline
    full_pipeline = Pipeline(
        [
            ("function_transformers", function_transformers),
            ("column_transformers", column_transformers),
            ("model", model),
        ]
    )

    return full_pipeline


def _column_rename(X):
    column_name_mapping = {
        "numer_sta": "Station Number",
        "pmer": "Sea Level Pressure (hPa)",
        "tend": "Pressure Tendency (hPa/3h)",
        "cod_tend": "Pressure Tendency Code",
        "dd": "Wind Direction (°)",
        "ff": "Wind Speed (m/s)",
        "t": "Air Temperature (°C)",
        "td": "Dew Point Temperature (°C)",
        "u": "Relative Humidity (%)",
        "vv": "Visibility (m)",
        "ww": "Present Weather Code",
        "w1": "Past Weather Code 1",
        "w2": "Past Weather Code 2",
        "n": "Total Cloud Cover (oktas)",
        "nbas": "Cloud Base Height (m)",
        "hbas": "Lowest Cloud Base Height (m)",
        "cl": "Low Cloud Type",
        "cm": "Medium Cloud Type",
        "ch": "High Cloud Type",
        "pres": "Station Level Pressure (hPa)",
        "niv_bar": "Barometer Altitude (m)",
        "geop": "Geopotential Height (m)",
        "tend24": "24h Pressure Tendency (hPa)",
        "tn12": "12h Minimum Temperature (°C)",
        "tn24": "24h Minimum Temperature (°C)",
        "tx12": "12h Maximum Temperature (°C)",
        "tx24": "24h Maximum Temperature (°C)",
        "tminsol": "Minimum Soil Temperature (°C)",
        "sw": "Sunshine Duration (hours)",
        "tw": "Wet Bulb Temperature (°C)",
        "raf10": "10min Max Wind Gust (m/s)",
        "rafper": "Max Wind Gust (m/s)",
        "per": "Measurement Period Duration",
        "etat_sol": "Ground State",
        "ht_neige": "Snow Height (cm)",
        "ssfrai": "New Snow Depth (cm)",
        "perssfrai": "New Snowfall Duration (hours)",
        "rr1": "Rainfall (1h, mm)",
        "rr3": "Rainfall (3h, mm)",
        "rr6": "Rainfall (6h, mm)",
        "rr12": "Rainfall (12h, mm)",
        "rr24": "Rainfall (24h, mm)",
        "phenspe1": "Special Weather Phenomenon 1",
        "phenspe2": "Special Weather Phenomenon 2",
        "phenspe3": "Special Weather Phenomenon 3",
        "phenspe4": "Special Weather Phenomenon 4",
        "nnuage1": "Layer 1 Cloud Cover (oktas)",
        "ctype1": "Layer 1 Cloud Type",
        "hnuage1": "Layer 1 Cloud Base Height (m)",
        "nnuage2": "Layer 2 Cloud Cover (oktas)",
        "ctype2": "Layer 2 Cloud Type",
        "hnuage2": "Layer 2 Cloud Base Height (m)",
        "nnuage3": "Layer 3 Cloud Cover (oktas)",
        "ctype3": "Layer 3 Cloud Type",
        "hnuage3": "Layer 3 Cloud Base Height (m)",
        "nnuage4": "Layer 4 Cloud Cover (oktas)",
        "ctype4": "Layer 4 Cloud Type",
        "hnuage4": "Layer 4 Cloud Base Height (m)",
    }
    external_conditions = X.rename(columns=column_name_mapping)
    return external_conditions


# def get_cv(X, y, random_state=0):
#     cv = TimeSeriesSplit(n_splits=8)
#     rng = np.random.RandomState(random_state)

#     for train_idx, test_idx in cv.split(X):
#         # Take a random sampling on test_idx so it's that samples are not consecutives.
#         yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


# def find_closest(value, non_nan_values):
#     """Find the closest value to the given value from a set of non-NaN values."""
#     if pd.isna(value):
#         # Ensure non_nan_values is not empty to avoid ValueError
#         if non_nan_values.empty:
#             return value  # Keep NaN if no values to compare
#         closest_value = non_nan_values.iloc[(non_nan_values - value).abs().argmin()]
#         return closest_value
#     return value


# # Function to fill NaN values with the closest value for all numeric columns
# def fill_closest_value_all_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Fill NaN values with the closest value for all numeric columns in the DataFrame."""
#     filled_df = df.copy()

#     for column in filled_df.columns:
#         if filled_df[column].dtype.kind in 'biufc':  # Check if column is numeric
#             non_nan_values = filled_df[column].dropna()

#             # Apply `find_closest` with `non_nan_values`
#             filled_df[column] = filled_df[column].apply(lambda x: find_closest(x, non_nan_values))

#     return filled_df


# def _fill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     """Fill missing rows and values in the DataFrame."""
#     # Ensure 'date' is in datetime format
#     df['date'] = pd.to_datetime(df['date'])

#     # Step 4: Create a complete date range from the minimum to the maximum date in the DataFrame
#     date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='H')

#     # Step 5: Create a DataFrame from the date_range
#     date_range_df = pd.DataFrame(date_range, columns=['date'])

#     # Step 6: Merge the date_range DataFrame with the original DataFrame on the 'date' column
#     full_external_conditions = pd.merge(date_range_df, df, on='date', how='left')

#     # Remove columns that are completely empty
#     full_external_conditions = full_external_conditions.dropna(axis=1, how='all')

#     # Fill missing values using the custom function
#     filled_external_conditions = fill_closest_value_all_columns(full_external_conditions)

#     return filled_external_conditions


# def _merge_data_with_external_data(external_conditions, data, test_data):
#     # Ensure datetime compatibility
#     external_conditions["Date and Time"] = pd.to_datetime(external_conditions["Date and Time"])
#     data["date"] = pd.to_datetime(data["date"])
#     test_data["date"] = pd.to_datetime(test_data["date"])

#     # Merge the dataframes
#     merged_data = pd.merge(data, external_conditions, left_on="date", right_on="Date and Time", how="left")
#     test_merged_data = pd.merge(test_data, external_conditions, left_on="date", right_on="Date and Time", how="left")

#     return merged_data, test_merged_data
