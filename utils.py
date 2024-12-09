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
                'date', 'counter_installation_date', 'Cloud_Base_Height_(m)',
                'counter_id', 'site_id', 'site_name', 'counter_technical_id',
                'coordinates', 'Station_Level_Pressure_(hPa)', 'Pressure_Tendency_Code',
                'Station_Number', 'Measurement_Period_Duration', 'Measurement_Period_Duration',
                'Layer_1_Cloud_Base_Height_(m)', 'Present_Weather_Code', 'Past_Weather_Code_1',
                'Past_Weather_Code_2', 'Rainfall_(1h,_mm)', 'Rainfall_(6h,_mm)',
                'Rainfall_(24h,_mm)', 'Wind_Direction_(°)', 'Dew_Point_Temperature_(°C)',
                'Lowest_Cloud_Base_Height_(m)', 'Low_Cloud_Type', 'Medium_Cloud_Type',
                'High_Cloud_Type', '10min_Max_Wind_Gust_(m/s)', 'Ground_State',
                'New_Snow_Depth_(cm)', 'New_Snowfall_Duration_(hours)',
                'Layer_1_Cloud_Cover_(oktas)', 'Layer_1_Cloud_Type', 'Layer_2_Cloud_Cover_(oktas)',
                'Layer_2_Cloud_Type', 'Layer_2_Cloud_Base_Height_(m)', 'Max_Wind_Gust_(m/s)',
                '24h_Pressure_Tendency_(hPa)', 'Sea_Level_Pressure_(hPa)', 

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
        "Sea_Level_Pressure_(hPa)",
        "Pressure_Tendency_(hPa/3h)",
        "Pressure_Tendency_Code",
        "Wind_Direction_(°)",
        "Wind_Speed_(m/s)",
        "Air_Temperature_(°C)",
        "Dew_Point_Temperature_(°C)",
        "Relative_Humidity_(%)",
        "Visibility_(m)",
        "Present_Weather_Code",
        "Past_Weather_Code_1",
        "Past_Weather_Code_2",
        "Total_Cloud_Cover_(oktas)",
        "Cloud_Base_Height_(m)",
        "Lowest_Cloud_Base_Height_(m)",
        "Low_Cloud_Type",
        "Station_Level_Pressure_(hPa)",
        "24h_Pressure_Tendency_(hPa)",
        "10min_Max_Wind_Gust_(m/s)",
        "Max_Wind_Gust_(m/s)",
        "Measurement_Period_Duration",
        "Ground_State",
        "Snow_Height_(cm)",
        "New_Snow_Depth_(cm)",
        "New_Snowfall_Duration_(hours)",
        "Rainfall_(1h,_mm)",
        "Rainfall_(3h,_mm)",
        "Rainfall_(6h,_mm)",
        "Rainfall_(12h,_mm)",
        "Rainfall_(24h,_mm)",
        "Layer_1_Cloud_Cover_(oktas)",
        "Layer_1_Cloud_Type",
        "Layer_1_Cloud_Base_Height_(m)",
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
        "numer_sta": "Station_Number",
        "pmer": "Sea_Level_Pressure_(hPa)",
        "tend": "Pressure_Tendency_(hPa/3h)",
        "cod_tend": "Pressure_Tendency_Code",
        "dd": "Wind_Direction_(°)",
        "ff": "Wind_Speed_(m/s)",
        "t": "Air_Temperature_(°C)",
        "td": "Dew_Point_Temperature_(°C)",
        "u": "Relative_Humidity_(%)",
        "vv": "Visibility_(m)",
        "ww": "Present_Weather_Code",
        "w1": "Past_Weather_Code_1",
        "w2": "Past_Weather_Code_2",
        "n": "Total_Cloud_Cover_(oktas)",
        "nbas": "Cloud_Base_Height_(m)",
        "hbas": "Lowest_Cloud_Base_Height_(m)",
        "cl": "Low_Cloud_Type",
        "cm": "Medium_Cloud_Type",
        "ch": "High_Cloud_Type",
        "pres": "Station_Level_Pressure_(hPa)",
        "niv_bar": "Barometer_Altitude_(m)",
        "geop": "Geopotential_Height_(m)",
        "tend24": "24h_Pressure_Tendency_(hPa)",
        "tn12": "12h_Minimum_Temperature_(°C)",
        "tn24": "24h_Minimum_Temperature_(°C)",
        "tx12": "12h_Maximum_Temperature_(°C)",
        "tx24": "24h_Maximum_Temperature_(°C)",
        "tminsol": "Minimum_Soil_Temperature_(°C)",
        "sw": "Sunshine_Duration_(hours)",
        "tw": "Wet_Bulb_Temperature_(°C)",
        "raf10": "10min_Max_Wind_Gust_(m/s)",
        "rafper": "Max_Wind_Gust_(m/s)",
        "per": "Measurement_Period_Duration",
        "etat_sol": "Ground_State",
        "ht_neige": "Snow_Height_(cm)",
        "ssfrai": "New_Snow_Depth_(cm)",
        "perssfrai": "New_Snowfall_Duration_(hours)",
        "rr1": "Rainfall_(1h,_mm)",
        "rr3": "Rainfall_(3h,_mm)",
        "rr6": "Rainfall_(6h,_mm)",
        "rr12": "Rainfall_(12h,_mm)",
        "rr24": "Rainfall_(24h,_mm)",
        "phenspe1": "Special_Weather_Phenomenon_1",
        "phenspe2": "Special_Weather_Phenomenon_2",
        "phenspe3": "Special_Weather_Phenomenon_3",
        "phenspe4": "Special_Weather_Phenomenon_4",
        "nnuage1": "Layer_1_Cloud_Cover_(oktas)",
        "ctype1": "Layer_1_Cloud_Type",
        "hnuage1": "Layer_1_Cloud_Base_Height_(m)",
        "nnuage2": "Layer_2_Cloud_Cover_(oktas)",
        "ctype2": "Layer_2_Cloud_Type",
        "hnuage2": "Layer_2_Cloud_Base_Height_(m)",
        "nnuage3": "Layer_3_Cloud_Cover_(oktas)",
        "ctype3": "Layer_3_Cloud_Type",
        "hnuage3": "Layer_3_Cloud_Base_Height_(m)",
        "nnuage4": "Layer_4_Cloud_Cover_(oktas)",
        "ctype4": "Layer_4_Cloud_Type",
        "hnuage4": "Layer_4_Cloud_Base_Height_(m)",
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
