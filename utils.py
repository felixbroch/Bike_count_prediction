import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from vacances_scolaires_france import SchoolHolidayDates
from datetime import date
from jours_feries_france import JoursFeries
from pathlib import Path

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge


problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
columnns_to_drop = ['counter_id', 'site_id', 'site_name', 'counter_technical_id',
                    'coordinates',
                    'Station Number', 'Measurement Period Duration',
                    'date', 'Date and Time', 'counter_installation_date']



def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

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


def _merge_external_data(X):
    file_path = "data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X




def _process_datetime_features(df):
    # Ensure "Date and Time" is in datetime format
    df["Date and Time"] = pd.to_datetime(df["Date and Time"], errors="coerce")

    # Drop rows with invalid datetime entries
    df = df.dropna(subset=["Date and Time"])

    # Extract date and time features
    df["year"] = df["Date and Time"].dt.year
    df["month"] = df["Date and Time"].dt.month
    df["weekday"] = df["Date and Time"].dt.dayofweek
    df["day"] = df["Date and Time"].dt.day
    df["hour"] = df["Date and Time"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Handle school and public holidays
    unique_dates = df["Date and Time"].dt.date.unique()
    d = SchoolHolidayDates()
    f = JoursFeries()

    try:
        dict_school_holidays = {date: d.is_holiday_for_zone(date, "C") for date in unique_dates}
        df["is_school_holiday"] = df["Date and Time"].dt.date.map(dict_school_holidays).fillna(0).astype(int)
    except Exception as e:
        print(f"Error with school holidays mapping: {e}")
        df["is_school_holiday"] = 0

    try:
        dict_public_holidays = {date: f.is_bank_holiday(date, zone="Métropole") for date in unique_dates}
        df["is_public_holiday"] = df["Date and Time"].dt.date.map(dict_public_holidays).fillna(0).astype(int)
    except Exception as e:
        print(f"Error with public holidays mapping: {e}")
        df["is_public_holiday"] = 0

    return df




def _get_estimator():
    date_encoder = FunctionTransformer(_process_datetime_features, validate=False)
    external_data_encoder = FunctionTransformer(_merge_external_data, validate=False)
    date_cols = ["year", "month", "weekday", "day", "hour", "is_weekend", "is_school_holiday", "is_public_holiday"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(
        external_data_encoder,
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe

















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
