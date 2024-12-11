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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import joblib




columns_to_drop_personal = [
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

columns_to_drop_boruta1 = [
        'counter_installation_date',
        'Station_Number', 'Sea_Level_Pressure_(hPa)',
        'Pressure_Tendency_(hPa/3h)', 'Pressure_Tendency_Code',
        'Wind_Direction_(°)', 'Wind_Speed_(m/s)',
        'Dew_Point_Temperature_(°C)', 'Relative_Humidity_(%)', 'Visibility_(m)',
        'Present_Weather_Code', 'Past_Weather_Code_1', 'Past_Weather_Code_2',
        'Total_Cloud_Cover_(oktas)', 'Cloud_Base_Height_(m)',
        'Lowest_Cloud_Base_Height_(m)', 'Low_Cloud_Type', 'Medium_Cloud_Type',
        'High_Cloud_Type', 'Station_Level_Pressure_(hPa)',
        '24h_Pressure_Tendency_(hPa)', '10min_Max_Wind_Gust_(m/s)',
        'Max_Wind_Gust_(m/s)', 'Measurement_Period_Duration', 'Ground_State',
        'Snow_Height_(cm)', 'New_Snow_Depth_(cm)',
        'New_Snowfall_Duration_(hours)', 'Rainfall_(1h,_mm)',
        'Rainfall_(3h,_mm)', 'Rainfall_(6h,_mm)', 'Rainfall_(12h,_mm)',
        'Rainfall_(24h,_mm)', 'Layer_1_Cloud_Cover_(oktas)',
        'Layer_1_Cloud_Type', 'Layer_1_Cloud_Base_Height_(m)',
        'Layer_2_Cloud_Cover_(oktas)', 'Layer_2_Cloud_Type',
        'Layer_2_Cloud_Base_Height_(m)', 'year', 'day',
        'is_school_holiday', 'is_public_holiday',
        ]


# Fonction qui fait ce qu'on voulait faire avec ffill et bfill mais a la place prends la valeur la plus proche
def fill_closest_value_all_columns(df):
    """Fill NaN values with the closest value for all numeric columns in the DataFrame."""
    filled_df = df.copy()
    
    for column in filled_df.columns:
        if filled_df[column].dtype.kind in 'biufc':  # Numeric columns
            non_nan_values = filled_df[column].dropna()
            
            def find_closest(value):
                if pd.isna(value):
                    closest_value = non_nan_values.iloc[(non_nan_values - value).abs().argmin()]
                    return closest_value
                return value
            
            filled_df[column] = filled_df[column].apply(find_closest)
    
    return filled_df


def _merge_external_data(X):
    external_conditions = pd.read_csv('data/external_data.csv')
    external_conditions['date'] = pd.to_datetime(external_conditions['date'])

    # Drop columns with more than 40% NaN values
    threshold = len(external_conditions) * 0.4
    external_conditions = external_conditions.dropna(thresh=threshold, axis=1)

    # Step 2: Remove duplicate entries based on the `date` column
    external_conditions = external_conditions.drop_duplicates(subset='date')

    # Step 3: Convert the 'date' column to datetime
    external_conditions['date'] = pd.to_datetime(external_conditions['date'])

    # Step 4: Create a complete date range from the minimum to the maximum date in the DataFrame
    date_range = pd.date_range(start=external_conditions['date'].min(), end=external_conditions['date'].max(), freq='h')

    # Step 5: Create a DataFrame from the date_range
    date_range_df = pd.DataFrame(date_range, columns=['date'])

    # Step 6: Merge the date_range DataFrame with the external_conditions DataFrame on the 'date' column
    full_external_conditions = pd.merge(date_range_df, external_conditions, on='date', how='left')

    # Apply the function to the DataFrame
    filled_external_conditions = fill_closest_value_all_columns(full_external_conditions)
    merged_conditions = pd.merge(X, filled_external_conditions, on='date', how='left')

    return merged_conditions


def _process_datetime_features(X):
    # Ensure "date" is in datetime format
    X["date"] = pd.to_datetime(X["date"], errors="coerce")

    # Drop rows with invalid datetime entries
    df = X.dropna(subset=["date"])

    # Extract date and time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Handle school and public holidays
    unique_dates = df["date"].dt.date.unique()
    d = SchoolHolidayDates()
    f = JoursFeries()

    try:
        dict_school_holidays = {date: d.is_holiday_for_zone(date, "C") for date in unique_dates}
        df["is_school_holiday"] = df["date"].dt.date.map(dict_school_holidays).fillna(0).astype(int)
    except Exception as e:
        print(f"Error with school holidays mapping: {e}")
        df["is_school_holiday"] = 0

    try:
        dict_public_holidays = {date: f.is_bank_holiday(date, zone="Métropole") for date in unique_dates}
        df["is_public_holiday"] = df["date"].dt.date.map(dict_public_holidays).fillna(0).astype(int)
    except Exception as e:
        print(f"Error with public holidays mapping: {e}")
        df["is_public_holiday"] = 0
    
    return df


def _add_construction_work(df, df_test):
    start_date_Monpar = "2021-01-25"
    end_date_Monpar = "2021-02-23"
    start_date_Clichy_NO_SE = "2021-04-09"
    end_date_Clichy = "2021-07-20"
    start_date_Clichy_SE_NO = "2021-03-23"
    start_date_Pompidou = "2021-03-13"
    end_date_Pompidou = "2021-04-01"

    df["road_work_Monpar_O_E"] = np.where(
        (df["date"] >= start_date_Monpar)
        & (df["date"] <= end_date_Monpar)
        & (df["counter_name"] == "152 boulevard du Montparnasse O-E"),
        1,
        0,
    )
    df["road_work_Monpar_E_O"] = np.where(
        (df["date"] >= start_date_Monpar)
        & (df["date"] <= end_date_Monpar)
        & (df["counter_name"] == "152 boulevard du Montparnasse E-O"),
        1,
        0,
    )
    df["road_work_Clichy_NO_SE"] = np.where(
        (df["date"] >= start_date_Clichy_NO_SE)
        & (df["date"] <= end_date_Clichy)
        & (df["counter_name"] == "20 Avenue de Clichy NO-SE"),
        1,
        0,
    )
    df["road_work_Clichy_SE_NO"] = np.where(
        (df["date"] >= start_date_Clichy_SE_NO)
        & (df["date"] <= end_date_Clichy)
        & (df["counter_name"] == "20 Avenue de Clichy SE-NO"),
        1,
        0,
    )
    df["road_work_Pompidou_NE_SO"] = np.where(
        (df["date"] >= start_date_Pompidou)
        & (df["date"] <= end_date_Pompidou)
        & (df["counter_name"] == "Voie Georges Pompidou NE-SO"),
        1,
        0,
    )
    df["road_work_Pompidou_SO_NE"] = np.where(
        (df["date"] >= start_date_Pompidou)
        & (df["date"] <= end_date_Pompidou)
        & (df["counter_name"] == "Voie Georges Pompidou SO-NE"),
        1,
        0,
    )

    df["road_work"] = (
        df["road_work_Monpar_E_O"]
        + df["road_work_Monpar_O_E"]
        + df["road_work_Clichy_NO_SE"]
        + df["road_work_Clichy_SE_NO"]
        + df["road_work_Pompidou_NE_SO"]
        + df["road_work_Pompidou_SO_NE"]
    )
    df.drop(
        [
            "road_work_Monpar_E_O",
            "road_work_Monpar_O_E",
            "road_work_Clichy_NO_SE",
            "road_work_Clichy_SE_NO",
            "road_work_Pompidou_NE_SO",
            "road_work_Pompidou_SO_NE",
        ],
        axis=1,
        inplace=True,
    )

    df["log_bike_count"][
        (df["date"] >= start_date_Monpar)
        & (df["date"] <= end_date_Monpar)
        & (df["counter_name"] == "152 boulevard du Montparnasse E-O")
    ] = 0
    df["log_bike_count"][
        (df["date"] >= start_date_Monpar)
        & (df["date"] <= end_date_Monpar)
        & (df["counter_name"] == "152 boulevard du Montparnasse O-E")
    ] = 0
    df["log_bike_count"][
        (df["date"] >= start_date_Clichy_NO_SE)
        & (df["date"] <= end_date_Clichy)
        & (df["counter_name"] == "20 Avenue de Clichy NO-SE")
    ] = 0
    df["log_bike_count"][
        (df["date"] >= start_date_Clichy_SE_NO)
        & (df["date"] <= end_date_Clichy)
        & (df["counter_name"] == "20 Avenue de Clichy SE-NO")
    ] = 0
    df["log_bike_count"][
        (df["date"] >= start_date_Pompidou)
        & (df["date"] <= end_date_Pompidou)
        & (df["counter_name"] == "Voie Georges Pompidou NE-SO")
    ] = 0
    df["log_bike_count"][
        (df["date"] >= start_date_Pompidou)
        & (df["date"] <= end_date_Pompidou)
        & (df["counter_name"] == "Voie Georges Pompidou SO-NE")
    ] = 0

    df_test['road_work'] = 0

    return df, df_test


def _confinement_and_couvre_feu(X, X_test):
    confinements = [
    ("2020-10-30", "2020-12-14", "confinement"),
    ("2021-04-03", "2021-05-19", "confinement"), 
    ]

    couvre_feux = [
        ("2020-10-17", "2020-10-29", "21:00", "06:00", "couvre-feu"),
        ("2020-12-15", "2021-01-15", "20:00", "06:00", "couvre-feu"),
        ("2021-01-16", "2021-03-19", "18:00", "06:00", "couvre-feu"),
        ("2021-03-20", "2021-05-18", "19:00", "06:00", "couvre-feu"),
        ("2021-05-19", "2021-06-08", "21:00", "06:00", "couvre-feu"),
        ("2021-06-09", "2021-06-19", "23:00", "06:00", "couvre-feu"),
    ]

    confinements = [(pd.to_datetime(start), pd.to_datetime(end), label) for start, end, label in confinements]
    couvre_feux = [(pd.to_datetime(start), pd.to_datetime(end), start_hour, end_hour, label) for start, end, start_hour, end_hour, label in couvre_feux]

    X["confinement"] = 0
    X["couvre_feu"] = 0

    X_test["confinement"] = 0
    X_test["couvre_feu"] = 0

    for start, end, label in confinements:
        X.loc[(X["date"] >= start) & (X["date"] <= end), "confinement"] = 1

    for start, end, start_hour, end_hour, label in couvre_feux:

        in_couvre_feu_period = (X["date"] >= start) & (X["date"] <= end)

        in_couvre_feu_hours = (X["date"].dt.time >= pd.to_datetime(start_hour).time()) | (X["date"].dt.time <= pd.to_datetime(end_hour).time())

        X.loc[in_couvre_feu_period & in_couvre_feu_hours, "couvre_feu"] = 1

    return X, X_test



def get_and_process_data():
    data = pd.read_parquet("data/train.parquet")
    data = _merge_external_data(data)
    data = _column_rename(data)
    data = _process_datetime_features(data)

    data_test = pd.read_parquet("data/final_test.parquet")
    data_test = _merge_external_data(data_test)
    data_test = _column_rename(data_test)
    data_test = _process_datetime_features(data_test)

    data, data_test = _add_construction_work(data, data_test)
    data, data_test = _confinement_and_couvre_feu(data, data_test)

    data = data.drop(columns=columns_to_drop_boruta1)
    data_test = data_test.drop(columns=columns_to_drop_boruta1)

    X = data.drop(columns=['log_bike_count', 'bike_count'])
    y = data['log_bike_count']

    return X, y, data_test


def create_pipeline(df, model=None):
    # Classify columns into categorical, numerical, and binary
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    numerical_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].unique()) > 2]
    binary_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].unique()) == 2]

    # Define preprocessing for each type of feature
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns),
            ('passthrough', 'passthrough', binary_columns),
        ]
    )

    # Use the provided model or default to RandomForestClassifier
    if model is None:
        best_params = joblib.load('xg_boost_best_params.pkl')
        if 'tree_method' in best_params:
            best_params['tree_method'] = 'hist'  # Ensure compatibility with CPU
        model = XGBRegressor(**best_params)

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline



def test_fit_and_submission(X_test, pipeline):
    y_pred = pipeline.predict(X_test)
    df_submission = pd.DataFrame(y_pred, columns=["log_bike_count"])
    df_submission.index = X_test.index
    df_submission.index.name = "Id"
    df_submission.to_csv("test_pipeline.csv", index=True)
    return df_submission






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
