"""
Utility functions for bike count prediction project.

This module contains data preprocessing, pipeline creation, and model training utilities
for predicting bike counts in Paris using historical data and weather information.

Author: Felix Brochier and Gregoire Bidault
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import joblib
from skrub import TableVectorizer
from lightgbm import LGBMRegressor
import best_params
import optuna
from sklearn.model_selection import cross_val_score


def get_and_process_data():
    """
    Load and preprocess training and test data for bike count prediction.
    
    This function performs the complete data preprocessing pipeline including:
    - Loading parquet files for training and test data
    - Merging external weather data
    - Renaming columns for better readability
    - Processing datetime features
    - Adding construction work indicators
    - Adding confinement and curfew indicators
    - Dropping unnecessary columns
    
    Returns:
        tuple: (X, y, data_test) where:
            - X: Preprocessed training features (pd.DataFrame)
            - y: Training target variable (log_bike_count, pd.Series)
            - data_test: Preprocessed test features (pd.DataFrame)
    """
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

    data = data.drop(columns=columns_to_drop_test)
    data_test = data_test.drop(columns=columns_to_drop_test)

    X = data.drop(columns=["log_bike_count", "bike_count"])
    y = data["log_bike_count"]

    return X, y, data_test


def create_pipeline(df, model=None):
    """
    Create a machine learning pipeline with custom preprocessing.
    
    This function creates a preprocessing pipeline that:
    - Classifies columns into categorical, numerical, and binary
    - Applies OneHotEncoder to categorical columns
    - Applies StandardScaler to numerical columns
    - Passes through binary columns unchanged
    
    Args:
        df (pd.DataFrame): Training data to determine column types
        model (sklearn estimator, optional): Model to use. If None, uses XGBoost with optimized parameters
    
    Returns:
        sklearn.pipeline.Pipeline: Complete preprocessing and modeling pipeline
    """
    # Classify columns into categorical, numerical, and binary
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    numerical_columns = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].unique()) > 2
    ]
    binary_columns = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].unique()) == 2
    ]

    # Define preprocessing for each type of feature
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numerical_transformer, numerical_columns),
            ("passthrough", "passthrough", binary_columns),
        ]
    )

    # Use the provided model or default to RandomForestClassifier
    if model is None:
        best_params_XGB = best_params.parameters_XGBoost
        model = XGBRegressor(**best_params_XGB)

    # Create the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline


def create_pipeline_TV(df, model=None):
    """
    Create a machine learning pipeline using TableVectorizer for preprocessing.
    
    This function creates a more sophisticated preprocessing pipeline using
    TableVectorizer, which automatically handles mixed data types and provides
    better performance than manual column classification.
    
    Args:
        df (pd.DataFrame): Training data
        model (sklearn estimator, optional): Model to use. If None, uses LightGBM with optimized parameters
    
    Returns:
        sklearn.pipeline.Pipeline: Complete preprocessing and modeling pipeline
    """
    # Define TableVectorizer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "vectorizer",
                TableVectorizer(),
                df.columns,
            )  # Apply TableVectorizer to all columns
        ]
    )

    # Use the provided model or default to RandomForestClassifier
    if model is None:
        best_params_LGBM = best_params.parameters_LGBM
        model = LGBMRegressor(**best_params_LGBM)

    # Create the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline


def objective(trial, X, y):
    """
    Optuna objective function for hyperparameter optimization.
    
    This function defines the search space for LightGBM hyperparameters
    and evaluates model performance using cross-validation.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions
        X (pd.DataFrame): Training features
        y (pd.Series): Training target variable
    
    Returns:
        float: Negative mean squared error (to minimize)
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 10.0),
    }

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "vectorizer",
                TableVectorizer(),
                list(X.columns),  # Use feature columns dynamically
            )
        ]
    )

    # Define the model and pipeline
    model = LGBMRegressor(**params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    return -scores.mean()

def create_pipeline_TV_with_optuna(X, y, n_trials=50):
    """
    Create an optimized pipeline using Optuna for hyperparameter tuning.
    
    This function combines TableVectorizer preprocessing with Optuna optimization
    to find the best LightGBM hyperparameters for the given dataset.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training target variable
        n_trials (int): Number of optimization trials to run
    
    Returns:
        sklearn.pipeline.Pipeline: Optimized preprocessing and modeling pipeline
    """
    # Initialize Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    # Get the best parameters
    best_params = study.best_params
    best_params["tree_method"] = "hist"  # Compatibility with CPU (in the case of training on the GPU before)

    # Create the final pipeline with the best parameters
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "vectorizer",
                TableVectorizer(),
                list(X.columns),  # Use feature columns dynamically
            )
        ]
    )
    model = LGBMRegressor(**best_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline


def test_fit_and_submission(X_test, pipeline):
    """
    Generate predictions and create submission file.
    
    This function fits the pipeline to test data, makes predictions,
    and saves the results in the required submission format.
    
    Args:
        X_test (pd.DataFrame): Test features
        pipeline (sklearn.pipeline.Pipeline): Trained pipeline
    
    Returns:
        pd.DataFrame: Predictions in submission format with columns ['log_bike_count']
    """
    y_pred = pipeline.predict(X_test)
    df_submission = pd.DataFrame(y_pred, columns=["log_bike_count"])
    df_submission.index = X_test.index
    df_submission.index.name = "Id"
    df_submission.to_csv("test_pipeline.csv", index=True)
    return df_submission


def _column_rename(X):
    """
    Rename weather data columns to more descriptive names.
    
    This function maps abbreviated weather column names to more descriptive,
    human-readable names including units where appropriate.
    
    Args:
        X (pd.DataFrame): DataFrame with weather data columns
    
    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
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


# Column sets for feature selection experimentation
# These lists define different feature selection strategies tested during model development

# Columns dropped based on domain knowledge and initial testing
columns_to_drop_test = [
    "Station_Number",
    "Measurement_Period_Duration",
    "site_name",
    "coordinates",
    "Layer_1_Cloud_Base_Height_(m)",
    "Sea_Level_Pressure_(hPa)",
    "counter_name",
]

# Columns dropped based on personal understanding of the problem domain
columns_to_drop_personal = [
    "date",
    "counter_installation_date",
    "Cloud_Base_Height_(m)",
    "counter_id",
    "site_id",
    "site_name",
    "counter_technical_id",
    "coordinates",
    "Station_Level_Pressure_(hPa)",
    "Pressure_Tendency_Code",
    "Station_Number",
    "Measurement_Period_Duration",
    "Measurement_Period_Duration",
    "Layer_1_Cloud_Base_Height_(m)",
    "Present_Weather_Code",
    "Past_Weather_Code_1",
    "Past_Weather_Code_2",
    "Rainfall_(1h,_mm)",
    "Rainfall_(6h,_mm)",
    "Rainfall_(24h,_mm)",
    "Wind_Direction_(°)",
    "Dew_Point_Temperature_(°C)",
    "Lowest_Cloud_Base_Height_(m)",
    "Low_Cloud_Type",
    "Medium_Cloud_Type",
    "High_Cloud_Type",
    "10min_Max_Wind_Gust_(m/s)",
    "Ground_State",
    "New_Snow_Depth_(cm)",
    "New_Snowfall_Duration_(hours)",
    "Layer_1_Cloud_Cover_(oktas)",
    "Layer_1_Cloud_Type",
    "Layer_2_Cloud_Cover_(oktas)",
    "Layer_2_Cloud_Type",
    "Layer_2_Cloud_Base_Height_(m)",
    "Max_Wind_Gust_(m/s)",
    "24h_Pressure_Tendency_(hPa)",
    "Sea_Level_Pressure_(hPa)",
]

# Columns dropped based on correlation analysis
columns_to_drop_correlation = [
    "site_name",
    "date",
    "counter_installation_date",
    "coordinates",
    "longitude",
    "Station_Number",
    "Sea_Level_Pressure_(hPa)",
    "Pressure_Tendency_(hPa/3h)",
    "Pressure_Tendency_Code",
    "Wind_Direction_(°)",
    "Wind_Speed_(m/s)",
    "Visibility_(m)",
    "Present_Weather_Code",
    "Past_Weather_Code_1",
    "Past_Weather_Code_2",
    "Total_Cloud_Cover_(oktas)",
    "Cloud_Base_Height_(m)",
    "Medium_Cloud_Type",
    "High_Cloud_Type",
    "Station_Level_Pressure_(hPa)",
    "24h_Pressure_Tendency_(hPa)",
    "Max_Wind_Gust_(m/s)",
    "Measurement_Period_Duration",
    "Snow_Height_(cm)",
    "New_Snow_Depth_(cm)",
    "New_Snowfall_Duration_(hours)",
    "Rainfall_(1h,_mm)",
    "Rainfall_(3h,_mm)",
    "Rainfall_(6h,_mm)",
    "Rainfall_(12h,_mm)",
    "Rainfall_(24h,_mm)",
    "Layer_1_Cloud_Base_Height_(m)",
    "Layer_2_Cloud_Type",
    "year",
    "month",
    "day",
    "is_school_holiday",
]

# Columns dropped based on random forest feature importance
columns_to_drop_random = [
    "site_id",
    "site_name",
    "counter_installation_date",
    "Station_Number",
    "Sea_Level_Pressure_(hPa)",
    "Pressure_Tendency_(hPa/3h)",
    "Pressure_Tendency_Code",
    "Wind_Direction_(°)",
    "Wind_Speed_(m/s)",
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
    "Medium_Cloud_Type",
    "High_Cloud_Type",
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
    "Layer_2_Cloud_Cover_(oktas)",
    "Layer_2_Cloud_Type",
    "Layer_2_Cloud_Base_Height_(m)",
    "year",
    "is_school_holiday",
    "is_public_holiday",
]

# Columns dropped based on Boruta feature selection algorithm
columns_to_drop_boruta1 = [
    "counter_installation_date",
    "Station_Number",
    "Sea_Level_Pressure_(hPa)",
    "Pressure_Tendency_(hPa/3h)",
    "Pressure_Tendency_Code",
    "Wind_Direction_(°)",
    "Wind_Speed_(m/s)",
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
    "Medium_Cloud_Type",
    "High_Cloud_Type",
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
    "Layer_2_Cloud_Cover_(oktas)",
    "Layer_2_Cloud_Type",
    "Layer_2_Cloud_Base_Height_(m)",
    "year",
    "day",
    "is_school_holiday",
    "is_public_holiday",
]


def fill_closest_value_all_columns(df):
    """
    Fill NaN values with the closest non-NaN value for all numeric columns.
    
    This function replaces NaN values in numeric columns by finding the closest
    non-NaN value in the same column based on absolute difference.
    
    Args:
        df (pd.DataFrame): DataFrame with potential NaN values
    
    Returns:
        pd.DataFrame: DataFrame with NaN values filled
    """
    """Fill NaN values with the closest value for all numeric columns in the DataFrame."""
    filled_df = df.copy()

    for column in filled_df.columns:
        if filled_df[column].dtype.kind in "biufc":  # Numeric columns
            non_nan_values = filled_df[column].dropna()

            def find_closest(value):
                if pd.isna(value):
                    closest_value = non_nan_values.iloc[
                        (non_nan_values - value).abs().argmin()
                    ]
                    return closest_value
                return value

            filled_df[column] = filled_df[column].apply(find_closest)

    return filled_df


def _merge_external_data(X):
    """
    Merge bike count data with external weather data.
    
    This function loads weather data, cleans it by removing columns with >40% NaN values,
    fills missing values, and merges it with the bike count data on date.
    
    Args:
        X (pd.DataFrame): Bike count data with 'date' column
    
    Returns:
        pd.DataFrame: Merged dataset with weather information
    """
    external_conditions = pd.read_csv("data/external_data.csv")
    external_conditions["date"] = pd.to_datetime(external_conditions["date"])

    # Drop columns with more than 40% NaN values
    threshold = len(external_conditions) * 0.4
    external_conditions = external_conditions.dropna(thresh=threshold, axis=1)

    external_conditions = external_conditions.drop_duplicates(subset="date")

    external_conditions["date"] = pd.to_datetime(external_conditions["date"])

    date_range = pd.date_range(
        start=external_conditions["date"].min(),
        end=external_conditions["date"].max(),
        freq="h",
    )

    date_range_df = pd.DataFrame(date_range, columns=["date"])

    full_external_conditions = pd.merge(
        date_range_df, external_conditions, on="date", how="left"
    )

    filled_external_conditions = fill_closest_value_all_columns(
        full_external_conditions
    )
    merged_conditions = pd.merge(X, filled_external_conditions, on="date", how="left")

    return merged_conditions


def _process_datetime_features(X):
    """
    Extract datetime features and add holiday indicators.
    
    This function processes the 'date' column to extract temporal features
    and adds indicators for school holidays and public holidays in France.
    
    Args:
        X (pd.DataFrame): DataFrame with 'date' column
    
    Returns:
        pd.DataFrame: DataFrame with additional temporal and holiday features
    """
    
    X["date"] = pd.to_datetime(X["date"], errors="coerce")

    df = X.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

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


def _add_construction_work(df, df_test):
    """
    Add construction work indicators for specific locations and time periods.
    
    This function adds binary indicators for known construction work periods
    that affected bike counts at specific counter locations in 2021.
    
    Args:
        df (pd.DataFrame): Training data
        df_test (pd.DataFrame): Test data
    
    Returns:
        tuple: (df, df_test) with construction work indicators added
    """
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

    df_test["road_work"] = 0

    return df, df_test


def _confinement_and_couvre_feu(X, X_test):
    """
    Add COVID-19 lockdown and curfew indicators.
    
    This function adds binary indicators for COVID-19 lockdowns (confinement)
    and curfews that affected mobility patterns in France during 2020-2021.
    
    Args:
        X (pd.DataFrame): Training data
        X_test (pd.DataFrame): Test data
    
    Returns:
        tuple: (X, X_test) with lockdown and curfew indicators added
    """
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

    confinements = [
        (pd.to_datetime(start), pd.to_datetime(end), label)
        for start, end, label in confinements
    ]
    couvre_feux = [
        (pd.to_datetime(start), pd.to_datetime(end), start_hour, end_hour, label)
        for start, end, start_hour, end_hour, label in couvre_feux
    ]

    X["confinement"] = 0
    X["couvre_feu"] = 0

    X_test["confinement"] = 0
    X_test["couvre_feu"] = 0

    for start, end, label in confinements:
        X.loc[(X["date"] >= start) & (X["date"] <= end), "confinement"] = 1

    for start, end, start_hour, end_hour, label in couvre_feux:
        in_couvre_feu_period = (X["date"] >= start) & (X["date"] <= end)

        in_couvre_feu_hours = (
            X["date"].dt.time >= pd.to_datetime(start_hour).time()
        ) | (X["date"].dt.time <= pd.to_datetime(end_hour).time())

        X.loc[in_couvre_feu_period & in_couvre_feu_hours, "couvre_feu"] = 1

    return X, X_test
