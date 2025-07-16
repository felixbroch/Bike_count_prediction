"""
Example estimator for bike count prediction using external weather data.

This module provides a simple example of how to create a machine learning pipeline
that incorporates external weather data for bike count prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge


def _encode_dates(X):
    """
    Extract datetime features from the date column.
    
    Args:
        X (pd.DataFrame): DataFrame with 'date' column
        
    Returns:
        pd.DataFrame: DataFrame with date features and 'date' column removed
    """
    X = X.copy()  # modify a copy of X
    # Encode the date information from the date column
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    """
    Merge bike count data with external weather data.
    
    This function performs an as-of merge to align weather data with bike count data
    based on timestamps.
    
    Args:
        X (pd.DataFrame): Bike count data with 'date' column
        
    Returns:
        pd.DataFrame: Merged dataset with weather information
    """
    file_path = Path(__file__).parent / "external_data.csv"
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


def get_estimator():
    """
    Create a simple Ridge regression pipeline with external weather data.
    
    This function creates a basic machine learning pipeline that:
    1. Merges external weather data
    2. Encodes datetime features
    3. Preprocesses categorical and date columns
    4. Fits a Ridge regression model
    
    Returns:
        sklearn.pipeline.Pipeline: Complete preprocessing and modeling pipeline
    """
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

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
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
