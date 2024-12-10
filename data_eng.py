"""Feature selection functions"""

from typing import List
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import logging

import constants
from src.configs.names import TARGET


def selecting_features_with_boruta(
    df: pd.DataFrame, features: List[str] | None, target: str | None = TARGET
) -> List[str]:
    """
    Select features using Boruta.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        features (List[str]): Features to select from.
        target (str): Target column.

    Returns:
        List[str]: Selected features by Boruta.
    """
    # Random forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=constants.NB_RF_CLASSIFIERS_BORUTA,
        random_state=constants.RANDOM_SEED,
    )
    # Boruta feature selection method
    boruta_selector = BorutaPy(
        rf_classifier,
        n_estimators=constants.NB_BORUTA_ESTIMATORS,
        random_state=constants.RANDOM_SEED,
    )
    boruta_selector.fit(df[features].values, df[target].values)
    selected_features = df[features].columns[boruta_selector.support_].tolist()
    logging.info(f"Selected features by Boruta : { selected_features }")
    return selected_features


def selecting_features_with_random_columns(
    df: pd.DataFrame, features: List[str] | None, target: str | None = TARGET
) -> List[str]:
    """
    Select features that have less importance than random ones.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        features (List[str]): List of features.
        target (str): Target column.

    Returns:
        List[str]: Selected features with random columns.
    """
    # Create 5 random columns
    for i in range(constants.NB_RANDOM_COLUMNS):
        df[f"random_{i}"] = np.random.uniform(
            low=constants.LOW_VALUE_FEATURE,
            high=constants.HIGH_VALUE_FEATURE,
            size=df.shape[0],
        )
    # Combine original features with random features
    all_features = features + [f"random_{i}" for i in range(5)]
    # Initialize and fit the RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        n_estimators=constants.NB_RF_CLASSIFIERS_RANDOM_COLUMNS,
        random_state=constants.RANDOM_SEED,
    )
    rf_classifier.fit(df[all_features].values, df[target].values)
    # Get feature importances
    feature_importances = rf_classifier.feature_importances_
    feature_importances_df = pd.DataFrame(
        {"feature": all_features, "importance": feature_importances}
    )
    # Select features with importance greater than the maximum random importance
    max_random_importance = feature_importances_df[
        feature_importances_df["feature"].str.contains("random")
    ]["importance"].max()
    selected_features = feature_importances_df[
        feature_importances_df["importance"] > max_random_importance
    ]["feature"].tolist()
    selected_features = [
        feature for feature in selected_features if not feature.startswith("random")
    ]
    logging.info(f"Selected features with random columns : { selected_features }")
    return selected_features


def selecting_features_with_correlations(
    df: pd.DataFrame, features: List[str] | None, target: str | None = TARGET
) -> list:
    """
    Select feature based on correlations.
    Keep features that are correlated with the target.
    Remove features that are correlated with other features.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        features (List[str] | None): List of features.
        target (str | None, optional): Target column. Defaults to TARGET.

    Returns:
        list: Selected features with correlations.
    """
    # Remove features that not correlated with the target
    correlation_with_label = (
        df[features].corr(method=constants.CORR_TYPE)[target].drop(target)
    )
    threshold_label = constants.THRESHOLD_CORR_LABEL
    selected_features = correlation_with_label[
        correlation_with_label.abs() > threshold_label
    ].index
    correlation_matrix = df[selected_features].corr()
    threshold_features = constants.THRESHOLD_CORR_FEATURE
    to_drop = set()
    # Remove features that are correlated with other features
    # Keep the one that is more correlated with the target
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold_features:
                feature_to_drop = (
                    correlation_matrix.columns[i]
                    if abs(correlation_with_label[correlation_matrix.columns[i]])
                    < abs(correlation_with_label[correlation_matrix.columns[j]])
                    else correlation_matrix.columns[j]
                )
                to_drop.add(feature_to_drop)
    selected_features = [f for f in selected_features if f not in to_drop]
    return selected_features
