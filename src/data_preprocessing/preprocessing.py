"""
Feature preprocessing pipeline for CSAT score prediction.

Transforms the cleaned CSAT dataset into model-ready features by:
    1. Handling missing values
    2. Extracting datetime features
    3. Encoding categoricals and scaling numerics via sklearn pipelines
"""

import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import (
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.utils.logger import logger
from src.utils.exception import CustomException


class DataPreprocessor:
    """
    End-to-end preprocessor for the CSAT classification task.

    Expects the raw CSV schema and produces a sparse numeric feature matrix
    suitable for sklearn or neural-network models.
    """

    def __init__(self):
        self.target_column = TARGET_COLUMN
        self.feature_engineer = FeatureEngineer()
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.identifier_columns = IDENTIFIER_COLUMNS
        self.numerical_columns = (
            self.feature_engineer.get_derived_column_names()
        )

    def create_numerical_pipeline(self):
        """
        Build the sklearn pipeline for datetime-derived numeric features.

        Returns:
            Pipeline with median imputation and standard scaling.
        """

        try:
            logger.info(
                "Creating numerical preprocessing pipeline"
            )

            return Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)

    def create_categorical_pipeline(self):
        """
        Build the sklearn pipeline for categorical features.

        Returns:
            Pipeline with mode imputation and one-hot encoding.
        """

        try:
            logger.info(
                "Creating categorical preprocessing pipeline"
            )

            return Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)

    def create_preprocessor(self):
        """
        Combine categorical and numerical pipelines into a ColumnTransformer.

        Returns:
            Fitted-ready sklearn ColumnTransformer.
        """

        try:
            logger.info(
                "Creating full preprocessing transformer"
            )

            transformers = [
                (
                    "categorical_pipeline",
                    self.create_categorical_pipeline(),
                    self.categorical_columns,
                ),
                (
                    "numerical_pipeline",
                    self.create_numerical_pipeline(),
                    self.numerical_columns,
                ),
            ]

            return ColumnTransformer(
                transformers=transformers
            )

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_csat_data(
        self,
        dataframe: pd.DataFrame,
        fit: bool = True,
        preprocessor=None,
    ):
        """
        Run the full CSAT preprocessing workflow on raw data.

        Pipeline:
            1. FeatureEngineer       — missing values + datetime features
            2. ColumnTransformer     — encode and scale features

        Args:
            dataframe: Raw CSAT dataframe loaded from CSV.
            fit: Whether to fit the preprocessor (True for training).
            preprocessor: Optional pre-fitted ColumnTransformer for inference.

        Returns:
            Tuple of (X_transformed, y, preprocessor) where:
                - X_transformed: numpy array of encoded features
                - y: CSAT Score target series, or None during inference
                - preprocessor: fitted ColumnTransformer for inference
        """

        try:
            logger.info("Starting csat preprocessing")

            dataframe = self.feature_engineer.engineer_features(
                dataframe
            )

            has_target = self.target_column in dataframe.columns
            y = None
            if has_target:
                y = dataframe[self.target_column].astype(int)

            columns_to_drop = list(self.identifier_columns)
            if has_target:
                columns_to_drop.insert(0, self.target_column)

            X = dataframe.drop(columns=columns_to_drop)

            if preprocessor is None:
                preprocessor = self.create_preprocessor()

            if fit:
                X_transformed = preprocessor.fit_transform(X)
            else:
                X_transformed = preprocessor.transform(X)

            logger.info(
                f"CSAT preprocessing completed. "
                f"Feature matrix shape: {X_transformed.shape}"
            )

            return X_transformed, y, preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def transform_csat_data(
        self,
        dataframe: pd.DataFrame,
        preprocessor,
    ):
        """Transform new data using a fitted preprocessor."""

        X_transformed, y, _ = self.preprocess_csat_data(
            dataframe,
            fit=False,
            preprocessor=preprocessor,
        )

        return X_transformed, y
