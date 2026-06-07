"""
Feature engineering pipeline for the CSAT dataset.

Orchestrates missing-value handling, text preprocessing, and datetime
feature extraction — matching the EDA workflow in the project notebook.
"""

import sys

import pandas as pd

from src.data.schema import (
    CATEGORICAL_COLUMNS,
    DATETIME_COLUMNS,
    HIGH_MISSING_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from src.data_preprocessing.missing_value_handler import (
    MissingValueHandler,
)
from src.data_preprocessing.text_preprocessing import (
    TextPreprocessor,
)
from src.data_preprocessing.datetime_features import (
    DatetimeFeatureExtractor,
)
from src.utils.logger import logger
from src.utils.exception import CustomException


class FeatureEngineer:
    """
    Transform raw CSAT CSV rows into a cleaned dataframe ready for encoding.

    Pipeline:
        1. MissingValueHandler  — drop sparse columns, impute categoricals
        2. TextPreprocessor     — impute Customer Remarks
        3. DatetimeFeatureExtractor — hour/day/month from timestamps
    """

    def __init__(
        self,
        missing_handler: MissingValueHandler | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        datetime_extractor: DatetimeFeatureExtractor | None = None,
    ):
        self.missing_handler = (
            missing_handler or MissingValueHandler()
        )
        self.text_preprocessor = (
            text_preprocessor or TextPreprocessor()
        )
        self.datetime_extractor = (
            datetime_extractor
            or DatetimeFeatureExtractor(
                datetime_columns=DATETIME_COLUMNS
            )
        )

        self.target_column = TARGET_COLUMN
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.identifier_columns = IDENTIFIER_COLUMNS
        self.high_missing_columns = HIGH_MISSING_COLUMNS

    def get_derived_column_names(self) -> list[str]:
        """Return datetime-derived numeric column names."""
        return self.datetime_extractor.get_derived_column_names()

    def engineer_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the full feature engineering workflow.

        Args:
            dataframe: Raw CSAT dataframe from CSV ingestion.

        Returns:
            Cleaned dataframe with derived datetime features.
            Typical shape after processing: (67675, 15).
        """

        try:
            logger.info("Starting feature engineering")

            dataframe = self.missing_handler.handle_missing_values(
                dataframe
            )
            dataframe = self.text_preprocessor.preprocess_text(
                dataframe
            )
            dataframe = (
                self.datetime_extractor.extract_datetime_features(
                    dataframe
                )
            )

            logger.info(
                f"Feature engineering completed. "
                f"Shape: {dataframe.shape}"
            )

            return dataframe

        except Exception as e:
            logger.error("Feature engineering failed")
            raise CustomException(e, sys)
