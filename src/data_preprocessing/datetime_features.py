"""
Datetime feature extraction for the CSAT dataset.

Converts remaining timestamp columns into numeric hour, day, and month
features after missing-value handling. order_date_time is excluded because
it is dropped upstream due to high missingness.
"""

import sys

import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException


class DatetimeFeatureExtractor:
    """
    Parses datetime columns and derives temporal features for modeling.

    Strategy (derived from the project notebook):
        - Parse mixed-format timestamps with day-first ordering
        - Extract hour, day, and month components
        - Drop raw datetime columns once derivatives are created
    """

    def __init__(
        self,
        datetime_columns: list[str] | None = None,
        temporal_features: tuple[str, ...] = ("hour", "day", "month"),
    ):
        """
        Args:
            datetime_columns: Timestamp columns to transform. Defaults to the
                three columns retained after missing-value handling.
            temporal_features: Datetime parts to extract from each column.
        """
        self.datetime_columns = datetime_columns or [
            "Issue_reported at",
            "issue_responded",
            "Survey_response_Date",
        ]
        self.temporal_features = temporal_features

    def get_derived_column_names(self) -> list[str]:
        """
        Return the numeric column names produced by feature extraction.

        Returns:
            List of derived feature names, e.g. 'Issue_reported at_hour'.
        """
        return [
            f"{column}_{feature}"
            for column in self.datetime_columns
            for feature in self.temporal_features
        ]

    def parse_datetime_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert datetime string columns to pandas datetime dtype.

        Args:
            dataframe: Dataframe containing raw datetime strings.

        Returns:
            Copy of the dataframe with parsed datetime columns.
        """

        try:
            logger.info("Parsing datetime columns")

            dataframe = dataframe.copy()

            for column in self.datetime_columns:
                if column not in dataframe.columns:
                    continue

                dataframe[column] = pd.to_datetime(
                    dataframe[column],
                    format="mixed",
                    dayfirst=True,
                )

            return dataframe

        except Exception as e:
            logger.error("Datetime parsing failed")
            raise CustomException(e, sys)

    def extract_datetime_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parse datetime columns and create numeric temporal features.

        Steps:
            1. Parse timestamp strings into datetime dtype
            2. Extract configured temporal components
            3. Drop the original datetime columns

        Args:
            dataframe: Dataframe after missing-value handling.

        Returns:
            Dataframe with derived numeric columns and raw datetime columns
            removed.
        """

        try:
            logger.info("Extracting datetime features")

            dataframe = self.parse_datetime_columns(dataframe)

            for column in self.datetime_columns:
                if column not in dataframe.columns:
                    continue

                for feature in self.temporal_features:
                    dataframe[f"{column}_{feature}"] = (
                        getattr(dataframe[column].dt, feature)
                    )

            datetime_columns_present = [
                column
                for column in self.datetime_columns
                if column in dataframe.columns
            ]

            if datetime_columns_present:
                dataframe = dataframe.drop(
                    columns=datetime_columns_present
                )

            logger.info(
                f"Datetime feature extraction completed. "
                f"Derived columns: {self.get_derived_column_names()}"
            )

            return dataframe

        except Exception as e:
            logger.error("Datetime feature extraction failed")
            raise CustomException(e, sys)
