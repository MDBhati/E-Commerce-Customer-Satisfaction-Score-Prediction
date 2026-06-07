"""
Missing value handling for the CSAT dataset.

EDA showed that several columns exceed 70% missingness and are dropped
entirely. Remaining gaps are imputed or resolved by row removal based on
column type and business meaning.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.schema import HIGH_MISSING_COLUMNS
from src.utils.logger import logger
from src.utils.exception import CustomException


class MissingValueHandler:
    """
    Cleans missing values in the raw CSAT dataset before feature engineering.

    Strategy (derived from EDA in the project notebook):
        - Drop columns with >70% missing values (too sparse to impute reliably)
        - Impute categorical text fields with a domain-specific placeholder
        - Drop rows with missing unique identifiers instead of imputing them
    """

    def __init__(
        self,
        missing_threshold: float = 0.70,
        plot_output_path: str = "reports/images/missing_values.png",
    ):
        """
        Args:
            missing_threshold: Minimum missing ratio used to justify column
                removal during EDA. Kept for reference/logging; columns in
                `columns_to_drop` were selected based on this threshold.
            plot_output_path: File path for the missing-values bar chart.
        """
        self.missing_threshold = missing_threshold
        self.plot_output_path = plot_output_path

        # Columns with >70% missing — retained values are insufficient for modeling
        self.columns_to_drop = list(HIGH_MISSING_COLUMNS)

        # Categorical columns where a placeholder preserves row count
        self.categorical_imputations = {
            "Customer Remarks": "No Remarks",  # 66.54% missing
        }

        # Identifier columns: drop rows rather than invent synthetic IDs
        self.identifier_columns_to_drop_rows = [
            "Order_id",  # 21.22% missing
        ]

    def get_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Return the count of missing values per column, highest first.

        Args:
            dataframe: Raw or partially cleaned CSAT dataframe.

        Returns:
            Series indexed by column name with missing-value counts.
        """

        try:
            logger.info("Calculating missing value counts")

            missing_values = (
                dataframe.isna()
                .sum()
                .sort_values(ascending=False)
            )

            logger.info(
                f"Missing values summary: "
                f"{missing_values.to_dict()}"
            )

            return missing_values

        except Exception as e:
            logger.error("Failed to calculate missing values")
            raise CustomException(e, sys)

    def save_missing_values_plot(
        self,
        dataframe: pd.DataFrame,
        top_n: int = 9,
    ) -> str:
        """
        Save a bar chart of the columns with the most missing values.

        Args:
            dataframe: Dataset to summarize (typically raw data before cleaning).
            top_n: Number of columns to include in the chart.

        Returns:
            Path to the saved PNG file.
        """

        try:
            logger.info("Saving missing values plot")

            missing_values = self.get_missing_values(dataframe)
            top_missing = missing_values.head(top_n)

            sns.set(style="whitegrid")

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=top_missing.index,
                y=top_missing.values,
                hue=top_missing.index,
                palette="mako",
                legend=False,
            )

            for index, value in enumerate(top_missing.values):
                plt.text(
                    index,
                    value + 1,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.ylabel("Number of Missing Values", fontsize=12)
            plt.xlabel("Features", fontsize=12)
            plt.title(
                f"Top {top_n} Features with Missing Values",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()

            os.makedirs(
                os.path.dirname(self.plot_output_path),
                exist_ok=True,
            )
            plt.savefig(self.plot_output_path)
            plt.close()

            logger.info(
                f"Missing values plot saved to "
                f"{self.plot_output_path}"
            )

            return self.plot_output_path

        except Exception as e:
            logger.error("Failed to save missing values plot")
            raise CustomException(e, sys)

    def handle_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the full missing-value cleaning pipeline.

        Steps:
            1. Drop high-missingness columns
            2. Impute categorical fields with placeholders
            3. Remove rows with missing identifier values

        Args:
            dataframe: Raw CSAT dataframe.

        Returns:
            Cleaned dataframe with no remaining missing values in retained
            columns. Shape typically changes from (85907, 20) to (67675, 15).
        """

        try:
            logger.info("Starting missing value handling")

            dataframe = dataframe.copy()

            columns_present = set(dataframe.columns)

            columns_to_drop = [
                column
                for column in self.columns_to_drop
                if column in columns_present
            ]

            if columns_to_drop:
                logger.info(
                    f"Dropping columns with >"
                    f"{self.missing_threshold:.0%} missing values: "
                    f"{columns_to_drop}"
                )
                dataframe = dataframe.drop(columns=columns_to_drop)

            for column, fill_value in (
                self.categorical_imputations.items()
            ):
                if column in dataframe.columns:
                    logger.info(
                        f"Imputing missing values in "
                        f"'{column}' with '{fill_value}'"
                    )
                    dataframe[column] = (
                        dataframe[column]
                        .fillna(fill_value)
                    )

            identifier_columns = [
                column
                for column in self.identifier_columns_to_drop_rows
                if column in dataframe.columns
            ]

            if identifier_columns:
                rows_before = len(dataframe)
                dataframe = dataframe.dropna(
                    subset=identifier_columns
                )
                rows_dropped = rows_before - len(dataframe)

                logger.info(
                    f"Dropped {rows_dropped} rows with missing "
                    f"values in {identifier_columns}"
                )

            logger.info(
                f"Missing value handling completed. "
                f"Remaining shape: {dataframe.shape}"
            )

            return dataframe

        except Exception as e:
            logger.error("Missing value handling failed")
            raise CustomException(e, sys)
