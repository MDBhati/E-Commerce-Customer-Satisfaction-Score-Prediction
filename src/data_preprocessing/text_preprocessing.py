"""
Text preprocessing for Customer Remarks in the CSAT dataset.

Customer Remarks is 66.54% missing in the raw CSV. The notebook strategy
imputes missing values with a placeholder and excludes the column from
modeling (free text is not one-hot encoded).
"""

import sys

import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException


class TextPreprocessor:
    """Handle missing and basic cleanup for Customer Remarks."""

    def __init__(
        self,
        text_column: str = "Customer Remarks",
        missing_placeholder: str = "No Remarks",
    ):
        self.text_column = text_column
        self.missing_placeholder = missing_placeholder

    def preprocess_text(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Impute missing Customer Remarks and strip whitespace.

        Args:
            dataframe: Raw or partially cleaned CSAT dataframe.

        Returns:
            Dataframe with imputed text column.
        """

        try:
            logger.info("Preprocessing Customer Remarks text column")

            dataframe = dataframe.copy()

            if self.text_column not in dataframe.columns:
                logger.info(
                    f"Column '{self.text_column}' not found; skipping"
                )
                return dataframe

            dataframe[self.text_column] = (
                dataframe[self.text_column]
                .fillna(self.missing_placeholder)
                .astype(str)
                .str.strip()
            )

            empty_mask = dataframe[self.text_column] == ""
            if empty_mask.any():
                dataframe.loc[
                    empty_mask, self.text_column
                ] = self.missing_placeholder

            logger.info(
                f"Text preprocessing completed for "
                f"'{self.text_column}'"
            )

            return dataframe

        except Exception as e:
            logger.error("Text preprocessing failed")
            raise CustomException(e, sys)

    def has_remarks(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """Return a binary flag indicating whether remarks were provided."""

        if self.text_column not in dataframe.columns:
            return pd.Series(0, index=dataframe.index)

        return (
            dataframe[self.text_column]
            .fillna(self.missing_placeholder)
            .ne(self.missing_placeholder)
            .astype(int)
        )
