import sys

import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.data.schema import (
    CSAT_SCHEMA,
    TARGET_COLUMN,
    VALID_CSAT_SCORES,
)


class DataValidation:
    """Validate raw CSAT data against the expected schema."""

    def validate_columns(
        self,
        dataframe: pd.DataFrame,
        required_schema: dict,
    ):

        try:
            logger.info("Validating dataset columns")

            dataframe_columns = set(dataframe.columns)
            required_columns = set(required_schema.keys())

            missing_columns = required_columns - dataframe_columns

            if missing_columns:
                raise ValueError(
                    f"Missing columns: {missing_columns}"
                )

            logger.info("Column validation successful")

            return True

        except Exception as e:
            logger.error("Column validation failed")
            raise CustomException(e, sys)

    def validate_dtypes(
        self,
        dataframe: pd.DataFrame,
        required_schema: dict,
    ):

        try:
            logger.info("Validating datatypes")

            for column, expected_dtype in required_schema.items():
                actual_dtype = str(dataframe[column].dtype)

                # Nullable integer columns may appear as Int64
                if (
                    expected_dtype == "int64"
                    and actual_dtype.startswith("Int")
                ):
                    continue

                if actual_dtype != expected_dtype:
                    raise ValueError(
                        f"Datatype mismatch in column "
                        f"{column}. "
                        f"Expected: {expected_dtype}, "
                        f"Found: {actual_dtype}"
                    )

            logger.info("Datatype validation successful")

            return True

        except Exception as e:
            logger.error("Datatype validation failed")
            raise CustomException(e, sys)

    def validate_target(
        self,
        dataframe: pd.DataFrame,
    ):

        try:
            logger.info("Validating CSAT Score target column")

            if TARGET_COLUMN not in dataframe.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found"
                )

            null_count = dataframe[TARGET_COLUMN].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"Target column has {null_count} missing values"
                )

            unique_scores = set(
                dataframe[TARGET_COLUMN].dropna().astype(int).unique()
            )
            invalid_scores = unique_scores - VALID_CSAT_SCORES

            if invalid_scores:
                raise ValueError(
                    f"Invalid CSAT scores found: {invalid_scores}"
                )

            logger.info("Target validation successful")

            return True

        except Exception as e:
            logger.error("Target validation failed")
            raise CustomException(e, sys)

    def check_missing_values(
        self,
        dataframe: pd.DataFrame,
    ):

        try:
            logger.info("Checking missing values")

            missing_values = dataframe.isnull().sum()

            logger.info(
                f"Missing values summary: "
                f"{missing_values.to_dict()}"
            )

            return missing_values

        except Exception as e:
            logger.error("Missing value check failed")
            raise CustomException(e, sys)

    def check_duplicates(
        self,
        dataframe: pd.DataFrame,
    ):

        try:
            logger.info("Checking duplicate rows")

            duplicate_count = dataframe.duplicated().sum()

            logger.info(
                f"Duplicate rows found: {duplicate_count}"
            )

            return duplicate_count

        except Exception as e:
            logger.error("Duplicate check failed")
            raise CustomException(e, sys)

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        required_schema: dict = CSAT_SCHEMA,
    ):
        """
        Run all validation checks on the raw CSAT dataset.

        Returns:
            Dict with validation summary statistics.
        """

        try:
            logger.info("Starting full dataset validation")

            self.validate_columns(dataframe, required_schema)
            self.validate_dtypes(dataframe, required_schema)
            self.validate_target(dataframe)

            missing_values = self.check_missing_values(dataframe)
            duplicate_count = self.check_duplicates(dataframe)

            summary = {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "missing_values": missing_values.to_dict(),
                "duplicate_rows": int(duplicate_count),
            }

            logger.info("Full dataset validation completed")

            return summary

        except Exception as e:
            logger.error("Full dataset validation failed")
            raise CustomException(e, sys)
