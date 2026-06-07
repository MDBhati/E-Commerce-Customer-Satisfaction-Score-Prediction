import sys

import pandas as pd
import yaml

from src.data.schema import CSAT_SCHEMA, TARGET_COLUMN, VALID_CSAT_SCORES
from src.utils.logger import logger
from src.utils.exception import CustomException


class DataIngestion:
    """Load and prepare the raw CSAT CSV dataset."""

    def __init__(self, config_path="configs/paths.yaml"):
        with open(config_path, "r") as file:
            self.paths = yaml.safe_load(file)

    def _coerce_dtypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Align dataframe dtypes with CSAT_SCHEMA after CSV load."""
        dataframe = dataframe.copy()

        for column, expected_dtype in CSAT_SCHEMA.items():
            if column not in dataframe.columns:
                continue

            if expected_dtype == "object":
                dataframe[column] = dataframe[column].astype(str)
                dataframe[column] = dataframe[column].replace(
                    {"nan": pd.NA, "None": pd.NA, "": pd.NA}
                )
            elif expected_dtype == "float64":
                dataframe[column] = pd.to_numeric(
                    dataframe[column], errors="coerce"
                )
            elif expected_dtype == "int64":
                dataframe[column] = pd.to_numeric(
                    dataframe[column], errors="coerce"
                ).astype("Int64")

        return dataframe

    def load_csat_data(self) -> pd.DataFrame:
        """
        Load eCommerce_Customer_support_data.csv and coerce dtypes.

        Returns:
            Raw CSAT dataframe with schema-aligned dtypes.
        """

        try:
            logger.info("Loading csat dataset")

            csv_path = self.paths["data_paths"]["csat_data"]
            csat_df = pd.read_csv(csv_path)
            csat_df.columns = csat_df.columns.str.strip()

            csat_df = self._coerce_dtypes(csat_df)

            logger.info(
                f"CSAT dataset loaded successfully "
                f"with shape {csat_df.shape}"
            )

            return csat_df

        except Exception as e:
            logger.error("Failed to load CSAT dataset")
            raise CustomException(e, sys)

    def save_processed_data(
        self,
        dataframe: pd.DataFrame,
    ) -> str:
        """Persist cleaned feature dataframe to disk."""

        try:
            output_path = self.paths["data_paths"]["processed_data"]
            import os

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True,
            )
            dataframe.to_csv(output_path, index=False)

            logger.info(
                f"Processed data saved to {output_path}"
            )

            return output_path

        except Exception as e:
            logger.error("Failed to save processed data")
            raise CustomException(e, sys)
