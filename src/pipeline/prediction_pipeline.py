"""
Inference pipeline for CSAT score prediction.

Loads saved model and preprocessor artifacts to predict CSAT scores
on new customer support records.
"""

import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.models import load_model

from src.data_preprocessing.preprocessing import DataPreprocessor
from src.utils.logger import logger
from src.utils.exception import CustomException


class PredictionPipeline:
    """Load artifacts and predict CSAT scores from raw CSV rows."""

    def __init__(
        self,
        paths_config: str = "configs/paths.yaml",
    ):
        with open(paths_config, "r") as file:
            self.paths = yaml.safe_load(file)

        self.preprocessor = DataPreprocessor()
        self.model = None
        self.fitted_preprocessor = None

    def load_artifacts(self):
        """Load trained Keras model and fitted sklearn preprocessor."""

        try:
            model_path = self.paths["artifact_paths"]["model"]
            preprocessor_path = self.paths["artifact_paths"][
                "preprocessor"
            ]

            logger.info("Loading prediction artifacts")

            self.model = load_model(model_path)
            self.fitted_preprocessor = joblib.load(
                preprocessor_path
            )

            logger.info("Prediction artifacts loaded successfully")

        except Exception as e:
            logger.error("Failed to load prediction artifacts")
            raise CustomException(e, sys)

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict CSAT scores (1-5) for raw input records.

        Args:
            dataframe: Raw CSAT dataframe matching CSV schema.

        Returns:
            Array of predicted CSAT scores as integers.
        """

        try:
            if self.model is None or self.fitted_preprocessor is None:
                self.load_artifacts()

            X_transformed, _ = (
                self.preprocessor.transform_csat_data(
                    dataframe,
                    self.fitted_preprocessor,
                )
            )

            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            X_transformed = X_transformed.astype("float32")

            probabilities = self.model.predict(
                X_transformed, verbose=0
            )
            predictions = np.argmax(probabilities, axis=1) + 1

            logger.info(
                f"Generated {len(predictions)} CSAT predictions"
            )

            return predictions

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)

    def predict_proba(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Return class probabilities for each CSAT score (1-5)."""

        try:
            if self.model is None or self.fitted_preprocessor is None:
                self.load_artifacts()

            X_transformed, _ = (
                self.preprocessor.transform_csat_data(
                    dataframe,
                    self.fitted_preprocessor,
                )
            )

            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            X_transformed = X_transformed.astype("float32")

            return self.model.predict(
                X_transformed, verbose=0
            )

        except Exception as e:
            logger.error("Probability prediction failed")
            raise CustomException(e, sys)
