"""
End-to-end training pipeline for CSAT score prediction.

Loads eCommerce_Customer_support_data.csv, validates, preprocesses,
trains a Keras ANN, and saves model artifacts.
"""

import json
import os
import sys

import joblib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data_preprocessing.preprocessing import DataPreprocessor
from src.models.csat_classifier import CSATClassifier
from src.utils.logger import logger
from src.utils.exception import CustomException


class TrainingPipeline:
    """Orchestrate data loading, preprocessing, and model training."""

    def __init__(
        self,
        paths_config: str = "configs/paths.yaml",
        training_config: str = "configs/training_config.yaml",
    ):
        with open(paths_config, "r") as file:
            self.paths = yaml.safe_load(file)

        with open(training_config, "r") as file:
            self.config = yaml.safe_load(file)

        self.ingestion = DataIngestion(paths_config)
        self.validation = DataValidation()
        self.preprocessor = DataPreprocessor()

    def _ensure_artifact_dirs(self):
        """Create artifact directories if they do not exist."""

        for key in ("model", "preprocessor", "training_history"):
            path = self.paths["artifact_paths"][key]
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def run(self):
        """
        Execute the full training pipeline.

        Returns:
            Dict with training metrics and artifact paths.
        """

        try:
            logger.info("Starting CSAT training pipeline")
            self._ensure_artifact_dirs()

            # 1. Load and validate raw data
            raw_df = self.ingestion.load_csat_data()
            validation_summary = self.validation.validate_dataset(
                raw_df
            )
            logger.info(
                f"Validation summary: {validation_summary}"
            )

            # 2. Preprocess features
            X, y, fitted_preprocessor = (
                self.preprocessor.preprocess_csat_data(raw_df)
            )

            # Keras requires dense float arrays
            if hasattr(X, "toarray"):
                X = X.toarray()

            X = X.astype("float32")

            # 3. Train/test split
            train_cfg = self.config["training"]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=train_cfg["test_size"],
                random_state=train_cfg["random_state"],
                stratify=y,
            )

            y_train_cat = to_categorical(y_train - 1)
            y_test_cat = to_categorical(y_test - 1)

            # 4. Build and train model
            model_cfg = self.config["model"]
            classifier = CSATClassifier(
                input_dim=X_train.shape[1],
                hidden_layers=model_cfg["hidden_layers"],
                dropout_rate=model_cfg["dropout_rate"],
                num_classes=model_cfg["num_classes"],
                optimizer=model_cfg["optimizer"],
                loss=model_cfg["loss"],
            )
            model = classifier.build_model()

            callbacks = CSATClassifier.get_callbacks(
                early_stopping_config=self.config["callbacks"][
                    "early_stopping"
                ],
                reduce_lr_config=self.config["callbacks"][
                    "reduce_lr"
                ],
            )

            history = model.fit(
                X_train,
                y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=train_cfg["epochs"],
                batch_size=train_cfg["batch_size"],
                callbacks=callbacks,
                verbose=1,
            )

            # 5. Evaluate
            loss, accuracy = model.evaluate(
                X_test, y_test_cat, verbose=0
            )
            logger.info(
                f"Test Loss: {loss:.4f}, "
                f"Test Accuracy: {accuracy:.4f}"
            )

            # 6. Save artifacts
            model_path = self.paths["artifact_paths"]["model"]
            preprocessor_path = self.paths["artifact_paths"][
                "preprocessor"
            ]
            history_path = self.paths["artifact_paths"][
                "training_history"
            ]

            model.save(model_path)
            joblib.dump(fitted_preprocessor, preprocessor_path)

            history_dict = {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            }
            with open(history_path, "w") as file:
                json.dump(history_dict, file, indent=2)

            result = {
                "test_loss": float(loss),
                "test_accuracy": float(accuracy),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "num_features": int(X.shape[1]),
                "model_path": model_path,
                "preprocessor_path": preprocessor_path,
                "history_path": history_path,
            }

            logger.info(
                f"Training pipeline completed: {result}"
            )

            return result

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
    print("\nTraining completed successfully")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model saved to: {results['model_path']}")
