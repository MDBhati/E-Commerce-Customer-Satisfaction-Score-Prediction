"""
Keras ANN classifier for CSAT score prediction (5-class classification).
"""

import sys

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from src.utils.logger import logger
from src.utils.exception import CustomException


class CSATClassifier:
    """Build and configure the neural network for CSAT prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        num_classes: int = 5,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def build_model(self) -> Sequential:
        """
        Build ANN architecture matching the project notebook.

        Architecture: Dense(128) -> Dropout -> Dense(64) -> Dropout
                      -> Dense(32) -> Dropout -> Dense(5, softmax)
        """

        try:
            logger.info(
                f"Building CSAT classifier with input_dim="
                f"{self.input_dim}"
            )

            layers = []

            for index, units in enumerate(self.hidden_layers):
                if index == 0:
                    layers.append(
                        Dense(
                            units,
                            input_dim=self.input_dim,
                            activation="relu",
                        )
                    )
                else:
                    layers.append(
                        Dense(units, activation="relu")
                    )
                layers.append(
                    Dropout(self.dropout_rate)
                )

            layers.append(
                Dense(
                    self.num_classes,
                    activation="softmax",
                )
            )

            model = Sequential(layers)
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["accuracy"],
            )

            self.model = model

            logger.info("CSAT classifier built successfully")

            return model

        except Exception as e:
            logger.error("Failed to build CSAT classifier")
            raise CustomException(e, sys)

    @staticmethod
    def get_callbacks(
        early_stopping_config: dict | None = None,
        reduce_lr_config: dict | None = None,
    ) -> list:
        """Create Keras training callbacks."""

        early_stopping_config = early_stopping_config or {
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True,
        }
        reduce_lr_config = reduce_lr_config or {
            "monitor": "val_loss",
            "factor": 0.2,
            "patience": 5,
            "min_lr": 0.00001,
        }

        return [
            EarlyStopping(**early_stopping_config),
            ReduceLROnPlateau(**reduce_lr_config),
        ]
