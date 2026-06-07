"""
Entry point for the E-Commerce CSAT Prediction project.

Usage:
    python main.py validate     — validate raw CSV data
    python main.py preprocess   — run preprocessing and show feature shape
    python main.py train        — train model and save artifacts
    python main.py predict      — predict on sample rows (requires trained model)
    python main.py serve        — start FastAPI inference server
"""

import argparse
import sys

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data_preprocessing.missing_value_handler import (
    MissingValueHandler,
)
from src.data_preprocessing.preprocessing import DataPreprocessor
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline


def validate_data():
    """Validate the raw CSAT CSV dataset."""
    ingestion = DataIngestion()
    validation = DataValidation()

    df = ingestion.load_csat_data()
    summary = validation.validate_dataset(df)

    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print("\nValidation completed successfully")


def preprocess_data():
    """Run preprocessing pipeline and print feature matrix shape."""
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor()

    df = ingestion.load_csat_data()
    X, y, _ = preprocessor.preprocess_csat_data(df)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print("\nPreprocessing completed successfully")


def train_model():
    """Train the CSAT classifier and save artifacts."""
    pipeline = TrainingPipeline()
    results = pipeline.run()

    print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Features: {results['num_features']}")
    print(f"Model saved to: {results['model_path']}")


def predict_sample():
    """Run prediction on the first 10 rows of the dataset."""
    ingestion = DataIngestion()
    pipeline = PredictionPipeline()

    df = ingestion.load_csat_data().head(10)
    actual = df["CSAT Score"].values
    predicted = pipeline.predict(df)

    print("Sample predictions (actual -> predicted):")
    for act, pred in zip(actual, predicted):
        print(f"  {act} -> {pred}")


def serve_api():
    """Start the FastAPI inference server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def generate_missing_values_plot():
    """Generate missing values EDA plot from raw data."""
    ingestion = DataIngestion()
    handler = MissingValueHandler()

    df = ingestion.load_csat_data()
    plot_path = handler.save_missing_values_plot(df)
    print(f"Missing values plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce CSAT Score Prediction"
    )
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "preprocess",
            "train",
            "predict",
            "plot-missing",
            "serve",
        ],
        help="Pipeline command to run",
    )

    args = parser.parse_args()

    commands = {
        "validate": validate_data,
        "preprocess": preprocess_data,
        "train": train_model,
        "predict": predict_sample,
        "plot-missing": generate_missing_values_plot,
        "serve": serve_api,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
