from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data_preprocessing.preprocessing import DataPreprocessor
from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.data_preprocessing.missing_value_handler import (
    MissingValueHandler,
)


def test_preprocessing_pipeline():
    """Verify preprocessing produces expected shape from raw CSV."""
    ingestion = DataIngestion()
    validation = DataValidation()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()

    raw_df = ingestion.load_csat_data()
    validation.validate_dataset(raw_df)

    engineered_df = feature_engineer.engineer_features(raw_df)
    # 15 cols after missing handling + 9 datetime-derived - 3 raw datetime cols
    assert engineered_df.shape == (67675, 21), (
        f"Expected (67675, 21), got {engineered_df.shape}"
    )

    X, y, fitted_preprocessor = preprocessor.preprocess_csat_data(
        raw_df
    )
    assert X.shape[0] == 67675
    assert len(y) == 67675
    assert set(y.unique()) == {1, 2, 3, 4, 5}
    assert X.shape[1] > 1000

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print("Preprocessing test passed")


def test_missing_value_handler():
    """Verify missing value handling matches notebook EDA."""
    ingestion = DataIngestion()
    handler = MissingValueHandler()

    raw_df = ingestion.load_csat_data()
    cleaned_df = handler.handle_missing_values(raw_df)

    assert cleaned_df.shape == (67675, 15)
    assert handler.columns_to_drop[0] == "connected_handling_time"
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("Missing value handler test passed")


if __name__ == "__main__":
    test_missing_value_handler()
    test_preprocessing_pipeline()
    print("\nAll preprocessing tests passed")
