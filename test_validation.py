from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data.schema import CSAT_SCHEMA


def main():
    ingestion = DataIngestion()
    validation = DataValidation()

    csat_df = ingestion.load_csat_data()
    summary = validation.validate_dataset(csat_df, CSAT_SCHEMA)

    print(f"\nRows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print("\nMissing values (top columns):")
    missing = summary["missing_values"]
    for col, count in sorted(
        missing.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {col}: {count:,}")

    print("\nValidation completed successfully")


if __name__ == "__main__":
    main()
