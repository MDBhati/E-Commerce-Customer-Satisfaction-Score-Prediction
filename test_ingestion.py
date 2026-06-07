from src.data.ingestion import DataIngestion

ingestion = DataIngestion()


csat_df = ingestion.load_csat_data()

print(csat_df.head())

print(csat_df.info())