import sys
import os
sys.path.append(os.path.abspath("."))

from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

config   = DataIngestionConfig()
ingester = DataIngestion(config=config)

try:
    artifact = ingester.initiate_data_ingestion()
    print(f"Ingestion done:")
    print(f"  train path    : {artifact.train_data_path}")
    print(f"  test path     : {artifact.test_data_path}")
    print(f"  total records : {artifact.total_records}")

except Exception as e:
    print(f"Error: {e}")