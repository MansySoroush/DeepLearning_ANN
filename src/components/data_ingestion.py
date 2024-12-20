import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.configs.configurations import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebooks/data/cleaned_data.csv')
            logging.info('Read the dataset as data-frame')

            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.cleaned_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(self.ingestion_config.cleaned_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
                
if __name__ == "__main__":
    obj=DataIngestion()
    raw_data = obj.initiate_data_ingestion()



