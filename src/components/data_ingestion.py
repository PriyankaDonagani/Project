
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/stud.csv')

            logging.info("Read the dataset as dataframe")

            # Create artifacts folder
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train test split initiated")

            # Split dataset into training and testing data
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save training data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save testing data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print("Training data path:", train_data)
    print("Testing data path:", test_data)
