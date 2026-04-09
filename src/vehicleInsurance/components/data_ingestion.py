import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from vehicleInsurance.data_access.vehicle_insurance_data import VehicleInsuranceData
from vehicleInsurance.entity.artifact_entity import DataIngestionArtifact
from vehicleInsurance.entity.config_entity import DataIngestionConfig
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logger.info("Exporting data from MongoDB")
            vehicle_insurance_data = VehicleInsuranceData()
            dataframe = vehicle_insurance_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logger.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Saving exported data to feature store: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            logger.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logger.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logger.info("Saving train and test sets")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
