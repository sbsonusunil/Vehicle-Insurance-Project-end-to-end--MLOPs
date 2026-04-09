import os
import sys
import pandas as pd
from vehicleInsurance.cloud_storage.aws_storage import SimpleStorageService
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.ml.model.estimator import VehicleInsuranceModel, S3Estimator
from vehicleInsurance.constants import MODEL_PUSHER_BUCKET_NAME, MODEL_PUSHER_S3_KEY


class VehicleData:
    def __init__(
        self,
        gender: str,
        age: int,
        driving_license: int,
        region_code: float,
        previously_insured: int,
        vehicle_age: str,
        vehicle_damage: str,
        annual_premium: float,
        policy_sales_channel: float,
        vintage: int,
    ):
        self.gender = gender
        self.age = age
        self.driving_license = driving_license
        self.region_code = region_code
        self.previously_insured = previously_insured
        self.vehicle_age = vehicle_age
        self.vehicle_damage = vehicle_damage
        self.annual_premium = annual_premium
        self.policy_sales_channel = policy_sales_channel
        self.vintage = vintage

    def get_data_as_dict(self):
        try:
            input_data = {
                "Gender": [self.gender],
                "Age": [self.age],
                "Driving_License": [self.driving_license],
                "Region_Code": [self.region_code],
                "Previously_Insured": [self.previously_insured],
                "Vehicle_Age": [self.vehicle_age],
                "Vehicle_Damage": [self.vehicle_damage],
                "Annual_Premium": [self.annual_premium],
                "Policy_Sales_Channel": [self.policy_sales_channel],
                "Vintage": [self.vintage],
            }
            return input_data
        except Exception as e:
            raise VehicleInsuranceException(e, sys)


class VehicleDataFrame:
    def __init__(self, vehicle_data: VehicleData):
        self.vehicle_data = vehicle_data

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        try:
            vehicle_data_dict = self.vehicle_data.get_data_as_dict()
            return pd.DataFrame(vehicle_data_dict)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = MODEL_PUSHER_BUCKET_NAME
        self.model_path = MODEL_PUSHER_S3_KEY
        self.storage = SimpleStorageService()

    def get_model(self) -> VehicleInsuranceModel:
        try:
            estimator = S3Estimator(
                bucket_name=self.bucket_name,
                model_path=self.model_path,
            )
            return estimator.load_model(model_path=self.model_path)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logger.info("Starting prediction pipeline")
            model = self.get_model()
            prediction = model.predict(dataframe)
            result = "Would be Interested" if prediction[0] == 1 else "Would Not be Interested"
            logger.info(f"Prediction: {result}")
            return result
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
