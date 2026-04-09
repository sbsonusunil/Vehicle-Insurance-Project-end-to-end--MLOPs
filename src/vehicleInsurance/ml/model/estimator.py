import os
import sys
import dill
from vehicleInsurance.cloud_storage.aws_storage import SimpleStorageService
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger


class TargetValueMapping:
    def __init__(self):
        self.Response: dict = {0: 0, 1: 1}

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class VehicleInsuranceModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise VehicleInsuranceException(e, sys)


class S3Estimator:
    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.storage = SimpleStorageService()

    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.storage.s3_key_path_available(
                bucket_name=self.bucket_name, s3_key=model_path
            )
        except Exception:
            return False

    def load_model(self, model_path: str) -> VehicleInsuranceModel:
        try:
            model_file = "loaded_model.pkl"
            self.storage.download_object(
                key=model_path,
                bucket_name=self.bucket_name,
                output_file_path=model_file,
            )
            with open(model_file, "rb") as f:
                model = dill.load(f)
            os.remove(model_file)
            return model
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def save_model(self, from_file: str, key: str) -> None:
        try:
            self.storage.upload_file(
                from_filename=from_file,
                to_filename=key,
                bucket_name=self.bucket_name,
                remove=False,
            )
            logger.info(f"Model uploaded to s3://{self.bucket_name}/{key}")
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def predict(self, dataframe):
        try:
            model = self.load_model(self.model_path)
            return model.predict(dataframe)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
