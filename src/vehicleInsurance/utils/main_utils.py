import os
import sys
import dill
import numpy as np
import yaml
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def write_yaml_file(
    file_path: str, content: object, replace: bool = False
) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info("Exited the save_object method of utils")
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)


def get_classification_score(y_true, y_pred):
    try:
        from vehicleInsurance.entity.artifact_entity import ClassificationMetricArtifact
        from sklearn.metrics import f1_score, precision_score, recall_score

        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
        )
        return classification_metric
    except Exception as e:
        raise VehicleInsuranceException(e, sys)
