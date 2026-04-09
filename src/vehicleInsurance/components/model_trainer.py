import os
import sys
from xgboost import XGBClassifier
from vehicleInsurance.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from vehicleInsurance.entity.config_entity import ModelTrainerConfig
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.ml.model.estimator import VehicleInsuranceModel
from vehicleInsurance.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    read_yaml_file,
    save_object,
    get_classification_score,
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            model_config = read_yaml_file(self.model_trainer_config.model_config_file_path)
            xgb_params = model_config.get("XGBClassifier", {}).get("params", {})
            xgb_clf = XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train, y_train)
            return xgb_clf
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(X_train, y_train)
            y_train_pred = model.predict(X_train)
            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            if train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                logger.info(
                    f"Trained model is not good enough. Expected: {self.model_trainer_config.expected_accuracy}, Got: {train_metric.f1_score}"
                )
                raise Exception(
                    f"Trained model is not good enough. Expected f1 score: {self.model_trainer_config.expected_accuracy}, Got: {train_metric.f1_score}"
                )

            y_test_pred = model.predict(X_test)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            vehicle_insurance_model = VehicleInsuranceModel(
                preprocessor=preprocessor, model=model
            )
            save_object(
                self.model_trainer_config.trained_model_file_path, obj=vehicle_insurance_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=test_metric,
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
