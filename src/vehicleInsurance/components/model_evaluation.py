import sys
import pandas as pd
from vehicleInsurance.constants import *
from vehicleInsurance.entity.artifact_entity import (
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from vehicleInsurance.entity.config_entity import ModelEvaluationConfig
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.ml.model.estimator import VehicleInsuranceModel, S3Estimator, TargetValueMapping
from vehicleInsurance.utils.main_utils import load_object, read_yaml_file, get_classification_score


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def get_best_model(self):
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.model_key_path
            estimator = S3Estimator(
                bucket_name=bucket_name,
                model_path=model_path,
            )
            if estimator.is_model_present(model_path=model_path):
                return estimator
            return None
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def evaluate_model(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation")
            schema_config = read_yaml_file(str(DATA_VALIDATION_SCHEMA_FILE_PATH))
            target_column = schema_config["target_column"]
            drop_columns = schema_config.get("drop_columns", [])

            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            y_true = df[target_column]

            drop_cols = [target_column] + [c for c in drop_columns if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)

            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model = load_object(file_path=trained_model_file_path)
            trained_model_metric_artifact = self.model_trainer_artifact.metric_artifact

            best_model_metric_artifact = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_pred_best = best_model.predict(df)
                best_model_metric_artifact = get_classification_score(
                    y_true=y_true, y_pred=y_pred_best
                )
                improved_accuracy = (
                    trained_model_metric_artifact.f1_score
                    - best_model_metric_artifact.f1_score
                )
                is_model_accepted = (
                    improved_accuracy >= self.model_eval_config.changed_threshold_score
                )
            else:
                improved_accuracy = None
                is_model_accepted = True
                best_model_metric_artifact = trained_model_metric_artifact

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=self.model_eval_config.model_key_path,
                trained_model_path=trained_model_file_path,
                train_model_metric_artifact=trained_model_metric_artifact,
                best_model_metric_artifact=best_model_metric_artifact,
            )
            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
