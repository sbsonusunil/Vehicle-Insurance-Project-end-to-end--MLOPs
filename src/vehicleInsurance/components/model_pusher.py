import sys
from vehicleInsurance.cloud_storage.aws_storage import SimpleStorageService
from vehicleInsurance.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from vehicleInsurance.entity.config_entity import ModelPusherConfig
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.ml.model.estimator import S3Estimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher")
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            s3_estimator = S3Estimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_path=self.model_pusher_config.s3_key,
            )
            s3_estimator.save_model(
                from_file=trained_model_path,
                key=self.model_pusher_config.s3_key,
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_key,
            )
            logger.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
