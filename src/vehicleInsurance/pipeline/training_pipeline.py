import sys
from vehicleInsurance.components.data_ingestion import DataIngestion
from vehicleInsurance.components.data_validation import DataValidation
from vehicleInsurance.components.data_transformation import DataTransformation
from vehicleInsurance.components.model_trainer import ModelTrainer
from vehicleInsurance.components.model_evaluation import ModelEvaluation
from vehicleInsurance.components.model_pusher import ModelPusher
from vehicleInsurance.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("Starting data validation")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(f"Data validation completed: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def start_data_transformation(self, data_validation_artifact):
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"Data transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logger.info("Starting model training")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"Model training completed: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def start_model_evaluation(self, data_validation_artifact, model_trainer_artifact):
        try:
            model_eval_config = ModelEvaluationConfig()
            logger.info("Starting model evaluation")
            model_evaluation = ModelEvaluation(
                model_eval_config=model_eval_config,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.evaluate_model()
            logger.info(f"Model evaluation completed: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact):
        try:
            model_pusher_config = ModelPusherConfig()
            logger.info("Starting model pusher")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info(f"Model pusher completed: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def run_pipeline(self):
        try:
            logger.info("Starting training pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_validation_artifact, model_trainer_artifact
            )
            if not model_evaluation_artifact.is_model_accepted:
                logger.info("Trained model is not better than the best model.")
                return model_evaluation_artifact
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            logger.info("Training pipeline completed successfully")
            return model_pusher_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
