import os
from dataclasses import dataclass
from vehicleInsurance.constants import *


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, PIPELINE_NAME)
    timestamp: str = None

    def __post_init__(self):
        from datetime import datetime
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(self.artifact_dir, self.timestamp)


@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.data_ingestion_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, "data.csv"
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, "train.csv"
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, "test.csv"
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = DATA_INGESTION_DATABASE_NAME


@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.data_validation_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_train_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR, "train.csv"
        )
        self.valid_test_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR, "test.csv"
        )
        self.invalid_train_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR, "train.csv"
        )
        self.invalid_test_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR, "test.csv"
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.data_transformation_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            DATA_TRANSFORMATION_TRAIN_FILE_PATH,
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            DATA_TRANSFORMATION_TEST_FILE_PATH,
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_FILE_NAME,
        )


@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig = None

    def __post_init__(self):
        self.model_trainer_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_TRAINER_TRAINED_MODEL_NAME,
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path: str = str(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)


@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    model_key_path: str = MODEL_PUSHER_S3_KEY


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_key: str = MODEL_PUSHER_S3_KEY
