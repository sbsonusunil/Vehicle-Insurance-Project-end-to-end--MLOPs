import os
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from vehicleInsurance.constants import *
from vehicleInsurance.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from vehicleInsurance.entity.config_entity import DataTransformationConfig
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.utils.main_utils import (
    read_yaml_file,
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(str(DATA_VALIDATION_SCHEMA_FILE_PATH))
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logger.info("Creating data transformer object")
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            target_column = self._schema_config["target_column"]
            drop_columns = self._schema_config.get("drop_columns", [])

            input_feature_train_df = train_df.drop(
                columns=[target_column] + [c for c in drop_columns if c in train_df.columns],
                axis=1,
            )
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(
                columns=[target_column] + [c for c in drop_columns if c in test_df.columns],
                axis=1,
            )
            target_feature_test_df = test_df[target_column]

            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[
                transformed_input_test_feature, np.array(target_feature_test_df)
            ]

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, array=test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_object,
            )

            logger.info("Data transformation completed")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logger.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
