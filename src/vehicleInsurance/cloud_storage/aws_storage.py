import os
import sys
import boto3
from botocore.exceptions import ClientError
from vehicleInsurance.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_REGION_NAME,
)
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger


class SimpleStorageService:
    def __init__(self):
        try:
            self.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=os.environ.get(AWS_ACCESS_KEY_ID_ENV_KEY),
                aws_secret_access_key=os.environ.get(AWS_SECRET_ACCESS_KEY_ENV_KEY),
                region_name=AWS_REGION_NAME,
            )
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get(AWS_ACCESS_KEY_ID_ENV_KEY),
                aws_secret_access_key=os.environ.get(AWS_SECRET_ACCESS_KEY_ENV_KEY),
                region_name=AWS_REGION_NAME,
            )
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:
        try:
            logger.info(f"Uploading {from_filename} to s3://{bucket_name}/{to_filename}")
            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )
            logger.info("Upload complete")
            if remove:
                os.remove(from_filename)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def download_object(
        self, key: str, bucket_name: str, output_file_path: str
    ) -> None:
        try:
            logger.info(f"Downloading s3://{bucket_name}/{key} to {output_file_path}")
            dir_path = os.path.dirname(output_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self.s3_resource.meta.client.download_file(bucket_name, key, output_file_path)
            logger.info("Download complete")
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def get_object(self, key: str, bucket_name: str) -> object:
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            return response["Body"]
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def create_bucket(self, bucket_name: str, region: str = AWS_REGION_NAME) -> None:
        try:
            if region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            logger.info(f"Bucket {bucket_name} created successfully.")
        except ClientError as e:
            raise VehicleInsuranceException(e, sys)

    def bucket_exists(self, bucket_name: str) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            return False
