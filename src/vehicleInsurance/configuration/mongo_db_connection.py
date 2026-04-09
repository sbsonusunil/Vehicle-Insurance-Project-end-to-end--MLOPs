import os
import certifi
import pymongo
from vehicleInsurance.constants import *
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
import sys

ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name: str = DATA_INGESTION_DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.environ.get("MONGO_DB_URL")
                if mongo_db_url is None:
                    raise Exception("MONGO_DB_URL environment variable is not set.")
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url, tlsCAFile=ca
                )
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logger.info(f"MongoDB connection established to database: {database_name}")
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
