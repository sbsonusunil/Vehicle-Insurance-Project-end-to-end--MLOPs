from vehicleInsurance.pipeline.training_pipeline import TrainingPipeline
from vehicleInsurance.logger import logger
from vehicleInsurance.exception import VehicleInsuranceException
import sys

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline from demo.py")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logger.info("Training pipeline completed successfully")
    except VehicleInsuranceException as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
