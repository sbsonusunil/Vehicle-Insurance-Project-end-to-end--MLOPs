import os
import sys
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, jsonify
from vehicleInsurance.pipeline.training_pipeline import TrainingPipeline
from vehicleInsurance.pipeline.prediction_pipeline import (
    PredictionPipeline,
    VehicleData,
    VehicleDataFrame,
)
from vehicleInsurance.exception import VehicleInsuranceException
from vehicleInsurance.logger import logger
from vehicleInsurance.constants import APP_HOST, APP_PORT

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET"])
def train():
    try:
        logger.info("Training pipeline triggered")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return jsonify({"status": "success", "message": "Training pipeline completed successfully"})
    except VehicleInsuranceException as e:
        logger.error(f"Training pipeline failed: {e}")
        return jsonify({"status": "failure", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"status": "failure", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = request.form.get("Gender")
        age = int(request.form.get("Age"))
        driving_license = int(request.form.get("Driving_License"))
        region_code = float(request.form.get("Region_Code"))
        previously_insured = int(request.form.get("Previously_Insured"))
        vehicle_age = request.form.get("Vehicle_Age")
        vehicle_damage = request.form.get("Vehicle_Damage")
        annual_premium = float(request.form.get("Annual_Premium"))
        policy_sales_channel = float(request.form.get("Policy_Sales_Channel"))
        vintage = int(request.form.get("Vintage"))

        vehicle_data = VehicleData(
            gender=gender,
            age=age,
            driving_license=driving_license,
            region_code=region_code,
            previously_insured=previously_insured,
            vehicle_age=vehicle_age,
            vehicle_damage=vehicle_damage,
            annual_premium=annual_premium,
            policy_sales_channel=policy_sales_channel,
            vintage=vintage,
        )

        vehicle_df = VehicleDataFrame(vehicle_data=vehicle_data)
        df = vehicle_df.get_vehicle_input_data_frame()

        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(df)

        return render_template("index.html", result=result)
    except VehicleInsuranceException as e:
        logger.error(f"Prediction failed: {e}")
        return render_template("index.html", result=f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return render_template("index.html", result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True)
