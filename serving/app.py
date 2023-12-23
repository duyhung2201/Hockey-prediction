"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
from comet_ml import API, ExistingExperiment, Experiment
import logging
import joblib
import json


app = Flask(__name__)

try:
    LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
    COMET_API_KEY = os.getenv("COMET_API_KEY")
except Exception as e:
    print(e)
    app.logger.info("Environment error")

model = None
model_name = None


def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global model, model_name

    try:
        # TODO: setup basic logging configuration
        logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
        # TODO: any other initialization before the first request (e.g. load default model)

        json = {
            "workspace": "duyhung2201",
            "model": "logisitc_regression_-shot_angle-net_distance",
            "version": "1.12.0",
        }
        model_name = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
        model_file_path = "log_reg_model['shot_angle', 'net_distance'].joblib"

        if not os.path.isfile(model_file_path):
            api = API(COMET_API_KEY)
            api.download_registry_model(
                json["workspace"],
                json["model"],
                json["version"],
                output_path="./",
                expand=True,
            )

            model = joblib.load(model_file_path)
            app.logger.info(
                f"succesfully downloaded and loaded default model ({model_name})"
            )
        else:
            model = joblib.load(model_file_path)
            app.logger.info(f"succesfully loaded default model ({model_name})")

    except Exception as e:
        app.logger.error(f"Error in initialization: {e}")


with app.app_context():
    before_first_request()


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    try:
        with open(LOG_FILE, "r") as f:
            log_data = f.readlines()
        response = [x.strip() for x in log_data]
    except Exception as e:
        app.logger.error(f"Error in reading log file: {e}")
        response = f"Error: {e}"
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data
    global model, model_name
    data = json.loads(request.data.decode())
    app.logger.info(data)
    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.
    # eg: app.logger.info(<LOG STRING>)

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    response = {}
    # json = {"workspace": "duyhung2201", "model": "logisitc_regression_-net_distance", "version":"1.25.0"}

    try:
        model_name = f'{data["workspace"]}_{data["model"]}_{data["version"]}'
        requested_model_path = "log_reg_model['net_distance'].joblib"

        if os.path.isfile(requested_model_path):
            model = joblib.load(requested_model_path)
            app.logger.info(f"Successfully loaded requested model ({model_name})")
            response = {
                "status": "success",
                "message": f"Model ({model_name}) is already downloaded.",
            }
        else:
            api = API(os.getenv("COMET_API_KEY"))
            api.download_registry_model(
                data["workspace"],
                data["model"],
                data["version"],
                output_path="./",
                expand=True,
            )

            model = joblib.load(requested_model_path)
            app.logger.info(
                f"Successfully downloaded and loaded requested model ({model_name})"
            )
            response = {
                "response = {'status': 'success', 'message': f'Model ({model_name}) downloaded and loaded.'}"
            }

    except Exception as e:
        app.logger.error(f"Error in downloading or loading the model: {e}")
        response = {
            "status": "error",
            "message": f"Error in downloading or loading the model: {e}",
        }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    try:
        # Get POST json data
        resp = json.loads(request.data.decode())
        app.logger.info(json)
        global model
        global model_name

        input_data = features(model_name, resp)
        # TODO:
        if hasattr(model, "predict"):
            predictions = model.predict(input_data)
            response = {
                "status": "Success",
                "Predictions": predictions.tolist()[0],
                "Model Used": model_name,
            }
            app.logger.info("Success in making the predictions.")

        else:
            response = {"status": "error: Model ia not loaded"}
            app.logger.error("Model is not able to do predictions")

    except Exception as e:
        response = {"status": "error", "message": f"Error in prediction: {e}"}
        app.logger.error(f"Error in prediction: {e}")

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


def features(model_name, resp):
    d = None
    if model_name == "duyhung2201_logisitc_regression_-shot_angle-net_distance_1.12.0":
        features_name = ["shot_angle", "net_distance"]
        d = [[resp[features_name[0]], resp[features_name[1]]]]

    if model_name == "duyhung2201_logisitc_regression_-net_distance_1.25.0":
        features_name = ["net_distance"]
        d = [[resp[features_name[0]]]]

    return d


@app.route("/", methods=["GET"])
def root():
    return jsonify("Voilla!!! Welcome to my API!!!!")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7777))
    app.run(debug=True, host="0.0.0.0", port=port)
