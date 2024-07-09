import logging
import joblib
from flask import Flask, request, jsonify
import os
from comet_ml import API
import pandas as pd


app = Flask(__name__)
try:
    LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    COMET_ML_PROJECT_NAME = os.getenv(
        "COMET_ML_PROJECT_NAME",
        "ift-6758-b03-project",
    )
    COMET_ML_WORKSPACE = os.getenv(
        "COMET_ML_WORKSPACE",
        "duyhung2201",
    )
except Exception as e:
    print(e)
    app.logger.info("Enviroment error")


class ModelHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.comet_ml_instance = API(api_key=COMET_API_KEY)
        self.model_path = None
        self.model = None
        self.fileNameParser()

    def load_model(self, model_name: str = None):
        # Load the model
        try:
            if model_name:
                self.model_name = model_name
                self.fileNameParser()
            version = "1.40.0" if self.model_name == "lr-distance" else "1.19.0"
            if not os.path.exists(self.model_path):
                self.comet_ml_instance.download_registry_model(
                    workspace=COMET_ML_WORKSPACE,
                    version=version,
                    registry_name=self.model_name,
                    output_path="models",
                )
                self.model = joblib.load(self.model_path)
                app.logger.info(f"Model {self.model_name} downloaded and loaded")
            else:
                self.model = joblib.load(self.model_path)
                app.logger.info(f"Model {self.model_name} loaded")
        except Exception as e:
            app.logger.error(f"Error loading model {self.model_name}: {e}")

    def predict(self, data: dict):
        # Predict
        try:
            data = self.dataParser(data)
            prediction = self.model.predict_proba(data)
            return prediction
        except Exception as e:
            app.logger.error(f"Error predicting {self.model_name}: {e}")
            return {}
        pass

    def fileNameParser(self):
        # Parse the file name
        self.model_path = (
            "log_reg_model['net_distance'].joblib"
            if self.model_name == "lr-distance"
            else "log_reg_model['shot_angle', 'net_distance'].joblib"
        )
        if not os.path.exists("models"):
            os.mkdir("models")
        self.model_path = os.path.join("models", self.model_path)

    def dataParser(self, data: dict):
        # Parse the data
        try:
            data = pd.DataFrame.from_dict(data, orient="index").T
            if data.empty:
                app.logger.info(f"Empty data for {self.model_name}")
                raise Exception("Empty data")
            return data.to_numpy()
        except Exception as e:
            app.logger.error(f"Error parsing data {self.model_name}: {e}")


model_handler = ModelHandler("lr-shot-distance")


def before_first_request():
    try:
        logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
        # Load the model
        model_handler.load_model()
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")


with app.app_context():
    before_first_request()


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    with open("flask.log", "r") as f:
        logs = "".join(f.readlines())
    return logs


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    resp = request.get_json()
    model_name = resp["model"]
    model_handler.load_model(model_name=model_name)
    return jsonify("Model downloaded and loaded")


@app.route("/predict", methods=["POST"])
def predict(resp):
    # resp = request.get_json()
    prediction = model_handler.predict(resp)
    return jsonify(prediction.tolist())


@app.route("/", methods=["GET"])
def root():
    return jsonify("Welcome to the NHL shot prediction API")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="7777", load_dotenv=True)
