import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        self.model = "lr-shot-distance"
        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        if X.empty:
            raise ValueError("Input dataframe is empty")

        try:
            df = X.copy()
            filtered_X, json_data = self.filter_events(df)
            response = requests.post(
                f"{self.base_url}/predict",
                data=json_data,
                headers={"Content-type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"Prediction request failed: {response.text}")
                return pd.DataFrame()

            predictions = response.json()
            df["goal_prob"] = None
            df.loc[filtered_X.index, "goal_prob"] = [probs[1] for probs in predictions]

            return df
        except requests.RequestException as e:
            logger.error(f"Network or server error occurred: {e}")
            return pd.DataFrame()
        except ValueError as e:
            logger.error(f"Error parsing response: {e}")
            return pd.DataFrame()

    def logs(self) -> dict:
        """Get server logs"""
        try:
            response = requests.get(f"{self.base_url}/logs")

            if response.status_code != 200:
                logger.error(f"Logs request failed: {response.text}")
                return {}

            return response.json()
        except requests.RequestException as e:
            logger.error(f"Network or server error occurred: {e}")
            return {}
        except ValueError as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def download_registry_model(self, workspace: str, model: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it.

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model

        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        try:
            response = requests.post(
                f"{self.base_url}/download_registry_model",
                json={"workspace": workspace, "model": model},
            )

            if response.status_code != 200:
                logger.error(f"Download registry model request failed: {response.text}")
                return {}
            self.model = model
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Network or server error occurred: {e}")
            return {}
        except ValueError as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def filter_events(self, events_df):
        if self.model == "lr-shot-distance":
            return (
                events_df[["net_distance", "shot_angle"]].dropna(),
                events_df[["net_distance", "shot_angle"]].dropna().to_json(),
            )
        elif self.model == "lr-distance":
            return (
                events_df[["net_distance"]].dropna(),
                events_df[["net_distance"]].dropna().reset_index(drop=True).to_json(),
            )
