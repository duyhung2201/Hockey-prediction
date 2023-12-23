from ift6758.ift6758.client.serving_client import ServingClient
from ift6758.ift6758.client.game_extractor import *
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GameClient:
    def __init__(
        self,
        ip: str = "0.0.0.0",
        port: int = 5000,
    ):
        self.serving_client = ServingClient(ip, port)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def ping_game(self, game_id):
        game_id = int(game_id)
        data_dir = os.path.join(self.base_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        path = os.path.join(data_dir, f"{game_id}.csv")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            logger.info(f"File {path} not found. Creating a new DataFrame.")
            df = pd.DataFrame()

        last_event_id = df["event_id"].max() if len(df) > 0 else 0
        try:
            game_data = download_game(game_id)
        except Exception as e:
            logger.error(f"Error downloading game data for game_id {game_id}: {e}")
            return df, pd.DataFrame()
        new_events = self.get_new_events(game_data, last_event_id)

        new_events_results = pd.DataFrame()
        if len(new_events) > 0:
            new_events_df = process_events(new_events, game_data)
            new_events_results = self.serving_client.predict(new_events_df)
            df = pd.concat([df, new_events_results])
            df.to_csv(path, index=False)

        return df, new_events_results

    def get_new_events(self, data, last_event_id):
        new_events = [
            event
            for event in data["plays"]
            if int(event["eventId"]) > last_event_id
            and event["typeDescKey"] in ["shot-on-goal", "goal"]
        ]
        return new_events
