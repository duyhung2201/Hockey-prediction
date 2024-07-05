import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def cal_game_seconds(period_time, period):
    return (
        int(period_time.split(":")[0]) * 60
        + int(period_time.split(":")[1])
        + (period - 1) * 1200
    )


def find_opponent_net(zoneCode, xCoord):
    if zoneCode == "O":
        return 89 if xCoord > 0 else -89
    elif zoneCode == "D":
        return 89 if xCoord < 0 else -89
    return None


def is_empty_net(situation_code, is_home_team_shot):
    return (
        1 - int(situation_code[0]) if is_home_team_shot else 1 - int(situation_code[3])
    )


def download_game(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    response = requests.get(url)
    data = json.loads(response.text)
    return data


def extract_from_events(events, metadata):
    result = []
    home_team = metadata["homeTeam"]["abbrev"]
    team_mapping = {
        metadata["homeTeam"]["id"]: home_team,
        metadata["awayTeam"]["id"]: metadata["awayTeam"]["abbrev"],
    }
    score = {
        metadata["homeTeam"]["abbrev"]: 0,
        metadata["awayTeam"]["abbrev"]: 0,
    }
    for event in events:
        try:
            if event["typeDescKey"] in ["shot-on-goal", "goal"]:
                shot = dict()

                shot["game_id"] = metadata["id"]
                shot["event_id"] = event["eventId"]
                shot["event"] = event["typeDescKey"]
                shot["period"] = event["periodDescriptor"]["number"]
                shot["period_time"] = event["timeInPeriod"]
                shot["game_seconds"] = cal_game_seconds(
                    shot["period_time"], shot["period"]
                )
                shot["time_remaining"] = event["timeRemaining"]

                shot["team"] = team_mapping[event["details"].get("eventOwnerTeamId")]
                shot["x_coordinate"] = event["details"].get("xCoord")
                shot["y_coordinate"] = event["details"].get("yCoord")

                if shot["event"] == "goal":
                    score[shot["team"]] += 1
                shot["home_score"] = score[metadata["homeTeam"]["abbrev"]]
                shot["away_score"] = score[metadata["awayTeam"]["abbrev"]]
                shot["shot_type"] = event["details"].get("shotType")
                shot["net_x"] = find_opponent_net(
                    event["details"]["zoneCode"], shot["x_coordinate"]
                )
                shot["is_empty_net"] = is_empty_net(
                    event["situationCode"], shot["team"] == home_team
                )

                result.append(shot)
        except Exception as e:
            print(e)
            print("game_id", metadata["id"])
            print(event)            

    return result


def extract_from_raw(data):
    result = []
    home_team = data["homeTeam"]["abbrev"]
    team_mapping = {
        data["homeTeam"]["id"]: home_team,
        data["awayTeam"]["id"]: data["awayTeam"]["abbrev"],
    }

    for event in data["plays"]:
        try:
            if event["typeDescKey"] in ["shot-on-goal", "goal"]:
                shot = dict()

                shot["game_id"] = data["id"]
                shot["event_id"] = event["eventId"]
                shot["event"] = event["typeDescKey"]
                shot["period"] = event["periodDescriptor"]["number"]
                shot["period_time"] = event["timeInPeriod"]
                shot["game_seconds"] = cal_game_seconds(
                    shot["period_time"], shot["period"]
                )
                shot["period_remaining"] = event["timeRemaining"]

                shot["team"] = team_mapping[event["details"].get("eventOwnerTeamId")]
                shot["x_coordinate"] = event["details"].get("xCoord")
                shot["y_coordinate"] = event["details"].get("yCoord")

                # shot['shooter_id'] = event['details'].get('shootingPlayerId')
                # shot['goalie_id'] = event['details'].get('goalieInNetId')
                shot["shot_type"] = event["details"].get("shotType")
                shot["net_x"] = find_opponent_net(
                    event["details"]["zoneCode"], shot["x_coordinate"]
                )
                shot["is_empty_net"] = is_empty_net(
                    event["situationCode"], shot["team"] == home_team
                )

                result.append(shot)
        except Exception as e:
            print(e)
            print("game_id", data["id"])
            print(event)

    return result


def engineer_feature(input_df):
    result = input_df.copy()
    mode_per_group = input_df.groupby(["team", "period"])["net_x"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )
    result["net_x"] = result["net_x"].fillna(mode_per_group)

    result["net_distance"] = np.sqrt(
        (result["x_coordinate"] - result["net_x"]) ** 2 + result["y_coordinate"] ** 2
    )

    result["shot_angle"] = np.degrees(
        np.arctan2(
            result["y_coordinate"], np.abs(result["x_coordinate"] - result["net_x"])
        )
    )

    result["is_goal"] = result["event"].apply(lambda e: 1 if e == "goal" else 0)

    return result


def process_gamedata(game_id):
    game_data = download_game(game_id)
    t = pd.DataFrame(extract_from_raw(game_data))
    return engineer_feature(t)


def process_events(events, metadata):
    t = pd.DataFrame(extract_from_events(events, metadata))
    return engineer_feature(t)

def get_metadata(game_data):
    metadata = {
        "homeTeam": game_data["homeTeam"],
        "awayTeam": game_data["awayTeam"],
        "id": game_data["id"],
    }
    return metadata