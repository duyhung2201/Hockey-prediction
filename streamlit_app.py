import streamlit as st
import pandas as pd
import numpy as np
from client.serving_client import *
from client.game_client import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as image

host = os.environ.get("CLIENT_HOST", "0.0.0.0")
port = os.environ.get("CLIENT_PORT", "8000")
serving_client = ServingClient(host, port)
game_client = GameClient(host, port)

st.title("Hockey Visualization App")
"""
Project for IFT 6758 - Data Science (Fall 2023) \n
App to extract data from the NHL API and predict goals based on different machine learning models \n
By Duy Hung Le, Fayçal Zine-Eddine, Gauransh Kumar and Prince Immanuel Joseph Arokiaraj
"""

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    workspace = st.text_input("Workspace", value="duyhung2201")
    model_name = st.text_input("Model", value=serving_client.model)
    version = st.text_input("Version", value="1.40.0")
    if st.button("Download Model"):
        if workspace and model_name:
            try:
                response = serving_client.download_registry_model(workspace, model_name)
                if response:
                    st.success(f"{model_name} downloaded successfully!")
                else:
                    st.error("Failed to download the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill all fields.")

# Game ID input
if "game_id" not in st.session_state:
    st.session_state["game_id"] = None

with st.container():
    st.header("Game ID")
    st.session_state["game_id"] = st.text_input(
        "Enter Game ID", value=st.session_state.get("game_id", "")
    )


# Function to display relevant game information
@st.cache_data
def display_game_info(game_data, metadata):
    home_team = metadata["homeTeam"]["abbrev"]
    away_team = metadata["awayTeam"]["abbrev"]
    period = game_data["period"].iloc[-1]
    time_left = game_data["time_remaining"].iloc[-1]

    # Calculating and displaying expected goals and score difference
    total_xg_home = game_data["home_xg"].iloc[-1]
    total_xg_away = game_data["away_xg"].iloc[-1]
    score_difference_home = total_xg_home - game_data["home_score"].iloc[-1]
    score_difference_away = total_xg_away - game_data["away_score"].iloc[-1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "period": period,
        "time_left": time_left,
        "total_xg_home": total_xg_home,
        "total_xg_away": total_xg_away,
        "home_score": game_data["home_score"].iloc[-1],
        "away_score": game_data["away_score"].iloc[-1],
        "score_difference_home": score_difference_home,
        "score_difference_away": score_difference_away,
    }


def show_game_info(game_info):
    if game_info:
        st.subheader(
            f"Game {st.session_state['game_id']} : {game_info['home_team']} vs {game_info['away_team']}"
        )
        st.write(f"Period: {game_info['period']} - {game_info['time_left']} left")
        col1, col2 = st.columns(2)
        col1.metric(
            f"{game_info['home_team']} xG (actual)",
            f"{int(game_info['total_xg_home'])} ({game_info['home_score']})",
            delta=float(game_info["score_difference_home"]),
        )
        col2.metric(
            f"{game_info['away_team']} xG (actual)",
            f"{int(game_info['total_xg_away'])} ({game_info['away_score']})",
            delta=float(game_info["score_difference_away"]),
        )


def get_predictions(new_events):
    predictions = serving_client.predict(new_events)
    return predictions


def display_prediction_features(new_events):
    filtered_events, _ = serving_client.filter_events(new_events)
    if not filtered_events.empty:
        st.write("Features Used for Prediction:")
        st.dataframe(filtered_events)
    else:
        st.write("No features available for prediction.")


def filter_cumulative_events(model, df):
    if model == "lr-distance":
        filtered_df = df[["net_distance", "goal_prob"]]
    elif model == "lr-shot-distance":
        filtered_df = df[["net_distance", "shot_angle", "goal_prob"]]

    return filtered_df


rink_image_np = image.imread("nhl_rink.png")


def update_event_plot(filtered_data, selected_team, selected_event_id):
    if filtered_data.empty:
        st.write("No data available for the selected event and team.")
        return

    # Get the first event
    event = filtered_data.iloc[0]

    # Extracting event details
    event_type = event["event"]
    shot_type = event["shot_type"]
    angle = event["shot_angle"]
    distance = event["net_distance"]
    period = event["period"]
    period_time = event["period_time"]
    x_coordinate = event["x_coordinate"]
    y_coordinate = event["y_coordinate"]
    if (
        -89 < x_coordinate < 0 or x_coordinate > 89
    ) and distance < 89:  # Flip the arrow if the shot is from the other side of the rink, on the other side of the goal
        arrow_dx = -np.abs(distance * np.cos(np.radians(angle))) / 2
    else:
        arrow_dx = np.abs(distance * np.cos(np.radians(angle))) / 2
    if y_coordinate < 0:
        arrow_dy = np.abs(distance * np.sin(np.radians(angle))) / 2
    else:
        arrow_dy = -np.abs(distance * np.sin(np.radians(angle))) / 2

    is_goal = event["is_goal"]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.imshow(rink_image_np, extent=[-100, 100, -42.5, 42.5])

    if is_goal:
        plt.scatter(x_coordinate, y_coordinate, c="green", marker="o")
    else:
        plt.scatter(x_coordinate, y_coordinate, c="red", marker="x")

    plt.arrow(
        x_coordinate,
        y_coordinate,
        arrow_dx,
        arrow_dy,
        head_width=2,
        head_length=3,
        fc="blue",
        ec="blue",
    )
    label_y_offset = (
        5 if y_coordinate > 0 else -5
    )  # Adjust label position based on y-coordinate
    plt.text(
        x_coordinate,
        y_coordinate + label_y_offset,
        f"Angle: {np.abs(angle.round(1))}°\nDistance: {distance.round(1)} ft",
        ha="center",  # Center align the text horizontally
        va="bottom"
        if y_coordinate > 0
        else "top",  # Adjust vertical alignment based on y-coordinate
        color="black",
    )
    plt.title(
        f"Team: {selected_team}, Type of shot: {shot_type}, Period: {period}, Time: {period_time}",
        y=1.1,
    )
    plt.ylim(-42.5, 42.5)
    plt.xlim(-100, 100)
    plt.xticks([-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0])
    plt.yticks([-42.5, -21.25, 0, 21.25, 42.5])
    plt.xlabel("Feet")
    plt.ylabel("Feet")
    st.pyplot(plt)


# Main functionality to ping game, get data, make predictions and display results
with st.container():
    if "ping_pressed" not in st.session_state:
        st.session_state.ping_pressed = False

    if st.button("Ping Game"):
        st.session_state.ping_pressed = True

        if st.session_state["game_id"]:
            try:
                game_data, new_events, metadata = game_client.ping_game(
                    st.session_state["game_id"]
                )
                if not game_data.empty:
                    # Storing game information and data used for prediction in the session state
                    st.session_state["game_info"] = display_game_info(
                        game_data, metadata
                    )
                    st.session_state["prediction_data"] = filter_cumulative_events(
                        model_name, game_data
                    )
                else:
                    st.write("Game Does not exist")
            except Exception as e:
                st.error(f"An error occurred while fetching game data: {e}")
        else:
            st.warning("Please enter a valid Game ID.")

    if "game_info" in st.session_state:
        show_game_info(st.session_state["game_info"])

    if (
        st.session_state.get("prediction_data") is not None
        and not st.session_state["prediction_data"].empty
    ):
        st.subheader("Data used for prediction with predictions:")
        st.dataframe(st.session_state["prediction_data"])
# Initialization of selected_event_index in session state
if "selected_event_index" not in st.session_state:
    st.session_state["selected_event_index"] = 0

with st.container():
    with st.sidebar:
        st.header("Plotting events")
        # Ensure game_data is available before trying to access it
        if "game_id" in st.session_state and st.session_state["game_id"]:
            game_data, _, _ = game_client.ping_game(st.session_state["game_id"])
            if not game_data.empty:
                selected_team = st.selectbox("Select Team", game_data["team"].unique())
                unique_event_ids = game_data[game_data["team"] == selected_team][
                    "event_id"
                ].unique()

                # Update selected_event_index based on the slider value
                st.session_state["selected_event_index"] = st.slider(
                    "Select Event Index",
                    0,
                    len(unique_event_ids) - 1,
                    st.session_state["selected_event_index"],
                )

    # Now outside of the sidebar block, but still inside the container
    if st.session_state.ping_pressed:
        st.subheader("Shot plotting")
        st.write(
            """
            Interactive tool to visualize the different shots in the selected game.\n 
            To include this functionality we used the extracted data that our model uses and plotted it on a rink image. We used a selectbox to select the team and a slider to go through every shot of that team. \n
            The green circles represent goals and the red cross represent missed shots. \n
            The arrow represents the vector of the angle and distance towards the goal.\n
            Finally, we added the angle and distance of the shot to the arrow using plt.arrow and a logic to make sure that the arrow pointed on the right direction based on the sign of the coordinates and the distance.
            """
        )

        # Make sure game_data is defined before accessing it
        if (
            "game_id" in st.session_state
            and st.session_state["game_id"]
            and not game_data.empty
        ):
            selected_event_index = st.session_state["selected_event_index"]
            selected_event_id = unique_event_ids[selected_event_index]
            filtered_data = game_data[
                (game_data["team"] == selected_team)
                & (game_data["event_id"] == selected_event_id)
            ]
            update_event_plot(filtered_data, selected_team, selected_event_id)
