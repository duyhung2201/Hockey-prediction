import streamlit as st
import pandas as pd
import numpy as np
from client.serving_client import *
from client.game_client import *
import matplotlib.pyplot as plt
import matplotlib.image as image
import os

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
    version = st.text_input("Version", value="")
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
with st.container():
    st.header("Game ID")
    game_id = st.text_input("Enter Game ID")


# Function to display relevant game information
def display_game_info(game_data, metadata):
    print(metadata)
    print(game_data.columns)
    home_team = metadata["homeTeam"]["abbrev"]
    away_team = metadata["awayTeam"]["abbrev"]
    period = game_data["period"].iloc[-1]
    time_left = game_data["time_remaining"].iloc[-1]
    current_score = f"{game_data['home_score'].iloc[-1]} - {game_data['away_score'].iloc[-1]}"  # Adjust column names as per your data

    # Display game information
    st.metric("Home Team", home_team)
    st.metric("Away Team", away_team)
    st.metric("Period", period)
    st.metric("Time Left in Period", time_left)
    st.metric("Current Score", current_score)

    # Calculating and displaying expected goals and score difference
    total_xg_home = game_data["home_xg"].iloc[
        -1
    ]  # Adjust column names as per your data
    total_xg_away = game_data["away_xg"].iloc[
        -1
    ]  # Adjust column names as per your data
    score_difference_home = total_xg_home - game_data["home_score"].iloc[-1]
    score_difference_away = total_xg_away - game_data["away_score"].iloc[-1]

    st.metric("Total Expected Goals - Home Team", f"{int(total_xg_home)}")
    st.metric("Total Expected Goals - Away Team", f"{int(total_xg_away)}")
    st.metric("Score Difference - Home Team", f"{int(score_difference_home)}")
    st.metric("Score Difference - Away Team", f"{int(score_difference_away)}")


def get_predictions(new_events):
    predictions = serving_client.predict(new_events)
    return predictions


def filter_cumulative_events(model, df):
    if model == "lr-distance":
        filtered_df = df[["net_distance", "goal_prob"]]
    elif model == "lr-shot-distance":
        filtered_df = df[["net_distance", "shot_angle", "goal_prob"]]

    return filtered_df


"""
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
"""

# Main functionality to ping game, get data, make predictions and display results
with st.container():
    if st.button("Ping Game"):
        if game_id:
            try:
                game_data, new_events, metadata = game_client.ping_game(game_id)
                if not game_data.empty:
                    # Displaying game information
                    display_game_info(game_data, metadata)

                    # Displaying data used for prediction and predictions
                    st.write("Data used for prediction with predictions:")
                    st.dataframe(filter_cumulative_events(model_name, game_data))
                    """
                    # Displaying shot plotting
                    ## Sidebar controls for team selection
                    st.sidebar.header("Plotting events")
                    selected_team = st.sidebar.selectbox(
                        "Select Team", game_data["team"].unique()
                    )

                    ## Get unique event IDs for the selected team
                    unique_event_ids = game_data[game_data["team"] == selected_team][
                        "event_id"
                    ].unique()

                    ## Sidebar slider for event ID selection based on index
                    selected_event_index = st.sidebar.slider(
                        "Select Event Index", 0, len(unique_event_ids) - 1, 0
                    )
                    selected_event_id = unique_event_ids[selected_event_index]

                    ## Filter data for the selected event ID
                    filtered_data = game_data[
                        (game_data["team"] == selected_team)
                        & (game_data["event_id"] == selected_event_id)
                    ]
                    st.header("Shot plotting")
                    st.write(
                    """
                    """
                    Interactive tool to visualize the different shots in the selected game.\n 
                    To include this functionality we used the extracted data that our model uses and plotted it on a rink image. We used a selectbox to select the team and a slider to go through every shot of that team. \n
                    The green circles represent goals and the red cross represent missed shots. \n
                    The arrow represents the vector of the angle and distance towards the goal.\n
                    Finally, we added the angle and distance of the shot to the arrow using plt.arrow and a logic to make sure that the arrow pointed on the right direction based on the sign of the coordinates and the distance.
                    """
                    """
                    )
                    update_event_plot(filtered_data, selected_team, selected_event_id)
                    """
                else:
                    # Displaying data used for prediction and predictions
                    st.write("Game Does not exist")

            except Exception as e:
                st.error(f"An error occurred while fetching game data: {e}")
        else:
            st.warning("Please enter a valid Game ID.")
