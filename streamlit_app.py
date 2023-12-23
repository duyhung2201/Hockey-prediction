import streamlit as st
import pandas as pd
import numpy as np
from ift6758.ift6758.client.serving_client import *
from ift6758.ift6758.client.game_client import *


serving_client = ServingClient("127.0.0.1", "8080")
game_client = GameClient("127.0.0.1", "8080")

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
        if workspace and model_name and version:
            try:
                response = serving_client.download_registry_model(
                    workspace, model_name, version
                )
                if response:
                    st.success(f"{model_name} downloaded successfully!")
                    st.json(response)
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
@st.cache(allow_output_mutation=True)
def display_game_info(game_data):
    home_team = game_data["home_team"].iloc[0]
    away_team = game_data["away_team"].iloc[0]
    period = game_data["period"].iloc[-1]
    time_left = game_data["time_left"].iloc[-1]
    current_score = f"{game_data['home_score'].iloc[-1]} - {game_data['away_score'].iloc[-1]}"  # Adjust column names as per your data

    # Display game information
    st.metric("Home Team", home_team)
    st.metric("Away Team", away_team)
    st.metric("Period", period)
    st.metric("Time Left in Period", time_left)
    st.metric("Current Score", current_score)

    # Calculating and displaying expected goals and score difference
    total_xg_home = game_data[game_data["team"] == home_team][
        "xG"
    ].sum()  # Adjust column names as per your data
    total_xg_away = game_data[game_data["team"] == away_team][
        "xG"
    ].sum()  # Adjust column names as per your data
    score_difference_home = total_xg_home - game_data["home_score"].iloc[-1]
    score_difference_away = total_xg_away - game_data["away_score"].iloc[-1]

    st.metric("Total Expected Goals - Home Team", f"{total_xg_home:.2f}")
    st.metric("Total Expected Goals - Away Team", f"{total_xg_away:.2f}")
    st.metric("Score Difference - Home Team", f"{score_difference_home:.2f}")
    st.metric("Score Difference - Away Team", f"{score_difference_away:.2f}")


@st.cache
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


# Main functionality to ping game, get data, make predictions and display results
with st.container():
    if st.button("Ping Game"):
        if game_id:
            try:
                game_data, new_events = game_client.ping_game(game_id)
                if not game_data.empty:
                    display_game_info(game_data)  # Display game info
                    display_prediction_features(
                        new_events
                    )  # Display prediction features

                    # Getting and displaying predictions
                    predictions = get_predictions(new_events)
                    new_events = new_events.assign(goal_prob=pd.Series(predictions))
                    st.write("New Events with Predictions:")
                    st.dataframe(new_events)
                else:
                    st.warning("No data available for the entered Game ID.")

                if not new_events.empty:
                    st.write("Data used for predictions and predictions:")
                    st.dataframe(new_events)
                else:
                    st.write("No new events since the last update.")

            except Exception as e:
                st.error(f"An error occurred while fetching game data: {e}")
        else:
            st.warning("Please enter a valid Game ID.")
