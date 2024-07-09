import streamlit as st
import pandas as pd
import numpy as np
from client.serving_client import *
from client.game_client import *
from streamlit_utils import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as image

host = os.environ.get("CLIENT_HOST", "0.0.0.0")
port = os.environ.get("CLIENT_PORT", "7777")
serving_client = ServingClient(host, port)
game_client = GameClient(host, port)

st.title("Hockey Visualization App")
"""
Project for IFT 6758 - Data Science (Fall 2023) \n
App to extract data from the NHL API and predict goals based on different machine learning models \n
By Hung Le, Fayçal Zine-Eddine, Gauransh Kumar and Prince Arokiaraj
"""

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    workspace = st.text_input("Workspace", value="duyhung2201")
    # model_name = st.text_input("Model", value=serving_client.model)
    model_options = ["lr-shot-distance", "lr-distance"]
    model_name = st.selectbox("Choose a model", options=model_options, index=model_options.index(serving_client.model) if serving_client.model in model_options else 0)
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
            We had to make extensive use of session state and caching to make sure the dataframe and game information did not reload during every change. \n
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
