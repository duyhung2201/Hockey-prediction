import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image


@st.cache_data
def display_game_info(game_data, metadata):
    """
    Function to display game information and calculate expected goals and score difference
    """
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
    """
    Function to display the calculated game information
    """
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


def filter_cumulative_events(model, df):
    """
    function to filter the dataframe to select relevant features based on the model
    """
    if model == "lr-distance":
        filtered_df = df[["net_distance", "goal_prob"]]
    elif model == "lr-shot-distance":
        filtered_df = df[["net_distance", "shot_angle", "goal_prob"]]

    return filtered_df


rink_image_np = image.imread("nhl_rink.png")


def update_event_plot(filtered_data, selected_team, selected_event_id):
    """
    function to create an interactive widget to visualize shots in a selected game
    """
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
