def update_game_id_slider(*args):
    global files
    global data
    files = getFiles(f"{seasons.value}{game_type_digits[game_type.value]}")
    game_id_slider.value = 1
    game_id_slider.max = len(files)

    update_event_slider()


def update_event_slider(*args):
    global data
    global files
    data = read_data(files[game_id_slider.value - 1])

    event_count = len(data["liveData"]["plays"]["allPlays"])
    if event_count:
        event_slider.max = event_count
        event_slider.value = 1
        event_slider.min = 1
    else:
        event_slider.value = 0
        event_slider.min = 0
        event_slider.max = event_count


def update_event_plot(season, game_type, game_id, event_index):
    events = data["liveData"]["plays"]["allPlays"]
    if not events:
        print("No event")
        return

    print("gameId: ", data["gamePk"])
    home = data["liveData"]["linescore"]["teams"]["home"]["team"]["abbreviation"]
    away = data["liveData"]["linescore"]["teams"]["away"]["team"]["abbreviation"]
    print(f"{home} vs. {away}")

    event_data = events[event_index - 1]

    coordinates = event_data["coordinates"]
    if not coordinates:
        return print(json.dumps(event_data, indent=4))

    period = event_data["about"]["period"]
    t = [i for i in data["liveData"]["linescore"]["periods"] if i["num"] == period]
    if t:
        isHomeOnRight = 1 if t[0]["home"]["rinkSide"] == "right" else -1

    summary = f"Event: {event_data['result']['event']}\nPeriod: {event_data['about']['period']}\nTime: {event_data['about']['periodTime']}\nTeam: {event_data['team']['name']}"

    print(summary)
    plt.title(event_data["result"]["description"], y=1.1)

    plt.imshow(rink_image_np, extent=[-100, 100, -42.5, 42.5])
    plt.ylim(-42.5, 42.5)
    plt.xlim(-100, 100)
    plt.xticks([-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0])
    plt.yticks([-42.5, -21.25, 0, 21.25, 42.5])
    plt.scatter(coordinates["x"], coordinates["y"])
    plt.text(isHomeOnRight * (-75), 47, away, ha="center", va="center", fontsize=12)
    plt.text(isHomeOnRight * (75), 47, home, ha="center", va="center", fontsize=12)
    plt.xlabel("Feet")
    plt.ylabel("Feet")

    plt.show()


seasons.observe(update_game_id_slider, "value")
game_type.observe(update_game_id_slider, "value")
game_id_slider.observe(update_event_slider, "value")

# Create interactive plot
interactive_plot = interactive(
    update_event_plot,
    season=seasons,
    game_type=game_type,
    game_id=game_id_slider,
    event_index=event_slider,
)
output = interactive_plot.children[-1]
output.layout.height = "450px"

display(interactive_plot)
