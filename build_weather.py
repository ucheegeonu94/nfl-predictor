"""
Fetch game-day weather for upcoming outdoor games.

Weather is a property of a game (venue + date), not a team, but the sidebar
surfaces it from whichever team's panel you open -- both teams in a game
see the same entry, keyed by game_id.

Real constraints this works around:
  - Forecasts are only meaningful ~16 days out (Open-Meteo's free daily-forecast
    horizon). Games further out are marked "unavailable" -- not a bug, weather
    two months from now genuinely isn't knowable yet.
  - Dome/closed-roof games don't need weather at all.
  - International neutral-site games (London, Munich, Sao Paulo, ...) are
    skipped -- their venues rotate yearly and aren't worth hardcoding
    coordinates for a handful of games a season.
  - Retractable-roof teams (ATL, DAL, HOU, IND) often show a NaN roof value
    on the future schedule until the decision is made close to kickoff; we
    fall back to that team's historical modal roof state to decide whether
    to bother fetching weather at all.

No API key needed -- Open-Meteo is free for non-commercial use.
"""

import json
from datetime import date, timedelta

import pandas as pd
import requests

GAMES_PATH = "games.csv"
OUT_PATH = "weather.json"
FORECAST_HORIZON_DAYS = 15

# Home-market coordinates, city-level (not exact stadium address -- plenty
# precise for day-ahead game weather, and stable across stadium renames).
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9738, -93.2581),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}
INDOOR_ROOFS = {"dome", "closed"}


def team_modal_roof(games):
    played = games.dropna(subset=["roof"])
    return played.groupby("home_team")["roof"].agg(lambda s: s.mode().iloc[0]).to_dict()


def effective_roof(row, modal_roof):
    if pd.notna(row["roof"]):
        return row["roof"]
    return modal_roof.get(row["home_team"])


def fetch_forecast(lat, lon, game_date):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max",
            "temperature_unit": "fahrenheit", "windspeed_unit": "mph", "timezone": "auto",
            "start_date": game_date, "end_date": game_date,
        },
        timeout=20,
    )
    r.raise_for_status()
    d = r.json().get("daily", {})
    if not d.get("time"):
        return None
    return {
        "temp_high_f": d["temperature_2m_max"][0],
        "temp_low_f": d["temperature_2m_min"][0],
        "precip_pct": d["precipitation_probability_max"][0],
        "wind_mph": d["windspeed_10m_max"][0],
    }


def main():
    games = pd.read_csv(GAMES_PATH, low_memory=False)
    modal_roof = team_modal_roof(games)

    upcoming = games[games["home_score"].isna()].copy()
    horizon = date.today() + timedelta(days=FORECAST_HORIZON_DAYS)

    out = {}
    for _, row in upcoming.iterrows():
        game_id = row["game_id"]
        if row.get("location") == "Neutral":
            out[game_id] = {"type": "neutral_site"}
            continue

        roof = effective_roof(row, modal_roof)
        if roof in INDOOR_ROOFS:
            out[game_id] = {"type": "indoor"}
            continue

        try:
            game_date = date.fromisoformat(row["gameday"])
        except (TypeError, ValueError):
            continue
        if game_date > horizon:
            out[game_id] = {"type": "too_far_out"}
            continue

        coords = TEAM_COORDS.get(row["home_team"])
        if not coords:
            continue
        forecast = fetch_forecast(coords[0], coords[1], row["gameday"])
        if forecast:
            out[game_id] = {"type": "forecast", **forecast}

    with open(OUT_PATH, "w") as f:
        json.dump(out, f)

    n_forecast = sum(1 for v in out.values() if v["type"] == "forecast")
    n_indoor = sum(1 for v in out.values() if v["type"] == "indoor")
    print(f"Wrote {OUT_PATH}: {len(out)} games ({n_forecast} with live forecast, {n_indoor} indoor)")


if __name__ == "__main__":
    main()
