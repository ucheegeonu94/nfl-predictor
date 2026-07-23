"""
Pull the current season's injury reports from nflverse and reduce them to
"latest known report per team" -- the data the sidebar needs.

nflverse publishes one file per season (injuries_<season>.parquet), built
from official weekly practice/game-status reports. Two important realities:

  - It only exists once the season has actually started -- there is no
    file for a season before Week 1's first practice report, so during the
    offseason this script writes an empty injuries.json and that's correct,
    not a failure.
  - Even in-season, "latest week in the file" can lag the *upcoming* game by
    a few days (e.g. a Tuesday pipeline run happens before that week's own
    Wed/Thu/Fri reports are out), so what's shown is genuinely "the most
    recent report we have," not necessarily "as of today."
"""

import json

import pandas as pd
import requests

GAMES_PATH = "games.csv"
OUT_PATH = "injuries.json"


def current_season():
    games = pd.read_csv(GAMES_PATH, low_memory=False)
    return int(games["season"].max())


def fetch_injuries(season):
    url = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    import io
    return pd.read_parquet(io.BytesIO(r.content))


def main():
    season = current_season()
    df = fetch_injuries(season)

    if df is None or df.empty:
        print(f"No injuries file for season {season} yet (offseason) -- writing empty report.")
        with open(OUT_PATH, "w") as f:
            json.dump({"season": season, "as_of_week": None, "teams": {}}, f)
        return

    reported = df[df["report_status"].notna()].copy()
    if reported.empty:
        with open(OUT_PATH, "w") as f:
            json.dump({"season": season, "as_of_week": None, "teams": {}}, f)
        print("Injuries file exists but has no report_status entries yet.")
        return

    latest_week = int(reported["week"].max())

    teams = {}
    for team, g in reported.groupby("team"):
        g_latest = g[g["week"] == g["week"].max()]
        players = [
            {
                "name": row["full_name"],
                "position": row["position"],
                "injury": row["report_primary_injury"] if pd.notna(row["report_primary_injury"]) else None,
                "status": row["report_status"],
            }
            for _, row in g_latest.sort_values("full_name").iterrows()
        ]
        teams[team] = {"week": int(g_latest["week"].iloc[0]), "players": players}

    out = {"season": season, "as_of_week": latest_week, "teams": teams}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Wrote {OUT_PATH}: season {season}, {len(teams)} teams with reports, most recent week {latest_week}")


if __name__ == "__main__":
    main()
