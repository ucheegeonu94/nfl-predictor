"""
Extract each team's *current* offensive/defensive efficiency for the league
rankings sidebar.

Every one of a team's upcoming games carries the same pre-game rolling
10-game EPA value (nothing new has been played yet to move it), so this
just takes each team's earliest upcoming game and reads its rolling EPA off
of that -- no separate modeling step, just a reduction of what features.py
already computed. Refreshed weekly like everything else, so it reflects
real games as the season plays out.
"""

import json

import pandas as pd

from features import build_features

OUT_PATH = "team_rankings.json"


def main():
    df = build_features()
    upcoming = df[df["home_score"].isna()].copy()
    if upcoming.empty:
        print("No upcoming games found -- writing empty rankings.")
        with open(OUT_PATH, "w") as f:
            json.dump({"teams": []}, f)
        return

    long_rows = []
    for _, row in upcoming.iterrows():
        long_rows.append({
            "team": row["home_team"], "gameday": row["gameday"],
            "off_epa": row["home_off_epa_per_play_10"], "def_epa": row["home_def_epa_per_play_10"],
        })
        long_rows.append({
            "team": row["away_team"], "gameday": row["gameday"],
            "off_epa": row["away_off_epa_per_play_10"], "def_epa": row["away_def_epa_per_play_10"],
        })
    long_df = pd.DataFrame(long_rows).sort_values("gameday")
    current = long_df.drop_duplicates(subset="team", keep="first").sort_values("team")

    teams = [
        {"team": r["team"], "off_epa": round(float(r["off_epa"]), 4), "def_epa": round(float(r["def_epa"]), 4)}
        for _, r in current.iterrows()
    ]
    with open(OUT_PATH, "w") as f:
        json.dump({"teams": teams}, f)
    print(f"Wrote {OUT_PATH}: {len(teams)} teams")


if __name__ == "__main__":
    main()
