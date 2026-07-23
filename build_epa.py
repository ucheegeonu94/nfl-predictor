"""
Download nflverse play-by-play data (1999-2025) and aggregate it down to
team-game level offensive/defensive EPA-per-play and success rate.

This produces `team_game_epa.csv`: one row per (game_id, team) from that
team's perspective, containing *that game's own* efficiency numbers. These
are POST-game stats — features.py turns them into pre-game features by
taking a rolling average of a team's past games (shifted so the current
game is excluded), the same way it already does for win% and point diff.

Only pass/run plays with a non-null EPA are counted (excludes kneels,
spikes, kickoffs, punts, penalties-with-no-play, etc.) so the numbers
reflect meaningful offensive snaps.
"""

import io
import pandas as pd
import pyarrow.parquet as pq
import requests

SEASONS = range(1999, 2026)
COLS = ["game_id", "season", "week", "posteam", "defteam", "epa", "success", "play_type"]
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


def fetch_season(season):
    url = BASE_URL.format(season=season)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    table = pq.read_table(io.BytesIO(r.content), columns=COLS)
    return table.to_pandas()


def aggregate_team_game(df):
    plays = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()].copy()

    off = (
        plays.groupby(["game_id", "posteam"])
        .agg(off_epa_per_play=("epa", "mean"), off_success_rate=("success", "mean"), off_plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defn = (
        plays.groupby(["game_id", "defteam"])
        .agg(def_epa_per_play=("epa", "mean"), def_success_rate=("success", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    merged = off.merge(defn, on=["game_id", "team"], how="outer")
    return merged


def main():
    all_frames = []
    for season in SEASONS:
        print(f"fetching {season}...", flush=True)
        raw = fetch_season(season)
        agg = aggregate_team_game(raw)
        all_frames.append(agg)

    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["team"])
    out.to_csv("team_game_epa.csv", index=False)
    print(f"wrote team_game_epa.csv: {out.shape}")


if __name__ == "__main__":
    main()
