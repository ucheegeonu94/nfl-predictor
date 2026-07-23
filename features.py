"""
Feature engineering for NFL game prediction.

Builds, for every game in nflverse's games.csv, a set of features that are
knowable *before kickoff* (no leakage from the game's own result):

  - Elo ratings for each team going into the game (538-style, with margin-of-
    victory multiplier and home-field adjustment), updated sequentially through
    history.
  - Rolling win% over each team's last 5 and 10 games.
  - Rolling offensive/defensive EPA-per-play and success rate (from play-by-play
    data, see build_epa.py) over each team's last 5 and 10 games.
  - QB continuity: whether the starting QB changed from that team's last game,
    and the starter's career start count (experience proxy).
  - Rest days for each team (from the source data).
  - Divisional-game flag.
  - Season week.

Targets:
  - home_win: 1 if home team won, else 0 (ties dropped for classification)
  - margin:   home_score - away_score (for the spread/regression model)
"""

import numpy as np
import pandas as pd

RAW_PATH = "games.csv"
EPA_PATH = "team_game_epa.csv"

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 55.0  # elo points added to home team's rating pre-game


def load_raw():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return df


def compute_elo(df):
    """Sequentially compute pre-game Elo ratings for every game."""
    ratings = {}
    pre_elo_home = np.zeros(len(df))
    pre_elo_away = np.zeros(len(df))

    for i, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, ELO_START)
        r_away = ratings.get(away, ELO_START)

        pre_elo_home[i] = r_home
        pre_elo_away[i] = r_away

        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue  # future/unplayed game: don't update ratings

        margin = row["home_score"] - row["away_score"]
        home_score_val = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

        elo_diff = (r_home + ELO_HOME_ADV) - r_away
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

        # 538-style margin-of-victory multiplier, dampened for blowouts
        winner_elo_diff = elo_diff if margin >= 0 else -elo_diff
        mov_mult = np.log(abs(margin) + 1) * (2.2 / ((winner_elo_diff * 0.001) + 2.2))

        delta = ELO_K * mov_mult * (home_score_val - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    df["elo_home_pre"] = pre_elo_home
    df["elo_away_pre"] = pre_elo_away
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    return df


def compute_rolling_form(df):
    """Rolling win% per team over last 5 and 10 games, computed only from
    games strictly before the current one (shift(1))."""
    long_rows = []
    for _, row in df.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        long_rows.append({
            "team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"],
            "win": 1.0 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0.0),
        })
        long_rows.append({
            "team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"],
            "win": 1.0 if row["away_score"] > row["home_score"] else (0.5 if row["home_score"] == row["away_score"] else 0.0),
        })
    long_df = pd.DataFrame(long_rows).sort_values(["team", "gameday"])

    for window in (5, 10):
        long_df[f"win_pct_{window}"] = (
            long_df.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    feat_cols = ["win_pct_5", "win_pct_10"]
    home_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols} | {"team": "home_team"}
    )
    away_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"away_{c}" for c in feat_cols} | {"team": "away_team"}
    )

    df = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["game_id", "away_team"], how="left")
    return df


def compute_rolling_epa(df):
    """Rolling offensive/defensive EPA-per-play and success rate per team,
    over the last 5 and 10 games, computed from team_game_epa.csv (built by
    build_epa.py from play-by-play data). Shifted so the current game's own
    play-by-play never leaks into its own features."""
    epa = pd.read_csv(EPA_PATH)
    epa = epa.merge(df[["game_id", "gameday"]], on="game_id", how="inner")
    epa = epa.sort_values(["team", "gameday"])

    raw_cols = ["off_epa_per_play", "def_epa_per_play", "off_success_rate", "def_success_rate"]
    for window in (5, 10):
        for col in raw_cols:
            epa[f"{col}_{window}"] = (
                epa.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    feat_cols = [
        "off_epa_per_play_5", "def_epa_per_play_5", "off_success_rate_5",
        "off_epa_per_play_10", "def_epa_per_play_10", "def_success_rate_10",
    ]
    home_feats = epa[["game_id", "team"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols} | {"team": "home_team"}
    )
    away_feats = epa[["game_id", "team"] + feat_cols].rename(
        columns={c: f"away_{c}" for c in feat_cols} | {"team": "away_team"}
    )

    df = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["game_id", "away_team"], how="left")
    return df


def compute_qb_features(df):
    """QB-continuity features from each team's starting QB id/name:
      - qb_change: 1 if the starter differs from that team's last known starter
      - qb_starts_career: career starts (in this dataset) prior to this game,
        a cheap experience/stability proxy

    For future games where the starter isn't yet known (qb_id is NaT/NaN,
    e.g. next season's schedule), we forward-fill each team's last known
    starter as the assumed starter -- i.e. we assume continuity absent any
    other signal. This is a real limitation: an offseason QB change or
    in-season injury announced after the schedule snapshot won't be
    reflected until games.csv is refreshed with actual starters.
    """
    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["home_qb_id"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["away_qb_id"]})
    long_df = pd.DataFrame(long_rows).sort_values(["team", "gameday"])

    # assume continuity for games with unknown starter (future schedule)
    long_df["qb_id_filled"] = long_df.groupby("team")["qb_id"].ffill()

    prev_qb = long_df.groupby("team")["qb_id_filled"].shift(1)
    long_df["qb_change"] = ((long_df["qb_id_filled"] != prev_qb) & prev_qb.notna()).astype(int)

    # career starts prior to this game, by QB identity (across any team)
    long_df = long_df.sort_values("gameday")
    known = long_df[long_df["qb_id_filled"].notna()].copy()
    known["qb_starts_career"] = known.groupby("qb_id_filled").cumcount()
    long_df = long_df.merge(known[["game_id", "team", "qb_starts_career"]], on=["game_id", "team"], how="left")
    long_df["qb_starts_career"] = long_df["qb_starts_career"].fillna(0)
    long_df = long_df.sort_values(["team", "gameday"])

    feat_cols = ["qb_change", "qb_starts_career"]
    home_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols} | {"team": "home_team"}
    )
    away_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"away_{c}" for c in feat_cols} | {"team": "away_team"}
    )

    df = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["game_id", "away_team"], how="left")
    return df


def build_features():
    df = load_raw()
    df = compute_elo(df)
    df = compute_rolling_form(df)
    df = compute_rolling_epa(df)
    df = compute_qb_features(df)

    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["div_game"] = df["div_game"].fillna(0).astype(int)

    for c in ["home_win_pct_5", "home_win_pct_10", "away_win_pct_5", "away_win_pct_10"]:
        df[c] = df[c].fillna(0.5)
    epa_epa_cols = [c for c in df.columns if c.endswith("_epa_per_play_5") or c.endswith("_epa_per_play_10")]
    for c in epa_epa_cols:
        df[c] = df[c].fillna(0.0)
    epa_success_cols = [c for c in df.columns if c.endswith("_success_rate_5") or c.endswith("_success_rate_10")]
    for c in epa_success_cols:
        df[c] = df[c].fillna(0.45)  # roughly league-average success rate
    for c in ["home_qb_change", "away_qb_change"]:
        df[c] = df[c].fillna(0).astype(int)
    for c in ["home_qb_starts_career", "away_qb_starts_career"]:
        df[c] = df[c].fillna(0)

    df["home_win"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] < df["away_score"], 0, np.nan)
    )
    df["margin"] = df["home_score"] - df["away_score"]

    return df


FEATURE_COLS = [
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "home_win_pct_5", "away_win_pct_5",
    "home_win_pct_10", "away_win_pct_10",
    "home_off_epa_per_play_5", "away_off_epa_per_play_5",
    "home_def_epa_per_play_5", "away_def_epa_per_play_5",
    "home_off_epa_per_play_10", "away_off_epa_per_play_10",
    "home_def_epa_per_play_10", "away_def_epa_per_play_10",
    "home_off_success_rate_5", "away_off_success_rate_5",
    "home_def_success_rate_10", "away_def_success_rate_10",
    "home_qb_change", "away_qb_change",
    "home_qb_starts_career", "away_qb_starts_career",
    "rest_diff", "div_game", "week",
]

if __name__ == "__main__":
    df = build_features()
    df.to_csv("features.csv", index=False)
    played = df.dropna(subset=["home_win"])
    print(f"Total games: {len(df)}, played (usable for training): {len(played)}")
    print(played[["season", "week", "home_team", "away_team", "elo_diff", "home_win", "margin"] + FEATURE_COLS[3:]].tail(5).to_string())
