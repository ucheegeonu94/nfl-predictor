"""
Feature engineering for NFL game prediction.

Builds, for every game in nflverse's games.csv, a set of features that are
knowable *before kickoff* (no leakage from the game's own result):

  - Elo ratings for each team going into the game (538-style, with margin-of-
    victory multiplier and home-field adjustment), updated sequentially through
    history.
  - Rolling win% and points allowed over each team's last 5 and 10 games.
  - Rolling offensive/defensive EPA-per-play and success rate (from play-by-play
    data, see build_epa.py) over each team's last 5 and 10 games.
  - Rolling penalty yards given (committed) and gained (drawn from the
    opponent) over each team's last 5 and 10 games, from play-by-play data.
  - QB continuity: whether the starting QB changed from that team's last game,
    and the starter's career start count (experience proxy).
  - QB-specific Elo: a per-QB rating (EWMA of the team's offensive EPA/play in
    games that QB started), capturing QB *quality* rather than just stability.
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

QB_ELO_START = 1500.0
QB_ELO_ALPHA = 0.18   # EWMA weight per start -- higher = faster-reacting rating
QB_ELO_SCALE = 600.0  # elo points per unit of offensive EPA/play


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
    """Rolling win% and points allowed per team over last 5 and 10 games,
    computed only from games strictly before the current one (shift(1)).

    Builds one row per (team, game) for *every* game, played or not -- an
    unplayed game gets NaN for its own win/points_allowed, which
    shift(1).rolling(...).mean() skips automatically. That's what makes a
    team's rolling value correctly carry forward into its future games
    instead of disappearing: if we only emitted rows for played games, a
    future game's game_id would never appear in this table at all, and the
    later merge back onto df would silently produce NaN -> fillna'd to a
    neutral default for every future prediction. (This bit us for real: it
    was true of every rolling feature here until this fix.)
    """
    long_rows = []
    for _, row in df.iterrows():
        played = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        if played:
            home_win = 1.0 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0.0)
            away_win = 1.0 if row["away_score"] > row["home_score"] else (0.5 if row["home_score"] == row["away_score"] else 0.0)
            home_pa, away_pa = row["away_score"], row["home_score"]
        else:
            home_win = away_win = home_pa = away_pa = np.nan
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"], "win": home_win, "points_allowed": home_pa})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"], "win": away_win, "points_allowed": away_pa})
    long_df = pd.DataFrame(long_rows).sort_values(["team", "gameday"])

    for window in (5, 10):
        long_df[f"win_pct_{window}"] = (
            long_df.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        long_df[f"points_allowed_{window}"] = (
            long_df.groupby("team")["points_allowed"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    feat_cols = ["win_pct_5", "win_pct_10", "points_allowed_5", "points_allowed_10"]
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
    play-by-play never leaks into its own features.

    One row per (team, game) for every game, played or not, with a *left*
    join onto team_game_epa.csv (which only has played games) so unplayed
    games get NaN and correctly carry forward the last real rolling value
    -- see compute_rolling_form's docstring for why this matters."""
    epa_hist = pd.read_csv(EPA_PATH)

    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"]})
    epa = pd.DataFrame(long_rows).merge(
        epa_hist[["game_id", "team", "off_epa_per_play", "def_epa_per_play", "off_success_rate", "def_success_rate"]],
        on=["game_id", "team"], how="left",
    )
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


def compute_rolling_penalties(df):
    """Rolling penalty yards given (committed by this team) and gained
    (committed by the opponent, i.e. yards this team benefited from) per
    team, over the last 5 and 10 games. Computed from team_game_epa.csv's
    penalty_yards_committed column (built by build_epa.py from every
    penalty on the play-by-play log, not just pass/run snaps).

    One row per (team, game) for every game, played or not -- see
    compute_rolling_form's docstring for why that's required for future
    games to correctly carry forward a team's rolling value."""
    pen_hist = pd.read_csv(EPA_PATH)[["game_id", "team", "penalty_yards_committed"]]

    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"], "home_team": row["home_team"], "away_team": row["away_team"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"], "home_team": row["home_team"], "away_team": row["away_team"]})
    pen = pd.DataFrame(long_rows).merge(pen_hist, on=["game_id", "team"], how="left")
    pen["opponent"] = np.where(pen["team"] == pen["home_team"], pen["away_team"], pen["home_team"])

    drawn = pen[["game_id", "team", "penalty_yards_committed"]].rename(
        columns={"team": "opponent", "penalty_yards_committed": "penalty_yards_gained"}
    )
    pen = pen.merge(drawn, on=["game_id", "opponent"], how="left")
    pen = pen.rename(columns={"penalty_yards_committed": "penalty_yards_given"})
    pen = pen.sort_values(["team", "gameday"])

    for window in (5, 10):
        for col in ("penalty_yards_given", "penalty_yards_gained"):
            pen[f"{col}_{window}"] = (
                pen.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    feat_cols = ["penalty_yards_given_5", "penalty_yards_gained_5", "penalty_yards_given_10", "penalty_yards_gained_10"]
    home_feats = pen[["game_id", "team"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols} | {"team": "home_team"}
    )
    away_feats = pen[["game_id", "team"] + feat_cols].rename(
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
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["home_qb_id"], "qb_name": row["home_qb_name"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["away_qb_id"], "qb_name": row["away_qb_name"]})
    long_df = pd.DataFrame(long_rows).sort_values(["team", "gameday"])

    # assume continuity for games with unknown starter (future schedule)
    long_df["qb_id_filled"] = long_df.groupby("team")["qb_id"].ffill()
    long_df["qb_name_presumed"] = long_df.groupby("team")["qb_name"].ffill()
    long_df["qb_confirmed"] = long_df["qb_id"].notna().astype(int)

    prev_qb = long_df.groupby("team")["qb_id_filled"].shift(1)
    long_df["qb_change"] = ((long_df["qb_id_filled"] != prev_qb) & prev_qb.notna()).astype(int)

    # career starts prior to this game, by QB identity (across any team)
    long_df = long_df.sort_values("gameday")
    known = long_df[long_df["qb_id_filled"].notna()].copy()
    known["qb_starts_career"] = known.groupby("qb_id_filled").cumcount()
    long_df = long_df.merge(known[["game_id", "team", "qb_starts_career"]], on=["game_id", "team"], how="left")
    long_df["qb_starts_career"] = long_df["qb_starts_career"].fillna(0)
    long_df = long_df.sort_values(["team", "gameday"])

    feat_cols = ["qb_change", "qb_starts_career", "qb_name_presumed", "qb_confirmed"]
    home_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"home_{c}" for c in feat_cols} | {"team": "home_team"}
    )
    away_feats = long_df[["game_id", "team"] + feat_cols].rename(
        columns={c: f"away_{c}" for c in feat_cols} | {"team": "away_team"}
    )

    df = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["game_id", "away_team"], how="left")
    return df


def compute_qb_elo(df):
    """QB-specific rating, distinct from qb_change/qb_starts_career: those
    capture *stability*, this captures *quality*. It's an EWMA of the team's
    offensive EPA-per-play in games this specific QB started, rescaled onto
    an Elo-like number so it reads on the same footing as team Elo.

    New/never-seen QBs start at the neutral prior (1500) and drift from
    there as they accumulate starts -- there's no draft-pedigree prior here
    (unlike 538's model, which seeds rookies by draft position), so a rookie
    reads as exactly average until his own play says otherwise. Computed
    sequentially in date order using only each QB's own starts strictly
    before the current game -- no leakage.
    """
    epa = pd.read_csv(EPA_PATH)

    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({"team": row["home_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["home_qb_id"]})
        long_rows.append({"team": row["away_team"], "gameday": row["gameday"], "game_id": row["game_id"], "qb_id": row["away_qb_id"]})
    long_df = pd.DataFrame(long_rows).sort_values(["team", "gameday"])
    long_df["qb_id_filled"] = long_df.groupby("team")["qb_id"].ffill()
    long_df = long_df.merge(epa[["game_id", "team", "off_epa_per_play"]], on=["game_id", "team"], how="left")
    long_df = long_df.sort_values("gameday").reset_index(drop=True)

    ratings = {}
    pre_elo = np.full(len(long_df), QB_ELO_START)
    for i, row in long_df.iterrows():
        qb = row["qb_id_filled"]
        if pd.isna(qb):
            continue
        rating = ratings.get(qb, QB_ELO_START)
        pre_elo[i] = rating
        epa_val = row["off_epa_per_play"]
        if pd.notna(epa_val):
            ratings[qb] = (1 - QB_ELO_ALPHA) * rating + QB_ELO_ALPHA * (QB_ELO_START + QB_ELO_SCALE * epa_val)

    long_df["qb_elo_pre"] = pre_elo

    home_feats = long_df[["game_id", "team", "qb_elo_pre"]].rename(
        columns={"qb_elo_pre": "home_qb_elo_pre", "team": "home_team"}
    )
    away_feats = long_df[["game_id", "team", "qb_elo_pre"]].rename(
        columns={"qb_elo_pre": "away_qb_elo_pre", "team": "away_team"}
    )

    df = df.merge(home_feats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["game_id", "away_team"], how="left")
    df["qb_elo_diff"] = df["home_qb_elo_pre"] - df["away_qb_elo_pre"]
    return df


def build_features():
    df = load_raw()
    df = compute_elo(df)
    df = compute_rolling_form(df)
    df = compute_rolling_epa(df)
    df = compute_rolling_penalties(df)
    df = compute_qb_features(df)
    df = compute_qb_elo(df)

    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["div_game"] = df["div_game"].fillna(0).astype(int)

    for c in ["home_win_pct_5", "home_win_pct_10", "away_win_pct_5", "away_win_pct_10"]:
        df[c] = df[c].fillna(0.5)
    for c in ["home_points_allowed_5", "home_points_allowed_10", "away_points_allowed_5", "away_points_allowed_10"]:
        df[c] = df[c].fillna(df[c].mean())
    penalty_cols = [c for c in df.columns if c.startswith(("home_penalty_yards_", "away_penalty_yards_"))]
    for c in penalty_cols:
        df[c] = df[c].fillna(df[c].mean())
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
    for c in ["home_qb_confirmed", "away_qb_confirmed"]:
        df[c] = df[c].fillna(0).astype(int)
    for c in ["home_qb_name_presumed", "away_qb_name_presumed"]:
        df[c] = df[c].fillna("TBD")
    for c in ["home_qb_elo_pre", "away_qb_elo_pre"]:
        df[c] = df[c].fillna(QB_ELO_START)
    df["qb_elo_diff"] = df["qb_elo_diff"].fillna(0.0)

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
    "home_qb_elo_pre", "away_qb_elo_pre", "qb_elo_diff",
    "home_points_allowed_5", "away_points_allowed_5",
    "home_points_allowed_10", "away_points_allowed_10",
    "home_penalty_yards_given_5", "away_penalty_yards_given_5",
    "home_penalty_yards_gained_5", "away_penalty_yards_gained_5",
    "home_penalty_yards_given_10", "away_penalty_yards_given_10",
    "home_penalty_yards_gained_10", "away_penalty_yards_gained_10",
    "rest_diff", "div_game", "week",
]

if __name__ == "__main__":
    df = build_features()
    df.to_csv("features.csv", index=False)
    played = df.dropna(subset=["home_win"])
    print(f"Total games: {len(df)}, played (usable for training): {len(played)}")
    print(played[["season", "week", "home_team", "away_team", "elo_diff", "home_win", "margin"] + FEATURE_COLS[3:]].tail(5).to_string())
