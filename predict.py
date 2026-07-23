"""
Generate predictions for not-yet-played games using the trained models.

Usage:
    python3 predict.py                 # all upcoming games
    python3 predict.py --week 1        # just a given week (current season)
"""

import argparse
import joblib
import pandas as pd

from features import build_features, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    df = build_features()
    upcoming = df[df["home_score"].isna()].copy()

    if args.season is not None:
        upcoming = upcoming[upcoming.season == args.season]
    if args.week is not None:
        upcoming = upcoming[upcoming.week == args.week]

    if upcoming.empty:
        print("No upcoming games found for the given filters.")
        return

    clf = joblib.load("model_win.joblib")
    reg = joblib.load("model_margin.joblib")

    X = upcoming[FEATURE_COLS]
    upcoming["home_win_prob"] = clf.predict_proba(X)[:, 1]
    upcoming["pred_margin"] = reg.predict(X)
    upcoming["favorite"] = upcoming.apply(
        lambda r: r.home_team if r.home_win_prob >= 0.5 else r.away_team, axis=1
    )
    upcoming["confidence"] = upcoming["home_win_prob"].apply(lambda p: max(p, 1 - p))

    out = upcoming[[
        "season", "week", "game_type", "gameday", "weekday", "gametime", "home_team", "away_team",
        "home_win_prob", "pred_margin", "favorite", "confidence"
    ]].sort_values(["season", "week", "gameday"])

    out["home_win_prob"] = out["home_win_prob"].round(3)
    out["pred_margin"] = out["pred_margin"].round(1)
    out["confidence"] = out["confidence"].round(3)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    out.to_csv("predictions.csv", index=False)
    out.to_json("predictions.json", orient="records", double_precision=3)
    print(f"\nWrote {len(out)} predictions to predictions.csv and predictions.json")


if __name__ == "__main__":
    main()
