"""
Train two models on the engineered NFL features:
  1. XGBClassifier -> P(home team wins)
  2. XGBRegressor  -> predicted margin (home_score - away_score)

Time-based split (no shuffling, no leakage across seasons):
  train: 1999-2023
  val:   2024   (used for early stopping)
  test:  2025   (untouched holdout, final reported numbers)

Baselines reported for context:
  - Always pick home team (home-field-only baseline)
  - Pick Vegas favorite from spread_line (closing-line baseline)
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error
import xgboost as xgb
import joblib

from features import build_features, FEATURE_COLS

RAW = pd.read_csv("games.csv", low_memory=False)


def main():
    df = build_features()
    df = df.merge(
        RAW[["game_id", "spread_line"]], on="game_id", how="left", suffixes=("", "_raw")
    )
    played = df.dropna(subset=["home_win"]).copy()
    played["home_win"] = played["home_win"].astype(int)

    train = played[played.season <= 2023]
    val = played[played.season == 2024]
    test = played[played.season == 2025]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    X_train, y_train = train[FEATURE_COLS], train["home_win"]
    X_val, y_val = val[FEATURE_COLS], val["home_win"]
    X_test, y_test = test[FEATURE_COLS], test["home_win"]

    # --- Win-probability classifier ---
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        eval_metric="logloss",
        early_stopping_rounds=40,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"classifier best_iteration={clf.best_iteration}")

    # --- Margin regressor ---
    reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        eval_metric="mae",
        early_stopping_rounds=40,
    )
    reg.fit(train[FEATURE_COLS], train["margin"], eval_set=[(X_val, val["margin"])], verbose=False)
    print(f"regressor best_iteration={reg.best_iteration}")

    def evaluate(name, sub):
        Xs = sub[FEATURE_COLS]
        p = clf.predict_proba(Xs)[:, 1]
        pred_win = (p >= 0.5).astype(int)
        m_pred = reg.predict(Xs)

        acc = accuracy_score(sub["home_win"], pred_win)
        ll = log_loss(sub["home_win"], p)
        brier = brier_score_loss(sub["home_win"], p)
        mae = mean_absolute_error(sub["margin"], m_pred)

        home_baseline = accuracy_score(sub["home_win"], np.ones(len(sub)))
        vegas_pick = (sub["spread_line"] > 0).astype(int)
        vegas_acc = accuracy_score(sub["home_win"], vegas_pick)

        print(f"\n== {name} (n={len(sub)}) ==")
        print(f"  model accuracy   : {acc:.4f}")
        print(f"  home-field base  : {home_baseline:.4f}")
        print(f"  vegas-line base  : {vegas_acc:.4f}")
        print(f"  log loss         : {ll:.4f}")
        print(f"  brier score      : {brier:.4f}")
        print(f"  margin MAE       : {mae:.3f} points")

        return {
            "n_games": int(len(sub)),
            "model_accuracy": round(float(acc), 4),
            "home_field_baseline": round(float(home_baseline), 4),
            "vegas_line_baseline": round(float(vegas_acc), 4),
            "log_loss": round(float(ll), 4),
            "brier_score": round(float(brier), 4),
            "margin_mae": round(float(mae), 3),
        }

    val_metrics = evaluate("VALIDATION 2024", val)
    test_metrics = evaluate("HOLDOUT TEST 2025", test)

    # refit on train+val for the artifact used going forward (test set stays untouched for reporting)
    full_train = pd.concat([train, val])
    clf_final = xgb.XGBClassifier(**{**clf.get_params(), "n_estimators": clf.best_iteration or clf.get_params()["n_estimators"]})
    clf_final.set_params(early_stopping_rounds=None)
    clf_final.fit(full_train[FEATURE_COLS], full_train["home_win"])

    reg_final = xgb.XGBRegressor(**{**reg.get_params(), "n_estimators": reg.best_iteration or reg.get_params()["n_estimators"]})
    reg_final.set_params(early_stopping_rounds=None)
    reg_final.fit(full_train[FEATURE_COLS], full_train["margin"])

    joblib.dump(clf_final, "model_win.joblib")
    joblib.dump(reg_final, "model_margin.joblib")

    importances = pd.Series(clf_final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance (win classifier):")
    print(importances.to_string())

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": "1999-2023",
        "validation_season": 2024,
        "holdout_season": 2025,
        "validation": val_metrics,
        "holdout": test_metrics,
        "top_features": importances.head(8).round(4).to_dict(),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nWrote metrics.json")


if __name__ == "__main__":
    main()
