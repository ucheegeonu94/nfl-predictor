# NFL Game Predictor

Predicts NFL game winners (win probability) and point margins using gradient-boosted
trees (XGBoost), trained on 1999-2025 game data from [nflverse](https://github.com/nflverse/nfldata).

## Setup

```
pip install -r requirements.txt
```
(On a machine without `python3-venv`, use `pip install --user --break-system-packages -r requirements.txt`.)

## Pipeline

1. **`games.csv`** — raw game-level data pulled from nflverse (scores, rest days,
   divisional flag, Vegas spread/total lines, weather, coaches, starting QB ids).
   Re-download anytime with:
   ```
   python3 -c "import requests; open('games.csv','wb').write(requests.get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv').content)"
   ```

2. **`build_epa.py`** — downloads nflverse play-by-play data for every season
   (1999-2025, ~450MB total) and aggregates it to **team-game level**: offensive
   and defensive EPA-per-play and success rate, counting only pass/run plays
   (kneels, spikes, kickoffs, punts, no-plays excluded). Writes
   `team_game_epa.csv`. Re-run only when new seasons/weeks of pbp data are
   published (takes a few minutes; ~13-20MB per season download).

3. **`features.py`** — builds pre-game features for every game, with no leakage:
   - **Elo ratings** (538-style, margin-of-victory-weighted, +55 pt home-field bump),
     computed sequentially game-by-game through history.
   - **Rolling win%** over each team's last 5 and 10 games.
   - **Rolling EPA efficiency**: offensive/defensive EPA-per-play (5- and
     10-game windows) and success rate (offense at 5 games, defense at 10 —
     the more stable window), computed from `team_game_epa.csv`. Replaces the
     old point-differential features with a much stronger, play-level measure
     of how good a team's offense/defense actually is, independent of garbage-time
     score effects.
   - **QB continuity**: `qb_change` (1 if the starter differs from that team's
     last game — a backup/injury signal) and `qb_starts_career` (cumulative
     career starts in this dataset, an experience/stability proxy), derived
     from `games.csv`'s `home_qb_id`/`away_qb_id`.
   - **Rest-day differential**, **divisional-game flag**, **week of season**.

   All rolling stats are shifted (`shift(1)`) so a game's own outcome/plays
   never leak into its own features. Run with `python3 features.py` → writes
   `features.csv`.

   **Limitation**: for future/unplayed games (e.g. next season's schedule
   before starters are known), `qb_change` and career starts assume the QB
   who started that team's most recent known game continues to start. A
   real in-season injury or benching won't be reflected until `games.csv`
   is refreshed with the actual starter for that week.

4. **`train.py`** — trains two XGBoost models:
   - `model_win.joblib`: classifier → P(home team wins)
   - `model_margin.joblib`: regressor → predicted home margin (points)

   Time-based split (train on the past, test on the future — never shuffled):
   - train: 1999–2023
   - validation: 2024 (early stopping)
   - holdout test: 2025 (final reported numbers, never touched during training)

   Run with `python3 train.py`.

5. **`predict.py`** — loads the saved models and scores upcoming (unplayed) games
   from the current schedule.
   ```
   python3 predict.py                    # all upcoming games
   python3 predict.py --season 2026 --week 1
   ```
   Writes `predictions.csv` and `predictions.json`.

6. **`render_page.py`** — assembles `template.html` + `predictions.json` +
   `metrics.json` (written by `train.py`) into the final static page,
   `nfl_predictions.html`. Publish that file with the Artifact tool, passing
   the existing artifact's URL so it updates in place. The accuracy stat tiles
   and "Updated <date>" tag on the page are pulled live from `metrics.json` at
   render time — never hardcoded — so they can't drift out of sync with the
   model that actually produced the predictions on the page.

### Full refresh, in order

```
python3 build_epa.py     # only needed if new pbp weeks are out; ~few min, ~450MB
python3 features.py
python3 train.py         # writes metrics.json
python3 predict.py       # writes predictions.json
python3 render_page.py   # writes nfl_predictions.html
```
Then publish `nfl_predictions.html` with the Artifact tool.

## Results (holdout test: 2025 season, n=284 games)

| Metric | EPA+QB model | Point-diff model (v1) | Home-field baseline | Vegas closing-line baseline |
|---|---|---|---|---|
| Accuracy (2024 val) | **70.2%** | 65.6% | 54.7% | 70.5% |
| Accuracy (2025 test) | **65.5%** | 64.8% | 53.5% | 65.9% |
| Log loss (2025 test) | 0.628 | 0.625 | — | — |
| Margin MAE (2025 test) | 10.1 pts | 10.1 pts | — | — |

**Reading this honestly**: swapping box-score point differential for play-by-play
EPA efficiency (offense/defense split) plus QB-continuity signal pushed 2024
validation accuracy to within 0.3pt of the Vegas closing line, and 2025 holdout
accuracy improved too, though it's noisier (log loss/margin MAE moved only
marginally). EPA and Elo now dominate feature importance — QB-change events
also show up as a real signal. Beating Vegas consistently is still very hard
(closing lines price in real-time injury news, weather, and sharp money this
model doesn't see), but the gap has closed meaningfully versus the box-score-only
version.

## Extending this

- QB-*specific* Elo (rating tied to the individual QB, not just the team) would
  likely add more than the simple change/experience flags used here — 538's
  NFL model treats this as one of its largest single factors.
- Add weather/roof/surface features for outdoor games.
- Recalibrate probabilities (`CalibratedClassifierCV`) if you need well-calibrated
  probabilities for betting-style decisions.
- Refresh `games.csv` and `team_game_epa.csv` periodically during the season so
  rolling-window features and QB starters stay current (see limitation note above).
