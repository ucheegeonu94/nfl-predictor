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
   - **QB-specific Elo** (`qb_elo_pre`): distinct from the continuity flags above
     — those capture *stability* (did the starter change), this captures
     *quality* (how good is he). It's an EWMA of the team's offensive EPA/play
     in games this specific QB started, rescaled onto an Elo-like number so it
     reads on the same footing as team Elo. New/unseen QBs start at a neutral
     1500 (no draft-pedigree prior, unlike 538's model) and drift from there.
   - **Points allowed** (5/10-game rolling average) — the opponent's score in
     each of a team's recent games.
   - **Penalty yards given/gained** (5/10-game rolling average): yards a team
     committed vs. yards its opponents committed (and thus it benefited from),
     from every penalty on the play-by-play log (including penalties on
     no-play snaps, which the EPA aggregation deliberately excludes).
   - **Rest-day differential**, **divisional-game flag**, **week of season**.

   All rolling stats are shifted (`shift(1)`) so a game's own outcome/plays
   never leak into its own features. Run with `python3 features.py` → writes
   `features.csv`.

   **A real bug found and fixed while adding the two features above**: every
   rolling feature (win%, EPA, points allowed, penalty yards) was computed by
   building a lookup table from *played games only*, then joining it back onto
   the full schedule by `game_id`. That works for historical games (their
   `game_id` is in the lookup table) but silently fails for every future game
   — its `game_id` was never in the played-games-only table, so the join
   produced nothing and the neutral-default fallback (league-average win%,
   zero EPA, etc.) quietly took over. **This affected every prediction for an
   upcoming game ever published from this repo** — Elo, QB Elo, and QB
   continuity were unaffected (they're computed differently) so predictions
   weren't random, but they were missing real recent-form signal the whole
   time. It did *not* affect training/validation/holdout accuracy numbers,
   since those only use played games, which always joined correctly. Fixed by
   building the lookup table from every game (played or not) with the current
   game's own stat as `NaN` when unplayed — pandas' rolling mean skips `NaN`,
   so a team's last real rolling value now correctly carries forward into all
   of its future games instead of disappearing.

   **Limitation**: for future/unplayed games (e.g. next season's schedule
   before starters are known), `qb_change`, career starts, and QB Elo all
   assume the QB who started that team's most recent known game continues to
   start (forward-filled). A real in-season injury or benching won't be
   reflected until `games.csv` is refreshed with the actual starter for that
   week — the page marks these as "(presumed starter)" rather than stating
   them as fact.

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

6. **`build_injuries.py`** — pulls the current season's injury reports from
   nflverse (`injuries_<season>.parquet`) and reduces them to each team's most
   recent weekly report (players with an Out/Doubtful/Questionable status).
   Writes `injuries.json`. **This file doesn't exist until the season has
   actually started** — nflverse has nothing to publish before Week 1's first
   practice report, so an empty report during the offseason is correct
   behavior, not a bug. Even in-season, "most recent report" can lag a few
   days behind an imminent game if the pipeline runs before that week's own
   Wed/Thu/Fri reports are out.

7. **`build_weather.py`** — fetches game-day weather (via the free
   [Open-Meteo](https://open-meteo.com) API, no key needed) for upcoming
   *outdoor* games within its ~15-day forecast horizon. Dome/closed-roof games
   are marked indoor (no forecast needed); international neutral-site games
   (London, Munich, São Paulo, ...) are skipped — their venues rotate yearly
   and aren't worth hardcoding; games further out than ~15 days are marked
   "too far out" since that weather genuinely isn't knowable yet. Team-market
   coordinates are a small hardcoded table (`TEAM_COORDS`) — city-level, not
   exact stadium addresses, which is precise enough for day-ahead game weather.
   Writes `weather.json`.

8. **`build_rankings.py`** — reduces `features.py`'s output to each team's
   *current* rolling offensive/defensive EPA-per-play (every one of a team's
   upcoming games carries the same value, since nothing new has been played
   yet to move it — this just reads it off the earliest upcoming game per
   team). Writes `team_rankings.json`, the data behind the "Team Efficiency
   Rankings" sheet on the page.

9. **`render_page.py`** — assembles `template.html` + `predictions.json` +
   `metrics.json` + `injuries.json` + `weather.json` + `team_rankings.json`
   into the final static page, `nfl_predictions.html`. Publish that file with
   the Artifact tool, passing the existing artifact's URL so it updates in
   place. The accuracy stat tiles and "Updated <date>" tag are pulled live
   from `metrics.json` at render time — never hardcoded — so they can't drift
   out of sync with the model that actually produced the predictions on the
   page. Tapping a team on any game card opens a detail sheet with that
   team's starting QB (name + Elo, flagged "(presumed starter)" when not yet
   officially confirmed), injury report, and that game's weather; a button in
   the header opens the league-wide efficiency rankings (offense/defense
   toggle, sorted lowest-to-highest EPA/play).

### Full refresh, in order

```
python3 build_epa.py       # only needed if new pbp weeks are out; ~few min, ~450MB
python3 features.py
python3 train.py           # writes metrics.json
python3 predict.py         # writes predictions.json
python3 build_injuries.py  # writes injuries.json (empty in the offseason)
python3 build_weather.py   # writes weather.json
python3 build_rankings.py  # writes team_rankings.json
python3 render_page.py     # writes nfl_predictions.html
```
Then publish `nfl_predictions.html` with the Artifact tool.

## Results (holdout test: 2025 season, n=284 games)

| Metric | Current (+pts allowed, +penalties, bug fix, fixed seed) | +QB Elo | EPA+QB (v2) | Point-diff (v1) | Home-field baseline | Vegas closing-line baseline |
|---|---|---|---|---|---|---|
| Accuracy (2024 val) | 69.8% | 70.2% | 70.2% | 65.6% | 54.7% | 70.5% |
| Accuracy (2025 test) | 63.4% | 64.1% | **65.5%** | 64.8% | 53.5% | 65.9% |
| Log loss (2025 test) | 0.631 | 0.630 | 0.628 | 0.625 | — | — |
| Margin MAE (2025 test) | 10.1 pts | 10.2 pts | 10.1 pts | 10.1 pts | — | — |

**Reading this honestly, across the whole series**: swapping box-score point
differential for play-by-play EPA efficiency plus QB-continuity (v1 → v2) was
a clean win. Every addition since (QB Elo, then points-allowed + penalty
yards) has nudged 2024 validation accuracy up while 2025 holdout accuracy
drifted down — a real pattern worth naming honestly rather than one iteration
that could be dismissed as noise. The most likely explanation is overfitting
to the 285-game validation set via repeated early-stopping selection as the
feature count grows, though a single 284-game holdout season is also
genuinely noisy on its own (~2.8pt standard error). **`v2` (EPA + QB
continuity, no QB Elo, no points-allowed/penalties) remains the best holdout
number produced so far** — everything after it added real, legitimate,
backtestable signal (confirmed non-trivial feature importance every time) but
hasn't yet produced a model that beats it on the one number that matters most.
The new features are still worth having in the model for the *reasons* the
user asked for them (points allowed and penalty discipline are real,
well-understood aspects of team performance), but this is flagged here rather
than glossed over. `train.py` now sets `random_state=42` on both models so
future comparisons are reproducible and this kind of before/after read is
trustworthy going forward — it wasn't set previously, so some of the
iteration-to-iteration swings reported earlier in this project's history may
have included training-seed noise on top of genuine feature effects.

## Extending this

- **Run an ablation, not just cumulative addition.** The results table above
  only ever tested "current best + one more feature." Given the validation/
  holdout divergence, the next real step is testing each new feature (QB Elo,
  points allowed, penalties) in isolation against the v2 baseline, and
  increasing regularization (`reg_lambda`, `max_depth`, `min_child_weight`) as
  features are added, rather than always keeping everything from every past
  addition.
- Tune `QB_ELO_ALPHA`/`QB_ELO_SCALE` (currently 0.18/600, chosen by inspection
  not a search).
- A draft-pedigree prior for rookie QBs (538 seeds by draft position instead of
  a flat 1500) would likely sharpen QB Elo in a player's first few starts.
- Recalibrate probabilities (`CalibratedClassifierCV`) if you need well-calibrated
  probabilities for betting-style decisions.
- Weather/injury data is currently sidebar-only (not fed into the model itself).
  Feeding wind speed into the margin regressor, or a team's Out/Doubtful count
  into the win classifier, are natural next features once there's enough
  in-season data to backtest them against.
- Refresh `games.csv` and `team_game_epa.csv` periodically during the season so
  rolling-window features and QB starters stay current (see limitation note above).
