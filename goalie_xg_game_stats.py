"""
Build a game-level goalie DataFrame with xG faced and Goals Saved Above Expected (GSAx).

For each goalie-game in the past 5 NHL regular seasons (2020-21 through 2024-25):
  - date, goalie name, team, opposing team, xG against, goals against, GSAx

Pipeline:
  1. Collect all regular-season game IDs from the NHL API.
  2. Fetch play-by-play data for each game.
  3. Extract fenwick events (shots on goal + goals + missed shots) with xG features.
  4. Train an XGBoost expected-goals model on the shot data.
  5. Apply the model to compute per-shot xG, then aggregate to goalie-game level.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://api-web.nhle.com/v1"
SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025"]
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data")
REQUEST_DELAY = 0.35  # seconds between API requests to be respectful

# NHL teams (3-letter abbreviations) — all 32 current teams
TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI",
    "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
    "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA",
    "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# Mapping for teams that relocated / rebranded mid-window
TEAM_RENAMES = {"ARI": "ARI", "UTA": "UTA"}  # ARI became UTA in 2024-25

# Shot types used as one-hot features
SHOT_TYPES = ["wrist", "slap", "snap", "backhand", "tip-in", "deflected", "wrap-around", "cradle", "bat"]

# Event types that count as unblocked shot attempts (Fenwick)
FENWICK_EVENTS = {"shot-on-goal", "goal", "missed-shot"}

# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GoalieXGScript/1.0"})


def _get_json(url: str, retries: int = 3) -> dict | None:
    """GET JSON from the NHL API with retries and caching."""
    cache_key = url.replace("/", "_").replace(":", "")
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            # Cache to disk
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            time.sleep(REQUEST_DELAY)
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt+1}/{retries} for {url}: {e} — waiting {wait}s")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# 1. Collect game IDs
# ---------------------------------------------------------------------------
def get_game_ids_for_season(season: str) -> list[int]:
    """Return sorted list of unique regular-season game IDs for a season.

    Uses club-schedule-season for each team, deduplicates across teams.
    """
    game_ids = set()

    # Determine which team abbreviations existed for this season
    teams_for_season = []
    season_start = int(season[:4])
    for t in TEAMS:
        # Arizona existed as ARI through 2023-24, became UTA in 2024-25
        if t == "UTA" and season_start < 2024:
            continue
        if t == "ARI" and season_start >= 2024:
            continue
        # Seattle entered in 2021-22
        if t == "SEA" and season_start < 2021:
            continue
        teams_for_season.append(t)

    print(f"Fetching schedule for {season} ({len(teams_for_season)} teams)...")
    for team in tqdm(teams_for_season, desc=f"  Schedules {season}"):
        url = f"{BASE_URL}/club-schedule-season/{team}/{season}"
        data = _get_json(url)
        if data is None:
            continue
        for game in data.get("games", []):
            game_type = game.get("gameType", 0)
            if game_type == 2:  # regular season only
                game_ids.add(game["id"])

    return sorted(game_ids)


def collect_all_game_ids() -> list[int]:
    """Collect game IDs across all configured seasons."""
    cache_path = CACHE_DIR / "all_game_ids.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    all_ids = []
    for season in SEASONS:
        ids = get_game_ids_for_season(season)
        print(f"  {season}: {len(ids)} regular-season games")
        all_ids.extend(ids)

    all_ids = sorted(set(all_ids))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_ids, f)
    print(f"Total unique game IDs: {len(all_ids)}")
    return all_ids


# ---------------------------------------------------------------------------
# 2. Fetch play-by-play and extract shot data
# ---------------------------------------------------------------------------
def _shot_distance(x: float, y: float) -> float:
    """Euclidean distance from shot location to centre of the net (89, 0)."""
    return math.sqrt((abs(x) - 89) ** 2 + y ** 2)


def _shot_angle(x: float, y: float) -> float:
    """Angle in degrees of the shot relative to the centre line of the net."""
    dx = 89 - abs(x)
    if dx <= 0:
        return 90.0
    return math.degrees(math.atan(abs(y) / dx))


def _parse_strength(situation_code: str) -> tuple[int, int, int, int]:
    """Parse situationCode like '1551' → (away_goalie, away_skaters, home_skaters, home_goalie)."""
    if not situation_code or len(situation_code) != 4:
        return (1, 5, 5, 1)
    return (
        int(situation_code[0]),
        int(situation_code[1]),
        int(situation_code[2]),
        int(situation_code[3]),
    )


def extract_shots_from_game(game_id: int) -> list[dict]:
    """Fetch PBP for one game and return a list of shot-level feature dicts."""
    url = f"{BASE_URL}/gamecenter/{game_id}/play-by-play"
    data = _get_json(url)
    if data is None:
        return []

    # Game metadata
    away_team = data.get("awayTeam", {})
    home_team = data.get("homeTeam", {})
    away_abbrev = away_team.get("abbrev", "UNK")
    home_abbrev = home_team.get("abbrev", "UNK")
    game_date = data.get("gameDate", "")

    # Build player name lookup from roster
    player_names: dict[int, str] = {}
    for roster_spot in data.get("rosterSpots", []):
        pid = roster_spot.get("playerId")
        first = roster_spot.get("firstName", {})
        last = roster_spot.get("lastName", {})
        if pid:
            fname = first.get("default", "") if isinstance(first, dict) else str(first)
            lname = last.get("default", "") if isinstance(last, dict) else str(last)
            player_names[pid] = f"{fname} {lname}".strip()

    shots = []
    prev_event_type = None
    prev_event_time = None
    prev_event_x = None
    prev_event_y = None

    for play in data.get("plays", []):
        event_type = play.get("typeDescKey", "")
        period_desc = play.get("periodDescriptor", {})
        period_num = period_desc.get("number", 0)
        period_type = period_desc.get("periodType", "REG")

        # Skip shootout events for xG purposes
        if period_type == "SO":
            prev_event_type = event_type
            continue

        time_str = play.get("timeInPeriod", "00:00")
        try:
            parts = time_str.split(":")
            elapsed_seconds = int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            elapsed_seconds = 0

        details = play.get("details", {})

        if event_type in FENWICK_EVENTS:
            x = details.get("xCoord")
            y = details.get("yCoord")
            if x is None or y is None:
                prev_event_type = event_type
                continue

            is_goal = 1 if event_type == "goal" else 0
            shot_on_goal = 1 if event_type in ("shot-on-goal", "goal") else 0

            # Determine shooting team and goalie
            shooting_player_id = details.get("shootingPlayerId") or details.get("scoringPlayerId")
            goalie_id = details.get("goalieInNetId")
            shot_type = details.get("shotType", "unknown")

            # Figure out which team is shooting
            event_owner_id = details.get("eventOwnerTeamId")
            if event_owner_id == away_team.get("id"):
                shooting_team = away_abbrev
                goalie_team = home_abbrev
            elif event_owner_id == home_team.get("id"):
                shooting_team = home_abbrev
                goalie_team = away_abbrev
            else:
                # Fallback: infer from coordinates
                if x > 0:
                    shooting_team = away_abbrev
                    goalie_team = home_abbrev
                else:
                    shooting_team = home_abbrev
                    goalie_team = away_abbrev

            # Strength / situation
            sit_code = play.get("situationCode", "1551")
            away_g, away_sk, home_sk, home_g = _parse_strength(sit_code)

            if shooting_team == away_abbrev:
                shooter_skaters = away_sk
                goalie_skaters = home_sk
                empty_net = home_g == 0
            else:
                shooter_skaters = home_sk
                goalie_skaters = away_sk
                empty_net = away_g == 0

            manpower_diff = shooter_skaters - goalie_skaters  # +1 = PP, -1 = SH

            # Distance and angle (normalize x to always be positive = offensive zone)
            dist = _shot_distance(x, y)
            angle = _shot_angle(x, y)

            # Rebound: previous event was a shot within 3 seconds
            is_rebound = 0
            time_since_last = None
            if prev_event_type in FENWICK_EVENTS and prev_event_time is not None:
                time_since_last = elapsed_seconds - prev_event_time
                if 0 < time_since_last <= 3:
                    is_rebound = 1

            # Rush: large change in x-coordinate from previous event
            is_rush = 0
            if prev_event_x is not None and prev_event_type is not None:
                x_change = abs(abs(x) - abs(prev_event_x))
                if prev_event_type not in FENWICK_EVENTS and x_change > 50:
                    is_rush = 1

            # Distance from last event (for sequencing)
            change_in_shot_angle = 0.0
            if prev_event_x is not None and prev_event_y is not None:
                prev_angle = _shot_angle(prev_event_x, prev_event_y)
                change_in_shot_angle = abs(angle - prev_angle)

            shot = {
                "game_id": game_id,
                "game_date": game_date,
                "period": period_num,
                "period_type": period_type,
                "time_in_period": elapsed_seconds,
                "event_type": event_type,
                "is_goal": is_goal,
                "shot_on_goal": shot_on_goal,
                "x_coord": x,
                "y_coord": y,
                "shot_distance": dist,
                "shot_angle": angle,
                "shot_type": shot_type,
                "is_rebound": is_rebound,
                "is_rush": is_rush,
                "time_since_last_event": time_since_last,
                "change_in_shot_angle": change_in_shot_angle,
                "manpower_diff": manpower_diff,
                "empty_net": int(empty_net),
                "shooting_player_id": shooting_player_id,
                "shooting_player_name": player_names.get(shooting_player_id, "Unknown"),
                "shooting_team": shooting_team,
                "goalie_id": goalie_id,
                "goalie_name": player_names.get(goalie_id, "Unknown") if goalie_id else "Empty Net",
                "goalie_team": goalie_team,
                "away_team": away_abbrev,
                "home_team": home_abbrev,
            }
            shots.append(shot)

        # Track previous event for rebound/rush detection
        prev_event_type = event_type
        prev_event_time = elapsed_seconds
        if details.get("xCoord") is not None:
            prev_event_x = details["xCoord"]
            prev_event_y = details.get("yCoord", 0)

    return shots


# ---------------------------------------------------------------------------
# 3. Build the full shot DataFrame
# ---------------------------------------------------------------------------
def build_shot_dataframe(game_ids: list[int]) -> pd.DataFrame:
    """Iterate over all games, extract shots, return a single DataFrame."""
    shots_cache = OUTPUT_DIR / "shots_all.parquet"
    if shots_cache.exists():
        print(f"Loading cached shot data from {shots_cache}")
        return pd.read_parquet(shots_cache)

    all_shots = []
    failed_games = []

    for gid in tqdm(game_ids, desc="Fetching play-by-play"):
        shots = extract_shots_from_game(gid)
        if shots:
            all_shots.extend(shots)
        else:
            failed_games.append(gid)

    if failed_games:
        print(f"Warning: {len(failed_games)} games returned no data.")

    df = pd.DataFrame(all_shots)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(shots_cache, index=False)
    print(f"Shot DataFrame: {len(df)} rows across {df['game_id'].nunique()} games")
    return df


# ---------------------------------------------------------------------------
# 4. Train xG model
# ---------------------------------------------------------------------------
XG_FEATURES = [
    "shot_distance",
    "shot_angle",
    "is_rebound",
    "is_rush",
    "time_since_last_event",
    "change_in_shot_angle",
    "manpower_diff",
    "empty_net",
    "period",
] + [f"shot_type_{st}" for st in SHOT_TYPES]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot shot type columns and fill NAs for model features."""
    df = df.copy()
    for st in SHOT_TYPES:
        df[f"shot_type_{st}"] = (df["shot_type"] == st).astype(int)
    df["time_since_last_event"] = df["time_since_last_event"].fillna(999)
    return df


def train_xg_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    """Train an XGBoost expected-goals model. Returns the fitted model."""
    model_path = OUTPUT_DIR / "xg_model.ubj"
    df = prepare_features(df)

    # Only train on non-empty-net shots (empty net goals are ~1.0 xG trivially)
    train_df = df[df["empty_net"] == 0].copy()

    X = train_df[XG_FEATURES].values
    y = train_df["is_goal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_path.exists():
        print(f"Loading cached xG model from {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        print("Training xG model...")
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"  Test Log Loss: {log_loss(y_test, y_pred):.4f}")
    print(f"  Test ROC AUC:  {roc_auc_score(y_test, y_pred):.4f}")

    return model


# ---------------------------------------------------------------------------
# 5. Score all shots and aggregate to goalie-game level
# ---------------------------------------------------------------------------
def score_shots(df: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    """Add xG column to the shot DataFrame."""
    df = prepare_features(df)
    X = df[XG_FEATURES].values

    # Predict xG for all shots
    xg_probs = model.predict_proba(X)[:, 1]

    # Override empty-net shots to xG = 1.0
    df["xG"] = xg_probs
    df.loc[df["empty_net"] == 1, "xG"] = 1.0

    return df


def build_goalie_game_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shot-level data to goalie-game level.

    Output columns:
      - game_date: date of the game
      - goalie_name: full name of the goalie
      - goalie_team: team the goalie played for
      - opposing_team: the other team
      - shots_against: number of unblocked shot attempts faced
      - goals_against: goals allowed
      - xG_against: total expected goals faced
      - GSAx: goals saved above expected (xG_against - goals_against)
    """
    # Exclude empty-net situations from goalie evaluation (no goalie in net)
    goalie_df = df[df["goalie_id"].notna() & (df["goalie_name"] != "Empty Net")].copy()

    grouped = goalie_df.groupby(["game_id", "game_date", "goalie_id", "goalie_name", "goalie_team"]).agg(
        shots_against=("is_goal", "count"),
        goals_against=("is_goal", "sum"),
        xG_against=("xG", "sum"),
        opposing_team=("shooting_team", "first"),
    ).reset_index()

    grouped["GSAx"] = grouped["xG_against"] - grouped["goals_against"]
    grouped["game_date"] = pd.to_datetime(grouped["game_date"]).dt.date

    # Select and order final columns
    result = grouped[[
        "game_date",
        "goalie_name",
        "goalie_team",
        "opposing_team",
        "shots_against",
        "goals_against",
        "xG_against",
        "GSAx",
        "game_id",
        "goalie_id",
    ]].sort_values(["game_date", "goalie_name"]).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Goalie Game-Level xG & GSAx Builder")
    print("Seasons:", ", ".join(s[:4] + "-" + s[4:] for s in SEASONS))
    print("=" * 60)

    # Step 1: Collect game IDs
    game_ids = collect_all_game_ids()

    # Step 2: Build shot-level DataFrame from play-by-play
    shots_df = build_shot_dataframe(game_ids)
    print(f"\nShot data shape: {shots_df.shape}")
    print(f"  Games: {shots_df['game_id'].nunique()}")
    print(f"  Goals: {shots_df['is_goal'].sum()}")
    print(f"  Goal rate: {shots_df['is_goal'].mean():.3%}")

    # Step 3: Train xG model
    model = train_xg_model(shots_df)

    # Step 4: Score all shots
    scored_df = score_shots(shots_df, model)

    # Step 5: Aggregate to goalie-game level
    goalie_games = build_goalie_game_dataframe(scored_df)

    # Save output
    output_path = OUTPUT_DIR / "goalie_game_stats.csv"
    goalie_games.to_csv(output_path, index=False)
    print(f"\nGoalie game stats saved to {output_path}")
    print(f"  Rows: {len(goalie_games)}")
    print(f"  Unique goalies: {goalie_games['goalie_name'].nunique()}")
    print(f"  Date range: {goalie_games['game_date'].min()} to {goalie_games['game_date'].max()}")

    # Also save as parquet for downstream analysis
    goalie_games.to_parquet(OUTPUT_DIR / "goalie_game_stats.parquet", index=False)

    print("\nSample output:")
    print(goalie_games.head(10).to_string(index=False))

    return goalie_games


if __name__ == "__main__":
    main()
