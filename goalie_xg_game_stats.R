# =============================================================================
# Goalie Game-Level xG & GSAx Builder (R / tidymodels)
#
# For each goalie-game in the past 5 NHL regular seasons (2020-21 to 2024-25):
#   game_date, goalie_name, goalie_team, opposing_team, xG_against, GSAx
#
# Pipeline:
#   1. Collect all regular-season game IDs from the NHL API
#   2. Fetch play-by-play for each game
#   3. Extract Fenwick events (shots on goal + goals + missed shots) with features
#   4. Train an XGBoost xG model via tidymodels
#   5. Score shots, aggregate to goalie-game level
# =============================================================================

library(tidyverse)
library(httr2)
library(jsonlite)
library(tidymodels)
library(xgboost)
library(arrow)
library(duckdb)
library(DBI)
library(cli)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL    <- "https://api-web.nhle.com/v1"
SEASONS     <- c("20202021", "20212022", "20222023", "20232024", "20242025")
CACHE_DIR   <- "data/cache"
OUTPUT_DIR  <- "data"
REQUEST_DELAY <- 0.35

TEAMS <- c(
 "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI",
 "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
 "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA",
 "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"
)

FENWICK_EVENTS <- c("shot-on-goal", "goal", "missed-shot")

dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

#' GET JSON from the NHL API with disk caching and retries
get_json <- function(url, retries = 3) {
  cache_key  <- url |> str_replace_all("[/:]", "_")
  cache_path <- file.path(CACHE_DIR, paste0(cache_key, ".json"))

  if (file.exists(cache_path)) {
    return(fromJSON(cache_path, simplifyVector = FALSE))
  }

  for (attempt in seq_len(retries)) {
    result <- tryCatch({
      resp <- request(url) |>
        req_headers(`User-Agent` = "GoalieXGScript/1.0") |>
        req_timeout(30) |>
        req_perform()

      if (resp_status(resp) == 404) return(NULL)

      data <- resp |> resp_body_json(simplifyVector = FALSE)

      # Cache to disk
      write_json(data, cache_path, auto_unbox = TRUE)
      Sys.sleep(REQUEST_DELAY)
      return(data)
    },
    error = function(e) {
      wait <- 2^(attempt + 1)
      cli_alert_warning("Retry {attempt}/{retries} for {url}: {e$message} -- waiting {wait}s")
      Sys.sleep(wait)
      return(NULL)
    })

    if (!is.null(result)) return(result)
  }

  return(NULL)
}

# ---------------------------------------------------------------------------
# 1. Collect game IDs
# ---------------------------------------------------------------------------

#' Get all regular-season game IDs for one season
get_game_ids_for_season <- function(season) {
  season_start <- as.integer(substr(season, 1, 4))

  teams_for_season <- TEAMS
  # Arizona became Utah in 2024-25
  if (season_start < 2024) {
    teams_for_season <- setdiff(teams_for_season, "UTA")
  }
  if (season_start >= 2024) {
    teams_for_season <- setdiff(teams_for_season, "ARI")
  }
  # Seattle entered in 2021-22
  if (season_start < 2021) {
    teams_for_season <- setdiff(teams_for_season, "SEA")
  }

  cli_alert_info("Fetching schedule for {season} ({length(teams_for_season)} teams)...")

  game_ids <- integer(0)

  for (team in teams_for_season) {
    url  <- paste0(BASE_URL, "/club-schedule-season/", team, "/", season)
    data <- get_json(url)
    if (is.null(data)) next

    for (game in data$games) {
      if (identical(game$gameType, 2L) || identical(game$gameType, 2)) {
        game_ids <- c(game_ids, game$id)
      }
    }
  }

  sort(unique(game_ids))
}

#' Collect game IDs across all configured seasons
collect_all_game_ids <- function() {
  cache_path <- file.path(CACHE_DIR, "all_game_ids.rds")
  if (file.exists(cache_path)) {
    cli_alert_success("Loading cached game IDs from {cache_path}")
    return(readRDS(cache_path))
  }

  all_ids <- integer(0)
  for (season in SEASONS) {
    ids <- get_game_ids_for_season(season)
    cli_alert_info("  {season}: {length(ids)} regular-season games")
    all_ids <- c(all_ids, ids)
  }

  all_ids <- sort(unique(all_ids))
  saveRDS(all_ids, cache_path)
  cli_alert_success("Total unique game IDs: {length(all_ids)}")
  all_ids
}

# ---------------------------------------------------------------------------
# 2. Fetch play-by-play and extract shot data
# ---------------------------------------------------------------------------

#' Euclidean distance from shot location to centre of net (89, 0)
shot_distance <- function(x, y) {
  sqrt((abs(x) - 89)^2 + y^2)
}

#' Shot angle in degrees relative to centre line of the net
shot_angle <- function(x, y) {
  dx <- 89 - abs(x)
  ifelse(dx <= 0, 90, atan(abs(y) / dx) * 180 / pi)
}

#' Parse situationCode "1551" -> list(away_g, away_sk, home_sk, home_g)
parse_strength <- function(sit_code) {
  if (is.null(sit_code) || nchar(as.character(sit_code)) != 4) {
    return(list(away_g = 1L, away_sk = 5L, home_sk = 5L, home_g = 1L))
  }
  digits <- as.integer(strsplit(as.character(sit_code), "")[[1]])
  list(away_g = digits[1], away_sk = digits[2], home_sk = digits[3], home_g = digits[4])
}

#' Extract shot-level features from one game's play-by-play
extract_shots_from_game <- function(game_id) {
  url  <- paste0(BASE_URL, "/gamecenter/", game_id, "/play-by-play")
  data <- get_json(url)
  if (is.null(data)) return(tibble())

 # Game metadata
  away_team   <- data$awayTeam
  home_team   <- data$homeTeam
  away_abbrev <- away_team$abbrev %||% "UNK"
  home_abbrev <- home_team$abbrev %||% "UNK"
  game_date   <- data$gameDate %||% ""

  # Build player name lookup from roster
  player_names <- list()
  for (spot in data$rosterSpots) {
    pid <- spot$playerId
    if (!is.null(pid)) {
      fname <- if (is.list(spot$firstName)) spot$firstName$default else as.character(spot$firstName)
      lname <- if (is.list(spot$lastName))  spot$lastName$default  else as.character(spot$lastName)
      player_names[[as.character(pid)]] <- trimws(paste(fname %||% "", lname %||% ""))
    }
  }

  lookup_name <- function(pid) {
    if (is.null(pid)) return(NA_character_)
    player_names[[as.character(pid)]] %||% "Unknown"
  }

  plays <- data$plays
  if (is.null(plays) || length(plays) == 0) return(tibble())

  shots <- vector("list", length(plays))
  shot_idx <- 0L

  prev_event_type <- NULL
  prev_event_time <- NULL
  prev_event_x    <- NULL
  prev_event_y    <- NULL

  for (play in plays) {
    event_type  <- play$typeDescKey %||% ""
    period_desc <- play$periodDescriptor
    period_num  <- period_desc$number %||% 0L
    period_type <- period_desc$periodType %||% "REG"

    # Skip shootout
    if (identical(period_type, "SO")) {
      prev_event_type <- event_type
      next
    }

    time_str <- play$timeInPeriod %||% "00:00"
    parts    <- as.integer(strsplit(time_str, ":")[[1]])
    elapsed  <- if (length(parts) == 2) parts[1] * 60L + parts[2] else 0L

    details <- play$details

    if (event_type %in% FENWICK_EVENTS) {
      x <- details$xCoord
      y <- details$yCoord
      if (is.null(x) || is.null(y)) {
        prev_event_type <- event_type
        next
      }

      is_goal     <- as.integer(event_type == "goal")
      shot_on_net <- as.integer(event_type %in% c("shot-on-goal", "goal"))

      shooting_player_id <- details$shootingPlayerId %||% details$scoringPlayerId
      goalie_id          <- details$goalieInNetId
      shot_type_val      <- details$shotType %||% "unknown"

      # Determine shooting team
      event_owner_id <- details$eventOwnerTeamId
      if (!is.null(event_owner_id) && identical(event_owner_id, away_team$id)) {
        shooting_team <- away_abbrev
        goalie_team   <- home_abbrev
      } else if (!is.null(event_owner_id) && identical(event_owner_id, home_team$id)) {
        shooting_team <- home_abbrev
        goalie_team   <- away_abbrev
      } else {
        if (x > 0) {
          shooting_team <- away_abbrev
          goalie_team   <- home_abbrev
        } else {
          shooting_team <- home_abbrev
          goalie_team   <- away_abbrev
        }
      }

      # Strength / situation
      str_info <- parse_strength(play$situationCode %||% "1551")

      if (identical(shooting_team, away_abbrev)) {
        shooter_skaters <- str_info$away_sk
        goalie_skaters  <- str_info$home_sk
        empty_net       <- str_info$home_g == 0L
      } else {
        shooter_skaters <- str_info$home_sk
        goalie_skaters  <- str_info$away_sk
        empty_net       <- str_info$away_g == 0L
      }

      manpower_diff <- shooter_skaters - goalie_skaters

      dist  <- shot_distance(x, y)
      angle <- shot_angle(x, y)

      # Rebound detection
      is_rebound         <- 0L
      time_since_last    <- NA_real_
      if (!is.null(prev_event_type) && prev_event_type %in% FENWICK_EVENTS && !is.null(prev_event_time)) {
        time_since_last <- elapsed - prev_event_time
        if (!is.na(time_since_last) && time_since_last > 0 && time_since_last <= 3) {
          is_rebound <- 1L
        }
      }

      # Rush detection
      is_rush <- 0L
      if (!is.null(prev_event_x) && !is.null(prev_event_type)) {
        x_change <- abs(abs(x) - abs(prev_event_x))
        if (!(prev_event_type %in% FENWICK_EVENTS) && x_change > 50) {
          is_rush <- 1L
        }
      }

      # Change in shot angle from previous event
      change_in_angle <- 0
      if (!is.null(prev_event_x) && !is.null(prev_event_y)) {
        prev_angle      <- shot_angle(prev_event_x, prev_event_y)
        change_in_angle <- abs(angle - prev_angle)
      }

      goalie_name <- if (is.null(goalie_id)) "Empty Net" else lookup_name(goalie_id)

      shot_idx <- shot_idx + 1L
      shots[[shot_idx]] <- tibble(
        game_id               = game_id,
        game_date             = game_date,
        period                = period_num,
        period_type           = period_type,
        time_in_period        = elapsed,
        event_type            = event_type,
        is_goal               = is_goal,
        shot_on_goal          = shot_on_net,
        x_coord               = x,
        y_coord               = y,
        shot_distance         = dist,
        shot_angle            = angle,
        shot_type             = shot_type_val,
        is_rebound            = is_rebound,
        is_rush               = is_rush,
        time_since_last_event = time_since_last,
        change_in_shot_angle  = change_in_angle,
        manpower_diff         = manpower_diff,
        empty_net             = as.integer(empty_net),
        shooting_player_id    = shooting_player_id %||% NA_integer_,
        shooting_player_name  = lookup_name(shooting_player_id),
        shooting_team         = shooting_team,
        goalie_id             = goalie_id %||% NA_integer_,
        goalie_name           = goalie_name,
        goalie_team           = goalie_team,
        away_team             = away_abbrev,
        home_team             = home_abbrev
      )
    }

    # Track previous event
    prev_event_type <- event_type
    prev_event_time <- elapsed
    if (!is.null(details$xCoord)) {
      prev_event_x <- details$xCoord
      prev_event_y <- details$yCoord %||% 0
    }
  }

  if (shot_idx == 0L) return(tibble())
  bind_rows(shots[seq_len(shot_idx)])
}

# ---------------------------------------------------------------------------
# 3. DuckDB connection (in-memory)
# ---------------------------------------------------------------------------

init_duckdb <- function() {
  con <- dbConnect(duckdb(), dbdir = ":memory:")
  cli_alert_success("Initialized in-memory DuckDB database")
  con
}

# ---------------------------------------------------------------------------
# 3b. Build the full shot data frame and store in DuckDB
# ---------------------------------------------------------------------------

build_shot_dataframe <- function(game_ids, con) {
  cache_path <- file.path(OUTPUT_DIR, "shots_all.parquet")
  if (file.exists(cache_path)) {
    cli_alert_success("Loading cached shot data into DuckDB from {cache_path}")
    dbExecute(con, sprintf(
      "CREATE TABLE shots AS SELECT * FROM read_parquet('%s')", cache_path
    ))
    row_count <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM shots")$n
    game_count <- dbGetQuery(con, "SELECT COUNT(DISTINCT game_id) AS n FROM shots")$n
    cli_alert_success("Shot data: {row_count} rows across {game_count} games (in DuckDB)")
    return(invisible(NULL))
  }

  n_games     <- length(game_ids)
  failed      <- integer(0)
  all_shots   <- vector("list", n_games)

  cli_alert_info("Fetching play-by-play for {n_games} games...")

  for (i in seq_along(game_ids)) {
    if (i %% 100 == 0 || i == n_games) {
      cli_alert_info("  Progress: {i}/{n_games}")
    }
    gid    <- game_ids[i]
    result <- extract_shots_from_game(gid)
    if (nrow(result) > 0) {
      all_shots[[i]] <- result
    } else {
      failed <- c(failed, gid)
    }
  }

  if (length(failed) > 0) {
    cli_alert_warning("{length(failed)} games returned no data.")
  }

  df <- bind_rows(all_shots)

  # Write parquet cache then load into DuckDB directly from parquet
  write_parquet(df, cache_path)
  dbExecute(con, sprintf(
    "CREATE TABLE shots AS SELECT * FROM read_parquet('%s')", cache_path
  ))

  row_count <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM shots")$n
  game_count <- dbGetQuery(con, "SELECT COUNT(DISTINCT game_id) AS n FROM shots")$n
  cli_alert_success("Shot data: {row_count} rows across {game_count} games (in DuckDB)")
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# 4. Train xG model via tidymodels
# ---------------------------------------------------------------------------

train_xg_model <- function(con) {
  model_path <- file.path(OUTPUT_DIR, "xg_workflow.rds")

  if (file.exists(model_path)) {
    cli_alert_success("Loading cached xG workflow from {model_path}")
    return(readRDS(model_path))
  }

  cli_alert_info("Preparing data for xG model (querying DuckDB)...")

  # Save training data as a persistent DuckDB table for reuse
  dbExecute(con, "DROP TABLE IF EXISTS training_shots")
  dbExecute(con, "
    CREATE TABLE training_shots AS
    SELECT shot_distance, shot_angle, is_rebound, is_rush,
           COALESCE(time_since_last_event, 999) AS time_since_last_event,
           change_in_shot_angle, manpower_diff, empty_net, period,
           shot_type, is_goal
    FROM shots
    WHERE empty_net = 0
  ")
  n_train <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM training_shots")$n
  cli_alert_success("Saved {n_train} training rows to DuckDB table 'training_shots'")

  train_data <- dbGetQuery(con, "SELECT * FROM training_shots") |>
    as_tibble() |>
    mutate(
      is_goal   = factor(is_goal, levels = c(1, 0), labels = c("goal", "no_goal")),
      shot_type = factor(shot_type)
    )

  # tidymodels split
  set.seed(42)
  split    <- initial_split(train_data, prop = 0.8, strata = is_goal)
  train_set <- training(split)
  test_set  <- testing(split)

  # Recipe: define preprocessing
  xg_recipe <- recipe(is_goal ~ shot_distance + shot_angle + is_rebound + is_rush +
                         time_since_last_event + change_in_shot_angle +
                         manpower_diff + empty_net + period + shot_type,
                       data = train_set) |>
    step_dummy(shot_type, one_hot = TRUE) |>
    step_zv(all_predictors())

  # Model spec
  xg_spec <- boost_tree(
    trees          = 500,
    tree_depth     = 5,
    learn_rate     = 0.05,
    sample_size    = 0.8,
    mtry           = 0.8,
    min_n          = 50
  ) |>
    set_engine("xgboost",
               objective    = "binary:logistic",
               eval_metric  = "logloss",
               colsample_bytree = 0.8) |>
    set_mode("classification")

  # Workflow
  xg_wf <- workflow() |>
    add_recipe(xg_recipe) |>
    add_model(xg_spec)

  cli_alert_info("Training xG model...")
  xg_fit <- tryCatch(
    xg_wf |> fit(data = train_set),
    error = function(e) {
      cli_alert_danger("Model training failed: {e$message}")
      cli_alert_warning("Play-by-play data is preserved in DuckDB tables 'shots' and 'training_shots'")
      return(NULL)
    }
  )

  if (is.null(xg_fit)) return(NULL)

  # Evaluate on test set
  test_preds <- predict(xg_fit, test_set, type = "prob") |>
    bind_cols(test_set |> select(is_goal))

  ll <- mn_log_loss(test_preds, truth = is_goal, .pred_goal) |> pull(.estimate)
  auc <- roc_auc(test_preds, truth = is_goal, .pred_goal)  |> pull(.estimate)

  cli_alert_success("  Test Log Loss: {round(ll, 4)}")
  cli_alert_success("  Test ROC AUC:  {round(auc, 4)}")

  # Save fitted workflow
  saveRDS(xg_fit, model_path)
  cli_alert_success("Model saved to {model_path}")

  xg_fit
}

# ---------------------------------------------------------------------------
# 5. Score all shots and aggregate to goalie-game level
# ---------------------------------------------------------------------------

score_shots <- function(con, xg_fit) {
  cli_alert_info("Scoring all shots with xG model (from DuckDB)...")

  # Pull all shots from DuckDB for scoring
  scored <- dbGetQuery(con, "
    SELECT *,
           COALESCE(time_since_last_event, 999) AS tse_filled
    FROM shots
  ") |>
    as_tibble() |>
    mutate(
      shot_type = factor(shot_type),
      time_since_last_event = tse_filled
    ) |>
    select(-tse_filled)

  # Predict xG (probability of goal)
  preds <- tryCatch(
    predict(xg_fit, scored, type = "prob"),
    error = function(e) {
      cli_alert_danger("Scoring failed: {e$message}")
      cli_alert_warning("Play-by-play data is preserved in DuckDB tables 'shots' and 'training_shots'")
      return(NULL)
    }
  )

  if (is.null(preds)) return(invisible(FALSE))

  scored <- scored |>
    mutate(
      xG = preds$.pred_goal,
      # Override empty-net shots to xG = 1.0
      xG = if_else(empty_net == 1L, 1.0, xG)
    )

  # Write scored shots back into DuckDB as a new table
  dbExecute(con, "DROP TABLE IF EXISTS scored_shots")
  dbWriteTable(con, "scored_shots", scored)
  cli_alert_success("Scored {nrow(scored)} shots -> DuckDB table 'scored_shots'")

  invisible(TRUE)
}

build_goalie_game_dataframe <- function(con) {
  cli_alert_info("Aggregating to goalie-game level (SQL in DuckDB)...")

  goalie_games <- dbGetQuery(con, "
    SELECT
      CAST(game_date AS DATE) AS game_date,
      goalie_name,
      goalie_team,
      FIRST(shooting_team) AS opposing_team,
      COUNT(*) AS shots_against,
      SUM(is_goal) AS goals_against,
      SUM(xG) AS xG_against,
      SUM(xG) - SUM(is_goal) AS GSAx,
      game_id,
      goalie_id
    FROM scored_shots
    WHERE goalie_id IS NOT NULL
      AND goalie_name != 'Empty Net'
    GROUP BY game_id, game_date, goalie_id, goalie_name, goalie_team
    ORDER BY game_date, goalie_name
  ") |> as_tibble()

  goalie_games
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  cli_h1("Goalie Game-Level xG & GSAx Builder")
  cli_alert_info("Seasons: {paste(SEASONS, collapse = ', ')}")

  # Initialize in-memory DuckDB
  con <- init_duckdb()
  on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)

  # Step 1: Collect game IDs
  game_ids <- collect_all_game_ids()

  # Step 2: Build shot-level data and load into DuckDB
  build_shot_dataframe(game_ids, con)

  cli_h2("Shot data summary (from DuckDB)")
  summary <- dbGetQuery(con, "
    SELECT COUNT(*) AS rows,
           COUNT(DISTINCT game_id) AS games,
           SUM(is_goal) AS goals,
           ROUND(AVG(is_goal) * 100, 2) AS goal_rate
    FROM shots
  ")
  cli_alert_info("Rows: {summary$rows}")
  cli_alert_info("Games: {summary$games}")
  cli_alert_info("Goals: {summary$goals}")
  cli_alert_info("Goal rate: {summary$goal_rate}%")

  # Step 3: Train xG model via tidymodels (pulls from DuckDB)
  xg_fit <- train_xg_model(con)

  if (is.null(xg_fit)) {
    cli_alert_danger("Exiting: model training failed. Play-by-play data remains in DuckDB.")
    return(invisible(NULL))
  }

  # Step 4: Score all shots (reads from DuckDB, writes scored_shots back)
  score_ok <- score_shots(con, xg_fit)

  if (identical(score_ok, FALSE)) {
    cli_alert_danger("Exiting: scoring failed. Play-by-play data remains in DuckDB.")
    return(invisible(NULL))
  }

  # Step 5: Aggregate to goalie-game level via SQL in DuckDB
  goalie_games <- build_goalie_game_dataframe(con)

  # Save outputs via DuckDB COPY for efficiency
  csv_path     <- file.path(OUTPUT_DIR, "goalie_game_stats.csv")
  parquet_path <- file.path(OUTPUT_DIR, "goalie_game_stats.parquet")

  write_csv(goalie_games, csv_path)
  write_parquet(goalie_games, parquet_path)

  cli_h2("Output")
  cli_alert_success("Saved to {csv_path}")
  cli_alert_info("Rows: {nrow(goalie_games)}")
  cli_alert_info("Unique goalies: {n_distinct(goalie_games$goalie_name)}")
  cli_alert_info("Date range: {min(goalie_games$game_date)} to {max(goalie_games$game_date)}")

  cat("\nSample output:\n")
  print(head(goalie_games, 10))

  goalie_games
}

# Run
goalie_games <- main()
