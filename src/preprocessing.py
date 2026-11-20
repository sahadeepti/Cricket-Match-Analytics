
# ============================================================
# CRICKET PERFORMANCE INTELLIGENCE SYSTEM
# DATA PREPROCESSING PIPELINE
# ============================================================

import os
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

BATTING_PATH = os.path.join(RAW_DIR, "Batting.csv")
RESULTS_PATH = os.path.join(RAW_DIR, "MatchResults refined .xlsx")
BALL_PATH = os.path.join(RAW_DIR, "ball_by_ball_t20_men.zip")


# ------------------------------------------------------------
# 2. GENERAL HELPER FUNCTIONS
# ------------------------------------------------------------

def clean_column_names(df):
    """Normalize column names."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_text(value):
    """Clean player/team names without removing meaningful characters."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value if value else np.nan


def normalize_team(team):
    """Standardize known team-name variations."""

    if pd.isna(team):
        return np.nan

    team = clean_text(team)
    key = team.lower()

    mapping = {
        "eng": "England",
        "england": "England",

        "afganisthan": "Afghanistan",
        "afghanistan": "Afghanistan",
        "afganistan": "Afghanistan",

        "new zealand": "New Zealand",
        "new zealand ": "New Zealand",
        "new zealand": "New Zealand",

        "sri lanka": "Sri Lanka",

        "zimbabwae": "Zimbabwe",
        "zimbabwe": "Zimbabwe",

        "netherland": "Netherlands",
        "netherlands": "Netherlands",

        "west indies": "West Indies",

        "usa": "USA",
        "u.s.a.": "USA",

        "uae": "UAE",
    }

    return mapping.get(key, team)


def normalize_player(player):
    """Basic player-name normalization."""

    if pd.isna(player):
        return np.nan

    player = clean_text(player)

    # Remove trailing question marks sometimes present in scraped data
    player = re.sub(r"\?+$", "", player)

    # Keep captain/wicketkeeper suffixes as metadata only if needed;
    # remove them from the canonical player name.
    player = re.sub(r"\s*\((c|wk|vc)\)\s*$", "", player,
                    flags=re.IGNORECASE)

    return player.strip()


def parse_cricket_date(series):
    """
    Parse dates such as:
    7.2.26
    07/02/2026
    2026-02-07

    For ambiguous numeric dates, day-first is assumed.
    """

    return pd.to_datetime(
        series.astype(str).str.strip(),
        dayfirst=True,
        errors="coerce"
    )


def save_csv(df, filename):
    """Save processed dataset."""

    path = os.path.join(PROCESSED_DIR, filename)

    df.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Rows: {len(df):,}")


def show_quality_report(df, dataset_name):
    """Display basic data-quality statistics."""

    print("\n" + "=" * 60)
    print(f"DATA QUALITY REPORT: {dataset_name}")
    print("=" * 60)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate rows:", df.duplicated().sum())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nColumn types:")
    print(df.dtypes)


# ------------------------------------------------------------
# 3. LOAD RAW DATASETS
# ------------------------------------------------------------

print("\nLoading raw datasets...")

batting_raw = pd.read_csv(BATTING_PATH)

results_raw = pd.read_excel(RESULTS_PATH)

ball_raw = pd.read_csv(BALL_PATH, low_memory=False)

print("Batting dataset:", batting_raw.shape)
print("Results dataset:", results_raw.shape)
print("Ball-by-ball dataset:", ball_raw.shape)


# ------------------------------------------------------------
# 4. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

batting_raw = clean_column_names(batting_raw)

results_raw = clean_column_names(results_raw)

ball_raw = clean_column_names(ball_raw)


# ------------------------------------------------------------
# 5. INSPECT BALL-BY-BALL SCHEMA
# ------------------------------------------------------------

print("\nBall-by-ball column names:")
print(ball_raw.columns.tolist())

print("\nFirst 5 ball-by-ball records:")
print(ball_raw.head())


# ------------------------------------------------------------
# 6. PREPROCESS BATTING DATA
# ------------------------------------------------------------

def preprocess_batting(df):

    df = df.copy()

    # Rename columns to normalized names
    column_mapping = {
        "date": "date",
        "team": "team",
        "vs": "opponent",
        "player": "player",
        "dismissal": "dismissal",
        "r": "runs",
        "b": "balls",
        "m": "minutes",
        "4s": "fours",
        "6s": "sixes",
        "sr": "strike_rate",
        "chase": "chase",
        "win": "win",
    }

    df = df.rename(columns=column_mapping)

    # Clean text
    for col in ["team", "opponent", "player", "dismissal"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normalize teams and players
    df["team"] = df["team"].apply(normalize_team)

    df["opponent"] = df["opponent"].apply(normalize_team)

    df["player"] = df["player"].apply(normalize_player)

    # Convert dates
    df["date"] = parse_cricket_date(df["date"])

    # Convert numeric columns
    numeric_cols = [
        "runs", "balls", "minutes",
        "fours", "sixes", "strike_rate"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Standardize chase/win flags
    for col in ["chase", "win"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1.0": 1,
                    "0.0": 0
                })
            )

            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Recalculate strike rate
    df["strike_rate_calculated"] = np.where(
        df["balls"] > 0,
        (df["runs"] / df["balls"]) * 100,
        np.nan
    )

    # Boundary runs
    df["boundary_runs"] = (
        df["fours"].fillna(0) * 4
        + df["sixes"].fillna(0) * 6
    )

    # Boundary percentage
    df["boundary_percentage"] = np.where(
        df["runs"] > 0,
        df["boundary_runs"] / df["runs"] * 100,
        0
    )

    # Balls per boundary
    df["balls_per_boundary"] = np.where(
        (df["fours"].fillna(0) + df["sixes"].fillna(0)) > 0,
        df["balls"] /
        (df["fours"].fillna(0) + df["sixes"].fillna(0)),
        np.nan
    )

    # Player score category
    df["score_category"] = pd.cut(
        df["runs"],
        bins=[-1, 9, 19, 29, 49, 10000],
        labels=[
            "Below 10",
            "10 to 19",
            "20 to 29",
            "30 to 49",
            "50+"
        ]
    )

    return df


batting_df = preprocess_batting(batting_raw)

show_quality_report(batting_df, "Batting")

save_csv(batting_df, "batting_processed.csv")


# ------------------------------------------------------------
# 7. PREPROCESS MATCH RESULTS
# ------------------------------------------------------------

def preprocess_results(df):

    df = df.copy()

    df = df.rename(columns={
        "date": "date",
        "team": "team",
        "vs": "opponent",
        "player": "player",
        "r": "runs",
        "b": "balls",
        "sr": "strike_rate",
        "win": "win",
        "chase": "chase"
    })

    if "date" in df.columns:
        df["date"] = parse_cricket_date(df["date"])

    for col in ["team", "opponent"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)

    if "player" in df.columns:
        df["player"] = df["player"].apply(normalize_player)

    for col in ["runs", "balls", "strike_rate", "win", "chase"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.drop_duplicates()

    return df


results_df = preprocess_results(results_raw)

show_quality_report(results_df, "Match Results")

save_csv(results_df, "match_results_processed.csv")


# ------------------------------------------------------------
# 8. BALL-BY-BALL COLUMN ALIAS MAPPING
# ------------------------------------------------------------

# Dataset headers differ across cricket data providers.
# Inspect the printed column list above and adjust aliases if needed.

BALL_ALIASES = {
    "match_id": [
        "match_id", "id", "game_id"
    ],

    "date": [
        "start_date", "date", "match_date"
    ],

    "innings": [
        "innings", "inning", "innings_number"
    ],

    "over": [
        "over", "over_number", "over_num"
    ],

    "ball": [
        "ball", "ball_number", "delivery"
    ],

    "batter": [
        "batter", "striker", "batsman"
    ],

    "bowler": [
        "bowler"
    ],

    "batting_team": [
        "batting_team", "team"
    ],

    "bowling_team": [
        "bowling_team"
    ],

    "runs_off_bat": [
        "runs_off_bat", "batter_runs", "runs_batter"
    ],

    "extras": [
        "extras", "extra_runs"
    ],

    "wides": [
        "wides", "wide"
    ],

    "noballs": [
        "noballs", "no_balls", "no_ball"
    ],

    "byes": [
        "byes", "bye"
    ],

    "legbyes": [
        "legbyes", "leg_byes", "legbye"
    ],

    "wicket_type": [
        "wicket_type", "dismissal", "kind"
    ],

    "player_dismissed": [
        "player_dismissed", "dismissed_player"
    ]
}


def apply_ball_aliases(df):

    df = df.copy()

    rename_map = {}

    for standard_name, candidates in BALL_ALIASES.items():

        for candidate in candidates:

            if candidate in df.columns:

                rename_map[candidate] = standard_name

                break

    df = df.rename(columns=rename_map)

    required = [
        "bowler",
        "batter",
        "runs_off_bat"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required ball-by-ball columns: {missing}\n"
            f"Actual columns: {df.columns.tolist()}\n"
            "Update BALL_ALIASES to match your CSV."
        )

    return df


ball_df = apply_ball_aliases(ball_raw)


# ------------------------------------------------------------
# 9. CLEAN BALL-BY-BALL DATA
# ------------------------------------------------------------

def preprocess_ball_by_ball(df):

    df = df.copy()

    # Normalize names
    for col in ["batter", "bowler", "player_dismissed"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_player)

    for col in ["batting_team", "bowling_team"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)

    # Parse dates
    if "date" in df.columns:
        df["date"] = parse_cricket_date(df["date"])

    # Numeric conversion
    numeric_cols = [
        "runs_off_bat",
        "extras",
        "wides",
        "noballs",
        "byes",
        "legbyes",
        "innings",
        "over",
        "ball"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Missing extras can be treated as zero
    if "extras" in df.columns:
        df["extras"] = df["extras"].fillna(0)

    # Runs off bat
    df["runs_off_bat"] = (
        df["runs_off_bat"].fillna(0)
    )

    # Missing wide/no-ball columns
    for col in ["wides", "noballs", "byes", "legbyes"]:

        if col not in df.columns:
            df[col] = 0

        df[col] = df[col].fillna(0)

    # Legal delivery
    df["is_legal_ball"] = (
        (df["wides"] == 0)
        & (df["noballs"] == 0)
    )

    # Bowling runs charged to bowler:
    # batter runs + wides + no-balls
    df["bowler_runs"] = (
        df["runs_off_bat"]
        + df["wides"]
        + df["noballs"]
    )

    # Total runs scored by batting team
    df["total_delivery_runs"] = (
        df["runs_off_bat"]
        + df["extras"]
    )

    # Dot ball
    df["is_dot_ball"] = (
        (df["runs_off_bat"] == 0)
        & (df["extras"] == 0)
    )

    # Bowler credited dismissals
    non_bowler_dismissals = [
        "run out",
        "retired hurt",
        "retired out",
        "obstructing the field"
    ]

    if "wicket_type" in df.columns:

        df["wicket_type"] = (
            df["wicket_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["bowler_wicket"] = (
            df["wicket_type"].notna()
            & ~df["wicket_type"].isin(
                non_bowler_dismissals
            )
            & ~df["wicket_type"].isin(
                ["", "nan", "none"]
            )
        )

    else:

        df["bowler_wicket"] = False

    return df


ball_df = preprocess_ball_by_ball(ball_df)

show_quality_report(ball_df, "Ball-by-Ball")

save_csv(ball_df, "ball_by_ball_processed.zip")


# ------------------------------------------------------------
# 10. CREATE BOWLING ANALYTICS
# ------------------------------------------------------------

def create_bowling_stats(df):

    group_cols = ["bowler"]

    if "match_id" in df.columns:
        group_cols.insert(0, "match_id")

    if "date" in df.columns:
        group_cols.append("date")

    if "bowling_team" in df.columns:
        group_cols.append("bowling_team")

    if "batting_team" in df.columns:
        group_cols.append("batting_team")

    bowling = df.groupby(
        group_cols,
        dropna=False
    ).agg(

        legal_balls=("is_legal_ball", "sum"),

        runs_conceded=("bowler_runs", "sum"),

        wickets=("bowler_wicket", "sum"),

        dot_balls=("is_dot_ball", "sum"),

    ).reset_index()

    bowling["overs"] = (
        bowling["legal_balls"] // 6
        + (bowling["legal_balls"] % 6) / 10
    )

    bowling["economy_rate"] = np.where(
        bowling["legal_balls"] > 0,
        bowling["runs_conceded"]
        / bowling["legal_balls"] * 6,
        np.nan
    )

    bowling["bowling_average"] = np.where(
        bowling["wickets"] > 0,
        bowling["runs_conceded"]
        / bowling["wickets"],
        np.nan
    )

    bowling["bowling_strike_rate"] = np.where(
        bowling["wickets"] > 0,
        bowling["legal_balls"]
        / bowling["wickets"],
        np.nan
    )

    bowling["dot_ball_percentage"] = np.where(
        bowling["legal_balls"] > 0,
        bowling["dot_balls"]
        / bowling["legal_balls"] * 100,
        0
    )

    return bowling


bowling_stats = create_bowling_stats(ball_df)

save_csv(bowling_stats, "bowling_processed.csv")


# ------------------------------------------------------------
# 11. CREATE MATCH SUMMARY
# ------------------------------------------------------------

def create_match_summary(df):

    group_cols = []

    for col in ["match_id", "date", "innings",
                "batting_team", "bowling_team"]:

        if col in df.columns:
            group_cols.append(col)

    if not group_cols:
        raise ValueError(
            "Match summary cannot be created: "
            "no match grouping columns found."
        )

    match_summary = df.groupby(
        group_cols,
        dropna=False
    ).agg(

        total_runs=("total_delivery_runs", "sum"),

        wickets=("player_dismissed", lambda x: x.notna().sum()),
        fours=("runs_off_bat", lambda x: (x == 4).sum()),
        sixes=("runs_off_bat", lambda x: (x == 6).sum()),

        legal_balls=("is_legal_ball", "sum"),



    ).reset_index()

    match_summary["overs"] = (
        match_summary["legal_balls"] // 6
        + (match_summary["legal_balls"] % 6) / 10
    )

    match_summary["run_rate"] = np.where(
        match_summary["legal_balls"] > 0,
        match_summary["total_runs"]
        / match_summary["legal_balls"] * 6,
        0
    )

    return match_summary


match_summary = create_match_summary(ball_df)

save_csv(match_summary, "match_summary_processed.csv")


# ------------------------------------------------------------
# 12. CREATE PLAYER BATTING STATS FROM BALL DATA
# ------------------------------------------------------------

def create_batter_stats(df):

    df = df.copy()

    group_cols = ["batter"]

    for col in ["match_id", "date", "innings",
                "batting_team", "bowling_team"]:

        if col in df.columns:
            group_cols.insert(0, col)

    # Batter faces every delivery except a wide
    if "wides" in df.columns:
        df["is_ball_faced"] = df["wides"].fillna(0) == 0
    else:
        df["is_ball_faced"] = df["is_legal_ball"]

    batter_stats = df.groupby(
        group_cols,
        dropna=False
    ).agg(
        runs=("runs_off_bat", "sum"),
        balls_faced=("is_ball_faced", "sum"),
        total_deliveries=("runs_off_bat", "count")
    ).reset_index()

    batter_stats["strike_rate"] = np.where(
        batter_stats["balls_faced"] > 0,
        batter_stats["runs"] / batter_stats["balls_faced"] * 100,
        np.nan
    )

    return batter_stats


batter_stats = create_batter_stats(ball_df)

save_csv(batter_stats, "batter_stats_from_ball_data.csv")


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed files created in:")
print(PROCESSED_DIR)

print("\nFiles:")

for filename in os.listdir(PROCESSED_DIR):
    print(" -", filename)

print("\nNext stage:")
print("Batting Analysis")
print("Bowling Analysis")
print("Player Impact Analysis")
print("Match Performance Review")