# ============================================================
# CRICKET PERFORMANCE INTELLIGENCE SYSTEM
# BOWLING ANALYSIS MODULE
# ============================================================

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

INPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "ball_by_ball_processed.zip"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_bowling_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    required_columns = [
        "bowler",
        "is_legal_ball",
        "bowler_runs",
        "bowler_wicket",
        "is_dot_ball"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

def clean_bowling_data(df):

    df = df.copy()

    df["bowler"] = (
        df["bowler"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["bowler"].notna()
        & (df["bowler"] != "")
    ]

    numeric_columns = [
        "bowler_runs",
        "runs_off_bat",
        "wides",
        "noballs",
        "over",
        "innings"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df["bowler_runs"] = (
        df["bowler_runs"].fillna(0)
    )

    df["is_legal_ball"] = (
        df["is_legal_ball"]
        .fillna(False)
        .astype(bool)
    )

    df["bowler_wicket"] = (
        df["bowler_wicket"]
        .fillna(False)
        .astype(bool)
    )

    df["is_dot_ball"] = (
        df["is_dot_ball"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ------------------------------------------------------------
# 4. BOWLING METRIC CALCULATIONS
# ------------------------------------------------------------

def calculate_bowling_metrics(stats):

    stats = stats.copy()

    # Overs in cricket notation: 4.2 means 4 overs, 2 balls
    stats["overs"] = (
        stats["legal_balls"] // 6
        + (stats["legal_balls"] % 6) / 10
    )

    # Economy rate
    stats["economy_rate"] = np.where(
        stats["legal_balls"] > 0,
        stats["runs_conceded"] * 6
        / stats["legal_balls"],
        np.nan
    )

    # Bowling average
    stats["bowling_average"] = np.where(
        stats["wickets"] > 0,
        stats["runs_conceded"]
        / stats["wickets"],
        np.nan
    )

    # Bowling strike rate
    stats["bowling_strike_rate"] = np.where(
        stats["wickets"] > 0,
        stats["legal_balls"]
        / stats["wickets"],
        np.nan
    )

    # Dot ball percentage
    stats["dot_ball_percentage"] = np.where(
        stats["legal_balls"] > 0,
        stats["dot_balls"]
        / stats["legal_balls"] * 100,
        0
    )

    return stats


# ------------------------------------------------------------
# 5. PLAYER-LEVEL BOWLING STATISTICS
# ------------------------------------------------------------

def calculate_player_bowling_stats(df):

    player_stats = df.groupby(
        "bowler",
        dropna=False
    ).agg(

        matches=("match_id", "nunique")
        if "match_id" in df.columns
        else ("bowler", "count"),

        legal_balls=("is_legal_ball", "sum"),

        runs_conceded=("bowler_runs", "sum"),

        wickets=("bowler_wicket", "sum"),

        dot_balls=("is_dot_ball", "sum")

    ).reset_index()

    player_stats = calculate_bowling_metrics(
        player_stats
    )

    player_stats = player_stats.sort_values(
        ["wickets", "economy_rate"],
        ascending=[False, True]
    )

    return player_stats


# ------------------------------------------------------------
# 6. MATCH-LEVEL BOWLING STATISTICS
# ------------------------------------------------------------

def calculate_match_bowling_stats(df):

    group_columns = ["bowler"]

    for col in [
        "match_id",
        "date",
        "innings",
        "bowling_team",
        "batting_team"
    ]:

        if col in df.columns:
            group_columns.append(col)

    match_stats = df.groupby(
        group_columns,
        dropna=False
    ).agg(

        legal_balls=("is_legal_ball", "sum"),

        runs_conceded=("bowler_runs", "sum"),

        wickets=("bowler_wicket", "sum"),

        dot_balls=("is_dot_ball", "sum")

    ).reset_index()

    match_stats = calculate_bowling_metrics(
        match_stats
    )

    return match_stats


# ------------------------------------------------------------
# 7. BOWLING PHASE ANALYSIS
# ------------------------------------------------------------

def analyze_bowling_phases(df):

    if "over" not in df.columns:

        print(
            "Over column unavailable. "
            "Skipping phase analysis."
        )

        return pd.DataFrame()

    df = df.copy()

    df["over"] = pd.to_numeric(
        df["over"],
        errors="coerce"
    )

    # Cricsheet commonly uses zero-based over numbering.
    # Convert 0-19 into 1-20.
    if df["over"].min() == 0:
        df["over_number"] = df["over"] + 1
    else:
        df["over_number"] = df["over"]

    df["phase"] = np.select(
        [
            df["over_number"].between(1, 6),
            df["over_number"].between(7, 15),
            df["over_number"].between(16, 20)
        ],
        [
            "Powerplay",
            "Middle Overs",
            "Death Overs"
        ],
        default="Other"
    )

    phase_stats = df.groupby(
        ["bowler", "phase"],
        dropna=False
    ).agg(

        legal_balls=("is_legal_ball", "sum"),

        runs_conceded=("bowler_runs", "sum"),

        wickets=("bowler_wicket", "sum"),

        dot_balls=("is_dot_ball", "sum")

    ).reset_index()

    phase_stats = calculate_bowling_metrics(
        phase_stats
    )

    return phase_stats


# ------------------------------------------------------------
# 8. TOP BOWLER RANKINGS
# ------------------------------------------------------------

def get_top_bowlers(player_stats):

    # Minimum 5 wickets for wicket rankings
    top_wickets = player_stats[
        player_stats["wickets"] >= 5
    ].sort_values(
        "wickets",
        ascending=False
    ).head(10)

    # Minimum 30 legal balls for economy rankings
    top_economy = player_stats[
        player_stats["legal_balls"] >= 30
    ].sort_values(
        "economy_rate",
        ascending=True
    ).head(10)

    # Minimum 5 wickets for bowling average rankings
    top_average = player_stats[
        player_stats["wickets"] >= 5
    ].sort_values(
        "bowling_average",
        ascending=True
    ).head(10)

    return top_wickets, top_economy, top_average


# ------------------------------------------------------------
# 9. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    print("\nLoading bowling data...")

    bowling_df = load_bowling_data()

    bowling_df = clean_bowling_data(
        bowling_df
    )

    print("Calculating player bowling statistics...")

    player_stats = calculate_player_bowling_stats(
        bowling_df
    )

    print("Calculating match-level bowling statistics...")

    match_stats = calculate_match_bowling_stats(
        bowling_df
    )

    print("Analyzing bowling phases...")

    phase_stats = analyze_bowling_phases(
        bowling_df
    )

    print("Generating top bowler rankings...")

    top_wickets, top_economy, top_average = (
        get_top_bowlers(player_stats)
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    player_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "player_bowling_statistics.csv"
        ),
        index=False
    )

    match_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "match_bowling_statistics.csv"
        ),
        index=False
    )

    phase_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "bowling_phase_analysis.csv"
        ),
        index=False
    )

    top_wickets.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_bowlers_wickets.csv"
        ),
        index=False
    )

    top_economy.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_bowlers_economy.csv"
        ),
        index=False
    )

    top_average.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_bowlers_average.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nTOP BOWLERS BY WICKETS")

    print(
        top_wickets[
            ["bowler", "wickets", "economy_rate"]
        ].to_string(index=False)
    )

    print("\nTOP BOWLERS BY ECONOMY")

    print(
        top_economy[
            ["bowler", "economy_rate", "wickets"]
        ].to_string(index=False)
    )

    print("\nTOP BOWLERS BY BOWLING AVERAGE")

    print(
        top_average[
            ["bowler", "bowling_average", "wickets"]
        ].to_string(index=False)
    )

    print("\nBowling analysis completed!")

    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()