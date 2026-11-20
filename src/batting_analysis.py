
# ============================================================
# CRICKET PERFORMANCE INTELLIGENCE SYSTEM
# BATTING ANALYSIS MODULE
# ============================================================

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

INPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "batting_processed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_batting_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Batting dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "player",
        "runs",
        "balls",
        "fours",
        "sixes",
        "strike_rate",
        "chase",
        "win"
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

def clean_batting_data(df):

    df = df.copy()

    numeric_columns = [
        "runs",
        "balls",
        "fours",
        "sixes",
        "strike_rate",
        "chase",
        "win"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["runs"] = df["runs"].fillna(0)
    df["balls"] = df["balls"].fillna(0)
    df["fours"] = df["fours"].fillna(0)
    df["sixes"] = df["sixes"].fillna(0)

    df["player"] = (
        df["player"]
        .astype(str)
        .str.strip()
    )

    df = df[df["player"] != ""]

    # Not-out detection
    if "dismissal" in df.columns:

        df["dismissal"] = (
            df["dismissal"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["not_out"] = (
            df["dismissal"].str.contains(
                "not out",
                regex=False
            )
        )

    else:

        df["not_out"] = False

    # Innings in which batter was dismissed
    df["dismissed"] = ~df["not_out"]

    # Calculated strike rate
    df["calculated_sr"] = np.where(
        df["balls"] > 0,
        df["runs"] / df["balls"] * 100,
        np.nan
    )

    # Boundary runs
    df["boundary_runs"] = (
        df["fours"] * 4
        + df["sixes"] * 6
    )

    # Percentage of runs from boundaries
    df["boundary_percentage"] = np.where(
        df["runs"] > 0,
        df["boundary_runs"] / df["runs"] * 100,
        0
    )

    return df


# ------------------------------------------------------------
# 4. PLAYER BATTING STATISTICS
# ------------------------------------------------------------

def calculate_player_statistics(df):

    player_stats = df.groupby(
        "player",
        dropna=False
    ).agg(

        innings=("runs", "count"),

        total_runs=("runs", "sum"),

        total_balls=("balls", "sum"),

        dismissals=("dismissed", "sum"),

        not_outs=("not_out", "sum"),

        highest_score=("runs", "max"),

        total_fours=("fours", "sum"),

        total_sixes=("sixes", "sum"),

        boundary_runs=("boundary_runs", "sum"),

        average_strike_rate=("calculated_sr", "mean"),

        total_boundary_percentage=(
            "boundary_percentage",
            "mean"
        )

    ).reset_index()

    # Batting average = runs / dismissals
    player_stats["batting_average"] = np.where(
        player_stats["dismissals"] > 0,
        player_stats["total_runs"]
        / player_stats["dismissals"],
        np.nan
    )

    # Overall strike rate = total runs / total balls
    player_stats["strike_rate"] = np.where(
        player_stats["total_balls"] > 0,
        player_stats["total_runs"]
        / player_stats["total_balls"] * 100,
        np.nan
    )

    # Number of innings scoring 50+
    fifties = (
        df.assign(fifty=df["runs"] >= 50)
        .groupby("player")["fifty"]
        .sum()
    )

    player_stats["fifties"] = (
        player_stats["player"]
        .map(fifties)
        .fillna(0)
        .astype(int)
    )

    # Number of innings scoring 30+
    thirty_plus = (
        df.assign(thirty_plus=df["runs"] >= 30)
        .groupby("player")["thirty_plus"]
        .sum()
    )

    player_stats["thirty_plus_scores"] = (
        player_stats["player"]
        .map(thirty_plus)
        .fillna(0)
        .astype(int)
    )

    # Boundary percentage based on total runs
    player_stats["boundary_percentage"] = np.where(
        player_stats["total_runs"] > 0,
        player_stats["boundary_runs"]
        / player_stats["total_runs"] * 100,
        0
    )

    # Sort by runs scored
    player_stats = player_stats.sort_values(
        "total_runs",
        ascending=False
    )

    return player_stats


# ------------------------------------------------------------
# 5. CHASE PERFORMANCE ANALYSIS
# ------------------------------------------------------------

def analyze_chase_performance(df):

    chase_df = df[df["chase"] == 1].copy()

    non_chase_df = df[df["chase"] == 0].copy()

    chase_stats = chase_df.groupby(
        "player"
    ).agg(

        chase_innings=("runs", "count"),

        chase_runs=("runs", "sum"),

        chase_average_sr=("calculated_sr", "mean"),

        chase_wins=("win", "sum")

    ).reset_index()

    non_chase_stats = non_chase_df.groupby(
        "player"
    ).agg(

        non_chase_innings=("runs", "count"),

        non_chase_runs=("runs", "sum"),

        non_chase_average_sr=("calculated_sr", "mean")

    ).reset_index()

    result = pd.merge(
        chase_stats,
        non_chase_stats,
        on="player",
        how="outer"
    )

    result = result.fillna(0)

    result["chase_average"] = np.where(
        result["chase_innings"] > 0,
        result["chase_runs"]
        / result["chase_innings"],
        0
    )

    result["non_chase_average"] = np.where(
        result["non_chase_innings"] > 0,
        result["non_chase_runs"]
        / result["non_chase_innings"],
        0
    )

    return result


# ------------------------------------------------------------
# 6. PLAYER CONSISTENCY ANALYSIS
# ------------------------------------------------------------

def analyze_consistency(df):

    consistency = df.groupby(
        "player"
    ).agg(

        mean_runs=("runs", "mean"),

        standard_deviation=("runs", "std"),

        innings=("runs", "count")

    ).reset_index()

    consistency["standard_deviation"] = (
        consistency["standard_deviation"].fillna(0)
    )

    # Coefficient of variation
    consistency["coefficient_of_variation"] = np.where(
        consistency["mean_runs"] > 0,
        consistency["standard_deviation"]
        / consistency["mean_runs"] * 100,
        np.nan
    )

    # Lower variation indicates more consistent scoring.
    consistency["consistency_score"] = np.where(
        consistency["mean_runs"] > 0,
        100 / (
            1 + consistency["coefficient_of_variation"]
        ),
        0
    )

    return consistency


# ------------------------------------------------------------
# 7. TOP BATTERS
# ------------------------------------------------------------

def get_top_batters(player_stats):

    top_runs = player_stats.sort_values(
        "total_runs",
        ascending=False
    ).head(10)

    top_average = player_stats[
        player_stats["innings"] >= 3
    ].sort_values(
        "batting_average",
        ascending=False
    ).head(10)

    top_strike_rate = player_stats[
        player_stats["total_balls"] >= 30
    ].sort_values(
        "strike_rate",
        ascending=False
    ).head(10)

    return top_runs, top_average, top_strike_rate


# ------------------------------------------------------------
# 8. RUN ANALYSIS
# ------------------------------------------------------------

def main():

    print("\nLoading batting data...")

    batting_df = load_batting_data()

    batting_df = clean_batting_data(batting_df)

    print("Calculating player batting statistics...")

    player_stats = calculate_player_statistics(
        batting_df
    )

    print("Analyzing chase performance...")

    chase_stats = analyze_chase_performance(
        batting_df
    )

    print("Analyzing player consistency...")

    consistency_stats = analyze_consistency(
        batting_df
    )

    print("Finding top batters...")

    top_runs, top_average, top_strike_rate = (
        get_top_batters(player_stats)
    )

    # Save outputs
    player_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "player_batting_statistics.csv"
        ),
        index=False
    )

    chase_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "chase_performance.csv"
        ),
        index=False
    )

    consistency_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "player_consistency.csv"
        ),
        index=False
    )

    top_runs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_batters_runs.csv"
        ),
        index=False
    )

    top_average.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_batters_average.csv"
        ),
        index=False
    )

    top_strike_rate.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_batters_strike_rate.csv"
        ),
        index=False
    )

    # Display results
    print("\nTOP 10 BATTERS BY RUNS")

    print(
        top_runs[
            ["player", "total_runs", "strike_rate"]
        ].to_string(index=False)
    )

    print("\nTOP 10 BATTERS BY AVERAGE")

    print(
        top_average[
            ["player", "batting_average", "total_runs"]
        ].to_string(index=False)
    )

    print("\nTOP 10 BATTERS BY STRIKE RATE")

    print(
        top_strike_rate[
            ["player", "strike_rate", "total_runs"]
        ].to_string(index=False)
    )

    print("\nBatting analysis completed!")

    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
