
# ============================================================
# CRICKET PERFORMANCE INTELLIGENCE SYSTEM
# PLAYER IMPACT ANALYSIS
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

ANALYSIS_DIR = os.path.join(
    BASE_DIR, "data", "analysis"
)

OUTPUT_DIR = ANALYSIS_DIR

BATTING_PATH = os.path.join(
    ANALYSIS_DIR,
    "player_batting_statistics.csv"
)

BOWLING_PATH = os.path.join(
    ANALYSIS_DIR,
    "player_bowling_statistics.csv"
)


# ------------------------------------------------------------
# 2. LOAD ANALYSIS DATA
# ------------------------------------------------------------

def load_data():

    if not os.path.exists(BATTING_PATH):
        raise FileNotFoundError(
            "Run batting_analysis.py first."
        )

    if not os.path.exists(BOWLING_PATH):
        raise FileNotFoundError(
            "Run bowling_analysis.py first."
        )

    batting = pd.read_csv(BATTING_PATH)
    bowling = pd.read_csv(BOWLING_PATH)

    batting["player"] = (
        batting["player"].astype(str).str.strip()
    )

    bowling["bowler"] = (
        bowling["bowler"].astype(str).str.strip()
    )

    return batting, bowling


# ------------------------------------------------------------
# 3. NORMALIZATION FUNCTION
# ------------------------------------------------------------

def percentile_score(series, higher_is_better=True):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if higher_is_better:
        scores = values.rank(pct=True) * 100
    else:
        scores = (
            values.rank(pct=True, ascending=False)
            * 100
        )

    return scores.fillna(0)


# ------------------------------------------------------------
# 4. BATTING IMPACT SCORE
# ------------------------------------------------------------

def calculate_batting_impact(batting):

    batting = batting.copy()

    metrics = {
        "total_runs": (30, True),
        "batting_average": (20, True),
        "strike_rate": (20, True),
        "boundary_percentage": (10, True),
        "fifties": (10, True),
        "consistency_score": (10, True)
    }

    batting["batting_impact_score"] = 0
    total_weight = 0

    for column, (weight, higher_is_better) in metrics.items():

        if column not in batting.columns:
            continue

        metric_score = percentile_score(
            batting[column],
            higher_is_better
        )

        batting["batting_impact_score"] += (
            metric_score * weight
        )

        total_weight += weight

    if total_weight > 0:

        batting["batting_impact_score"] /= (
            total_weight
        )

    return batting[
        ["player", "batting_impact_score"]
    ]


# ------------------------------------------------------------
# 5. BOWLING IMPACT SCORE
# ------------------------------------------------------------

def calculate_bowling_impact(bowling):

    bowling = bowling.copy()

    metrics = {
        "wickets": (35, True),
        "economy_rate": (25, False),
        "bowling_average": (15, False),
        "bowling_strike_rate": (15, False),
        "dot_ball_percentage": (10, True)
    }

    bowling["bowling_impact_score"] = 0
    total_weight = 0

    for column, (weight, higher_is_better) in metrics.items():

        if column not in bowling.columns:
            continue

        metric_score = percentile_score(
            bowling[column],
            higher_is_better
        )

        bowling["bowling_impact_score"] += (
            metric_score * weight
        )

        total_weight += weight

    if total_weight > 0:

        bowling["bowling_impact_score"] /= (
            total_weight
        )

    return bowling[
        ["bowler", "bowling_impact_score"]
    ]


# ------------------------------------------------------------
# 6. COMBINE BATTING AND BOWLING SCORES
# ------------------------------------------------------------

def calculate_player_impact(batting, bowling):

    batting_impact = calculate_batting_impact(
        batting
    )

    bowling_impact = calculate_bowling_impact(
        bowling
    )

    bowling_impact = bowling_impact.rename(
        columns={"bowler": "player"}
    )

    impact = pd.merge(
        batting_impact,
        bowling_impact,
        on="player",
        how="outer"
    )

    has_batting = impact[
        "batting_impact_score"
    ].notna()

    has_bowling = impact[
        "bowling_impact_score"
    ].notna()

    impact["player_role"] = np.select(
        [
            has_batting & has_bowling,
            has_batting,
            has_bowling
        ],
        [
            "All-Rounder",
            "Batter",
            "Bowler"
        ],
        default="Unknown"
    )

    # Calculate impact score according to available role
    impact["player_impact_score"] = np.select(
        [
            has_batting & has_bowling,
            has_batting,
            has_bowling
        ],
        [
            (
                impact["batting_impact_score"]
                + impact["bowling_impact_score"]
            ) / 2,

            impact["batting_impact_score"],

            impact["bowling_impact_score"]
        ],
        default=0
    )

    impact["player_impact_score"] = (
        impact["player_impact_score"].round(2)
    )

    impact = impact.sort_values(
        "player_impact_score",
        ascending=False
    )

    return impact


# ------------------------------------------------------------
# 7. PLAYER RANKING
# ------------------------------------------------------------

def rank_players(impact):

    impact = impact.copy()

    impact["rank"] = (
        impact["player_impact_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return impact.sort_values("rank")


# ------------------------------------------------------------
# 8. ROLE-BASED RANKINGS
# ------------------------------------------------------------

def role_based_rankings(impact):

    batters = impact[
        impact["player_role"] == "Batter"
    ].sort_values(
        "player_impact_score",
        ascending=False
    )

    bowlers = impact[
        impact["player_role"] == "Bowler"
    ].sort_values(
        "player_impact_score",
        ascending=False
    )

    all_rounders = impact[
        impact["player_role"] == "All-Rounder"
    ].sort_values(
        "player_impact_score",
        ascending=False
    )

    return batters, bowlers, all_rounders


# ------------------------------------------------------------
# 9. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    print("\nLoading batting and bowling analytics...")

    batting, bowling = load_data()

    print("Calculating batting impact...")

    batting_impact = calculate_batting_impact(
        batting
    )

    print("Calculating bowling impact...")

    bowling_impact = calculate_bowling_impact(
        bowling
    )

    print("Combining player impact scores...")

    impact = calculate_player_impact(
        batting,
        bowling
    )

    impact = rank_players(impact)

    batters, bowlers, all_rounders = (
        role_based_rankings(impact)
    )

    # Save results
    impact.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "player_impact_scores.csv"
        ),
        index=False
    )

    batters.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "batter_impact_ranking.csv"
        ),
        index=False
    )

    bowlers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "bowler_impact_ranking.csv"
        ),
        index=False
    )

    all_rounders.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_rounder_impact_ranking.csv"
        ),
        index=False
    )

    # Display top players
    print("\nTOP 10 PLAYERS BY IMPACT SCORE")

    print(
        impact[
            [
                "rank",
                "player",
                "player_role",
                "player_impact_score"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTOP BATTERS")

    print(
        batters[
            ["player", "player_impact_score"]
        ].head(10).to_string(index=False)
    )

    print("\nTOP BOWLERS")

    print(
        bowlers[
            ["player", "player_impact_score"]
        ].head(10).to_string(index=False)
    )

    print("\nTOP ALL-ROUNDERS")

    print(
        all_rounders[
            ["player", "player_impact_score"]
        ].head(10).to_string(index=False)
    )

    print("\nPlayer Impact Analysis completed!")

    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
