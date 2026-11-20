
# ============================================================
# CRICKET PERFORMANCE INTELLIGENCE SYSTEM
# MATCH PERFORMANCE ANALYSIS
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

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "analysis"
)

MATCH_SUMMARY_PATH = os.path.join(
    PROCESSED_DIR,
    "match_summary_processed.csv"
)

BALL_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "ball_by_ball_processed.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_match_data():

    if not os.path.exists(MATCH_SUMMARY_PATH):
        raise FileNotFoundError(
            "match_summary_processed.csv not found. "
            "Run preprocessing.py first."
        )

    summary = pd.read_csv(MATCH_SUMMARY_PATH)

    required_columns = [
        "match_id",
        "innings",
        "batting_team",
        "total_runs",
        "legal_balls"
    ]

    missing = [
        col for col in required_columns
        if col not in summary.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in match summary: {missing}"
        )

    ball_data = None

    if os.path.exists(BALL_DATA_PATH):
        ball_data = pd.read_csv(BALL_DATA_PATH)

    return summary, ball_data


# ------------------------------------------------------------
# 3. CALCULATE TOTAL WICKETS
# ------------------------------------------------------------

def calculate_total_wickets(ball_data):

    if ball_data is None:
        return pd.DataFrame()

    if "player_dismissed" not in ball_data.columns:
        print("player_dismissed column unavailable.")
        return pd.DataFrame()

    grouping_columns = [
        col for col in [
            "match_id",
            "innings",
            "batting_team"
        ]
        if col in ball_data.columns
    ]

    wickets = ball_data[
        ball_data["player_dismissed"].notna()
        & (
            ball_data["player_dismissed"]
            .astype(str).str.strip() != ""
        )
    ]

    wicket_summary = wickets.groupby(
        grouping_columns
    ).size().reset_index(name="wickets_lost")

    return wicket_summary


# ------------------------------------------------------------
# 4. PREPARE MATCH INNINGS
# ------------------------------------------------------------

def prepare_match_data(summary, ball_data):

    summary = summary.copy()

    summary["innings"] = pd.to_numeric(
        summary["innings"],
        errors="coerce"
    )

    summary["total_runs"] = pd.to_numeric(
        summary["total_runs"],
        errors="coerce"
    )

    summary["legal_balls"] = pd.to_numeric(
        summary["legal_balls"],
        errors="coerce"
    )

    # Match innings ordering
    summary = summary.sort_values(
        ["match_id", "innings"]
    )

    # Attach total wickets lost from ball-by-ball records
    wicket_summary = calculate_total_wickets(
        ball_data
    )

    if not wicket_summary.empty:

        summary = summary.merge(
            wicket_summary,
            on=[
                col for col in [
                    "match_id",
                    "innings",
                    "batting_team"
                ]
                if col in summary.columns
                and col in wicket_summary.columns
            ],
            how="left"
        )

        summary["wickets_lost"] = (
            summary["wickets_lost"].fillna(0)
        )

    else:

        summary["wickets_lost"] = np.nan

    return summary


# ------------------------------------------------------------
# 5. CREATE MATCH COMPARISON
# ------------------------------------------------------------

def create_match_comparison(summary):

    match_records = []

    for match_id, group in summary.groupby(
        "match_id",
        dropna=False
    ):

        group = group.sort_values("innings")

        # This module compares the first two innings only.
        if len(group) < 2:
            continue

        first = group.iloc[0]
        second = group.iloc[1]

        first_runs = first["total_runs"]
        second_runs = second["total_runs"]

        run_difference = abs(
            first_runs - second_runs
        )

        first_team = first["batting_team"]
        second_team = second["batting_team"]

        first_wickets = first["wickets_lost"]
        second_wickets = second["wickets_lost"]

        first_overs = first["legal_balls"] / 6
        second_overs = second["legal_balls"] / 6

        match_records.append({

            "match_id": match_id,

            "date": first.get("date", np.nan),

            "first_innings_team": first_team,

            "second_innings_team": second_team,

            "first_innings_runs": first_runs,

            "second_innings_runs": second_runs,

            "first_innings_wickets": first_wickets,

            "second_innings_wickets": second_wickets,

            "first_innings_overs": first_overs,

            "second_innings_overs": second_overs,

            "run_difference": run_difference,

            "first_innings_run_rate": (
                first_runs / first_overs
                if first_overs > 0 else np.nan
            ),

            "second_innings_run_rate": (
                second_runs / second_overs
                if second_overs > 0 else np.nan
            )

        })

    return pd.DataFrame(match_records)


# ------------------------------------------------------------
# 6. CLASSIFY MATCH COMPETITIVENESS
# ------------------------------------------------------------

def classify_match(row):

    difference = row["run_difference"]

    first_runs = row["first_innings_runs"]
    second_runs = row["second_innings_runs"]

    # Equal scores
    if first_runs == second_runs:
        return "Tie / Equal Scores"

    # Heuristic classification based on score margin
    if difference <= 5:
        return "Very Close Match"

    elif difference <= 15:
        return "Close Match"

    elif difference <= 30:
        return "Competitive Match"

    elif difference <= 60:
        return "Dominant Performance"

    else:
        return "One-Sided Match"


# ------------------------------------------------------------
# 7. MATCH PERFORMANCE ANALYSIS
# ------------------------------------------------------------

def analyze_matches(match_comparison):

    match_comparison = match_comparison.copy()

    match_comparison["competitiveness"] = (
        match_comparison.apply(
            classify_match,
            axis=1
        )
    )

    match_comparison["run_rate_difference"] = (
        match_comparison[
            "first_innings_run_rate"
        ]
        - match_comparison[
            "second_innings_run_rate"
        ]
    ).abs()

    return match_comparison


# ------------------------------------------------------------
# 8. COMPETITIVENESS SUMMARY
# ------------------------------------------------------------

def competitiveness_summary(match_analysis):

    summary = match_analysis[
        "competitiveness"
    ].value_counts().reset_index()

    summary.columns = [
        "match_category",
        "number_of_matches"
    ]

    summary["percentage"] = (
        summary["number_of_matches"]
        / summary["number_of_matches"].sum()
        * 100
    )

    return summary


# ------------------------------------------------------------
# 9. TOP CLOSE MATCHES
# ------------------------------------------------------------

def get_close_matches(match_analysis):

    close_matches = match_analysis[
        match_analysis["competitiveness"].isin([
            "Very Close Match",
            "Close Match"
        ])
    ].sort_values(
        "run_difference",
        ascending=True
    )

    return close_matches


# ------------------------------------------------------------
# 10. MAIN EXECUTION
# ------------------------------------------------------------

def main():

    print("\nLoading match data...")

    summary, ball_data = load_match_data()

    print("Preparing innings data...")

    summary = prepare_match_data(
        summary,
        ball_data
    )

    print("Comparing match innings...")

    match_comparison = create_match_comparison(
        summary
    )

    if match_comparison.empty:

        print(
            "No matches with two innings found. "
            "Check match_id and innings columns."
        )

        return

    print("Analyzing match competitiveness...")

    match_analysis = analyze_matches(
        match_comparison
    )

    category_summary = competitiveness_summary(
        match_analysis
    )

    close_matches = get_close_matches(
        match_analysis
    )

    # Save output
    match_analysis.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "match_performance_analysis.csv"
        ),
        index=False
    )

    category_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "match_competitiveness_summary.csv"
        ),
        index=False
    )

    close_matches.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "close_matches.csv"
        ),
        index=False
    )

    # Display
    print("\nMATCH COMPETITIVENESS SUMMARY")

    print(
        category_summary.to_string(
            index=False
        )
    )

    print("\nCLOSE MATCHES")

    print(
        close_matches[
            [
                "match_id",
                "first_innings_team",
                "second_innings_team",
                "first_innings_runs",
                "second_innings_runs",
                "run_difference",
                "competitiveness"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nMatch analysis completed!")

    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
