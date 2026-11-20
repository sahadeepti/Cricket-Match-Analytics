
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"

BATTING_FILE = PROCESSED_DIR / "batting_processed.csv"
MATCH_FILE = PROCESSED_DIR / "match_summary_processed.csv"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_data():

    if not BATTING_FILE.exists():
        raise FileNotFoundError(
            f"Batting file not found: {BATTING_FILE}"
        )

    batting = pd.read_csv(BATTING_FILE)

    batting.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in batting.columns
    ]

    match_summary = None

    if MATCH_FILE.exists():
        match_summary = pd.read_csv(MATCH_FILE)

        match_summary.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in match_summary.columns
        ]

    print("Datasets loaded successfully.")
    print("Batting columns:", batting.columns.tolist())

    if match_summary is not None:
        print("Match summary columns:", match_summary.columns.tolist())

    return batting, match_summary


# ==================================================
# 3. TEAM BATTING PERFORMANCE
# ==================================================

def analyze_team_batting(batting):

    df = batting.copy()

    required_columns = ["team", "runs", "balls"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert numeric values
    for col in ["runs", "balls"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    for col in ["fours", "sixes"]:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    # Identify dismissals
    if "dismissal" in df.columns:

        dismissal = (
            df["dismissal"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        not_out_values = [
            "", "nan", "not out", "not_out",
            "retired hurt", "retired_hurt"
        ]

        df["dismissed"] = (
            ~dismissal.isin(not_out_values)
        ).astype(int)

    else:
        df["dismissed"] = 0

    # Team aggregation
    team_stats = df.groupby("team").agg(
        total_runs=("runs", "sum"),
        total_balls=("balls", "sum"),
        total_dismissals=("dismissed", "sum"),
        total_fours=("fours", "sum"),
        total_sixes=("sixes", "sum"),
        player_innings=("runs", "count")
    ).reset_index()

    # Batting average
    team_stats["batting_average"] = (
        team_stats["total_runs"]
        / team_stats["total_dismissals"].replace(0, pd.NA)
    )

    # Strike rate
    team_stats["strike_rate"] = (
        team_stats["total_runs"]
        / team_stats["total_balls"].replace(0, pd.NA)
        * 100
    )

    # Boundary statistics
    team_stats["boundary_runs"] = (
        team_stats["total_fours"] * 4
        + team_stats["total_sixes"] * 6
    )

    team_stats["boundary_percentage"] = (
        team_stats["boundary_runs"]
        / team_stats["total_runs"].replace(0, pd.NA)
        * 100
    )

    team_stats = team_stats.sort_values(
        "total_runs",
        ascending=False
    )

    output_file = (
        ANALYSIS_DIR / "team_performance_summary.csv"
    )

    team_stats.to_csv(output_file, index=False)

    print("\nTeam batting analysis completed.")
    print(team_stats.head())

    print(f"Saved to: {output_file}")

    return team_stats


# ==================================================
# 4. BATTING FIRST VS CHASING
# ==================================================

def analyze_batting_first_vs_chase(batting):

    df = batting.copy()

    if "chase" not in df.columns:
        print("Chase column not found. Skipping chase analysis.")
        return None

    df["runs"] = pd.to_numeric(
        df["runs"], errors="coerce"
    ).fillna(0)

    chase_values = (
        df["chase"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "1": "Chasing",
        "1.0": "Chasing",
        "true": "Chasing",
        "yes": "Chasing",
        "chase": "Chasing",
        "chasing": "Chasing",

        "0": "Batting First",
        "0.0": "Batting First",
        "false": "Batting First",
        "no": "Batting First",
        "batting first": "Batting First",
        "first innings": "Batting First"
    }

    df["innings_type"] = chase_values.map(mapping)

    df = df.dropna(subset=["innings_type"])

    if df.empty:
        print("No recognized values in chase column.")
        return None

    team_chase_stats = df.groupby(
        ["team", "innings_type"]
    ).agg(
        total_runs=("runs", "sum"),
        player_innings=("runs", "count"),
        average_player_runs=("runs", "mean")
    ).reset_index()

    output_file = (
        ANALYSIS_DIR / "team_batting_first_vs_chase.csv"
    )

    team_chase_stats.to_csv(output_file, index=False)

    print("\nBatting first vs chasing analysis completed.")
    print(team_chase_stats.head())

    print(f"Saved to: {output_file}")

    return team_chase_stats


# ==================================================
# 5. TEAM WIN RATE
# ==================================================

def analyze_team_win_rate(batting):

    required_columns = [
        "date", "team", "opponent", "win"
    ]

    missing = [
        col for col in required_columns
        if col not in batting.columns
    ]

    if missing:
        print(
            f"Skipping win-rate analysis. Missing columns: {missing}"
        )
        return None

    df = batting.copy()

    # Normalize win values
    win_values = (
        df["win"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    win_mapping = {
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
        "win": 1,
        "won": 1,

        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "loss": 0,
        "lost": 0
    }

    df["win_numeric"] = win_values.map(win_mapping)

    # One row per team-match, not one row per player
    team_match = df.drop_duplicates(
        subset=["date", "team", "opponent"]
    )

    team_match = team_match.dropna(
        subset=["win_numeric"]
    )

    if team_match.empty:
        print("No valid win/loss values found.")
        return None

    win_stats = team_match.groupby("team").agg(
        matches_played=("win_numeric", "count"),
        matches_won=("win_numeric", "sum")
    ).reset_index()

    win_stats["matches_lost"] = (
        win_stats["matches_played"]
        - win_stats["matches_won"]
    )

    win_stats["win_percentage"] = (
        win_stats["matches_won"]
        / win_stats["matches_played"]
        * 100
    )

    win_stats = win_stats.sort_values(
        "win_percentage",
        ascending=False
    )

    output_file = (
        ANALYSIS_DIR / "team_win_rate.csv"
    )

    win_stats.to_csv(output_file, index=False)

    print("\nTeam win-rate analysis completed.")
    print(win_stats.head())

    print(f"Saved to: {output_file}")

    return win_stats


# ==================================================
# 6. TEAM MATCH STATISTICS
# ==================================================

def analyze_team_match_statistics(match_summary):

    if match_summary is None:
        print("Match summary unavailable.")
        return None

    df = match_summary.copy()

    required_columns = [
        "batting_team", "total_runs"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(
            f"Missing match summary columns: {missing}"
        )
        return None

    df["total_runs"] = pd.to_numeric(
        df["total_runs"], errors="coerce"
    )

    aggregation = {
        "total_runs": [
            "sum", "mean", "max", "min", "count"
        ]
    }

    for col in ["wickets", "legal_balls", "run_rate"]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

            aggregation[col] = ["mean"]

    stats = df.groupby("batting_team").agg(
        aggregation
    )

    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]

    stats = stats.reset_index()

    output_file = (
        ANALYSIS_DIR / "team_match_innings_statistics.csv"
    )

    stats.to_csv(output_file, index=False)

    print("\nTeam match statistics completed.")
    print(stats.head())

    print(f"Saved to: {output_file}")

    return stats


# ==================================================
# 7. MAIN FUNCTION
# ==================================================

def main():

    batting, match_summary = load_data()

    print("\n" + "=" * 50)
    print("TEAM PERFORMANCE ANALYSIS")
    print("=" * 50)

    analyze_team_batting(batting)

    analyze_batting_first_vs_chase(batting)

    analyze_team_win_rate(batting)

    analyze_team_match_statistics(match_summary)

    print("\n" + "=" * 50)
    print("TEAM ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 50)


# ==================================================
# 8. EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
