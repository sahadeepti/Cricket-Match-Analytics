
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"

VISUALIZATION_DIR = ANALYSIS_DIR / "visualizations"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. GENERAL HELPER FUNCTIONS
# ==================================================

def load_csv(filename):
    """Load an analysis CSV file if it exists."""

    filepath = ANALYSIS_DIR / filename

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            print(f"File is empty: {filename}")
            return None

        df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df.columns
        ]

        return df

    except Exception as error:
        print(f"Error loading {filename}: {error}")
        return None


def find_column(df, possible_columns):
    """Find the first matching column from possible names."""

    if df is None:
        return None

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def save_chart(filename):
    """Save the current Matplotlib chart."""

    output_path = VISUALIZATION_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart saved: {output_path}")


def create_bar_chart(
    df,
    label_column,
    value_column,
    title,
    xlabel,
    ylabel,
    filename,
    top_n=10,
    ascending=False
):
    """Create a reusable bar chart."""

    if df is None:
        return

    if label_column not in df.columns:
        print(f"Missing label column: {label_column}")
        return

    if value_column not in df.columns:
        print(f"Missing value column: {value_column}")
        return

    chart_data = df[[label_column, value_column]].copy()

    chart_data[value_column] = pd.to_numeric(
        chart_data[value_column],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=[label_column, value_column]
    )

    if chart_data.empty:
        print(f"No valid data for chart: {title}")
        return

    chart_data = chart_data.sort_values(
        value_column,
        ascending=ascending
    ).head(top_n)

    # Horizontal bar charts are easier to read for player names
    chart_data = chart_data.sort_values(
        value_column,
        ascending=True
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        chart_data[label_column].astype(str),
        chart_data[value_column]
    )

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    save_chart(filename)


# ==================================================
# 3. BATTING VISUALIZATIONS
# ==================================================

def visualize_batting():

    print("\nCreating batting visualizations...")

    batting = load_csv("player_batting_statistics.csv")

    if batting is None:
        batting = load_csv("top_batters_runs.csv")

    if batting is None:
        print("Batting analysis data unavailable.")
        return

    player_col = find_column(
        batting,
        ["player", "batter", "batsman"]
    )

    runs_col = find_column(
        batting,
        ["total_runs", "runs", "r"]
    )

    average_col = find_column(
        batting,
        ["batting_average", "average", "avg"]
    )

    strike_rate_col = find_column(
        batting,
        ["strike_rate", "sr"]
    )

    if player_col is None:
        print("Player column not found in batting dataset.")
        return

    if runs_col:
        create_bar_chart(
            batting,
            player_col,
            runs_col,
            "Top 10 Batsmen by Total Runs",
            "Total Runs",
            "Player",
            "top_batsmen_runs.png"
        )

    if average_col:
        create_bar_chart(
            batting,
            player_col,
            average_col,
            "Top 10 Batsmen by Batting Average",
            "Batting Average",
            "Player",
            "top_batsmen_average.png"
        )

    if strike_rate_col:
        create_bar_chart(
            batting,
            player_col,
            strike_rate_col,
            "Top 10 Batsmen by Strike Rate",
            "Strike Rate",
            "Player",
            "top_batsmen_strike_rate.png"
        )


# ==================================================
# 4. BOWLING VISUALIZATIONS
# ==================================================

def visualize_bowling():

    print("\nCreating bowling visualizations...")

    bowling = load_csv("player_bowling_statistics.csv")

    if bowling is None:
        bowling = load_csv("top_bowlers_wickets.csv")

    if bowling is None:
        print("Bowling analysis data unavailable.")
        return

    player_col = find_column(
        bowling,
        ["player", "bowler"]
    )

    wickets_col = find_column(
        bowling,
        ["total_wickets", "wickets", "bowler_wickets"]
    )

    economy_col = find_column(
        bowling,
        ["economy_rate", "economy", "econ"]
    )

    average_col = find_column(
        bowling,
        ["bowling_average", "average"]
    )

    if player_col is None:
        print("Bowler column not found.")
        return

    if wickets_col:
        create_bar_chart(
            bowling,
            player_col,
            wickets_col,
            "Top 10 Bowlers by Wickets",
            "Wickets",
            "Bowler",
            "top_bowlers_wickets.png"
        )

    if economy_col:
        create_bar_chart(
            bowling,
            player_col,
            economy_col,
            "Best 10 Bowlers by Economy Rate",
            "Economy Rate",
            "Bowler",
            "top_bowlers_economy.png",
            ascending=True
        )

    if average_col:
        create_bar_chart(
            bowling,
            player_col,
            average_col,
            "Best 10 Bowlers by Bowling Average",
            "Bowling Average",
            "Bowler",
            "top_bowlers_average.png",
            ascending=True
        )


# ==================================================
# 5. PLAYER IMPACT VISUALIZATIONS
# ==================================================

def visualize_player_impact():

    print("\nCreating player impact visualizations...")

    impact = load_csv("player_impact_scores.csv")

    if impact is None:
        impact = load_csv("batter_impact_ranking.csv")

    if impact is None:
        print("Player impact dataset unavailable.")
        return

    player_col = find_column(
        impact,
        ["player", "name"]
    )

    impact_col = find_column(
        impact,
        [
            "impact_score",
            "overall_impact_score",
            "batting_impact_score",
            "impact"
        ]
    )

    if player_col is None or impact_col is None:
        print("Player or impact score column not found.")
        return

    create_bar_chart(
        impact,
        player_col,
        impact_col,
        "Top 10 Players by Impact Score",
        "Impact Score",
        "Player",
        "top_players_impact.png"
    )


# ==================================================
# 6. TEAM VISUALIZATIONS
# ==================================================

def visualize_team_performance():

    print("\nCreating team visualizations...")

    team_data = load_csv("team_performance_summary.csv")

    if team_data is not None:

        team_col = find_column(
            team_data,
            ["team", "batting_team"]
        )

        runs_col = find_column(
            team_data,
            ["total_runs", "runs"]
        )

        if team_col and runs_col:

            create_bar_chart(
                team_data,
                team_col,
                runs_col,
                "Team Comparison by Total Runs",
                "Total Runs",
                "Team",
                "team_total_runs.png",
                top_n=15
            )

    win_data = load_csv("team_win_rate.csv")

    if win_data is not None:

        team_col = find_column(
            win_data,
            ["team", "batting_team"]
        )

        win_col = find_column(
            win_data,
            ["win_percentage", "win_rate", "winning_percentage"]
        )

        if team_col and win_col:

            create_bar_chart(
                win_data,
                team_col,
                win_col,
                "Team Win Percentage",
                "Win Percentage (%)",
                "Team",
                "team_win_percentage.png",
                top_n=15
            )

    chase_data = load_csv(
        "team_batting_first_vs_chase.csv"
    )

    if chase_data is not None:

        team_col = find_column(
            chase_data,
            ["team", "batting_team"]
        )

        innings_col = find_column(
            chase_data,
            ["innings_type"]
        )

        runs_col = find_column(
            chase_data,
            ["total_runs", "runs"]
        )

        if team_col and innings_col and runs_col:

            pivot = chase_data.pivot_table(
                index=team_col,
                columns=innings_col,
                values=runs_col,
                aggfunc="sum"
            )

            if not pivot.empty:

                plt.figure(figsize=(13, 7))

                pivot.plot(
                    kind="bar",
                    figsize=(13, 7)
                )

                plt.title(
                    "Team Batting First vs Chasing"
                )

                plt.xlabel("Team")
                plt.ylabel("Total Player Runs")
                plt.xticks(rotation=45)
                plt.grid(axis="y", linestyle="--", alpha=0.4)

                save_chart(
                    "team_batting_first_vs_chasing.png"
                )


# ==================================================
# 7. MATCH VISUALIZATIONS
# ==================================================

def visualize_match_analysis():

    print("\nCreating match visualizations...")

    match_data = load_csv(
        "match_performance_analysis.csv"
    )

    if match_data is None:
        print("Match analysis data unavailable.")
        return

    run_difference_col = find_column(
        match_data,
        [
            "run_difference",
            "runs_difference",
            "score_difference"
        ]
    )

    if run_difference_col:

        values = pd.to_numeric(
            match_data[run_difference_col],
            errors="coerce"
        ).dropna()

        if not values.empty:

            plt.figure(figsize=(10, 6))

            plt.hist(
                values,
                bins=20,
                edgecolor="black"
            )

            plt.title(
                "Distribution of Match Run Differences"
            )

            plt.xlabel("Run Difference")
            plt.ylabel("Number of Matches")

            save_chart(
                "match_run_difference_distribution.png"
            )

    competitiveness_col = find_column(
        match_data,
        [
            "competitiveness",
            "match_competitiveness",
            "match_category"
        ]
    )

    if competitiveness_col:

        categories = (
            match_data[competitiveness_col]
            .value_counts()
        )

        if not categories.empty:

            plt.figure(figsize=(10, 6))

            categories.plot(
                kind="bar"
            )

            plt.title(
                "Match Competitiveness Distribution"
            )

            plt.xlabel("Match Category")
            plt.ylabel("Number of Matches")
            plt.xticks(rotation=45)

            save_chart(
                "match_competitiveness_distribution.png"
            )


# ==================================================
# 8. CORRELATION HEATMAP
# ==================================================

def visualize_correlation():

    print("\nCreating correlation heatmap...")

    batting = load_csv("player_batting_statistics.csv")

    if batting is None:
        print("Batting statistics unavailable for correlation.")
        return

    numeric_data = batting.select_dtypes(
        include=["number"]
    )

    if numeric_data.shape[1] < 2:
        print("Not enough numeric columns for correlation.")
        return

    correlation = numeric_data.corr()

    plt.figure(figsize=(12, 9))

    image = plt.imshow(
        correlation,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "Correlation Heatmap of Batting Statistics"
    )

    save_chart(
        "batting_correlation_heatmap.png"
    )


# ==================================================
# 9. MAIN FUNCTION
# ==================================================

def main():

    print("\n" + "=" * 60)
    print("CRICKET PERFORMANCE VISUALIZATION")
    print("=" * 60)

    visualize_batting()

    visualize_bowling()

    visualize_player_impact()

    visualize_team_performance()

    visualize_match_analysis()

    visualize_correlation()

    print("\n" + "=" * 60)
    print("VISUALIZATION PROCESS COMPLETED")
    print(f"Charts saved in: {VISUALIZATION_DIR}")
    print("=" * 60)


# ==================================================
# 10. EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
