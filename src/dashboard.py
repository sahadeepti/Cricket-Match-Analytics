import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = PROJECT_DIR / "data" / "analysis"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
VISUALIZATION_DIR = ANALYSIS_DIR / "visualizations"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cricket Match Analytics",
    page_icon="🏏",
    layout="wide"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #123c69;
    }

    .sub-title {
        font-size: 18px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def load_analysis_files():
    data = {}

    if ANALYSIS_DIR.exists():
        for file in ANALYSIS_DIR.glob("*.csv"):
            data[file.stem] = load_csv(file)

    return data


def load_processed_files():
    data = {}

    if PROCESSED_DIR.exists():
        for file in PROCESSED_DIR.glob("*.csv"):
            data[file.stem] = load_csv(file)

    return data


analysis_data = load_analysis_files()
processed_data = load_processed_files()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_dataset(keyword):
    """
    Find a CSV dataset whose filename contains a keyword.
    """
    for name, dataframe in analysis_data.items():
        if keyword.lower() in name.lower():
            return dataframe

    return pd.DataFrame()


def find_column(dataframe, possible_names):
    """
    Find a column by matching possible column names.
    Matching ignores capitalization and spaces.
    """
    if dataframe.empty:
        return None

    normalized_columns = {
        str(column).lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = name.lower().replace(" ", "_")

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def show_dataframe(dataframe, title=None):
    if title:
        st.subheader(title)

    if dataframe.empty:
        st.info("No data available for this section.")
    else:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


def show_metric(label, value):
    st.metric(label=label, value=value)


def show_bar_chart(dataframe, category_column, value_column, title):
    """
    Display a bar chart when both requested columns are available.
    """
    if dataframe.empty:
        return

    if category_column not in dataframe.columns:
        return

    if value_column not in dataframe.columns:
        return

    chart_data = dataframe[
        [category_column, value_column]
    ].dropna()

    if chart_data.empty:
        return

    chart_data = chart_data.sort_values(
        value_column,
        ascending=False
    ).head(10)

    chart_data = chart_data.set_index(category_column)

    st.subheader(title)

    st.bar_chart(chart_data[value_column])


def show_existing_visualizations():
    """
    Display PNG charts created by visualization.py.
    """
    if not VISUALIZATION_DIR.exists():
        st.info(
            "Visualization folder not found. "
            "Run visualization.py first."
        )
        return

    image_files = list(VISUALIZATION_DIR.glob("*.png"))

    if not image_files:
        st.info("No visualization images found.")
        return

    selected_image = st.selectbox(
        "Select a visualization",
        image_files,
        format_func=lambda path: path.stem.replace("_", " ").title()
    )

    st.image(
        str(selected_image),
        caption=selected_image.stem.replace("_", " ").title(),
        use_container_width=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏏 Cricket Match Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Interactive Cricket Data Analytics Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Dashboard Overview",
        "Batting Analysis",
        "Bowling Analysis",
        "Player Impact Analysis",
        "Team Analysis",
        "Match Analysis",
        "Visualizations",
        "Explore CSV Data"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Cricket Match Analytics Project"
)



# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    total_analysis_files = len(analysis_data)
    total_processed_files = len(processed_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric(
            "Analysis Datasets",
            total_analysis_files
        )

    with col2:
        show_metric(
            "Processed Datasets",
            total_processed_files
        )

    with col3:
        show_metric(
            "Visualization Charts",
            len(list(VISUALIZATION_DIR.glob("*.png")))
            if VISUALIZATION_DIR.exists()
            else 0
        )

    st.divider()

    st.subheader("Project Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Batting Analysis
            - Runs scored
            - Batting average
            - Strike rate
            - Boundary performance

            ### Bowling Analysis
            - Wickets
            - Economy rate
            - Bowling average
            - Bowling performance
            """
        )

    with col2:
        st.markdown(
            """
            ### Player Impact Analysis
            - Batter impact rankings
            - Bowler impact rankings
            - All-rounder rankings

            ### Team and Match Analysis
            - Team performance
            - Batting first vs chasing
            - Match competitiveness
            """
        )

    st.divider()

    st.subheader("Available Analysis Datasets")

    if analysis_data:
        dataset_summary = []

        for name, dataframe in analysis_data.items():
            dataset_summary.append({
                "Dataset": name,
                "Rows": len(dataframe),
                "Columns": len(dataframe.columns)
            })

        st.dataframe(
            pd.DataFrame(dataset_summary),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "No analysis CSV files found. "
            "Run the analysis modules first."
        )


# ============================================================
# BATTING ANALYSIS
# ============================================================

elif page == "Batting Analysis":

    st.header("🏏 Batting Performance Analysis")

    batting_data = find_dataset("batting")

    if batting_data.empty:
        st.warning(
            "Batting analysis dataset was not found."
        )

        st.write("Available datasets:", list(analysis_data.keys()))

    else:
        st.write(
            "Explore batting statistics and player performance."
        )

        player_column = find_column(
            batting_data,
            ["player", "batter", "batsman", "name"]
        )

        runs_column = find_column(
            batting_data,
            ["runs", "total_runs", "runs_scored"]
        )

        average_column = find_column(
            batting_data,
            ["average", "batting_average", "avg"]
        )

        strike_rate_column = find_column(
            batting_data,
            ["strike_rate", "sr"]
        )

        if player_column and runs_column:
            show_bar_chart(
                batting_data,
                player_column,
                runs_column,
                "Top 10 Players by Runs"
            )

        if player_column and average_column:
            show_bar_chart(
                batting_data,
                player_column,
                average_column,
                "Top 10 Players by Batting Average"
            )

        if player_column and strike_rate_column:
            show_bar_chart(
                batting_data,
                player_column,
                strike_rate_column,
                "Top 10 Players by Strike Rate"
            )

        st.divider()

        show_dataframe(
            batting_data,
            "Batting Statistics"
        )


# ============================================================
# BOWLING ANALYSIS
# ============================================================

elif page == "Bowling Analysis":

    st.header("🎯 Bowling Performance Analysis")

    bowling_data = find_dataset("bowling")

    if bowling_data.empty:
        st.warning(
            "Bowling analysis dataset was not found."
        )

        st.write("Available datasets:", list(analysis_data.keys()))

    else:
        player_column = find_column(
            bowling_data,
            ["player", "bowler", "name"]
        )

        wickets_column = find_column(
            bowling_data,
            ["wickets", "total_wickets"]
        )

        economy_column = find_column(
            bowling_data,
            ["economy", "economy_rate", "econ"]
        )

        average_column = find_column(
            bowling_data,
            ["bowling_average", "average", "avg"]
        )

        if player_column and wickets_column:
            show_bar_chart(
                bowling_data,
                player_column,
                wickets_column,
                "Top 10 Bowlers by Wickets"
            )

        if player_column and economy_column:
            show_bar_chart(
                bowling_data,
                player_column,
                economy_column,
                "Bowling Economy Rate"
            )

        if player_column and average_column:
            show_bar_chart(
                bowling_data,
                player_column,
                average_column,
                "Bowling Average"
            )

        st.divider()

        show_dataframe(
            bowling_data,
            "Bowling Statistics"
        )


# ============================================================
# PLAYER IMPACT ANALYSIS
# ============================================================

elif page == "Player Impact Analysis":

    st.header("⭐ Player Impact Analysis")

    impact_data = find_dataset("impact")

    if impact_data.empty:
        st.warning(
            "Player impact dataset was not found."
        )

        st.write("Available datasets:", list(analysis_data.keys()))

    else:
        st.write(
            "Player impact scores are calculated using "
            "the project's custom analytical scoring method."
        )

        player_column = find_column(
            impact_data,
            ["player", "name"]
        )

        score_column = find_column(
            impact_data,
            ["impact_score", "total_impact_score", "score"]
        )

        if player_column and score_column:
            show_bar_chart(
                impact_data,
                player_column,
                score_column,
                "Top 10 Players by Impact Score"
            )

        st.divider()

        show_dataframe(
            impact_data,
            "Player Impact Rankings"
        )


# ============================================================
# TEAM ANALYSIS
# ============================================================

elif page == "Team Analysis":

    st.header("🏆 Team Performance Analysis")

    team_data = find_dataset("team_performance_summary")

    if team_data.empty:
        team_data = find_dataset("team")

    if team_data.empty:
        st.warning(
            "Team analysis dataset was not found."
        )

        st.write("Available datasets:", list(analysis_data.keys()))

    else:
        team_column = find_column(
            team_data,
            ["team", "batting_team", "name"]
        )

        runs_column = find_column(
            team_data,
            ["total_runs", "runs", "average_runs"]
        )

        win_column = find_column(
            team_data,
            ["win_percentage", "win_rate", "win_percent"]
        )

        if team_column and runs_column:
            show_bar_chart(
                team_data,
                team_column,
                runs_column,
                "Team Runs Comparison"
            )

        if team_column and win_column:
            show_bar_chart(
                team_data,
                team_column,
                win_column,
                "Team Win Rate Comparison"
            )

        st.divider()

        show_dataframe(
            team_data,
            "Team Performance Summary"
        )

        st.subheader("Additional Team Datasets")

        for name, dataframe in analysis_data.items():
            if "team" in name.lower() and name != team_data:
                with st.expander(name.replace("_", " ").title()):
                    show_dataframe(dataframe)


# ============================================================
# MATCH ANALYSIS
# ============================================================

elif page == "Match Analysis":

    st.header("📊 Match Analysis")

    match_data = find_dataset("match_performance")

    if match_data.empty:
        match_data = find_dataset("match")

    if match_data.empty:
        st.warning(
            "Match analysis dataset was not found."
        )

        st.write("Available datasets:", list(analysis_data.keys()))

    else:
        st.write(
            "Explore match statistics and competitiveness."
        )

        show_dataframe(
            match_data,
            "Match Performance Data"
        )

        competitiveness_data = find_dataset(
            "competitiveness"
        )

        if not competitiveness_data.empty:
            st.subheader("Match Competitiveness Summary")

            show_dataframe(
                competitiveness_data
            )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "Visualizations":

    st.header("📈 Cricket Analytics Visualizations")

    st.write(
        "View charts generated by the visualization module."
    )

    show_existing_visualizations()


# ============================================================
# EXPLORE CSV DATA
# ============================================================

elif page == "Explore CSV Data":

    st.header("📁 Explore Analysis CSV Files")

    if not analysis_data:
        st.warning(
            "No analysis CSV files found in data/analysis."
        )

    else:
        selected_dataset = st.selectbox(
            "Select Dataset",
            list(analysis_data.keys())
        )

        selected_data = analysis_data[selected_dataset]

        st.subheader(
            selected_dataset.replace("_", " ").title()
        )

        st.write(
            f"Rows: {len(selected_data)} | "
            f"Columns: {len(selected_data.columns)}"
        )

        st.dataframe(
            selected_data,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Download Dataset")

        csv_data = selected_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{selected_dataset}.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cricket Performance Intelligence System | "
    "MCA Data Analytics Project"
)