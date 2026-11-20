# 🏏 Cricket Match Performance Analysis System

### A Data Analytics Approach to Player, Team and Match Performance

## 1. Project Overview

The **Cricket Match Performance Analysis System** is a Python-based data analytics project developed to analyze T20 cricket performance using historical cricket datasets.

The system processes batting, bowling, ball-by-ball and match-result data to generate meaningful insights about individual players, teams and matches.

It applies statistical analysis, data aggregation, feature engineering, percentile-based ranking and data visualization to evaluate cricket performance.

The project also provides an interactive web dashboard developed using Streamlit, allowing users to explore analytical results through tables, charts and player rankings.

## 2. Project Objectives

The major objectives are:

- Analyze batting performance using runs, batting average, strike rate and boundary contribution.
- Evaluate bowling performance using wickets, economy rate, bowling average and bowling strike rate.
- Identify high-impact batsmen, bowlers and all-rounders using a weighted scoring approach.
- Compare team performance and batting-first versus chasing patterns.
- Analyze match performance and classify matches based on competitiveness.
- Generate graphical representations of cricket statistics.
- Develop an interactive dashboard for exploring cricket performance data.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data cleaning, manipulation and aggregation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |
| OpenPyXL | Reading Excel datasets, if required |

## 4. Project Structure

```text
CricketMatchAnalysis/
│
├── data/
│   ├── raw/
│   │   ├── Batting.csv
│   │   ├── MatchResults refined .xlsx
│   │   └── ball_by_ball_t20_men.csv
│   │
│   ├── processed/
│   │   ├── ball_by_ball_processed.csv
│   │   ├── batter_stats_from_ball_data.csv
│   │   ├── batting_processed.csv
│   │   ├── bowling_processed.csv
│   │   ├── match_results_processed.csv
│   │   └── match_summary_processed.csv
│   │
│   └── analysis/
│       ├── player_impact_scores.csv
│       ├── batter_impact_ranking.csv
│       ├── bowler_impact_ranking.csv
│       ├── all_rounder_impact_ranking.csv
│       ├── team_performance_summary.csv
│       ├── team_batting_first_vs_chase.csv
│       ├── team_win_rate.csv
│       ├── team_match_innings_statistics.csv
│       ├── match_performance_analysis.csv
│       ├── match_competitiveness_summary.csv
│       ├── close_matches.csv
│       │
│       └── visualizations/
│           └── Generated PNG charts
│
├── src/
│   ├── preprocessing.py
│   ├── batting_analysis.py
│   ├── bowling_analysis.py
│   ├── player_impact.py
│   ├── match_analysis.py
│   ├── team_analysis.py
│   ├── visualization.py
│   └── dashboard.py
│
├── requirements.txt
└── README.md
```

*Note: The folder structure above describes the expected project organization. Some generated CSV filenames may vary depending on the implementation.*

## 5. Modules and Their Functionality

### 5.1 Data Preprocessing (`preprocessing.py`)

This module prepares raw cricket datasets for analysis.

Main operations:

- Load raw CSV and Excel datasets.
- Handle missing and duplicate records.
- Standardize column names and data types.
- Process ball-by-ball cricket records.
- Identify legal deliveries and calculate derived statistics.
- Generate processed batting, bowling and match datasets.

**Output:** Cleaned datasets stored in `data/processed/`.

### 5.2 Batting Analysis (`batting_analysis.py`)

Evaluates individual batting performance using statistical measures.

Key analyses:

- Total runs scored.
- Batting average.
- Batting strike rate.
- Boundary contribution.
- Player consistency.
- Batting performance while chasing.
- Player performance rankings.

### 5.3 Bowling Analysis (`bowling_analysis.py`)

Evaluates bowling performance using delivery-level and match-level statistics.

Key analyses:

- Total wickets.
- Bowling average.
- Economy rate.
- Bowling strike rate.
- Dot-ball percentage.
- Match-level bowling performance.
- Bowling phase analysis.

### 5.4 Player Impact Analysis (`player_impact.py`)

Evaluates player contribution using multiple performance indicators.

The module applies:

- Percentile-based ranking.
- Weighted scoring.
- Batting impact calculation.
- Bowling impact calculation.
- All-rounder impact calculation.

The generated rankings help compare players across multiple performance dimensions.

### 5.5 Match Analysis (`match_analysis.py`)

Examines match-level performance and competitiveness.

Key analyses:

- First-innings and second-innings scoring comparison.
- Run difference calculation.
- Match competitiveness classification.
- Identification of close matches.
- Match performance summaries.

### 5.6 Team Analysis (`team_analysis.py`)

Evaluates team-level performance.

Key analyses:

- Team performance summaries.
- Team win percentage.
- Batting-first versus chasing comparison.
- Team innings statistics.
- Average runs and run-rate analysis.

### 5.7 Data Visualization (`visualization.py`)

Generates graphical representations of analytical results using Matplotlib.

Examples include:

- Top batsmen by runs.
- Batting average and strike-rate comparisons.
- Top bowlers by wickets.
- Bowling economy comparisons.
- Player impact rankings.
- Team performance comparisons.
- Match competitiveness distributions.
- Batting correlation heatmap.

**Output:** PNG charts stored in `data/analysis/visualizations/`.

### 5.8 Interactive Dashboard (`dashboard.py`)

Provides an interactive interface using Streamlit.

Dashboard sections:

1. Dashboard Overview
2. Batting Analysis
3. Bowling Analysis
4. Player Impact Analysis
5. Team Analysis
6. Match Analysis
7. Visualizations
8. Explore CSV Data

Users can inspect analytical tables, compare players and teams, view charts and download selected analysis datasets.

## 6. Statistical Formulas and Analytical Techniques

The project uses the following formulas and techniques.

### Batting Statistics

**Batting Average**

```text
Batting Average = Total Runs / Times Dismissed
```

**Batting Strike Rate**

```text
Strike Rate = (Runs Scored / Balls Faced) × 100
```

**Boundary Runs**

```text
Boundary Runs = (4 × Fours) + (6 × Sixes)
```

**Boundary Percentage**

```text
Boundary Percentage = (Boundary Runs / Total Runs) × 100
```

**Coefficient of Variation**

```text
CV = (Standard Deviation / Mean) × 100
```

### Bowling Statistics

**Economy Rate**

```text
Economy Rate = (Runs Conceded / Legal Balls) × 6
```

**Bowling Average**

```text
Bowling Average = Runs Conceded / Wickets
```

**Bowling Strike Rate**

```text
Bowling Strike Rate = Legal Balls / Wickets
```

**Dot-Ball Percentage**

```text
Dot-Ball Percentage = (Dot Balls / Legal Balls) × 100
```

### Team and Match Statistics

**Win Percentage**

```text
Win Percentage = (Matches Won / Matches Played) × 100
```

**Run Difference**

```text
Run Difference = Absolute Difference Between Innings Runs
```

### Player Impact Analysis

The project uses percentile ranking and weighted scoring to combine different performance indicators.

The batting and bowling impact scores are calculated using project-defined weights. This allows different statistics to be compared on a common scale.

These impact scores are analytical indicators developed for this project, not official cricket ratings.

## 7. Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or later (a compatible Python environment is required).
- Anaconda or Miniconda (recommended).
- Visual Studio Code or another Python IDE.

### Step 1: Clone or Download the Project

Download the project and open its main directory in PowerShell or the terminal.

```powershell
cd path\to\CricketMatchAnalysis
```

### Step 2: Create a Conda Environment

```powershell
conda create -n cricket_analysis python=3.11 -y
```

### Step 3: Activate the Environment

```powershell
conda activate cricket_analysis
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not yet created, install the main packages using:

```powershell
pip install pandas numpy matplotlib streamlit openpyxl
```

## 8. Execution Instructions

Run the modules in the following order.

### Step 1: Preprocess the datasets

```powershell
python src/preprocessing.py
```

This generates cleaned datasets in `data/processed/`.

### Step 2: Run Batting Analysis

```powershell
python src/batting_analysis.py
```

### Step 3: Run Bowling Analysis

```powershell
python src/bowling_analysis.py
```

### Step 4: Run Player Impact Analysis

```powershell
python src/player_impact.py
```

### Step 5: Run Match Analysis

```powershell
python src/match_analysis.py
```

### Step 6: Run Team Analysis

```powershell
python src/team_analysis.py
```

### Step 7: Generate Visualizations

```powershell
python src/visualization.py
```

### Step 8: Launch the Streamlit Dashboard

```powershell
streamlit run src/dashboard.py
```

The dashboard will normally open
