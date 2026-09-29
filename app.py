import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="India vs West Indies | Cricket Analytics",
    page_icon="🏏",
    layout="wide"
)


# ============================================================
# CUSTOM CSS (works in both light and dark themes)
# ============================================================

st.markdown("""
<style>
.metric-card {
    background: var(--secondary-background-color);
    padding: 18px 12px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    height: 100%;
}
.metric-title {
    font-size: 13px;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    margin-top: 4px;
}
.metric-sub {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 2px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 8px;
}
.result-banner {
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    color: white;
    padding: 16px 22px;
    border-radius: 14px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin: 6px 0 18px 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = Path("data/processed")

REQUIRED_FILES = {
    "match_summary": "match_summary.csv",
    "team_comparison": "team_comparison.csv",
    "batting": "batting_summary.csv",
    "bowling": "bowling_summary.csv",
    "phase_impact": "phase_impact.csv",
    "over_analysis": "over_analysis_insights.csv",
    "wicket_overs": "wicket_overs.csv",
    "highest_pressure": "highest_pressure_overs.csv",
}

OPTIONAL_FILES = {
    "highest_scoring": "highest_scoring_overs.csv",
    "phase_batting": "phase_batting_analysis.csv",
    "phase_bowling": "phase_bowling_analysis.csv",
    "batter_bowler": "batter_bowler_matchup.csv",
}


# ============================================================
# HELPERS
# ============================================================

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces from headers and text values so team names match."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )
    return df


@st.cache_data
def load_data():
    data, missing = {}, []

    for name, fname in REQUIRED_FILES.items():
        path = DATA_PATH / fname
        data[name] = clean_df(pd.read_csv(path)) if path.exists() else None
        if data[name] is None:
            missing.append(fname)

    for name, fname in OPTIONAL_FILES.items():
        path = DATA_PATH / fname
        data[name] = clean_df(pd.read_csv(path)) if path.exists() else pd.DataFrame()

    return data, missing


data, missing_files = load_data()

if missing_files:
    st.error(
        "Missing required data files in "
        f"`{DATA_PATH}`: {', '.join(missing_files)}"
    )
    st.stop()

match_summary = data["match_summary"]
team_comparison = data["team_comparison"]
batting = data["batting"]
bowling = data["bowling"]
phase_impact = data["phase_impact"]
over_analysis = data["over_analysis"]
wicket_overs = data["wicket_overs"]
highest_pressure = data["highest_pressure"]
highest_scoring = data["highest_scoring"]
phase_batting = data["phase_batting"]
phase_bowling = data["phase_bowling"]
batter_bowler = data["batter_bowler"]


def first_col(df, candidates):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def available(df, candidates):
    return [c for c in candidates if c in df.columns]


def teams_of(df, col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def metric_card(title, value, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def empty_note(name):
    st.info(f"No data available for **{name}**.")


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


def bar_chart(df, x, y, title, color=None, fmt=None, barmode="group", sort=None):
    """Safe bar chart: skips gracefully if a column is missing."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        empty_note(title)
        return
    if sort is not None:
        df = df.sort_values(y, ascending=(sort == "asc"))
    fig = px.bar(
        df, x=x, y=y, color=color if color in df.columns else None,
        text=y, title=title, barmode=barmode
    )
    fig.update_traces(
        texttemplate=fmt or "%{text}",
        textposition="outside",
        cliponaxis=False
    )
    fig.update_layout(margin=dict(t=60, b=20))
    show(fig)


def download_button(df, label, filename):
    st.download_button(
        label=f"⬇️ {label}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# ============================================================
# DERIVED COLUMNS
# ============================================================

FOURS = first_col(batting, ["4's", "Fours", "4s"])
SIXES = first_col(batting, ["6's", "Sixes", "6s"])

if "Dot %" not in batting.columns:
    if "Dot_Balls" in batting.columns and "Balls" in batting.columns:
        batting["Dot %"] = np.where(
            batting["Balls"] > 0,
            batting["Dot_Balls"] / batting["Balls"] * 100, 0
        ).round(2)

if "Dot %" not in bowling.columns:
    if "Dot_Balls" in bowling.columns and "Legal_Balls" in bowling.columns:
        bowling["Dot %"] = np.where(
            bowling["Legal_Balls"] > 0,
            bowling["Dot_Balls"] / bowling["Legal_Balls"] * 100, 0
        ).round(2)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏏 Cricket Analytics")
st.sidebar.markdown("### India 🇮🇳 vs West Indies 🌴")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏟️ Match Centre",
        "🟢 Batting Analysis",
        "🔴 Bowling Analysis",
        "🟡 Match Insights",
        "⚔️ Matchups"
    ],
    key="nav_page"
)

st.sidebar.divider()
st.sidebar.caption("Ball-by-ball ODI analytics")


# ============================================================
# MATCH CENTRE
# ============================================================

if page == "🏟️ Match Centre":

    st.title("🏟️ Match Centre")
    st.markdown("### What happened in the match?")

    if len(match_summary) < 2:
        st.warning("Match summary needs two innings rows.")
        st.stop()

    inn1 = match_summary.iloc[0]   # batting first
    inn2 = match_summary.iloc[1]   # chasing

    runs1, runs2 = int(inn1["Runs"]), int(inn2["Runs"])
    wk1, wk2 = int(inn1["Wickets"]), int(inn2["Wickets"])

    if runs2 > runs1:
        result = f'{inn2["Batting Team"]} won by {10 - wk2} wickets'
    elif runs1 > runs2:
        result = f'{inn1["Batting Team"]} won by {runs1 - runs2} runs'
    else:
        result = "Match Tied"

    st.markdown(f'<div class="result-banner">🏆 {result}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            f'{inn1["Batting Team"]} (1st inns)',
            f"{runs1}/{wk1}",
            f'{inn1["Overs"]} overs' if "Overs" in match_summary.columns else ""
        )
    with c2:
        metric_card(
            f'{inn2["Batting Team"]} (2nd inns)',
            f"{runs2}/{wk2}",
            f'{inn2["Overs"]} overs' if "Overs" in match_summary.columns else ""
        )
    with c3:
        metric_card("Run difference", abs(runs1 - runs2))

    st.divider()

    # ---------------- Run progression ----------------
    section("📈 Run Progression")

    fig = px.line(
        over_analysis, x="Over", y="Cumulative Runs",
        color="Batting Team", markers=True,
        title="Cumulative Runs by Over"
    )
    fig.update_layout(
        xaxis_title="Over", yaxis_title="Cumulative Runs",
        hovermode="x unified"
    )
    show(fig)

    # ---------------- Team comparison ----------------
    section("📊 Team Comparison")

    comp_options = available(
        team_comparison,
        ["Runs", "Wickets", "Fours", "Sixes", "Run Rate",
         "Boundary Runs", "Boundary %"]
    )
    if comp_options:
        metric = st.selectbox(
            "Select comparison metric", comp_options, key="comp_metric"
        )
        bar_chart(
            team_comparison, "Batting Team", metric,
            f"{metric} Comparison", sort=None
        )
    else:
        empty_note("Team comparison")

    # ---------------- Wicket timeline ----------------
    section("🎯 Wicket Timeline")

    wk_size = first_col(wicket_overs, ["Wickets"])
    hover = available(wicket_overs, ["Runs", "Wickets", "Cumulative Runs"])
    fig = px.scatter(
        wicket_overs, x="Over", y="Cumulative Wickets",
        color="Batting Team", size=wk_size, hover_data=hover,
        title="Wickets by Over"
    )
    fig.update_layout(xaxis_title="Over", yaxis_title="Cumulative Wickets")
    show(fig)

    # ---------------- Scorecard ----------------
    section("📋 Match Scorecard")
    st.dataframe(match_summary, use_container_width=True, hide_index=True)


# ============================================================
# BATTING ANALYSIS
# ============================================================

elif page == "🟢 Batting Analysis":

    st.title("🟢 Batting Analysis")
    st.markdown("### Who contributed with the bat?")

    teams = teams_of(batting, "Batting Team")
    if not teams:
        st.warning("No batting teams found.")
        st.stop()

    top_l, top_r = st.columns([2, 1])
    with top_l:
        selected_team = st.selectbox(
            "Select Batting Team", teams, key="bat_team"
        )

    team_batting = batting[batting["Batting Team"] == selected_team].copy()

    if team_batting.empty:
        st.warning("No batting data for this team.")
        st.stop()

    with top_r:
        max_balls = int(team_batting["Balls"].max())
        min_balls = st.slider(
            "Minimum balls faced", 0, max(max_balls, 1), 0, key="bat_min_balls"
        )

    team_batting = team_batting[team_batting["Balls"] >= min_balls]
    if team_batting.empty:
        st.warning("No batters meet the minimum balls filter.")
        st.stop()

    # ---------------- KPIs ----------------
    total_runs = team_batting["Runs"].sum()
    total_balls = team_batting["Balls"].sum()
    team_sr = (total_runs / total_balls * 100) if total_balls else 0
    boundaries = sum(
        team_batting[c].sum() for c in (FOURS, SIXES) if c
    )
    best = team_batting.loc[team_batting["Runs"].idxmax()]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Team Runs", int(total_runs))
    with c2:
        metric_card("Best Batter", best["Striker"])
    with c3:
        metric_card("Best Score", int(best["Runs"]),
                    f'off {int(best["Balls"])} balls')
    with c4:
        metric_card("Team Strike Rate", f"{team_sr:.1f}")
    with c5:
        metric_card("Boundaries", int(boundaries))

    st.divider()

    # ---------------- Charts ----------------
    section("📊 Runs by Batter")
    bar_chart(team_batting, "Striker", "Runs",
              f"Runs scored by {selected_team}", sort="desc")

    section("⚡ Strike Rate")
    bar_chart(team_batting, "Striker", "Strike Rate",
              "Strike Rate by Batter", fmt="%{text:.1f}", sort="desc")

    section("🎯 Runs vs Strike Rate")
    fig = px.scatter(
        team_batting, x="Balls", y="Runs", size="Strike Rate",
        color="Striker", hover_data=available(team_batting, ["Strike Rate"]),
        title="Batter Impact (bubble size = strike rate)"
    )
    show(fig)

    if FOURS and SIXES:
        section("💥 Boundaries")
        bnd = team_batting[["Striker", FOURS, SIXES]].melt(
            id_vars="Striker", var_name="Type", value_name="Count"
        )
        fig = px.bar(
            bnd, x="Striker", y="Count", color="Type",
            barmode="group", title=f"{selected_team} — Fours and Sixes"
        )
        show(fig)

    if "Dot %" in team_batting.columns:
        section("🔵 Dot Ball Percentage")
        bar_chart(team_batting, "Striker", "Dot %",
                  "Dot Ball % by Batter", fmt="%{text:.1f}%", sort="desc")

    section("🔥 Batting by Phase")
    if not phase_batting.empty and "Batting Team" in phase_batting.columns:
        pb = phase_batting[phase_batting["Batting Team"] == selected_team]
        bar_chart(pb, "Phase", "Runs", "Runs by Phase", color="Phase")
    else:
        empty_note("Batting by phase")

    section("📋 Batting Scorecard")
    st.dataframe(team_batting, use_container_width=True, hide_index=True)
    download_button(team_batting, "Download batting data",
                    f"{selected_team}_batting.csv")


# ============================================================
# BOWLING ANALYSIS
# ============================================================

elif page == "🔴 Bowling Analysis":

    st.title("🔴 Bowling Analysis")
    st.markdown("### How did the bowling attack control the batters?")

    teams = teams_of(bowling, "Bowling Team")
    if not teams:
        st.warning("No bowling teams found.")
        st.stop()

    selected_team = st.selectbox(
        "Select Bowling Team", teams, key="bowl_team"
    )

    team_bowling = bowling[bowling["Bowling Team"] == selected_team].copy()
    if team_bowling.empty:
        st.warning("No bowling data for this team.")
        st.stop()

    # ---------------- KPIs ----------------
    total_wickets = team_bowling["Wickets"].sum()
    total_runs = team_bowling["Runs_Conceded"].sum()
    best = team_bowling.loc[team_bowling["Wickets"].idxmax()]
    econ_col = "Economy" if "Economy" in team_bowling.columns else None
    best_econ = (
        team_bowling.loc[team_bowling["Economy"].idxmin()]
        if econ_col else None
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Wickets", int(total_wickets))
    with c2:
        metric_card("Top Wicket-Taker", best["Bowler"])
    with c3:
        metric_card("Best Wickets", int(best["Wickets"]))
    with c4:
        metric_card("Runs Conceded", int(total_runs))
    with c5:
        if best_econ is not None:
            metric_card("Best Economy", f'{best_econ["Economy"]:.2f}',
                        best_econ["Bowler"])
        else:
            metric_card("Best Economy", "-")

    st.divider()

    section("🎯 Wickets by Bowler")
    bar_chart(team_bowling, "Bowler", "Wickets", "Wickets Taken", sort="desc")

    section("📉 Economy Rate")
    bar_chart(team_bowling, "Bowler", "Economy", "Bowler Economy Rate",
              fmt="%{text:.2f}", sort="asc")

    if econ_col:
        section("🎯 Economy vs Wickets")
        fig = px.scatter(
            team_bowling, x="Economy", y="Wickets", color="Bowler",
            size="Runs_Conceded",
            title="Bowler Impact (bubble size = runs conceded)"
        )
        show(fig)

    if "Dot %" in team_bowling.columns:
        section("🔵 Dot Ball Percentage")
        bar_chart(team_bowling, "Bowler", "Dot %", "Dot Ball % by Bowler",
                  fmt="%{text:.1f}%", sort="desc")

    section("🔥 Bowling by Phase")
    if not phase_bowling.empty and "Bowling Team" in phase_bowling.columns:
        pbw = phase_bowling[phase_bowling["Bowling Team"] == selected_team]
        bar_chart(pbw, "Phase", "Runs_Conceded", "Runs Conceded by Phase",
                  color="Phase")
    else:
        empty_note("Bowling by phase")

    section("📋 Bowling Scorecard")
    st.dataframe(team_bowling, use_container_width=True, hide_index=True)
    download_button(team_bowling, "Download bowling data",
                    f"{selected_team}_bowling.csv")


# ============================================================
# MATCH INSIGHTS
# ============================================================

elif page == "🟡 Match Insights":

    st.title("🟡 Match Insights")
    st.markdown("### Why did the match unfold this way?")

    # ---------------- Momentum ----------------
    section("📈 Scoring Momentum")

    mom_col = first_col(over_analysis, ["Over Run Rate", "Runs"])
    smooth = st.checkbox("Smooth with 3-over rolling average", value=False,
                         key="smooth")

    mom = over_analysis.sort_values(["Batting Team", "Over"]).copy()
    if smooth and mom_col:
        mom[mom_col] = (
            mom.groupby("Batting Team")[mom_col]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
            .round(2)
        )
    if mom_col:
        fig = px.line(
            mom, x="Over", y=mom_col, color="Batting Team", markers=True,
            title="Runs Scored Per Over"
        )
        fig.update_layout(xaxis_title="Over", yaxis_title="Runs per Over",
                          hovermode="x unified")
        show(fig)

    # ---------------- Phase impact ----------------
    section("🔥 Phase Impact")

    phase_options = available(
        phase_impact, ["Runs", "Wickets", "Run Rate", "Run Contribution %"]
    )
    if phase_options:
        phase_metric = st.selectbox(
            "Select phase metric", phase_options, key="phase_metric"
        )
        bar_chart(phase_impact, "Phase", phase_metric,
                  f"{phase_metric} by Phase", color="Batting Team")
    else:
        empty_note("Phase impact")

    # ---------------- Pressure ----------------
    section("🚨 Pressure Index")

    if "Pressure Index" in over_analysis.columns:
        fig = px.bar(
            over_analysis.sort_values(["Batting Team", "Over"]),
            x="Over", y="Pressure Index", color="Batting Team",
            barmode="group", title="Custom Pressure Index by Over"
        )
        fig.update_layout(xaxis_title="Over", yaxis_title="Pressure Index")
        show(fig)
    st.info(
        "Pressure Index is a custom analytical metric based on "
        "dot-ball percentage, wickets and over run rate."
    )

    # ---------------- Highest scoring ----------------
    section("💥 Highest Scoring Overs")
    if not highest_scoring.empty:
        hs = highest_scoring.copy()
        hs["Over"] = hs["Over"].astype(str)
        bar_chart(hs, "Over", "Runs", "Highest Scoring Overs",
                  color="Batting Team", sort="desc")
    else:
        empty_note("Highest scoring overs")

    # ---------------- Tables ----------------
    section("🎯 Wicket Overs")
    st.dataframe(wicket_overs, use_container_width=True, hide_index=True)

    section("🚨 Highest Pressure Overs")
    st.dataframe(highest_pressure, use_container_width=True, hide_index=True)


# ============================================================
# MATCHUPS
# ============================================================

elif page == "⚔️ Matchups":

    st.title("⚔️ Batter vs Bowler Matchups")
    st.markdown("### Who won the individual battles?")

    if batter_bowler.empty:
        st.warning("`batter_bowler_matchup.csv` was not found or is empty.")
        st.stop()

    bat_col = first_col(batter_bowler, ["Striker", "Batter", "Batsman"])
    bowl_col = first_col(batter_bowler, ["Bowler"])
    runs_col = first_col(batter_bowler, ["Runs", "Runs_Scored"])
    balls_col = first_col(batter_bowler, ["Balls", "Legal_Balls"])
    wk_col = first_col(batter_bowler, ["Wickets", "Dismissals", "Wicket"])

    if not (bat_col and bowl_col):
        st.dataframe(batter_bowler, use_container_width=True, hide_index=True)
        st.stop()

    f1, f2 = st.columns(2)
    with f1:
        batters = sorted(batter_bowler[bat_col].dropna().unique())
        sel_batters = st.multiselect("Batters", batters, default=[],
                                     key="mu_bat",
                                     placeholder="All batters")
    with f2:
        bowlers = sorted(batter_bowler[bowl_col].dropna().unique())
        sel_bowlers = st.multiselect("Bowlers", bowlers, default=[],
                                     key="mu_bowl",
                                     placeholder="All bowlers")

    mu = batter_bowler.copy()
    if sel_batters:
        mu = mu[mu[bat_col].isin(sel_batters)]
    if sel_bowlers:
        mu = mu[mu[bowl_col].isin(sel_bowlers)]

    if mu.empty:
        st.info("No matchups for this selection.")
        st.stop()

    value_options = available(mu, [runs_col, balls_col, wk_col, "Strike Rate"])
    if runs_col and value_options:
        value = st.selectbox("Heatmap value", value_options, key="mu_value")

        pivot = mu.pivot_table(
            index=bat_col, columns=bowl_col, values=value,
            aggfunc="sum", fill_value=0
        )
        fig = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            title=f"{value}: batter (rows) vs bowler (columns)"
        )
        show(fig)

    section("📋 Matchup Table")
    sort_col = runs_col or mu.columns[-1]
    st.dataframe(
        mu.sort_values(sort_col, ascending=False),
        use_container_width=True, hide_index=True
    )
    download_button(mu, "Download matchups", "matchups.csv")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.6;">
    🏏 India vs West Indies ODI | Cricket Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
