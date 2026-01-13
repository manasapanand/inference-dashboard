import streamlit as st
import pandas as pd
import json
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Post University Chat Inference Analytics Dashboard",
    page_icon="🎓",
    layout="wide"
)

DATA_FILES = [
    "/Users/manasa/Desktop/langfuse/output/GPT_inference_edtech_gold_langfuse.json",
    "/Users/manasa/Desktop/langfuse/output/GPT_inference_edtech_gold_fresh_test100.json"
]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(files):
    all_sessions = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
            sessions = data.get("sessions", [])
            for s in sessions:
                s["_source_file"] = path.split("/")[-1]  # optional lineage
            all_sessions.extend(sessions)
    return all_sessions

sessions = load_data(DATA_FILES)

# ---------------- NORMALIZE ----------------
rows = []

for s in sessions:
    inf = s["session_inference"]

    timestamps = [
        pd.to_datetime(m.get("timestamp"), errors="coerce")
        for m in s.get("messages", [])
        if m.get("timestamp")
    ]
    if not timestamps:
        continue

    session_ts = min(timestamps)

    rows.append({
        "sessionId": s["sessionId"],
        "source": inf["source"],
        "primary_intent": inf["primary_intent"],
        "sentiment": inf["sentiment"],
        "urgency": inf["urgency"],
        "escalation": inf["escalation"]["required"],
        "escalation_level": inf["escalation"]["level"],
        "complexity_score": inf["complexity_score"],
        "resolution_confidence": inf["resolution_confidence"],
        "intent_count": len(inf["topics"]),
        "intent_flow": inf["intent_flow"],
        "risk_flags": ", ".join(inf["risk_flags"]),
        "session_time": session_ts,
        "session_date": session_ts.date(),
        "session_week": session_ts.to_period("W").start_time,
        "data_source": s["_source_file"]
    })

df = pd.DataFrame(rows)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

source_filter = st.sidebar.multiselect(
    "Support Source",
    options=df["source"].unique(),
    default=df["source"].unique()
)

intent_filter = st.sidebar.multiselect(
    "Primary Intent",
    options=df["primary_intent"].unique(),
    default=df["primary_intent"].unique()
)

data_source_filter = st.sidebar.multiselect(
    "Input File",
    options=df["data_source"].unique(),
    default=df["data_source"].unique()
)

escalation_filter = st.sidebar.selectbox(
    "Escalation Required",
    options=["All", True, False]
)

filtered = df[
    (df["source"].isin(source_filter)) &
    (df["primary_intent"].isin(intent_filter)) &
    (df["data_source"].isin(data_source_filter))
]

if escalation_filter != "All":
    filtered = filtered[filtered["escalation"] == escalation_filter]

# ---------------- KPI COMPUTATION ----------------
total_sessions = len(filtered)
it_support_count = len(filtered[filtered["source"] == "it_support"])
it_helpdesk_count = len(filtered[filtered["source"] == "it_helpdesk"])
escalation_rate = filtered["escalation"].mean() * 100 if total_sessions else 0

# ---------------- HEADER ----------------
st.title("🎓 Post University Chat Inference Analytics Dashboard")
st.markdown("""
This dashboard provides **end-to-end observability** into Post University chat interactions  
covering **intent detection, sentiment, urgency, escalation, risk, and time-series trends**  
across **production + synthetic test sessions**.
""")

# ---------------- KPI ROW ----------------
st.subheader("📊 Session Overview KPIs")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sessions", total_sessions)
k2.metric("IT Support Sessions", it_support_count)
k3.metric("IT Helpdesk Sessions", it_helpdesk_count)
k4.metric("Escalation Rate", f"{escalation_rate:.1f}%")

# =========================================================
# 🎯 INTENT ANALYTICS
# =========================================================
st.subheader("🎯 Intent Analytics")

c1, c2 = st.columns(2)

c1.plotly_chart(
    px.bar(
        filtered,
        x="primary_intent",
        title="Sessions by Primary Intent",
        color="primary_intent"
    ),
    use_container_width=True
)

c2.plotly_chart(
    px.bar(
        filtered.groupby("intent_count").size().reset_index(name="sessions"),
        x="intent_count",
        y="sessions",
        title="Single vs Multi-Intent Sessions"
    ),
    use_container_width=True
)

# =========================================================
# 😊 SENTIMENT & 🚨 URGENCY
# =========================================================
st.subheader("😊 Sentiment & 🚨 Urgency")

c3, c4 = st.columns(2)

c3.plotly_chart(
    px.pie(
        filtered,
        names="sentiment",
        title="Session Sentiment Distribution"
    ),
    use_container_width=True
)

c4.plotly_chart(
    px.bar(
        filtered,
        x="urgency",
        title="Session Urgency Distribution",
        color="urgency"
    ),
    use_container_width=True
)

# =========================================================
# ⚠️ RISK & ESCALATION
# =========================================================
st.subheader("⚠️ Risk & Escalation Analysis")

c5, c6 = st.columns(2)

risk_counts = (
    filtered["risk_flags"]
    .str.split(", ")
    .explode()
    .value_counts()
    .reset_index()
)
risk_counts.columns = ["risk_flag", "count"]

c5.plotly_chart(
    px.bar(
        risk_counts,
        x="risk_flag",
        y="count",
        title="Risk Flag Distribution"
    ),
    use_container_width=True
)

c6.plotly_chart(
    px.bar(
        filtered,
        x="escalation_level",
        title="Escalation Levels (L1 vs L2)",
        color="escalation_level"
    ),
    use_container_width=True
)

# =========================================================
# 🧠 COMPLEXITY VS QUALITY
# =========================================================
st.subheader("🧠 Complexity vs Resolution Confidence")

st.plotly_chart(
    px.scatter(
        filtered,
        x="complexity_score",
        y="resolution_confidence",
        color="primary_intent",
        size="intent_count",
        hover_data=["source", "intent_flow"],
        title="Complexity vs Resolution Confidence"
    ),
    use_container_width=True
)

# =========================================================
# 📆 TIME-SERIES ANALYTICS
# =========================================================
st.subheader("📆 Time-Series Trends")

daily = (
    filtered
    .groupby("session_date")
    .agg(
        sessions=("sessionId", "count"),
        escalation_rate=("escalation", "mean"),
        avg_complexity=("complexity_score", "mean"),
        avg_resolution=("resolution_confidence", "mean")
    )
    .reset_index()
)

st.plotly_chart(
    px.line(
        daily,
        x="session_date",
        y=["sessions", "escalation_rate"],
        title="Daily Session Volume & Escalation Rate"
    ),
    use_container_width=True
)

# =========================================================
# 📊 SOURCE-WISE TRENDS
# =========================================================
st.subheader("📊 Source-Wise Trends")

source_daily = (
    filtered
    .groupby(["session_date", "source"])
    .size()
    .reset_index(name="sessions")
)

st.plotly_chart(
    px.line(
        source_daily,
        x="session_date",
        y="sessions",
        color="source",
        title="Daily Sessions by Support Source"
    ),
    use_container_width=True
)

# =========================================================
# 🚨 ESCALATION SPIKE ALERTS
# =========================================================
st.subheader("🚨 Escalation Spike Alerts")

alert_threshold = st.slider(
    "Escalation Spike Threshold (%)",
    min_value=10,
    max_value=80,
    value=30,
    step=5
)

alert_df = daily.copy()
alert_df["escalation_pct"] = alert_df["escalation_rate"] * 100

spikes = alert_df[alert_df["escalation_pct"] >= alert_threshold]

if spikes.empty:
    st.success("✅ No escalation spikes detected above threshold.")
else:
    st.error("🚨 Escalation Spikes Detected")
    st.dataframe(
        spikes[["session_date", "sessions", "escalation_pct"]],
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(
            spikes,
            x="session_date",
            y="escalation_pct",
            title="Escalation Spike Days (%)"
        ),
        use_container_width=True
    )

# =========================================================
# 📄 DETAIL TABLE
# =========================================================
st.subheader("📄 Session-Level Inference Table")
st.dataframe(filtered, use_container_width=True)
