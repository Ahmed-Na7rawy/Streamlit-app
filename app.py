import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from download_model import download_model

st.set_page_config(page_title="Sleep Health & Lifestyle", page_icon="🌙", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("sleep_health_dataset.csv")

@st.cache_resource
def load_model():
    if not os.path.exists("random_forest_model.pkl"):
        with st.spinner("Downloading model... This may take a moment."):
            download_model()
    return joblib.load("random_forest_model.pkl")

df = load_data()
model = load_model()

ENCODE = {
    "gender":                  {"Female": 0, "Male": 1, "Other": 2},
    "occupation":              {"Doctor": 0, "Driver": 1, "Freelancer": 2, "Homemaker": 3,
                                "Lawyer": 4, "Manager": 5, "Nurse": 6,
                                "Software Engineer": 7, "Student": 8, "Teacher": 9},
    "country":                 {"Australia": 0, "Brazil": 1, "Germany": 2, "India": 3,
                                "Japan": 4, "Netherlands": 5, "South Korea": 6,
                                "Spain": 7, "UK": 8, "USA": 9},
    "chronotype":              {"Evening": 0, "Morning": 1, "Neutral": 2},
    "mental_health_condition": {"Anxiety": 0, "Both": 1, "Depression": 2, "Healthy": 3},
    "season":                  {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3},
    "day_type":                {"Weekday": 0, "Weekend": 1},
}

RISK_COLORS = {"Healthy": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22", "Severe": "#e74c3c"}
RISK_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]

st.sidebar.title("Sleep Health")
page = st.sidebar.radio("Navigate", ["Overview", "Data Explorer", "Risk Predictor"])

if page == "Overview":
    st.title(" Sleep Health & Lifestyle Dashboard")
    st.markdown("Analyzing **100,000** individuals across sleep quality, stress, and health metrics.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",      f"{len(df):,}")
    c2.metric("Avg Sleep Duration", f"{df['sleep_duration_hrs'].mean():.1f} hrs")
    c3.metric("Avg Sleep Quality",  f"{df['sleep_quality_score'].mean():.1f} / 10")
    c4.metric("Avg Stress Score",   f"{df['stress_score'].mean():.1f} / 10")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sleep Disorder Risk Distribution")
        counts = df["sleep_disorder_risk"].value_counts().reindex(RISK_ORDER).reset_index()
        counts.columns = ["Risk Level", "Count"]
        fig = px.bar(counts, x="Risk Level", y="Count", color="Risk Level",
                     color_discrete_map=RISK_COLORS, category_orders={"Risk Level": RISK_ORDER})
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Sleep Quality by Gender")
        fig = px.box(df, x="gender", y="sleep_quality_score", color="gender",
                     color_discrete_sequence=["#3498db", "#e74c3c", "#9b59b6"])
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix — Key Health Metrics")
    num_cols = ["age", "sleep_duration_hrs", "sleep_quality_score",
                "stress_score", "heart_rate_resting_bpm", "cognitive_performance_score"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                               colorscale="RdBu", zmid=0, text=corr.values,
                               texttemplate="%{text}", textfont={"size": 11}))
    fig.update_layout(margin=dict(t=20, b=20), height=420)
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Sleep Duration vs. Cognitive Performance")
        sample = df.sample(2000, random_state=42)
        fig = px.scatter(sample, x="sleep_duration_hrs", y="cognitive_performance_score",
                         opacity=0.3, trendline="ols", color_discrete_sequence=["#3498db"],
                         labels={"sleep_duration_hrs": "Sleep Duration (hrs)",
                                 "cognitive_performance_score": "Cognitive Performance"})
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Disorder Risk by Occupation")
        occ_risk = df.groupby(["occupation", "sleep_disorder_risk"]).size().reset_index(name="count")
        totals = occ_risk.groupby("occupation")["count"].transform("sum")
        occ_risk["pct"] = (occ_risk["count"] / totals * 100).round(1)
        fig = px.bar(occ_risk, x="occupation", y="pct", color="sleep_disorder_risk",
                     color_discrete_map=RISK_COLORS, category_orders={"sleep_disorder_risk": RISK_ORDER},
                     labels={"pct": "% of Occupation", "occupation": ""})
        fig.update_layout(xaxis_tickangle=-45, legend_title="Risk", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Explorer":
    st.title(" Data Explorer")

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        risk_filter   = col1.multiselect("Sleep Disorder Risk", RISK_ORDER, default=RISK_ORDER)
        gender_filter = col2.multiselect("Gender", sorted(df["gender"].unique()), default=list(df["gender"].unique()))
        occ_filter    = col3.multiselect("Occupation", sorted(df["occupation"].unique()), default=list(df["occupation"].unique()))

    filtered = df[df["sleep_disorder_risk"].isin(risk_filter) &
                  df["gender"].isin(gender_filter) &
                  df["occupation"].isin(occ_filter)]

    st.caption(f"Showing **{len(filtered):,}** records")
    st.dataframe(filtered.head(200), use_container_width=True)
    st.divider()
    st.subheader("Distribution of Any Numeric Column")

    num_col = st.selectbox("Choose a column to plot", options=[
        "sleep_duration_hrs", "sleep_quality_score", "stress_score",
        "cognitive_performance_score", "heart_rate_resting_bpm", "bmi",
        "caffeine_mg_before_bed", "sleep_latency_mins", "wake_episodes_per_night", "nap_duration_mins"])

    fig = px.histogram(filtered, x=num_col, color="sleep_disorder_risk", barmode="overlay", opacity=0.6,
                       color_discrete_map=RISK_COLORS, category_orders={"sleep_disorder_risk": RISK_ORDER},
                       labels={num_col: num_col.replace("_", " ").title()})
    fig.update_layout(legend_title="Risk Level", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Predictor":
    st.title("Sleep Disorder Risk Predictor")
    st.markdown("Fill in your details and the Random Forest model will estimate your sleep disorder risk.")

    with st.form("predict_form"):
        st.subheader("Personal Info")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 70, 30)
        gender = c2.selectbox("Gender", sorted(ENCODE["gender"].keys()))
        occupation = c3.selectbox("Occupation", sorted(ENCODE["occupation"].keys()))

        c4, c5, c6 = st.columns(3)
        bmi = c4.number_input("BMI", 16.0, 45.0, 25.0, step=0.1)
        country = c5.selectbox("Country", sorted(ENCODE["country"].keys()))
        chronotype = c6.selectbox("Chronotype", sorted(ENCODE["chronotype"].keys()))

        st.subheader("Sleep Metrics")
        c7, c8, c9 = st.columns(3)
        sleep_duration = c7.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.0, 0.5)
        sleep_quality  = c8.slider("Sleep Quality Score (1–10)", 1.0, 10.0, 7.0, 0.1)
        sleep_latency  = c9.number_input("Sleep Latency (mins)", 0, 120, 15)

        c10, c11, c12 = st.columns(3)
        rem_pct  = c10.slider("REM % of Sleep", 0.0, 40.0, 20.0, 0.5)
        deep_pct = c11.slider("Deep Sleep %", 0.0, 40.0, 20.0, 0.5)
        wake_eps = c12.number_input("Wake Episodes / Night", 0, 15, 2)

        c13, c14 = st.columns(2)
        nap_dur      = c13.number_input("Nap Duration (mins)", 0, 180, 0)
        weekend_diff = c14.number_input("Weekend Sleep Diff (hrs)", -4.0, 6.0, 1.0, 0.5)

        st.subheader("Lifestyle")
        c15, c16, c17 = st.columns(3)
        caffeine    = c15.number_input("Caffeine before bed (mg)", 0, 400, 0)
        alcohol     = c16.number_input("Alcohol units before bed", 0.0, 10.0, 0.0, 0.5)
        screen_time = c17.number_input("Screen time before bed (mins)", 0, 300, 60)

        c18, c19, c20 = st.columns(3)
        exercise = c18.selectbox("Exercised today?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        steps    = c19.number_input("Steps that day", 0, 30000, 5000, step=500)
        work_hrs = c20.number_input("Work hours today", 0.0, 18.0, 8.0, 0.5)

        c21, c22, c23 = st.columns(3)
        stress     = c21.slider("Stress Score (1–10)", 1.0, 10.0, 5.0, 0.1)
        heart_rate = c22.number_input("Resting Heart Rate (bpm)", 40, 120, 70)
        cog_score  = c23.slider("Cognitive Performance Score", 0.0, 10.0, 7.0, 0.1)

        c24, c25, c26 = st.columns(3)
        mental_health = c24.selectbox("Mental Health Condition", sorted(ENCODE["mental_health_condition"].keys()))
        season        = c25.selectbox("Current Season", sorted(ENCODE["season"].keys()))
        day_type      = c26.selectbox("Day Type", sorted(ENCODE["day_type"].keys()))

        c27, c28, c29 = st.columns(3)
        sleep_aid  = c27.selectbox("Sleep Aid Used?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        shift_work = c28.selectbox("Shift Worker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        room_temp  = c29.number_input("Room Temperature (°C)", 15.0, 35.0, 20.0, 0.5)

        felt_rested = st.selectbox("Felt rested this morning?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        submitted = st.form_submit_button("Predict My Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "age": age, "gender": ENCODE["gender"][gender],
            "occupation": ENCODE["occupation"][occupation], "bmi": bmi,
            "country": ENCODE["country"][country], "sleep_duration_hrs": sleep_duration,
            "sleep_quality_score": sleep_quality, "rem_percentage": rem_pct,
            "deep_sleep_percentage": deep_pct, "sleep_latency_mins": sleep_latency,
            "wake_episodes_per_night": wake_eps, "caffeine_mg_before_bed": caffeine,
            "alcohol_units_before_bed": alcohol, "screen_time_before_bed_mins": screen_time,
            "exercise_day": exercise, "steps_that_day": steps, "nap_duration_mins": nap_dur,
            "stress_score": stress, "work_hours_that_day": work_hrs,
            "chronotype": ENCODE["chronotype"][chronotype],
            "mental_health_condition": ENCODE["mental_health_condition"][mental_health],
            "heart_rate_resting_bpm": heart_rate, "sleep_aid_used": sleep_aid,
            "shift_work": shift_work, "room_temperature_celsius": room_temp,
            "weekend_sleep_diff_hrs": weekend_diff, "season": ENCODE["season"][season],
            "day_type": ENCODE["day_type"][day_type], "cognitive_performance_score": cog_score,
            "felt_rested": felt_rested,
        }])

        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]
        color      = RISK_COLORS[prediction]

        st.markdown(f"""
        <div style="background:{color}22; border-left:6px solid {color};
                    padding:20px; border-radius:10px; margin-top:20px;">
            <h2 style="color:{color}; margin:0;">Predicted Risk: {prediction}</h2>
        </div>""", unsafe_allow_html=True)

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({"Risk Level": model.classes_,
                                 "Probability": (proba * 100).round(1)}).sort_values("Probability", ascending=True)
        fig = px.bar(prob_df, x="Probability", y="Risk Level", orientation="h",
                     color="Risk Level", color_discrete_map=RISK_COLORS, text="Probability")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_range=[0, 110],
                          xaxis_title="Probability (%)", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)