# ==========================================================
# AI POWERED STUDENT ALCOHOL ANALYTICS SYSTEM
# FINAL YEAR MAJOR PROJECT
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
from scipy.stats import pearsonr

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Alcohol Analytics System",
    page_icon="🍺",
    layout="wide"
)

# ==========================================================
# PREMIUM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #0B1120;
    color: white;
}

h1, h2, h3, h4, h5 {
    color: white !important;
}

.stMetric {
    background: linear-gradient(135deg,#111827,#1E293B);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    text-align: center;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(90deg,#1D4ED8,#6D28D9);
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOADING ANIMATION
# ==========================================================

with st.spinner("Loading AI Analytics System..."):
    time.sleep(2)

# ==========================================================
# LOAD DATA
# ==========================================================

df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

df_mat["subject"] = "Math"
df_por["subject"] = "Portuguese"

df = pd.concat([df_mat, df_por], ignore_index=True)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df["total_alcohol"] = df["Dalc"] + df["Walc"]
df["grade_avg"] = (df["G1"] + df["G2"] + df["G3"]) / 3

# ==========================================================
# LOAD MODEL
# ==========================================================

model_loaded = False

try:
    model = pickle.load(open("model.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1046/1046784.png",
    width=100
)

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📈 Deep Analysis",
        "🤖 AI Prediction",
        "📉 Statistics",
        "📚 Feature Insights",
        "📥 Download"
    ]
)

st.sidebar.markdown("---")

subject_filter = st.sidebar.selectbox(
    "Filter Subject",
    ["All", "Math", "Portuguese"]
)

if subject_filter != "All":
    df = df[df["subject"] == subject_filter]

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.markdown("""
    <div style='padding:30px;border-radius:20px;
    background: linear-gradient(135deg,#111827,#1E3A8A);
    text-align:center;'>

    <h1 style='color:white;font-size:50px;'>
    🍺 AI-Powered Student Alcohol Analytics System
    </h1>

    <h3 style='color:#CBD5E1;'>
    Final Year Major Project Using Machine Learning
    </h3>

    <p style='color:#94A3B8;font-size:18px;'>
    Analyzing student behavior and academic performance using AI.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        len(df)
    )

    col2.metric(
        "Average Alcohol",
        round(df["total_alcohol"].mean(), 2)
    )

    col3.metric(
        "Average Grade",
        round(df["G3"].mean(), 2)
    )

    high_risk = len(df[df["total_alcohol"] >= 8])

    col4.metric(
        "High Risk Students",
        high_risk
    )

    st.markdown("---")

    st.subheader("🎯 Research Objective")

    st.write("""
    This project analyzes the relationship between alcohol
    consumption and academic performance using machine learning,
    statistical analysis, and intelligent dashboards.
    """)

    st.markdown("---")

    st.subheader("⚙️ System Workflow")

    st.markdown("""
    1. Data Collection  
    2. Data Preprocessing  
    3. Feature Engineering  
    4. Statistical Analysis  
    5. Machine Learning Training  
    6. Prediction Generation  
    7. Visualization Dashboard  
    8. AI Insight Generation  
    """)

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Missing Values: {df.isnull().sum().sum()}")

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Dashboard":

    st.title("📊 Analytics Dashboard")

    fig1 = px.histogram(
        df,
        x="total_alcohol",
        nbins=20,
        color_discrete_sequence=["#636EFA"]
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="total_alcohol",
        y="G3",
        color="subject",
        size="studytime",
        hover_data=["age"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        fig3 = px.box(
            df,
            x="subject",
            y="total_alcohol",
            color="subject"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col2:

        fig4 = px.pie(
            df,
            names="failures"
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    st.subheader("📌 Key Insights")

    st.success(
        "Students with lower study time show higher alcohol trends."
    )

    st.warning(
        "High social activity may increase alcohol consumption."
    )

    st.info(
        "Academic failures correlate with behavioral patterns."
    )

# ==========================================================
# DEEP ANALYSIS
# ==========================================================

elif page == "📈 Deep Analysis":

    st.title("📈 Deep Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        fig5 = px.box(
            df,
            x="studytime",
            y="total_alcohol",
            color="studytime"
        )

        st.plotly_chart(fig5, use_container_width=True)

    with col2:

        fig6 = px.scatter(
            df,
            x="age",
            y="total_alcohol",
            color="failures"
        )

        st.plotly_chart(fig6, use_container_width=True)

# ==========================================================
# AI PREDICTION
# ==========================================================

elif page == "🤖 AI Prediction":

    st.title("🤖 AI Alcohol Consumption Prediction")

    st.write(
        "Enter student details to predict alcohol consumption level."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 15, 22, 18)

        studytime = st.slider(
            "Study Time",
            1,
            4,
            2
        )

        failures = st.slider(
            "Failures",
            0,
            4,
            0
        )

    with col2:

        absences = st.slider(
            "Absences",
            0,
            50,
            5
        )

        goout = st.slider(
            "Going Out Frequency",
            1,
            5,
            3
        )

        freetime = st.slider(
            "Free Time",
            1,
            5,
            3
        )

    if st.button("Predict Alcohol Consumption"):

        if model_loaded:

            try:

                input_data = np.zeros((1, 34))

                input_data[0][0] = age
                input_data[0][1] = studytime
                input_data[0][2] = failures
                input_data[0][3] = absences
                input_data[0][4] = goout
                input_data[0][5] = freetime

                prediction = model.predict(input_data)

                pred = round(prediction[0], 2)

                st.success(
                    f"Predicted Alcohol Level: {pred}"
                )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    title={'text': "Alcohol Risk Level"},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'steps': [
                            {'range': [0, 4], 'color': "green"},
                            {'range': [4, 7], 'color': "orange"},
                            {'range': [7, 10], 'color': "red"}
                        ]
                    }
                ))

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                if pred <= 4:

                    st.success("""
                    🟢 LOW RISK

                    Student shows controlled alcohol behavior.
                    """)

                elif pred <= 7:

                    st.warning("""
                    🟠 MEDIUM RISK

                    Student may require monitoring.
                    """)

                else:

                    st.error("""
                    🔴 HIGH RISK

                    High alcohol consumption detected.
                    """)

                st.markdown("---")

                st.subheader("🧠 AI Insights")

                if failures >= 3:
                    st.warning(
                        "High failure count detected."
                    )

                if goout >= 4:
                    st.info(
                        "Frequent social outings detected."
                    )

                if studytime <= 1:
                    st.error(
                        "Low study time detected."
                    )

                st.markdown("---")

                st.subheader("🧠 AI Recommendations")

                if pred > 7:

                    st.error(
                        "Reduce social outing frequency."
                    )

                    st.warning(
                        "Increase study time."
                    )

                    st.info(
                        "Academic counseling recommended."
                    )

                elif pred > 4:

                    st.warning(
                        "Maintain balanced lifestyle."
                    )

                    st.info(
                        "Monitor academic performance."
                    )

                else:

                    st.success(
                        "Healthy academic behavior detected."
                    )

            except Exception as e:

                st.error(f"Prediction Error: {e}")

        else:

            st.error("Model not loaded.")

# ==========================================================
# STATISTICS
# ==========================================================

elif page == "📉 Statistics":

    st.title("📉 Statistical Analysis")

    st.dataframe(
        df[
            [
                "total_alcohol",
                "G3",
                "studytime",
                "failures"
            ]
        ].describe(),
        use_container_width=True
    )

    st.markdown("---")

    corr, p = pearsonr(
        df["total_alcohol"],
        df["G3"]
    )

    st.metric(
        "Correlation",
        round(corr, 3)
    )

    st.metric(
        "P-Value",
        round(p, 5)
    )

    if p < 0.05:

        st.success(
            "Significant relationship exists."
        )

    else:

        st.warning(
            "No statistically significant relationship found."
        )

# ==========================================================
# FEATURE INSIGHTS
# ==========================================================

elif page == "📚 Feature Insights":

    st.title("📚 Machine Learning Feature Insights")

    if model_loaded and hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        features = [
            f"Feature {i+1}"
            for i in range(len(importances))
        ]

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })

        imp_df = imp_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            imp_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance not available."
        )

# ==========================================================
# DOWNLOAD
# ==========================================================

elif page == "📥 Download":

    st.title("📥 Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download CSV File",
        data=csv,
        file_name="alcohol_analytics_dataset.csv",
        mime="text/csv"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;'>

<h3>🎓 Developed By Pratham Taak</h3>

<h4>B.Tech Computer Science Engineering</h4>

<h4>Final Year Major Project 2026</h4>

</div>
""", unsafe_allow_html=True)
