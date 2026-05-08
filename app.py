# ==========================================================
# AI-POWERED STUDENT ALCOHOL ANALYTICS SYSTEM
# FINAL YEAR MAJOR PROJECT (PREMIUM UI VERSION)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
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
# CUSTOM CSS (PREMIUM UI)
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4 {
    color: white !important;
}

.stMetric {
    background: linear-gradient(135deg,#1f2937,#111827);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #374151;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(90deg,#1d4ed8,#6d28d9);
}

.css-1d391kg {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

df_mat["subject"] = "Math"
df_por["subject"] = "Portuguese"

df = pd.concat([df_mat, df_por], ignore_index=True)

# Feature Engineering
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

st.sidebar.title("🎓 Navigation Panel")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Analytics Dashboard",
        "📈 Deep Analysis",
        "🤖 AI Prediction",
        "📉 Statistical Report",
        "📚 Feature Insights",
        "📥 Download Dataset"
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

    st.title("🍺 AI-Powered Student Alcohol Analytics System")

    st.markdown("""
    ### 🎯 Final Year Major Project
    
    This intelligent analytics system analyzes the relationship between:
    
    - Student alcohol consumption
    - Academic performance
    - Study behavior
    - Failure trends
    - Lifestyle patterns
    
    using:
    
    ✅ Machine Learning  
    ✅ Statistical Analysis  
    ✅ Data Visualization  
    ✅ Predictive Analytics  
    ✅ AI-Based Insights
    """)

    st.markdown("---")

    # METRICS

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

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    # CHART 1

    st.subheader("Alcohol Consumption Distribution")

    fig1 = px.histogram(
        df,
        x="total_alcohol",
        nbins=20,
        color_discrete_sequence=["#636EFA"]
    )

    st.plotly_chart(fig1, use_container_width=True)

    # CHART 2

    st.subheader("Alcohol vs Final Grade")

    fig2 = px.scatter(
        df,
        x="total_alcohol",
        y="G3",
        color="subject",
        size="studytime",
        hover_data=["age"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # CHART 3

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Subject Wise Alcohol Usage")

        fig3 = px.box(
            df,
            x="subject",
            y="total_alcohol",
            color="subject"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col2:

        st.subheader("Failures Distribution")

        fig4 = px.pie(
            df,
            names="failures"
        )

        st.plotly_chart(fig4, use_container_width=True)

# ==========================================================
# DEEP ANALYSIS
# ==========================================================

elif page == "📈 Deep Analysis":

    st.title("📈 Deep Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    st.subheader("Correlation Heatmap")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Study Time vs Alcohol")

        fig5 = px.box(
            df,
            x="studytime",
            y="total_alcohol",
            color="studytime"
        )

        st.plotly_chart(fig5, use_container_width=True)

    with col2:

        st.subheader("Age vs Alcohol")

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

    st.markdown("""
    Enter student details to predict alcohol consumption level.
    """)

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

                # CREATE 34 FEATURES INPUT

                input_data = np.zeros((1, 34))

                # Put important values in first positions
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

                # RISK ANALYSIS

                if pred <= 4:

                    st.success("""
                    🟢 LOW RISK
                    
                    Student shows controlled alcohol consumption behavior.
                    """)

                elif pred <= 7:

                    st.warning("""
                    🟠 MEDIUM RISK
                    
                    Student may require monitoring and counseling.
                    """)

                else:

                    st.error("""
                    🔴 HIGH RISK
                    
                    High alcohol consumption detected.
                    Academic performance may be affected.
                    """)

                # AI INSIGHTS

                st.markdown("---")

                st.subheader("🧠 AI Insights")

                if failures >= 3:
                    st.warning(
                        "High failure count may correlate with poor academic habits."
                    )

                if goout >= 4:
                    st.info(
                        "Frequent social outings may increase alcohol exposure."
                    )

                if studytime <= 1:
                    st.error(
                        "Low study time detected."
                    )

            except Exception as e:

                st.error(f"Prediction Error: {e}")

        else:

            st.error("Model not loaded")

# ==========================================================
# STATISTICS
# ==========================================================

elif page == "📉 Statistical Report":

    st.title("📉 Statistical Analysis Report")

    st.subheader("Descriptive Statistics")

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

    st.subheader("Pearson Correlation Test")

    corr, p = pearsonr(
        df["total_alcohol"],
        df["G3"]
    )

    st.metric(
        "Correlation Value",
        round(corr, 3)
    )

    st.metric(
        "P-Value",
        round(p, 5)
    )

    if p < 0.05:

        st.success("""
        Significant relationship exists between
        alcohol consumption and academic performance.
        """)

    else:

        st.warning("""
        No statistically significant relationship found.
        """)

# ==========================================================
# FEATURE INSIGHTS
# ==========================================================

elif page == "📚 Feature Insights":

    st.title("📚 Machine Learning Feature Insights")

    if model_loaded and hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        numeric_df = df.select_dtypes(include=np.number)

        features = numeric_df.drop(
            columns=[
                "Dalc",
                "Walc",
                "total_alcohol"
            ],
            errors="ignore"
        ).columns

        min_len = min(
            len(importances),
            len(features)
        )

        imp_df = pd.DataFrame({
            "Feature": features[:min_len],
            "Importance": importances[:min_len]
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

elif page == "📥 Download Dataset":

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
<center>

### 🎓 Developed By Pratham Taak  
B.Tech CSE Final Year Major Project

</center>
""", unsafe_allow_html=True)
