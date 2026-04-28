# ==========================================================

# ALCOHOL CONSUMPTION ANALYSIS SYSTEM (FINAL YEAR PROJECT)

# Advanced EDA + Statistics + ML + Dashboard + Export

# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr

# ----------------------------------------------------------

# CONFIG

# ----------------------------------------------------------

st.set_page_config(page_title="Alcohol Analytics Pro", layout="wide")

# ----------------------------------------------------------

# LOAD DATA (SAFE - NO FUNCTION ERRORS)

# ----------------------------------------------------------

df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

df_mat["subject"] = "math"
df_por["subject"] = "portuguese"

df = pd.concat([df_mat, df_por], ignore_index=True)

# Feature Engineering

df["total_alcohol"] = df["Dalc"] + df["Walc"]
df["grade_avg"] = (df["G1"] + df["G2"] + df["G3"]) / 3

# ----------------------------------------------------------

# LOAD MODEL (SAFE TRY BLOCK)

# ----------------------------------------------------------

model_loaded = False
try:
    model = pickle.load(open("model.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# ----------------------------------------------------------

# SIDEBAR NAVIGATION

# ----------------------------------------------------------

st.sidebar.title("⚙️ Navigation Panel")

page = st.sidebar.radio(
"Go to",
[
"🏠 Overview",
"📊 Dashboard",
"📈 Deep Analysis",
"📉 Statistics",
"📚 Feature Insights",
"🔮 Prediction",
"📥 Download"
]
)

# ----------------------------------------------------------

# FILTER

# ----------------------------------------------------------

st.sidebar.subheader("Filter Data")
subject_filter = st.sidebar.selectbox("Subject", ["All", "math", "portuguese"])

if subject_filter != "All":
df = df[df["subject"] == subject_filter]

# ----------------------------------------------------------

# OVERVIEW PAGE

# ----------------------------------------------------------

if page == "🏠 Overview":
st.title("🍺 Alcohol Consumption Analytics System")

```
st.write("""
This is an advanced data science project that includes:
- Data Analysis
- Statistical Testing
- Machine Learning
- Visualization Dashboard
""")

st.subheader("Dataset Preview")
st.dataframe(df.head(20))

st.subheader("Dataset Shape")
st.write(df.shape)
```

# ----------------------------------------------------------

# DASHBOARD PAGE

# ----------------------------------------------------------

elif page == "📊 Dashboard":
st.title("📊 Dashboard")

```
col1, col2, col3 = st.columns(3)

col1.metric("Avg Alcohol", round(df["total_alcohol"].mean(), 2))
col2.metric("Avg Grade", round(df["G3"].mean(), 2))
col3.metric("Max Alcohol", df["total_alcohol"].max())

st.subheader("Alcohol Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df["total_alcohol"], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Subject Comparison")
fig2, ax2 = plt.subplots()
sns.boxplot(x="subject", y="total_alcohol", data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Alcohol vs Grades")
fig3, ax3 = plt.subplots()
sns.scatterplot(x="total_alcohol", y="G3", data=df, ax=ax3)
st.pyplot(fig3)
```

# ----------------------------------------------------------

# DEEP ANALYSIS

# ----------------------------------------------------------

elif page == "📈 Deep Analysis":
st.title("📈 Deep Analysis")

```
st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.subheader("Study Time vs Alcohol")
fig5, ax5 = plt.subplots()
sns.boxplot(x="studytime", y="total_alcohol", data=df, ax=ax5)
st.pyplot(fig5)

st.subheader("Failures vs Alcohol")
fig6, ax6 = plt.subplots()
sns.boxplot(x="failures", y="total_alcohol", data=df, ax=ax6)
st.pyplot(fig6)

st.subheader("Age vs Alcohol")
fig7, ax7 = plt.subplots()
sns.scatterplot(x="age", y="total_alcohol", data=df, ax=ax7)
st.pyplot(fig7)
```

# ----------------------------------------------------------

# STATISTICS

# ----------------------------------------------------------

elif page == "📉 Statistics":
st.title("📉 Statistical Analysis")

```
st.subheader("Descriptive Statistics")
st.write(df[["total_alcohol", "G3"]].describe())

st.subheader("Correlation Test")
corr, p = pearsonr(df["total_alcohol"], df["G3"])

st.write("Correlation:", round(corr, 3))
st.write("P-value:", p)

if p < 0.05:
    st.success("Significant relationship exists")
else:
    st.warning("No significant relationship")
```

# ----------------------------------------------------------

# FEATURE INSIGHTS

# ----------------------------------------------------------

elif page == "📚 Feature Insights":
st.title("📚 Feature Insights")

```
st.subheader("Top Influencing Features")

if model_loaded and hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    features = df.drop(["Dalc", "Walc", "total_alcohol"], axis=1).columns

    imp_df = pd.Series(importances, index=features)
    imp_df = imp_df.sort_values(ascending=False).head(10)

    fig8, ax8 = plt.subplots()
    imp_df.plot(kind="barh", ax=ax8)
    ax8.invert_yaxis()
    st.pyplot(fig8)
else:
    st.info("Feature importance not available")
```

# ----------------------------------------------------------

# PREDICTION

# ----------------------------------------------------------

elif page == "🔮 Prediction":
st.title("🔮 Alcohol Prediction")

```
age = st.slider("Age", 15, 22)
studytime = st.slider("Study Time", 1, 4)
failures = st.slider("Failures", 0, 3)

if st.button("Predict"):
    if model_loaded:
        try:
            input_data = np.array([[age, studytime, failures]])
            pred = model.predict(input_data)
            st.success(f"Predicted Alcohol Level: {pred[0]:.2f}")
        except:
            st.warning("Model trained on more features")
    else:
        st.error("Model not loaded")
```

# ----------------------------------------------------------

# DOWNLOAD

# ----------------------------------------------------------

elif page == "📥 Download":
st.title("📥 Download Data")

```
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Dataset",
    csv,
    "alcohol_data.csv",
    "text/csv"
)
```
