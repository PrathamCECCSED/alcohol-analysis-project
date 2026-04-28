# ================================

# ADVANCED ALCOHOL ANALYSIS APP

# ================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

st.set_page_config(page_title="Alcohol Analytics Pro", layout="wide")

# ================================

# LOAD DATA & MODEL

# ================================

@st.cache_data
def load_data():
df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

```
df_mat['subject'] = 'math'
df_por['subject'] = 'portuguese'

df = pd.concat([df_mat, df_por], ignore_index=True)

df['total_alcohol'] = df['Dalc'] + df['Walc']
df['grade_avg'] = (df['G1'] + df['G2'] + df['G3']) / 3

return df
```

df = load_data()

# Load model safely

try:
model = pickle.load(open("model.pkl", "rb"))
model_loaded = True
except:
model_loaded = False

# ================================

# SIDEBAR CONTROLS

# ================================

st.sidebar.title("⚙️ Controls")

page = st.sidebar.radio(
"Navigate",
["🏠 Overview", "📊 Dashboard", "📈 Deep Analysis", "📉 Statistics", "🔮 Prediction"]
)

selected_subject = st.sidebar.selectbox(
"Filter Subject",
["All", "math", "portuguese"]
)

if selected_subject != "All":
df = df[df['subject'] == selected_subject]

# ================================

# OVERVIEW PAGE

# ================================

if page == "🏠 Overview":
st.title("🍺 Alcohol Consumption Analytics")

```
st.write("""
This project analyzes student alcohol consumption behavior using:
- Data Analytics
- Statistical Analysis
- Machine Learning
- Interactive Dashboard
""")

st.subheader("Dataset Preview")
st.dataframe(df.head())
```

# ================================

# DASHBOARD

# ================================

elif page == "📊 Dashboard":
st.title("📊 Dashboard")

```
col1, col2, col3 = st.columns(3)
col1.metric("Avg Alcohol", round(df['total_alcohol'].mean(),2))
col2.metric("Avg Grade", round(df['G3'].mean(),2))
col3.metric("Max Alcohol", df['total_alcohol'].max())

st.subheader("Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['total_alcohol'], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Subject Comparison")
fig2, ax2 = plt.subplots()
sns.boxplot(x='subject', y='total_alcohol', data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Alcohol vs Grades")
fig3, ax3 = plt.subplots()
sns.scatterplot(x='total_alcohol', y='G3', data=df, ax=ax3)
st.pyplot(fig3)
```

# ================================

# DEEP ANALYSIS

# ================================

elif page == "📈 Deep Analysis":
st.title("📈 Deep Analysis")

```
st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

st.subheader("Study Time vs Alcohol")
fig5, ax5 = plt.subplots()
sns.boxplot(x='studytime', y='total_alcohol', data=df, ax=ax5)
st.pyplot(fig5)

st.subheader("Failures vs Alcohol")
fig6, ax6 = plt.subplots()
sns.boxplot(x='failures', y='total_alcohol', data=df, ax=ax6)
st.pyplot(fig6)

st.subheader("Age vs Alcohol")
fig7, ax7 = plt.subplots()
sns.scatterplot(x='age', y='total_alcohol', data=df, ax=ax7)
st.pyplot(fig7)
```

# ================================

# STATISTICS PAGE

# ================================

elif page == "📉 Statistics":
st.title("📉 Statistical Analysis")

```
st.subheader("Basic Statistics")
st.write(df[['total_alcohol','G3']].describe())

st.subheader("Correlation Test")
corr, p = pearsonr(df['total_alcohol'], df['G3'])
st.write(f"Correlation: {corr:.3f}")
st.write(f"P-value: {p:.5f}")

if p < 0.05:
    st.success("Statistically Significant Relationship")
else:
    st.warning("No Significant Relationship")
```

# ================================

# PREDICTION PAGE

# ================================

elif page == "🔮 Prediction":
st.title("🔮 Prediction")

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
            st.warning("Model expects more features (demo mode).")
    else:
        st.error("Model not loaded")

st.subheader("Feature Importance")
if model_loaded and hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    features = df.drop(['Dalc','Walc','total_alcohol'], axis=1).columns

    imp_df = pd.Series(importances, index=features).sort_values(ascending=False)[:10]

    fig8, ax8 = plt.subplots()
    imp_df.plot(kind='barh', ax=ax8)
    ax8.invert_yaxis()
    st.pyplot(fig8)
else:
    st.info("Feature importance unavailable")
```

# ================================

# DOWNLOAD SECTION

# ================================

st.sidebar.subheader("Download Data")

csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
label="Download Dataset",
data=csv,
file_name='alcohol_data.csv',
mime='text/csv',
)
