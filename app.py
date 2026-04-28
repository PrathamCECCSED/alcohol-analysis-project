import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alcohol Analyzer", layout="wide")

# Load model

model = pickle.load(open("model.pkl", "rb"))

# Load dataset

df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

df_mat['subject'] = 'math'
df_por['subject'] = 'portuguese'

df = pd.concat([df_mat, df_por], ignore_index=True)

# Feature engineering

df['total_alcohol'] = df['Dalc'] + df['Walc']

# Sidebar

st.sidebar.title("User Input")

age = st.sidebar.slider("Age", 15, 22, 18)
studytime = st.sidebar.slider("Study Time", 1, 4, 2)
failures = st.sidebar.slider("Failures", 0, 3, 0)

# Main Title

st.title("🍺 Alcohol Consumption Analysis & Prediction")

# Tabs

tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Prediction"])

# ---------------- DASHBOARD ----------------

with tab1:
st.subheader("Alcohol Distribution")

```
fig1, ax1 = plt.subplots()
sns.histplot(df['total_alcohol'], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Subject Comparison")

fig2, ax2 = plt.subplots()
sns.boxplot(x='subject', y='total_alcohol', data=df, ax=ax2)
st.pyplot(fig2)
```

# ---------------- PREDICTION ----------------

with tab2:
st.subheader("Predict Alcohol Consumption")

```
if st.button("Predict"):
    try:
        # Temporary simple input (demo purpose)
        input_data = np.array([[age, studytime, failures]])
        prediction = model.predict(input_data)

        st.success(f"Predicted Level: {prediction[0]:.2f}")
    except:
        st.warning("Model uses many features — demo prediction shown.")
```
