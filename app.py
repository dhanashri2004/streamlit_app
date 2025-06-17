# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Load model and encoders
model = joblib.load("incident_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset
df = pd.read_csv("incident_event_log.csv")
df.dropna(inplace=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Incident Analyzer",
        options=["Home", "Data Preview", "Visualizations", "Predict Incident State", "Report"],
        icons=["house", "table", "bar-chart", "magic", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
    )

# Page 1: Home
if selected == "Home":
    st.title("🔐 Incident Event Log Analyzer")
    st.markdown("This app uses a machine learning model to predict the state of an incident from a service log dataset.")
    st.markdown("Navigate using the sidebar to explore features.")

# Page 2: Data Preview
elif selected == "Data Preview":
    st.header("📊 Dataset Preview")
    st.dataframe(df.head(50))
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Page 3: Visualizations
elif selected == "Visualizations":
    st.header("📈 Graphical Insights")

    col_to_plot = st.selectbox("Select column to plot", df.select_dtypes(include='object').columns.tolist())
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col_to_plot, order=df[col_to_plot].value_counts().index, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Page 4: Prediction
elif selected == "Predict Incident State":
    st.header("🔮 Predict Incident State")

    input_data = {}
    for col in df.columns:
        if col != "incident_state":
            if df[col].dtype == 'object':
                options = df[col].unique().tolist()
                input_data[col] = st.selectbox(f"{col}", options)
            else:
                input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), step=1.0)

    # Encode input
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        decoded = label_encoders["incident_state"].inverse_transform([prediction])[0]
        st.success(f"✅ Predicted Incident State: **{decoded}**")

# Page 5: Report
elif selected == "Report":
    st.header("📝 Dataset Summary Report")

    st.markdown("**1. Columns Summary**")
    st.write(df.describe(include='all'))

    st.markdown("**2. Target Balance (incident_state)**")
    fig2, ax2 = plt.subplots()
    df["incident_state"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, startangle=90, shadow=True)
    plt.ylabel("")
    st.pyplot(fig2)

    st.markdown("**3. Null Values**")
    st.write(df.isnull().sum())
