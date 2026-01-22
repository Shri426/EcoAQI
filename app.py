import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="EcoAQI – Green AI Air Quality Monitoring",
    page_icon="🌱",
    layout="wide"
)

# ==================================================
# UI STYLING
# ==================================================
st.markdown("""
<style>
.stApp { background-color: #0f172a; }
h1,h2,h3,h4,h5,p,span,label,div {
    color: #FFFFFF !important;
    font-weight: 700;
}
.stButton > button {
    background-color: #16A34A;
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    padding: 12px;
    border-radius: 12px;
    width: 100%;
    border: none;
}
.stButton > button:hover { background-color: #15803D; }
[data-testid="metric-container"] {
    background-color: #020617;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}
div[data-testid="stDownloadButton"] > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 700;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================
model = joblib.load("data/processed/model/aqi_model.pkl")

# ==================================================
# AQI LOGIC
# ==================================================
def aqi_details(aqi):
    if aqi <= 50:
        return "Good", "🟢", "Air quality is satisfactory.", "No action required."
    elif aqi <= 100:
        return "Moderate", "🟡", "Minor discomfort to sensitive groups.", "Limit prolonged outdoor activity."
    elif aqi <= 200:
        return "Unhealthy", "🟠", "Breathing discomfort possible.", "Reduce outdoor exposure."
    elif aqi <= 300:
        return "Very Unhealthy", "🔴", "High risk of respiratory effects.", "Avoid outdoor activity."
    else:
        return "Hazardous", "🟣", "Serious health impacts.", "Emergency response required."

def status_card(title, value, unit, limit):
    safe = value <= limit
    color = "#22C55E" if safe else "#EF4444"
    glow = "0 0 18px rgba(34,197,94,0.6)" if safe else "0 0 18px rgba(239,68,68,0.6)"
    label = "SAFE" if safe else "UNSAFE"

    st.markdown(
        f"""
        <div style="background:#020617;padding:22px;border-radius:16px;
        text-align:center;border:1px solid {color};box-shadow:{glow};">
            <h4>{title}</h4>
            <h2 style="color:{color};">{value} {unit}</h2>
            <span style="padding:6px 16px;border-radius:20px;
            background:{color};color:white;font-size:14px;">
                {label}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# HISTORY FILE
# ==================================================
CSV_FILE = "aqi_history.csv"
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "Time","Temperature","Humidity","Wind Speed",
        "AQI","Category","Health Impact","Action"
    ]).to_csv(CSV_FILE, index=False)

# ==================================================
# HEADER
# ==================================================
st.title("🌱 EcoAQI – Green AI Air Quality Monitoring")
st.caption("Explainable AI • Sustainable Computing • Public Health Insight")
st.divider()

# ==================================================
# INPUT SECTION
# ==================================================
st.subheader("🌍 Environmental Conditions")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider("🌡 Temperature (°C)", 0.0, 50.0, 25.0)
with col2:
    humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 60.0)
with col3:
    wind = st.slider("🌬 Wind Speed (m/s)", 0.0, 10.0, 2.0)

st.subheader("Environmental Safety Status")
c1, c2, c3 = st.columns(3)
with c1: status_card("🌡 Temperature", round(temp,2), "°C", 35)
with c2: status_card("💧 Humidity", round(humidity,2), "%", 80)
with c3: status_card("🌬 Wind Speed", round(wind,2), "m/s", 6)

# ==================================================
# PREDICT BUTTON
# ==================================================
st.divider()
colA, colB, colC = st.columns([1,2,1])
with colB:
    predict = st.button("🔍 Predict AQI")

# ==================================================
# PREDICTION
# ==================================================
if predict:
    X = np.array([[temp, humidity, wind]])
    aqi = float(model.predict(X)[0])
    category, emoji, impact, action = aqi_details(aqi)

    st.divider()
    st.subheader("📊 AQI Prediction Result")

    colX, colY = st.columns(2)
    with colX:
        st.markdown(
            f"""
            <div style="background:#020617;padding:24px;border-radius:16px;text-align:center;">
                <h2>AQI</h2>
                <h1 style="color:#F87171;">{aqi:.2f}</h1>
                <h3>{emoji} {category}</h3>
            </div>
            """, unsafe_allow_html=True
        )
    with colY:
        st.info(f"**Health Impact:** {impact}")
        st.warning(f"**Recommended Action:** {action}")

    # Save history
    pd.DataFrame([{
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature": temp,
        "Humidity": humidity,
        "Wind Speed": wind,
        "AQI": round(aqi,2),
        "Category": category,
        "Health Impact": impact,
        "Action": action
    }]).to_csv(CSV_FILE, mode="a", header=False, index=False)

# ==================================================
# TREND FIRST
# ==================================================
df = pd.read_csv(CSV_FILE)

if not df.empty:
    st.divider()
    st.subheader("📈 AQI Trend Over Time")
    df["Time"] = pd.to_datetime(df["Time"])
    st.line_chart(df.set_index("Time")["AQI"])

    st.divider()
    st.subheader("📂 Historical AQI Records")

    selected_date = st.date_input(
        "🔍 Filter by Date",
        df["Time"].dt.date.iloc[-1]
    )
    filtered_df = df[df["Time"].dt.date == selected_date]

    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        "⬇ Download AQI History",
        df.to_csv(index=False),
        "aqi_history.csv"
    )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("EcoAQI • Green AI • Explainable Air Quality Monitoring")

