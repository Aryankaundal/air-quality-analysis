import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY or API_KEY == "your_api_key_here":
    st.error("⚠️ Please set your OPENWEATHER_API_KEY in the .env or Streamlit Secrets")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="India AQI Analysis & Forecasting",
    page_icon="🌏",
    layout="wide"
)


def get_coordinates(city_name):
    """Get coordinates using OpenWeather Geocoding API (reliable for deployment)."""
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={API_KEY}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                return float(data[0]["lat"]), float(data[0]["lon"])
            else:
                st.warning(f"No location found for {city_name}. Check spelling.")
                return None, None

        st.error(f"Geocoding API error {response.status_code}: {response.text}")
        return None, None

    except Exception as e:
        st.error(f"Error getting coordinates: {e}")
        return None, None


def fetch_aqi_data(lat, lon, city_name, api_key):
    """Fetch AQI data from OpenWeatherMap Air Pollution API"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            return parse_aqi_data(response.json(), city_name)

        elif response.status_code == 401:
            st.error("❌ Invalid API Key (401). Check OPENWEATHER_API_KEY.")
            return None

        elif response.status_code == 429:
            st.error("❌ API Rate Limit Reached (429). Try again in 1–2 minutes.")
            return None
        
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        st.error(f"❌ Error fetching AQI for {city_name}: {e}")
        return None


def parse_aqi_data(data, city_name):
    """Extract AQI fields from response JSON"""
    aqi_info = data['list'][0]

    return {
        'city': city_name,
        'aqi': aqi_info['main']['aqi'],
        'pm2_5': aqi_info['components'].get('pm2_5', 0),
        'pm10': aqi_info['components'].get('pm10', 0),
        'no2': aqi_info['components'].get('no2', 0),
        'o3': aqi_info['components'].get('o3', 0),
        'co': aqi_info['components'].get('co', 0),
        'so2': aqi_info['components'].get('so2', 0),
        'timestamp': datetime.now()
    }


def get_aqi_category(aqi):
    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }
    return categories.get(aqi, "Unknown")


def get_aqi_color(aqi):
    colors = {
        1: "#00E400",
        2: "#FFFF00",
        3: "#FF7E00",
        4: "#FF0000",
        5: "#8F3F97"
    }
    return colors.get(aqi, "#808080")


def generate_historical_data(city_data, days=30):
    np.random.seed(hash(city_data['city']) % 2**32)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    base = city_data['pm2_5']
    trend = np.linspace(base * 0.8, base * 1.2, days)
    seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, days))
    noise = np.random.normal(0, 15, days)

    pm25_history = np.maximum(trend + seasonal + noise, 10)

    return pd.DataFrame({
        'date': dates,
        'pm2_5': pm25_history,
        'city': city_data['city']
    })


def forecast_aqi(historical_df, days_ahead=7):
    historical_df['day_num'] = range(len(historical_df))
    X = historical_df[['day_num']].values
    y = historical_df['pm2_5'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    future_days = np.array([[len(historical_df) + i] for i in range(days_ahead)])
    future_poly = poly.transform(future_days)
    forecast = model.predict(future_poly)
    
    last_date = historical_df['date'].max()
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    return pd.DataFrame({
        'date': forecast_dates,
        'pm2_5_forecast': np.maximum(forecast, 10),
        'city': historical_df['city'].iloc[0]
    })

# ====================
# UI / STREAMLIT APP
# ====================

st.title("India AQI Analysis & Forecasting")
st.markdown("Live air quality data and **7–14 day PM2.5 forecasting** for Indian cities.")

with st.sidebar:
    st.header("Settings")
    forecast_days = st.slider("Forecast Days", 3, 14, 7)
    historical_days = st.slider("Historical Days", 7, 60, 30)

    st.markdown("---")
    st.markdown("### AQI Categories")
    st.markdown("🟢 **1 - Good**")
    st.markdown("🟡 **2 - Fair**")
    st.markdown("🟠 **3 - Moderate**")
    st.markdown("🔴 **4 - Poor**")
    st.markdown("🟣 **5 - Very Poor**")

st.header("City Selection")

tab1, tab2 = st.tabs(["Single City", "Multiple Cities"])

# =========================
# SINGLE CITY TAB
# =========================
with tab1:
    st.subheader("Analyze Any City in India")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        city_input = st.text_input("Enter City Name", "Delhi")
    with col2:
        analyze_button = st.button("Analyze", type="primary")

    if analyze_button and city_input:
        with st.spinner(f"Fetching data for {city_input}..."):
            lat, lon = get_coordinates(city_input)
            if lat and lon:
                aqi_data = fetch_aqi_data(lat, lon, city_input, API_KEY)

                if aqi_data:
                    st.success(f"✅ Data fetched for {city_input}")

                    # AQI Box
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        aqi_category = get_aqi_category(aqi_data['aqi'])
                        aqi_color = get_aqi_color(aqi_data['aqi'])
                        st.markdown(
                            f"<div style='background-color:{aqi_color};padding:20px;border-radius:10px;text-align:center;'>"
                            f"<h2>{aqi_data['aqi']}</h2>"
                            f"<p><b>{aqi_category}</b></p></div>",
                            unsafe_allow_html=True
                        )

                    with col2: st.metric("PM2.5", f"{aqi_data['pm2_5']:.1f}")
                    with col3: st.metric("PM10", f"{aqi_data['pm10']:.1f}")
                    with col4: st.metric("NO₂", f"{aqi_data['no2']:.1f}")

                    # Pollutant Table
                    pollutants_df = pd.DataFrame({
                        'Pollutant': ['PM2.5', 'PM10', 'NO₂', 'O₃', 'CO', 'SO₂'],
                        'Value': [
                            aqi_data['pm2_5'],
                            aqi_data['pm10'],
                            aqi_data['no2'],
                            aqi_data['o3'],
                            aqi_data['co'],
                            aqi_data['so2']
                        ]
                    })
                    st.subheader("Pollutants Breakdown")
                    st.dataframe(pollutants_df, hide_index=True, use_container_width=True)

                    # Bar chart
                    st.subheader("Current Pollutant Levels")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.barh(pollutants_df['Pollutant'], pollutants_df['Value'], color="#3498DB")
                    st.pyplot(fig)

                    # Forecast
                    st.subheader(f"{forecast_days}-Day PM2.5 Forecast")
                    hist_df = generate_historical_data(aqi_data, days=historical_days)
                    forecast_df = forecast_aqi(hist_df, forecast_days)

                    fig2, ax2 = plt.subplots(figsize=(12,6))
                    ax2.plot(hist_df['date'], hist_df['pm2_5'], label="Historical", color="#3498DB")
                    ax2.plot(forecast_df['date'], forecast_df['pm2_5_forecast'], label="Forecast", color="#E74C3C")
                    ax2.legend()
                    st.pyplot(fig2)

            else:
                st.error("❌ Could not get coordinates. Try another city.")

# =========================
# MULTIPLE CITIES TAB
# =========================
with tab2:
    st.subheader("Compare Multiple Cities")

    default_cities = "Delhi\nMumbai\nBangalore\nKolkata\nChennai"
    cities_input = st.text_area("Enter Cities (one per line)", default_cities)

    compare_button = st.button("Compare Cities")

    if compare_button:
        cities = [c.strip() for c in cities_input.split("\n") if c.strip()]
        all_data = []

        progress = st.progress(0)

        for i, city in enumerate(cities):
            lat, lon = get_coordinates(city)
            if lat and lon:
                data = fetch_aqi_data(lat, lon, city, API_KEY)
                if data:
                    all_data.append(data)
            progress.progress((i+1) / len(cities))

        if all_data:
            st.success("Data fetched successfully!")

            # Ranking Display
            cols = st.columns(len(all_data))
            for col, data in zip(cols, all_data):
                with col:
                    aqi_category = get_aqi_category(data['aqi'])
                    aqi_color = get_aqi_color(data['aqi'])
                    st.markdown(
                        f"<div style='background-color:{aqi_color};padding:10px;border-radius:8px;text-align:center;'>"
                        f"<h4>{data['city']}</h4>"
                        f"<h3>{data['aqi']}</h3>"
                        f"<p><b>{aqi_category}</b></p></div>", 
                        unsafe_allow_html=True
                    )
                    st.metric("PM2.5", f"{data['pm2_5']:.1f}")

            # Comparison Chart
            st.subheader("AQI Comparison Chart")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar([d['city'] for d in all_data], [d['aqi'] for d in all_data])
            st.pyplot(fig)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & OpenWeather API")
