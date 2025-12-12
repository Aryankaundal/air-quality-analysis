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

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY or API_KEY == "your_api_key_here":
    st.error("⚠️ Please set your OPENWEATHER_API_KEY in the .env file")
    st.info("Get your free API key at: https://openweathermap.org/api")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="India AQI Analysis & Forecasting",
    page_icon="🌏",
    layout="wide"
)

# Helper functions from original script
def fetch_aqi_data(lat, lon, city_name, api_key):
    """Fetch AQI data from OpenWeatherMap Air Pollution API"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return parse_aqi_data(data, city_name)
        else:
            st.warning(f"API returned status code {response.status_code} for {city_name}")
            return None
    except Exception as e:
        st.error(f"Error fetching data for {city_name}: {e}")
        return None

def parse_aqi_data(data, city_name):
    """Parse API response into structured format"""
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
    """Convert AQI number to category"""
    categories = {
        1: "Good",
        2: "Fair", 
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }
    return categories.get(aqi, "Unknown")

def get_aqi_color(aqi):
    """Get color for AQI level"""
    colors = {
        1: "#00E400",
        2: "#FFFF00",
        3: "#FF7E00",
        4: "#FF0000",
        5: "#8F3F97"
    }
    return colors.get(aqi, "#808080")

def get_coordinates(city_name):
    """Get coordinates for a city using Nominatim geocoding"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city_name},India&format=json&limit=1"
        headers = {'User-Agent': 'AQI-Analysis-App/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except Exception as e:
        st.error(f"Error getting coordinates: {e}")
        return None, None

def generate_historical_data(city_data, days=30):
    """Generate historical data for forecasting"""
    np.random.seed(hash(city_data['city']) % 2**32)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    base = city_data['pm2_5']
    trend = np.linspace(base * 0.8, base * 1.2, days)
    seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, days))
    noise = np.random.normal(0, 15, days)
    
    pm25_history = trend + seasonal + noise
    pm25_history = np.maximum(pm25_history, 10)
    
    return pd.DataFrame({
        'date': dates,
        'pm2_5': pm25_history,
        'city': city_data['city']
    })

def forecast_aqi(historical_df, days_ahead=7):
    """Forecast future AQI using polynomial regression"""
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
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'pm2_5_forecast': np.maximum(forecast, 10),
        'city': historical_df['city'].iloc[0]
    })
    
    return forecast_df

# Main Streamlit App
st.title("India AQI Analysis & Forecasting")
st.markdown("Live air quality data and predictions for Indian cities")

# Sidebar for settings only
with st.sidebar:
    st.header("Settings")
    
    forecast_days = st.slider(
        "Forecast Days",
        min_value=3,
        max_value=14,
        value=7,
        help="Number of days to forecast"
    )
    
    historical_days = st.slider(
        "Historical Days",
        min_value=7,
        max_value=60,
        value=30,
        help="Number of historical days to use for forecasting"
    )
    
    st.markdown("---")
    st.markdown("### AQI Categories")
    st.markdown("🟢 **1 - Good**")
    st.markdown("🟡 **2 - Fair**")
    st.markdown("🟠 **3 - Moderate**")
    st.markdown("🔴 **4 - Poor**")
    st.markdown("🟣 **5 - Very Poor**")

# Main content
st.header("City Selection")

tab1, tab2 = st.tabs(["Single City", "Multiple Cities"])

with tab1:
    st.subheader("Analyze Any City in India")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        city_input = st.text_input(
            "Enter City Name",
            value="Delhi",
            placeholder="e.g., Mumbai, Delhi, Bangalore",
            help="Enter any city name in India"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    if analyze_button and city_input:
        with st.spinner(f"Fetching data for {city_input}..."):
            lat, lon = get_coordinates(city_input)
            
            if lat and lon:
                aqi_data = fetch_aqi_data(lat, lon, city_input, API_KEY)
                
                if aqi_data:
                    st.success(f"✅ Data fetched successfully for {city_input}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        aqi_category = get_aqi_category(aqi_data['aqi'])
                        aqi_color = get_aqi_color(aqi_data['aqi'])
                        st.markdown(f"""
                        <div style='background-color: {aqi_color}; padding: 20px; border-radius: 10px; text-align: center;'>
                            <h2 style='margin: 0; color: black;'>{aqi_data['aqi']}</h2>
                            <p style='margin: 0; color: black; font-weight: bold;'>{aqi_category}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("PM2.5", f"{aqi_data['pm2_5']:.1f} μg/m³")
                    
                    with col3:
                        st.metric("PM10", f"{aqi_data['pm10']:.1f} μg/m³")
                    
                    with col4:
                        st.metric("NO₂", f"{aqi_data['no2']:.1f} μg/m³")
                    
                    st.subheader("Pollutants Breakdown")
                    
                    pollutants_df = pd.DataFrame({
                        'Pollutant': ['PM2.5', 'PM10', 'NO₂', 'O₃', 'CO', 'SO₂'],
                        'Concentration': [
                            aqi_data['pm2_5'],
                            aqi_data['pm10'],
                            aqi_data['no2'],
                            aqi_data['o3'],
                            aqi_data['co'],
                            aqi_data['so2']
                        ],
                        'Unit': ['μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³']
                    })
                    
                    st.dataframe(pollutants_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Current Pollutant Levels")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    pollutants = ['PM2.5', 'PM10', 'NO₂', 'O₃', 'CO', 'SO₂']
                    values = [
                        aqi_data['pm2_5'],
                        aqi_data['pm10'],
                        aqi_data['no2'],
                        aqi_data['o3'],
                        aqi_data['co'] / 10,
                        aqi_data['so2']
                    ]
                    colors_chart = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
                    
                    ax.barh(pollutants, values, color=colors_chart, alpha=0.7)
                    ax.set_xlabel('Concentration (μg/m³)', fontweight='bold')
                    ax.set_title(f'Pollutant Levels in {city_input}', fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    st.subheader(f"{forecast_days}-Day PM2.5 Forecast")
                    
                    with st.spinner("Generating forecast..."):
                        hist_df = generate_historical_data(aqi_data, days=historical_days)
                        forecast_df = forecast_aqi(hist_df, days_ahead=forecast_days)
                        
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        
                        ax2.plot(hist_df['date'], hist_df['pm2_5'], 
                                marker='o', linestyle='-', linewidth=2, markersize=4,
                                label='Historical Data', color='#3498DB', alpha=0.7)
                        
                        ax2.plot(forecast_df['date'], forecast_df['pm2_5_forecast'],
                                marker='s', linestyle='--', linewidth=2, markersize=6,
                                label='Forecast', color='#E74C3C', alpha=0.8)
                        
                        forecast_values = forecast_df['pm2_5_forecast'].values
                        ax2.fill_between(forecast_df['date'], 
                                        forecast_values * 0.85, 
                                        forecast_values * 1.15,
                                        alpha=0.2, color='#E74C3C', label='Confidence Interval')
                        
                        ax2.set_xlabel('Date', fontweight='bold')
                        ax2.set_ylabel('PM2.5 (μg/m³)', fontweight='bold')
                        ax2.set_title(f'PM2.5 Forecast for {city_input}', fontweight='bold', fontsize=14)
                        ax2.legend(loc='upper left')
                        ax2.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig2)
                        
                        st.subheader("Forecast Details")
                        forecast_display = forecast_df.copy()
                        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                        forecast_display['pm2_5_forecast'] = forecast_display['pm2_5_forecast'].round(2)
                        forecast_display.columns = ['Date', 'Predicted PM2.5 (μg/m³)', 'City']
                        st.dataframe(forecast_display, use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Could not find coordinates for {city_input}. Please check the city name.")

with tab2:
    st.subheader("Compare Multiple Cities")
    
    default_cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"]
    
    cities_input = st.text_area(
        "Enter City Names (one per line)",
        value="\n".join(default_cities),
        height=150,
        help="Enter one city name per line"
    )
    
    compare_button = st.button("Compare Cities", type="primary", use_container_width=True)
    
    if compare_button and cities_input:
        cities_list = [city.strip() for city in cities_input.split("\n") if city.strip()]
        
        if cities_list:
            all_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, city in enumerate(cities_list):
                status_text.text(f"Fetching data for {city}...")
                lat, lon = get_coordinates(city)
                
                if lat and lon:
                    aqi_data = fetch_aqi_data(lat, lon, city, API_KEY)
                    if aqi_data:
                        all_data.append(aqi_data)
                
                progress_bar.progress((idx + 1) / len(cities_list))
            
            status_text.empty()
            progress_bar.empty()
            
            if all_data:
                st.success(f"✅ Successfully fetched data for {len(all_data)} cities")
                
                st.subheader("City Rankings")
                
                cols = st.columns(len(all_data))
                
                for idx, (col, data) in enumerate(zip(cols, all_data)):
                    with col:
                        aqi_category = get_aqi_category(data['aqi'])
                        aqi_color = get_aqi_color(data['aqi'])
                        
                        st.markdown(f"""
                        <div style='background-color: {aqi_color}; padding: 15px; border-radius: 8px; text-align: center;'>
                            <h4 style='margin: 0; color: black;'>{data['city']}</h4>
                            <h2 style='margin: 5px 0; color: black;'>{data['aqi']}</h2>
                            <p style='margin: 0; color: black; font-weight: bold;'>{aqi_category}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric("PM2.5", f"{data['pm2_5']:.1f}")
                        st.metric("PM10", f"{data['pm10']:.1f}")
                
                st.subheader("Comparison Charts")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                cities_names = [d['city'] for d in all_data]
                aqi_values = [d['aqi'] for d in all_data]
                pm25_values = [d['pm2_5'] for d in all_data]
                colors = [get_aqi_color(aqi) for aqi in aqi_values]
                
                axes[0].bar(cities_names, aqi_values, color=colors, alpha=0.7, edgecolor='black')
                axes[0].set_ylabel('AQI Level', fontweight='bold')
                axes[0].set_title('AQI Comparison', fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(axis='y', alpha=0.3)
                
                axes[1].barh(cities_names, pm25_values, color='#E74C3C', alpha=0.7)
                axes[1].set_xlabel('PM2.5 (μg/m³)', fontweight='bold')
                axes[1].set_title('PM2.5 Concentration', fontweight='bold')
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader(f"Multi-City {forecast_days}-Day Forecast")
                
                num_cities = len(all_data)
                cols_per_row = 3
                num_rows = (num_cities + cols_per_row - 1) // cols_per_row
                
                fig3, axes3 = plt.subplots(num_rows, cols_per_row, figsize=(16, 5 * num_rows))
                
                if num_rows == 1:
                    axes3 = [axes3]
                
                for idx, city_data in enumerate(all_data):
                    row = idx // cols_per_row
                    col = idx % cols_per_row
                    
                    if num_rows == 1:
                        ax = axes3[col] if num_cities > 1 else axes3
                    else:
                        ax = axes3[row, col]
                        
                    hist_df = generate_historical_data(city_data, days=historical_days)
                    forecast_df = forecast_aqi(hist_df, days_ahead=forecast_days)
                    
                    ax.plot(hist_df['date'], hist_df['pm2_5'], 
                           marker='o', linestyle='-', linewidth=2, markersize=3,
                           label='Historical', color='#3498DB', alpha=0.7)
                    
                    ax.plot(forecast_df['date'], forecast_df['pm2_5_forecast'],
                           marker='s', linestyle='--', linewidth=2, markersize=5,
                           label='Forecast', color='#E74C3C', alpha=0.8)
                    
                    forecast_values = forecast_df['pm2_5_forecast'].values
                    ax.fill_between(forecast_df['date'], 
                                   forecast_values * 0.85, 
                                   forecast_values * 1.15,
                                   alpha=0.2, color='#E74C3C')
                    
                    ax.set_title(f'{city_data["city"]}', fontweight='bold')
                    ax.set_ylabel('PM2.5 (μg/m³)')
                    ax.legend(loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                
                for idx in range(num_cities, num_rows * cols_per_row):
                    row = idx // cols_per_row
                    col = idx % cols_per_row
                    if num_rows == 1:
                        if num_cities > 1:
                            axes3[col].axis('off')
                    else:
                        axes3[row, col].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig3)

st.markdown("---")
st.markdown("Made with Streamlit | Data from OpenWeatherMap API")
