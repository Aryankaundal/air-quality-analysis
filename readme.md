# AQI Analysis & Forecasting System

A real-time Air Quality Index (AQI) monitoring and forecasting web application for Indian cities, built with Streamlit and powered by OpenWeatherMap API.

## Features

### Real-Time AQI Monitoring
- Live air quality data for any city in India
- Comprehensive pollutant tracking (PM2.5, PM10, NO₂, O₃, CO, SO₂)
- Color-coded AQI categories (Good, Moderate, Unhealthy, etc.)
- Detailed pollutant breakdowns with health implications

### Advanced Forecasting
- 7-day AQI predictions using polynomial regression
- Configurable forecast periods (3-14 days)
- Adjustable historical data windows (7-60 days)
- Confidence intervals for prediction accuracy

### Multi-City Comparison
- Side-by-side comparison of multiple cities
- Interactive visualizations with Plotly
- City rankings by AQI levels
- Comparative pollutant analysis

## Tech Stack

- **Frontend**: Streamlit
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Forecasting**: Scikit-learn (Polynomial Regression)
- **API**: OpenWeatherMap Air Pollution API

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenWeatherMap API key (free tier available)

### Step 1: Clone the Repository
\`\`\`bash
git clone https://github.com/yourusername/aqi-analysis.git
cd aqi-analysis
\`\`\`

### Step 2: Create Virtual Environment (Recommended)
\`\`\`bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
\`\`\`

### Step 3: Install Dependencies
\`\`\`bash
pip install -r scripts/requirements.txt
\`\`\`

### Step 4: Set Up API Key

1. Get your free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create a `.env` file in the root directory:
\`\`\`bash
cp .env.example .env
\`\`\`
3. Open `.env` and add your API key:
\`\`\`
OPENWEATHER_API_KEY=your_actual_api_key_here
\`\`\`

### Step 5: Run the Application
\`\`\`bash
streamlit run app.py
\`\`\`

The app will open automatically in your browser at `http://localhost:8501`

## Usage

### Single City Analysis
1. Select "Single City Analysis" tab
2. Enter any Indian city name (e.g., Delhi, Mumbai, Bangalore)
3. View current AQI, pollutant levels, and health recommendations
4. Scroll down for 7-day forecast predictions

### Multi-City Comparison
1. Select "Multi-City Comparison" tab
2. Enter multiple cities separated by commas
3. Compare AQI levels, pollutants, and rankings
4. Analyze trends across different regions

### Customization Options
- **Forecast Days**: Adjust prediction period (3-14 days)
- **Historical Data**: Change data window for better predictions (7-60 days)
- **Refresh Data**: Click refresh button for latest readings

## Project Structure

\`\`\`
aqi-analysis/
├── app.py                      # Main Streamlit application
├── .env                        # Environment variables (not in git)
├── .env.example               # Example environment file
├── scripts/
│   └── requirements.txt       # Python dependencies
└── README.md                  # This file
\`\`\`

## API Information

This application uses the [OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution) which provides:
- Current air pollution data
- Historical data (past 5 days)
- Forecast data (next 5 days)
- Free tier: 60 calls/minute, 1,000,000 calls/month

## Data & Methodology

### AQI Calculation
The app uses the US EPA Air Quality Index standard:
- **Good (0-50)**: Air quality is satisfactory
- **Moderate (51-100)**: Acceptable for most people
- **Unhealthy for Sensitive Groups (101-150)**: May affect sensitive individuals
- **Unhealthy (151-200)**: Everyone may begin to experience effects
- **Very Unhealthy (201-300)**: Health alert
- **Hazardous (301+)**: Emergency conditions

### Forecasting Model
- Polynomial regression (degree 2) on historical PM2.5 data
- Configurable training window (7-60 days)
- Confidence intervals calculated using prediction variance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenWeatherMap for providing the Air Pollution API
- Streamlit for the excellent web framework
- The open-source community for the amazing Python libraries

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---
