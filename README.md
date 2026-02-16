# 🚢 Full Sail Finance Volume Predictor

A comprehensive Python application for predicting liquidity pool volumes on Full Sail Finance (Sui-based DEX) with advanced charting and data visualization capabilities.

> See `AGENTS.md` for agentic navigation.

## 🌟 Features

- **📊 Real-time Data Fetching**: DefiLlama and CoinGecko APIs for historical/real-time Sui DEX volumes
- **🧠 AI-Powered Predictions**: Prophet and ARIMA time-series forecasting models
- **📈 Interactive Visualizations**: Plotly and Altair charts with educational features
- **🎛️ User-friendly Dashboard**: Streamlit web interface with pool selector and prediction controls
- **☁️ Cloud-Ready**: Optimized for Google Cloud Platform deployment
- **💰 Cost-Optimized**: Designed to use GCP free tier and credits efficiently

## 🏗️ Architecture

```
├── Data Layer
│   ├── DefiLlama API (DEX volumes)
│   ├── CoinGecko API (Sui metrics)
│   └── CSV/Pandas caching
├── Processing Layer
│   ├── Data cleaning & aggregation
│   ├── Feature engineering
│   └── Technical indicators
├── ML Layer
│   ├── Prophet forecasting
│   ├── ARIMA modeling
│   └── Ensemble predictions
├── Visualization Layer
│   ├── Plotly interactive charts
│   ├── Altair declarative viz
│   └── Educational annotations
└── Deployment Layer
    ├── Streamlit dashboard
    ├── Docker containerization
    └── GCP Cloud Run/App Engine
```

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone https://github.com/arigatoexpress/full-sail-volume-calculator-2.0.git
   cd full-sail-volume-calculator-2.0
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Dashboard**
   - Open browser to `http://localhost:8501`
   - Use the sidebar to configure data and predictions
   - Explore different pools and time ranges

### Testing

```bash
# Run all tests
python test_application.py

# Run specific component tests
python -m pytest test_application.py::TestDataFetcher -v

# Run integration test
python test_application.py
```

## ☁️ GCP Deployment

### Prerequisites

1. **Google Cloud Account**
   - Sign up at [cloud.google.com](https://cloud.google.com)
   - $300 free credits for new users
   - Enable billing (required for deployment)

2. **Install Google Cloud CLI**
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Windows
   # Download installer from cloud.google.com/sdk
   
   # Linux
   curl https://sdk.cloud.google.com | bash
   ```

3. **Authenticate and Setup**
   ```bash
   gcloud auth login
   export GCP_PROJECT_ID="your-project-id"
   gcloud config set project "$GCP_PROJECT_ID"
   gcloud auth configure-docker
   ```

### Deployment Options

#### Option 1: Google App Engine (Recommended for beginners)

1. **Deploy Application**
   ```bash
   gcloud app deploy app.yaml
   ```

2. **Access Application**
   ```bash
   gcloud app browse
   ```

3. **Monitor and Scale**
   ```bash
   gcloud app logs tail -s default
   gcloud app versions list
   ```

#### Option 2: Cloud Run (Recommended for flexibility)

1. **Build and Push Container**
   ```bash
   # Set your project ID
   export PROJECT_ID=your-project-id
   
   # Build container
   docker build -t gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest .
   
   # Push to Container Registry
   docker push gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest
   ```

2. **Deploy to Cloud Run**
   ```bash
   # Update PROJECT_ID in cloud_run_config.yaml
   sed -i "s/PROJECT_ID/$PROJECT_ID/g" cloud_run_config.yaml
   
   # Deploy
   gcloud run services replace cloud_run_config.yaml --region=us-central1
   ```

3. **Get Service URL**
   ```bash
   gcloud run services describe full-sail-volume-predictor --region=us-central1 --format="value(status.url)"
   ```

#### Option 3: Docker Locally

1. **Build and Run**
   ```bash
   docker build -t full-sail-predictor .
   docker run -p 8080:8080 full-sail-predictor
   ```

2. **Access Application**
   - Open browser to `http://localhost:8080`

### Cost Optimization

The application is designed to minimize GCP costs:

- **App Engine**: 
  - `min_instances: 0` (scales to zero when not in use)
  - `max_instances: 3` (limits maximum cost)
  - 1 CPU, 2GB RAM (minimal resources)

- **Cloud Run**:
  - Auto-scaling from 0 to 5 instances
  - CPU throttling enabled
  - 1 vCPU, 2GB RAM allocation

- **Free Tier Benefits**:
  - App Engine: 28 instance hours/day free
  - Cloud Run: 2 million requests/month free
  - Container Registry: 0.5GB storage free

### Estimated Costs (Beyond Free Tier)

- **Light Usage** (< 100 daily users): $0-5/month
- **Medium Usage** (< 1000 daily users): $5-20/month
- **Heavy Usage** (> 1000 daily users): $20-50/month

## 📊 Usage Guide

### Dashboard Overview

1. **📥 Data Management**
   - Refresh data from APIs
   - View cached data status
   - Monitor data quality

2. **🏊 Pool Selection**
   - Choose specific pools or view all
   - Compare different trading pairs
   - Analyze pool-specific metrics

3. **🔮 Prediction Settings**
   - Select forecast horizon (1-14 days)
   - Choose prediction model (Ensemble, Prophet, ARIMA)
   - Configure confidence intervals

4. **📈 Visualization Options**
   - Interactive Plotly charts
   - Declarative Altair visualizations
   - Technical indicators overlay
   - Event highlighting

### Educational Features

- **💡 Chart Explanations**: Tooltips explain volume spikes, moving averages, and correlation patterns
- **🧠 Model Insights**: Understand how Prophet and ARIMA work
- **📚 Trading Education**: Learn about liquidity, slippage, and market patterns
- **⚠️ Risk Awareness**: Important disclaimers about prediction limitations

### Data Sources

- **DefiLlama API**: Historical DEX volumes, TVL data
- **CoinGecko API**: Sui token metrics, market data
- **Real-time Updates**: Cached for 1 hour, configurable refresh

## 🔧 Configuration

### Environment Variables

```bash
# Optional: API rate limiting
DEFILLAMA_RATE_LIMIT=0.1
COINGECKO_RATE_LIMIT=0.1

# Optional: Cache settings
DATA_CACHE_HOURS=1
DEFAULT_HISTORY_DAYS=60

# Optional: Model settings
DEFAULT_FORECAST_DAYS=7
CONFIDENCE_LEVEL=0.95
```

### Advanced Configuration

Edit `utils.py` for default settings:

```python
config = {
    "api_settings": {
        "request_timeout": 30,
        "rate_limit_delay": 0.1
    },
    "model_settings": {
        "ensemble_weights": {"prophet": 0.6, "arima": 0.4}
    }
}
```

## 🛠️ Development

### Project Structure

```
Full Sail Volume Calculator 2.0/
├── app.py                 # Main Streamlit application
├── data_fetcher.py       # API data fetching
├── data_processor.py     # Data cleaning & feature engineering
├── prediction_models.py  # Prophet & ARIMA models
├── visualization.py      # Plotly & Altair charts
├── utils.py              # Utilities & error handling
├── test_application.py   # Test suite
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── app.yaml            # App Engine config
├── cloud_run_config.yaml # Cloud Run config
└── README.md           # This file
```

### Key Components

1. **DataFetcher**: Handles API calls with caching and retry logic
2. **DataProcessor**: Cleans data, adds technical indicators, detects events
3. **VolumePredictor**: Implements Prophet, ARIMA, and ensemble forecasting
4. **VolumeVisualizer**: Creates interactive educational charts
5. **FullSailDashboard**: Main Streamlit application with UI logic

### Adding New Features

1. **New Data Sources**: Extend `DataFetcher` class
2. **New Models**: Add to `VolumePredictor` class
3. **New Charts**: Extend `VolumeVisualizer` class
4. **New UI Elements**: Modify `FullSailDashboard` class

## 📈 Model Performance

- **Prophet**: Best for seasonal patterns, handles holidays
- **ARIMA**: Good for stationary data, classical approach
- **Ensemble**: Combines both models for robustness

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination
- **MAE**: Mean Absolute Error

## ⚠️ Important Notes

### Limitations

- Predictions are based on historical data
- Cannot predict black swan events
- Accuracy decreases for longer forecasts
- DeFi markets are highly volatile

### Risk Disclaimer

This tool is for educational purposes only. Predictions should not be used as financial advice. Always do your own research (DYOR) and never invest more than you can afford to lose.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

### Common Issues

1. **Prophet Installation Fails**
   ```bash
   # Try conda instead of pip
   conda install -c conda-forge prophet
   ```

2. **GCP Deployment Fails**
   ```bash
   # Check quotas and billing
   gcloud compute quotas list
   gcloud billing accounts list
   ```

3. **Memory Issues**
   - Reduce data cache size
   - Use smaller prediction horizons
   - Consider upgrading GCP resources

### Getting Help

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/arigatoexpress/full-sail-volume-calculator-2.0/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/arigatoexpress/full-sail-volume-calculator-2.0/discussions)

---

🚢 **Built with ❤️ for the Full Sail Finance community**

*Empowering DeFi traders with AI-powered volume predictions and educational visualizations.*
