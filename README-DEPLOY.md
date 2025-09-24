# 🚢 Full Sail Volume Predictor - GCP Deployment

Simple, working app that predicts volume for all Full Sail pools and generates vote weights for the next epoch.

## 🚀 Quick Deploy to Google Cloud Run

### Prerequisites
1. **Google Cloud CLI** installed: https://cloud.google.com/sdk/docs/install
2. **Google Cloud Project** with billing enabled
3. **Docker** (optional, for local testing)

### One-Command Deployment

```bash
# Make sure you're in the project directory
cd "/Users/aribs/Documents/Cursor/Full Sail Volume Calculator 2.0"

# Run the deployment script
./deploy.sh
```

### Manual Deployment Steps

If you prefer to deploy manually:

1. **Set your project ID:**
```bash
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

2. **Enable required APIs:**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

3. **Build and deploy:**
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/full-sail-predictor
gcloud run deploy full-sail-predictor \
    --image gcr.io/$PROJECT_ID/full-sail-predictor \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10
```

## 🎯 What the App Does

- **Predicts volume** for all 8 Full Sail pools for the next 7-day epoch
- **Shows confidence scores** and prediction ranges
- **Generates vote weights** proportional to predicted volume
- **Provides copy-paste vote slate** ready for Full Sail voting
- **Downloads results** as CSV or JSON

## 📊 Full Sail Pools Supported

1. SAIL/USDC
2. SUI/USDC  
3. IKA/SUI
4. ALKIMI/SUI
5. USDZ/USDC
6. USDT/USDC
7. wBTC/USDC
8. ETH/USDC

## 🔧 Configuration

The deployment uses:
- **Container**: Python 3.11 slim
- **Framework**: Streamlit 1.49.1
- **Dependencies**: pandas, numpy, plotly, altair
- **Resources**: 1GB RAM, 1 CPU, max 10 instances
- **Region**: us-central1 (adjustable)

## 💰 Cost Estimation

Cloud Run pricing (approximate):
- **Free tier**: 2M requests/month, 400k GB-seconds compute
- **Beyond free**: ~$0.40 per 1M requests, $0.00002400 per GB-second
- **Expected cost**: $0-5/month for typical usage

## 🔒 Security

- Runs as non-root user
- No authentication required (public app)
- No persistent storage (stateless)
- All data generated in-memory

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements-deploy.txt

# Run locally
streamlit run simple_app.py --server.port 8502
```

## 📝 Notes

- The app generates sample data for demonstration
- Predictions are based on trend analysis of recent 30-day patterns
- Confidence scores are volatility-adjusted
- Vote weights sum to 100% for easy submission

## 🆘 Troubleshooting

**Build fails**: Check that all required APIs are enabled
**Deploy fails**: Verify project ID and permissions
**App won't start**: Check Cloud Run logs in GCP Console
**High costs**: Adjust max-instances or add authentication

## 📞 Support

Check Cloud Run logs:
```bash
gcloud logs read --service=full-sail-predictor --limit=50
```

View service details:
```bash
gcloud run services describe full-sail-predictor --region=us-central1
```
