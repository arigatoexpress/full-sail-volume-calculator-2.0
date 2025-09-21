# 🚀 GCP Deployment Instructions for Full Sail Finance Volume Predictor

Complete step-by-step guide for deploying the Full Sail Finance Volume Predictor to Google Cloud Platform using your $300 free credits.

## 📋 Prerequisites Checklist

- [ ] Google Cloud Account created
- [ ] $300 free credits activated
- [ ] Billing account enabled
- [ ] Google Cloud CLI installed
- [ ] Docker installed (for Cloud Run option)
- [ ] Project created in GCP Console

## 🎯 Step 1: Google Cloud Setup

### 1.1 Create Google Cloud Account

1. Go to [cloud.google.com](https://cloud.google.com)
2. Click "Get started for free"
3. Complete signup process
4. Verify $300 credit balance in billing

### 1.2 Create New Project

```bash
# Set project variables
export PROJECT_ID="full-sail-predictor-$(date +%s)"
export REGION="us-central1"

# Create project
gcloud projects create $PROJECT_ID --name="Full Sail Volume Predictor"

# Set as default project
gcloud config set project $PROJECT_ID

# Enable billing (required for deployment)
# Note: You'll need to link billing account via console first
```

### 1.3 Enable Required APIs

```bash
# Enable necessary Google Cloud APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    run.googleapis.com \
    appengine.googleapis.com \
    compute.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled
```

## 🎯 Step 2: Local Preparation

### 2.1 Prepare Application

```bash
# Navigate to project directory
cd "Full Sail Volume Calculator 2.0"

# Test application locally first
pip install -r requirements.txt
streamlit run app.py

# Verify app works at http://localhost:8501
# Stop with Ctrl+C when satisfied
```

### 2.2 Update Configuration Files

Update `cloud_run_config.yaml` with your project ID:

```bash
# Replace PROJECT_ID placeholder
sed -i "s/PROJECT_ID/$PROJECT_ID/g" cloud_run_config.yaml

# Verify the change
grep "gcr.io" cloud_run_config.yaml
```

## 🎯 Step 3: Deployment Option A - App Engine (Easiest)

### 3.1 Initialize App Engine

```bash
# Create App Engine application
gcloud app create --region=$REGION

# Check status
gcloud app describe
```

### 3.2 Deploy to App Engine

```bash
# Deploy application
gcloud app deploy app.yaml --quiet

# Get application URL
gcloud app browse
```

### 3.3 Monitor App Engine

```bash
# View logs
gcloud app logs tail -s default

# Check versions
gcloud app versions list

# Monitor costs
gcloud billing budgets list
```

### 3.4 App Engine Cost Optimization

```bash
# Set traffic to latest version only
gcloud app services set-traffic default --splits=$(gcloud app versions list --service=default --format="value(id)" --limit=1)=1.00

# Stop old versions to save costs
gcloud app versions stop $(gcloud app versions list --service=default --format="value(id)" --limit=1 --sort-by=~version.createTime | tail -n +2)
```

## 🎯 Step 4: Deployment Option B - Cloud Run (More Flexible)

### 4.1 Authenticate Docker

```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Verify authentication
docker info
```

### 4.2 Build and Push Container

```bash
# Build container image
docker build -t gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest .

# Test container locally (optional)
docker run -p 8080:8080 gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest

# Verify image in registry
gcloud container images list
```

### 4.3 Deploy to Cloud Run

```bash
# Deploy using configuration file
gcloud run services replace cloud_run_config.yaml --region=$REGION

# Alternative: Deploy directly
# gcloud run deploy full-sail-volume-predictor \
#   --image gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest \
#   --region=$REGION \
#   --platform managed \
#   --allow-unauthenticated \
#   --memory 2Gi \
#   --cpu 1 \
#   --min-instances 0 \
#   --max-instances 5
```

### 4.4 Get Service URL

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe full-sail-volume-predictor --region=$REGION --format="value(status.url)")
echo "Application URL: $SERVICE_URL"

# Test the deployment
curl -I $SERVICE_URL
```

### 4.5 Cloud Run Management

```bash
# View logs
gcloud run services logs read full-sail-volume-predictor --region=$REGION

# Update service (after code changes)
docker build -t gcr.io/$PROJECT_ID/full-sail-volume-predictor:v2 .
docker push gcr.io/$PROJECT_ID/full-sail-volume-predictor:v2
gcloud run services update full-sail-volume-predictor --image gcr.io/$PROJECT_ID/full-sail-volume-predictor:v2 --region=$REGION
```

## 🎯 Step 5: Cost Monitoring and Optimization

### 5.1 Set Up Billing Alerts

```bash
# Create budget (replace BILLING_ACCOUNT_ID with yours)
BILLING_ACCOUNT_ID=$(gcloud billing accounts list --format="value(name)" --limit=1)

# Create budget alert for $50/month
gcloud billing budgets create \
    --billing-account=$BILLING_ACCOUNT_ID \
    --display-name="Full Sail App Budget" \
    --budget-amount=50USD \
    --threshold-rule=percent=0.8,basis=CURRENT_SPEND \
    --threshold-rule=percent=1.0,basis=CURRENT_SPEND

# View current costs
gcloud billing budgets list --billing-account=$BILLING_ACCOUNT_ID
```

### 5.2 Monitor Resource Usage

```bash
# App Engine monitoring
gcloud app versions list --service=default
gcloud logging read "resource.type=gae_app" --limit=10

# Cloud Run monitoring
gcloud run services describe full-sail-volume-predictor --region=$REGION
gcloud logging read "resource.type=cloud_run_revision" --limit=10

# Check current billing
gcloud billing accounts get-iam-policy $BILLING_ACCOUNT_ID
```

### 5.3 Cost Optimization Tips

```bash
# Stop App Engine when not needed
gcloud app versions stop $(gcloud app versions list --service=default --format="value(id)")

# Delete unused Cloud Run revisions
gcloud run revisions delete OLD_REVISION_NAME --region=$REGION

# Clean up unused container images
gcloud container images list-tags gcr.io/$PROJECT_ID/full-sail-volume-predictor
gcloud container images delete gcr.io/$PROJECT_ID/full-sail-volume-predictor:OLD_TAG --quiet
```

## 🎯 Step 6: Production Setup

### 6.1 Custom Domain (Optional)

```bash
# For App Engine
gcloud app domain-mappings create your-domain.com

# For Cloud Run
gcloud run domain-mappings create --service=full-sail-volume-predictor --domain=your-domain.com --region=$REGION
```

### 6.2 SSL Certificate

```bash
# Managed SSL certificate (free)
gcloud compute ssl-certificates create full-sail-ssl \
    --domains=your-domain.com \
    --global

# Verify certificate
gcloud compute ssl-certificates describe full-sail-ssl --global
```

### 6.3 Environment Variables

```bash
# Set environment variables for production
gcloud run services update full-sail-volume-predictor \
    --set-env-vars="ENVIRONMENT=production,CACHE_DURATION=3600" \
    --region=$REGION
```

## 🎯 Step 7: Maintenance and Updates

### 7.1 Automated Deployment Script

Create `deploy.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment process..."

# Build and test locally
echo "📦 Building application..."
pip install -r requirements.txt
python test_application.py

# Deploy based on target
if [ "$1" = "appengine" ]; then
    echo "🎯 Deploying to App Engine..."
    gcloud app deploy app.yaml --quiet
    gcloud app browse
elif [ "$1" = "cloudrun" ]; then
    echo "🎯 Deploying to Cloud Run..."
    docker build -t gcr.io/$PROJECT_ID/full-sail-volume-predictor:$(date +%s) .
    docker push gcr.io/$PROJECT_ID/full-sail-volume-predictor:$(date +%s)
    gcloud run services update full-sail-volume-predictor --image gcr.io/$PROJECT_ID/full-sail-volume-predictor:$(date +%s) --region=$REGION
else
    echo "❌ Please specify deployment target: appengine or cloudrun"
    exit 1
fi

echo "✅ Deployment completed!"
```

Make executable and use:

```bash
chmod +x deploy.sh
./deploy.sh cloudrun
```

### 7.2 Backup and Recovery

```bash
# Export App Engine configuration
gcloud app describe --format="export" > app_backup.yaml

# Backup Cloud Run configuration
gcloud run services describe full-sail-volume-predictor --region=$REGION --format="export" > cloudrun_backup.yaml

# List all container images for backup
gcloud container images list --repository=gcr.io/$PROJECT_ID
```

## 🎯 Step 8: Troubleshooting

### 8.1 Common Issues

**Issue: Billing not enabled**
```bash
# Check billing status
gcloud billing accounts list
gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

**Issue: API not enabled**
```bash
# Re-enable required APIs
gcloud services enable cloudbuild.googleapis.com containerregistry.googleapis.com
```

**Issue: Insufficient quotas**
```bash
# Check quotas
gcloud compute quotas list --filter="metric:CPUs"
# Request quota increase in console if needed
```

**Issue: App won't start**
```bash
# Check logs
gcloud app logs tail -s default
gcloud run services logs read full-sail-volume-predictor --region=$REGION

# Test locally first
docker run -p 8080:8080 gcr.io/$PROJECT_ID/full-sail-volume-predictor:latest
```

### 8.2 Health Checks

```bash
# Test App Engine health
curl -I https://$PROJECT_ID.appspot.com/_stcore/health

# Test Cloud Run health  
curl -I $SERVICE_URL/_stcore/health

# Check service status
gcloud app describe
gcloud run services describe full-sail-volume-predictor --region=$REGION
```

## 🎯 Step 9: Cost Estimates

### Free Tier Limits (Monthly)

**App Engine Standard:**
- 28 instance hours/day free
- 1GB egress free
- No startup costs

**Cloud Run:**
- 2 million requests free
- 400,000 GB-seconds free
- 200,000 CPU-seconds free

**Container Registry:**
- 0.5GB storage free

### Beyond Free Tier (Estimated)

**Light Usage** (< 100 daily users):
- App Engine: $0-10/month
- Cloud Run: $0-5/month

**Medium Usage** (< 1000 daily users):
- App Engine: $10-30/month  
- Cloud Run: $5-20/month

**Heavy Usage** (> 1000 daily users):
- App Engine: $30-100/month
- Cloud Run: $20-50/month

## 🎯 Step 10: Next Steps

### 10.1 Enhance Application

- Add user authentication
- Implement data persistence with Cloud SQL
- Add BigQuery for large-scale analytics
- Set up CI/CD with Cloud Build

### 10.2 Monitor and Scale

- Set up Cloud Monitoring
- Configure alerting policies
- Implement auto-scaling policies
- Add performance profiling

### 10.3 Security

- Implement IAM policies
- Add VPC security
- Enable audit logging
- Set up security scanning

---

## 📞 Support

If you encounter issues:

1. Check the logs first
2. Verify billing and quotas
3. Test locally before deploying
4. Consult GCP documentation
5. Use Google Cloud Support (included with credits)

**Remember**: Your $300 credits should last 3-12 months depending on usage. Monitor costs regularly!

🚢 **Happy deploying with Full Sail Finance!**
