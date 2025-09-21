#!/bin/bash

# 🚀 LIQUIDITY PREDICTOR - GCP DEPLOYMENT SCRIPT
# Automated deployment to Google Cloud Platform

set -e  # Exit on any error

echo "🚀 Starting Liquidity Predictor deployment to GCP..."

# Configuration
PROJECT_ID="quant-ai-trader-credits"
SERVICE_NAME="liquidity-predictor"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/liquidity-predictor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo_error "gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo_info "Checking gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
    echo_warning "Please authenticate with gcloud:"
    gcloud auth login
fi

echo_info "Setting up GCP project: $PROJECT_ID"

# Create project if it doesn't exist
if ! gcloud projects describe $PROJECT_ID &> /dev/null; then
    echo_info "Creating new GCP project: $PROJECT_ID"
    gcloud projects create $PROJECT_ID
    echo_success "Project created successfully"
else
    echo_success "Project already exists"
fi

# Set the project
gcloud config set project $PROJECT_ID
echo_success "Project set to $PROJECT_ID"

echo_info "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
echo_success "APIs enabled"

echo_info "Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME
echo_success "Docker image built and pushed"

echo_info "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 16Gi \
    --cpu 8 \
    --max-instances 50 \
    --min-instances 2 \
    --timeout 3600 \
    --concurrency 100 \
    --cpu-boost \
    --port 8080 \
    --set-env-vars="STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0"

echo_success "Deployment completed!"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo_success "Service URL: $SERVICE_URL"
echo_info "Region: $REGION"
echo_info "Max Instances: 10"
echo_info "Memory: 4Gi"
echo_info "CPU: 2 cores"
echo ""
echo_info "Your Liquidity Predictor is now live! 🚀"
echo ""

# Optional: Set up custom domain
read -p "🌐 Would you like to set up a custom domain? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your domain (e.g., app.liquiditypredictor.com): " DOMAIN
    echo_info "Setting up domain mapping for $DOMAIN..."
    gcloud run domain-mappings create \
        --service $SERVICE_NAME \
        --domain $DOMAIN \
        --region $REGION
    echo_success "Domain mapping created. Please update your DNS records:"
    echo "  Add a CNAME record: $DOMAIN -> ghs.googlehosted.com"
fi

echo ""
echo "📊 Next Steps:"
echo "1. Test your application at: $SERVICE_URL"
echo "2. Monitor performance in GCP Console"
echo "3. Set up monitoring and alerting"
echo "4. Configure custom domain (if not done above)"
echo ""
echo_success "Deployment complete! Your Liquidity Predictor is ready to use! 🎯"
