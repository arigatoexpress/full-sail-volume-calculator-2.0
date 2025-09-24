#!/bin/bash

# Full Sail Volume Predictor - GCP Deployment Script
set -e

# Configuration
PROJECT_ID="full-sail-predictor"
SERVICE_NAME="full-sail-volume-predictor"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚢 Deploying Full Sail Volume Predictor to GCP Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
echo "📋 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push the container
echo "🏗️ Building and pushing container..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --port 8080

# Get the service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo ""
echo "🌐 Your Full Sail Volume Predictor is live at:"
echo "   ${SERVICE_URL}"
echo ""
echo "📊 Features:"
echo "   - Predicts volume for all Full Sail pools"
echo "   - Generates vote weights for next epoch"
echo "   - Copy-paste ready vote slates"
echo ""
echo "🎯 Ready for Full Sail voting!"
