#!/bin/bash

# 🔒 LIQUIDITY PREDICTOR - VIRTUAL ENVIRONMENT SETUP SCRIPT
# This script sets up a secure, isolated Python environment for the project

set -e  # Exit on any error

echo "🔒 Setting up Liquidity Predictor Virtual Environment..."

# Colors for output
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

# Step 1: Create virtual environment
echo_info "Creating virtual environment..."
if [ ! -d "liquidity_predictor_env" ]; then
    python3 -m venv liquidity_predictor_env
    echo_success "Virtual environment created"
else
    echo_warning "Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo_info "Activating virtual environment..."
source liquidity_predictor_env/bin/activate

# Step 3: Upgrade pip
echo_info "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo_info "Installing project dependencies..."
pip install -r requirements.txt

# Step 5: Install additional development tools
echo_info "Installing development tools..."
pip install jupyter ipykernel pytest-cov black flake8

# Step 6: Create kernel for Jupyter (optional)
echo_info "Creating Jupyter kernel..."
python -m ipykernel install --user --name=liquidity_predictor --display-name="Liquidity Predictor"

echo ""
echo "🎉 VIRTUAL ENVIRONMENT SETUP COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo_success "Virtual environment: liquidity_predictor_env"
echo_info "Python version: $(python --version)"
echo_info "Pip version: $(pip --version)"
echo ""
echo "🚀 TO ACTIVATE THE ENVIRONMENT:"
echo "source liquidity_predictor_env/bin/activate"
echo ""
echo "🎯 TO RUN THE APPLICATION:"
echo "streamlit run app.py"
echo ""
echo "🌐 TO DEPLOY TO GCP:"
echo "./gcp_deploy.sh"
echo ""
echo_success "Your Liquidity Predictor is now running in a secure, isolated environment! 🔒💧"
