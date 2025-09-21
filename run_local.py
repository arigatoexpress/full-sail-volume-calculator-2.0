#!/usr/bin/env python3
"""
Local development runner for Full Sail Finance Volume Predictor.
Handles environment setup and graceful startup.
"""

import sys
import os
import subprocess
import importlib.util

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'altair',
        'requests', 'statsmodels'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_tests():
    """Run basic tests to ensure application is working."""
    print("🧪 Running basic tests...")
    
    try:
        # Import main modules to check for syntax errors
        from data_fetcher import DataFetcher
        from data_processor import DataProcessor
        from visualization import VolumeVisualizer
        
        # Test data fetching
        fetcher = DataFetcher()
        test_data = fetcher._create_sample_historical_data(7)
        
        if len(test_data) > 0:
            print("✅ Data fetching test passed")
        else:
            print("❌ Data fetching test failed")
            return False
        
        # Test data processing
        processor = DataProcessor()
        processed_data = processor.process_pool_data(test_data)
        
        if len(processed_data) > 0:
            print("✅ Data processing test passed")
        else:
            print("❌ Data processing test failed")
            return False
        
        print("✅ All basic tests passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def start_streamlit():
    """Start the Streamlit application."""
    print("🚀 Starting Full Sail Finance Volume Predictor...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    
    try:
        # Run streamlit with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print("💡 Try running manually: streamlit run app.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main entry point for local development."""
    print("🚢 Full Sail Finance Volume Predictor - Local Development")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if in correct directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Run tests
    if "--skip-tests" not in sys.argv:
        if not run_tests():
            print("⚠️  Some tests failed, but continuing anyway...")
            print("💡 Use --skip-tests to skip testing")
    
    # Start application
    start_streamlit()

if __name__ == "__main__":
    main()
