"""
Test suite for the Full Sail Finance application.
Tests all major components and functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from prediction_models import VolumePredictor
from visualization import VolumeVisualizer
from utils import (
    validate_dataframe, safe_divide, safe_percentage_change,
    clean_numeric_data, calculate_model_metrics, generate_summary_stats
)


class TestDataFetcher:
    """Test cases for DataFetcher class."""
    
    def test_init(self):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher()
        assert fetcher.cache_dir == "data_cache"
        assert fetcher.base_defillama_url == "https://api.llama.fi"
        assert fetcher.base_coingecko_url == "https://api.coingecko.com/api/v3"
    
    def test_create_sample_dex_data(self):
        """Test sample DEX data creation."""
        fetcher = DataFetcher()
        sample_data = fetcher._create_sample_dex_data()
        
        assert isinstance(sample_data, pd.DataFrame)
        assert 'date' in sample_data.columns
        assert 'volume_24h' in sample_data.columns
        assert 'tvl' in sample_data.columns
        assert len(sample_data) > 0
        assert sample_data['volume_24h'].min() >= 0  # No negative volumes
    
    def test_create_sample_historical_data(self):
        """Test sample historical data creation."""
        fetcher = DataFetcher()
        sample_data = fetcher._create_sample_historical_data(30)
        
        assert isinstance(sample_data, pd.DataFrame)
        assert 'date' in sample_data.columns
        assert 'pool' in sample_data.columns
        assert 'volume_24h' in sample_data.columns
        assert len(sample_data) == 30 * 4  # 30 days * 4 pools (SAIL/USDC, SUI/USDC, IKA/SUI, ALKIMI/SUI)
        assert sample_data['volume_24h'].min() >= 0
    
    def test_fetch_historical_volumes(self):
        """Test fetching historical volumes."""
        fetcher = DataFetcher()
        data = fetcher.fetch_historical_volumes(30)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'date' in data.columns
        assert 'volume_24h' in data.columns


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = DataProcessor()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'volume_24h': np.random.rand(30) * 1000 + 100,
            'tvl': np.random.rand(30) * 10000 + 1000,
            'pool': ['SAIL/USDC'] * 30
        })
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some problematic data
        dirty_data = self.test_data.copy()
        dirty_data.loc[5, 'volume_24h'] = np.nan
        dirty_data.loc[10, 'tvl'] = np.inf
        dirty_data.loc[15, 'volume_24h'] = -100  # Negative value
        
        cleaned_data = self.processor.clean_data(dirty_data)
        
        assert not cleaned_data['volume_24h'].isna().any()
        assert not np.isinf(cleaned_data['tvl']).any()
        assert cleaned_data['volume_24h'].min() >= 0  # Should handle negatives
    
    def test_aggregate_by_epoch(self):
        """Test data aggregation by time periods."""
        weekly_data = self.processor.aggregate_by_epoch(self.test_data, 'W')
        
        assert isinstance(weekly_data, pd.DataFrame)
        assert len(weekly_data) <= len(self.test_data)  # Should be fewer rows
        assert 'date' in weekly_data.columns
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        stats = self.processor.compute_statistics(self.test_data)
        
        assert isinstance(stats, dict)
        assert 'volume_24h' in stats
        assert 'mean' in stats['volume_24h']
        assert 'std' in stats['volume_24h']
        assert stats['volume_24h']['mean'] > 0
    
    def test_add_technical_indicators(self):
        """Test technical indicators addition."""
        data_with_indicators = self.processor.add_technical_indicators(self.test_data)
        
        assert 'volume_24h_ma7' in data_with_indicators.columns
        assert 'volume_24h_ma30' in data_with_indicators.columns
        assert 'volume_24h_rsi' in data_with_indicators.columns
        assert len(data_with_indicators) == len(self.test_data)
    
    def test_create_features(self):
        """Test feature engineering."""
        featured_data = self.processor.create_features(self.test_data)
        
        assert 'day_of_week' in featured_data.columns
        assert 'month' in featured_data.columns
        assert 'is_weekend' in featured_data.columns
        assert len(featured_data) == len(self.test_data)
    
    def test_detect_events(self):
        """Test event detection."""
        # Create data with a clear spike
        spike_data = self.test_data.copy()
        spike_data.loc[15, 'volume_24h'] = spike_data['volume_24h'].mean() * 5  # Create spike
        
        data_with_events = self.processor.detect_events(spike_data)
        
        assert 'volume_spike' in data_with_events.columns
        assert 'volume_z_score' in data_with_events.columns
        assert data_with_events['volume_spike'].sum() > 0  # Should detect the spike


class TestVolumePredictor:
    """Test cases for VolumePredictor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.predictor = VolumePredictor()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=60),
            'volume_24h': np.random.rand(60) * 1000 + 500 + np.sin(np.arange(60) * 0.1) * 100,  # With trend
        })
    
    def test_prepare_prophet_data(self):
        """Test Prophet data preparation."""
        if not hasattr(self.predictor, '_simple_forecast'):
            pytest.skip("Prophet not available")
        
        prophet_data = self.predictor.prepare_prophet_data(self.test_data)
        
        assert isinstance(prophet_data, pd.DataFrame)
        assert 'ds' in prophet_data.columns
        assert 'y' in prophet_data.columns
        assert len(prophet_data) == len(self.test_data)
    
    def test_simple_forecast(self):
        """Test simple forecasting method."""
        predictions = self.predictor._simple_forecast(self.test_data, 'volume_24h', 7)
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'date' in predictions.columns
        assert 'predicted' in predictions.columns
        assert 'lower_bound' in predictions.columns
        assert 'upper_bound' in predictions.columns
        assert len(predictions) == 7
        assert predictions['predicted'].min() >= 0
    
    def test_ensemble_predict(self):
        """Test ensemble prediction."""
        predictions = self.predictor.ensemble_predict(self.test_data, 'volume_24h', 7)
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 7
        assert 'predicted' in predictions.columns
        assert predictions['predicted'].min() >= 0
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        performance = self.predictor.evaluate_model_performance(self.test_data, 'volume_24h', 7)
        
        assert isinstance(performance, dict)
        # Should either have metrics or an error
        assert 'mape' in performance or 'error' in performance


class TestVolumeVisualizer:
    """Test cases for VolumeVisualizer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.visualizer = VolumeVisualizer()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'volume_24h': np.random.rand(30) * 1000 + 100,
            'pool': ['SAIL/USDC'] * 15 + ['SAIL/SUI'] * 15
        })
    
    def test_create_volume_timeseries(self):
        """Test volume time series chart creation."""
        fig = self.visualizer.create_volume_timeseries(self.test_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0  # Should have traces
    
    def test_create_pool_comparison(self):
        """Test pool comparison chart creation."""
        fig = self.visualizer.create_pool_comparison(self.test_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation."""
        # Create correlation matrix
        correlation_data = self.test_data[['volume_24h']].corr()
        fig = self.visualizer.create_correlation_heatmap(correlation_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_educational_annotations(self):
        """Test educational annotations."""
        annotations = self.visualizer.create_educational_annotations()
        
        assert isinstance(annotations, dict)
        assert 'volume_spike' in annotations
        assert 'moving_average' in annotations
        assert len(annotations) > 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'volume': np.random.rand(20) * 1000
        })
        
        assert validate_dataframe(valid_df, ['date', 'volume'], min_rows=10) == True
        
        # Invalid DataFrame (missing column)
        assert validate_dataframe(valid_df, ['date', 'price'], min_rows=10) == False
        
        # Invalid DataFrame (insufficient rows)
        assert validate_dataframe(valid_df, ['date', 'volume'], min_rows=50) == False
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=999) == 999
        assert safe_divide(10, np.nan) == 0.0
    
    def test_safe_percentage_change(self):
        """Test safe percentage change calculation."""
        assert safe_percentage_change(110, 100) == 10.0
        assert safe_percentage_change(90, 100) == -10.0
        assert safe_percentage_change(100, 0) == 0.0
        assert safe_percentage_change(np.nan, 100) == 0.0
    
    def test_clean_numeric_data(self):
        """Test numeric data cleaning."""
        dirty_series = pd.Series([1, 2, np.nan, np.inf, -np.inf, 1000, 5, 6])
        cleaned_series = clean_numeric_data(dirty_series)
        
        assert not cleaned_series.isna().any()
        assert not np.isinf(cleaned_series).any()
        assert len(cleaned_series) == len(dirty_series)
    
    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 520])
        
        metrics = calculate_model_metrics(actual, predicted)
        
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2_score' in metrics
        assert metrics['mae'] > 0
    
    def test_generate_summary_stats(self):
        """Test summary statistics generation."""
        test_df = pd.DataFrame({
            'volume': np.random.rand(100) * 1000,
            'price': np.random.rand(100) * 10,
            'name': ['test'] * 100
        })
        
        stats = generate_summary_stats(test_df)
        
        assert isinstance(stats, dict)
        assert 'volume' in stats
        assert 'price' in stats
        assert 'name' not in stats  # Should exclude non-numeric
        assert 'mean' in stats['volume']
        assert 'std' in stats['volume']


def run_integration_test():
    """Run integration test of the full pipeline."""
    print("Running integration test...")
    
    try:
        # Initialize components
        fetcher = DataFetcher()
        processor = DataProcessor()
        predictor = VolumePredictor()
        visualizer = VolumeVisualizer()
        
        # Fetch data
        historical_data = fetcher.fetch_historical_volumes(30)
        print(f"✓ Fetched {len(historical_data)} rows of historical data")
        
        # Process data
        processed_data = processor.process_pool_data(historical_data)
        print(f"✓ Processed data for {len(processed_data)} pools")
        
        # Generate predictions for first pool
        first_pool = list(processed_data.keys())[0]
        pool_data = processed_data[first_pool]
        
        if len(pool_data) >= 14:  # Need sufficient data
            predictions = predictor.ensemble_predict(pool_data, 'volume_24h', 7)
            print(f"✓ Generated {len(predictions)} predictions")
            
            # Create visualization
            fig = visualizer.create_volume_timeseries(pool_data)
            print("✓ Created visualization")
            
            print("🎉 Integration test passed!")
            return True
        else:
            print("⚠️ Insufficient data for prediction test")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests if pytest is available
    try:
        import pytest
        print("Running pytest...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available, running manual tests...")
        
        # Run manual tests
        test_classes = [TestDataFetcher, TestDataProcessor, TestVolumePredictor, 
                       TestVolumeVisualizer, TestUtils]
        
        for test_class in test_classes:
            print(f"\nTesting {test_class.__name__}...")
            test_instance = test_class()
            
            # Run setup if it exists
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            # Run all test methods
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    try:
                        method = getattr(test_instance, method_name)
                        method()
                        print(f"  ✓ {method_name}")
                    except Exception as e:
                        print(f"  ❌ {method_name}: {e}")
    
    # Run integration test
    print("\n" + "="*50)
    run_integration_test()
