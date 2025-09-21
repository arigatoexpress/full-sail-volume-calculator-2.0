"""
Utility functions and helpers for the Full Sail Finance application.
Includes error handling, logging, and common operations.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import functools
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_sail_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def error_handler(func):
    """Decorator for error handling and logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function calls on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], min_rows: int = 1) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        logger.warning(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"DataFrame missing required columns: {missing_columns}")
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, handling division by zero."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def safe_percentage_change(current: float, previous: float) -> float:
    """Safely calculate percentage change."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def format_currency(amount: float, decimals: int = 0) -> str:
    """Format number as currency string."""
    try:
        if pd.isna(amount):
            return "N/A"
        return f"${amount:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage string."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def clean_numeric_data(series: pd.Series) -> pd.Series:
    """Clean numeric data by removing outliers and handling missing values."""
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Remove infinite values
    numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median
    if numeric_series.notna().any():  # Check if there are any non-NaN values
        median_value = numeric_series.median()
        numeric_series = numeric_series.fillna(median_value)
    
    # Handle outliers using IQR method
    Q1 = numeric_series.quantile(0.25)
    Q3 = numeric_series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        numeric_series = numeric_series.clip(lower=lower_bound, upper=upper_bound)
    
    # Ensure no negative values for volume/financial data
    # Replace negative values with their absolute value
    numeric_series = numeric_series.abs().clip(lower=0)
    
    return numeric_series


def calculate_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate moving average with error handling."""
    try:
        if len(series) < window:
            return series.rolling(window=len(series), min_periods=1).mean()
        return series.rolling(window=window, min_periods=1).mean()
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        return series.fillna(series.mean())


def detect_anomalies(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Detect anomalies using z-score method."""
    try:
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return pd.Series([False] * len(series), index=series.index)


def save_config(config: Dict, filename: str = "app_config.json") -> None:
    """Save configuration to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Configuration saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


def load_config(filename: str = "app_config.json") -> Dict:
    """Load configuration from JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {filename}")
            return config
        else:
            logger.warning(f"Configuration file {filename} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def get_default_config() -> Dict:
    """Get default application configuration."""
    return {
        "api_settings": {
            "defillama_base_url": "https://api.llama.fi",
            "coingecko_base_url": "https://api.coingecko.com/api/v3",
            "request_timeout": 30,
            "rate_limit_delay": 0.1
        },
        "data_settings": {
            "cache_duration_hours": 1,
            "default_history_days": 60,
            "min_data_points": 14
        },
        "model_settings": {
            "default_forecast_days": 7,
            "confidence_level": 0.95,
            "ensemble_weights": {"prophet": 0.6, "arima": 0.4}
        },
        "ui_settings": {
            "default_chart_height": 500,
            "color_scheme": "plotly",
            "show_technical_indicators": True
        }
    }


def validate_prediction_input(df: pd.DataFrame, target_col: str, forecast_days: int) -> Tuple[bool, str]:
    """
    Validate input data for prediction models.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        forecast_days: Number of forecast days
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check DataFrame
    if df is None or df.empty:
        return False, "Input data is empty"
    
    # Check target column
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found"
    
    # Check data quality
    if df[target_col].isna().all():
        return False, f"All values in '{target_col}' are missing"
    
    # Check sufficient data
    valid_data_points = df[target_col].notna().sum()
    if valid_data_points < 14:
        return False, f"Insufficient data points: {valid_data_points} (minimum: 14)"
    
    # Check forecast horizon
    if forecast_days < 1 or forecast_days > 30:
        return False, "Forecast days must be between 1 and 30"
    
    return True, "Validation passed"


def calculate_model_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Ensure arrays are the same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {"error": "No valid data points for metrics calculation"}
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (handle division by zero)
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "r2_score": r2,
            "data_points": len(actual)
        }
        
    except Exception as e:
        logger.error(f"Error calculating model metrics: {e}")
        return {"error": str(e)}


def generate_summary_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
    """
    Generate summary statistics for DataFrame columns.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to analyze (if None, uses all numeric columns)
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {}
        
        for col in columns:
            if col in df.columns:
                series = df[col].dropna()
                
                if len(series) > 0:
                    summary[col] = {
                        "count": len(series),
                        "mean": series.mean(),
                        "median": series.median(),
                        "std": series.std(),
                        "min": series.min(),
                        "max": series.max(),
                        "q25": series.quantile(0.25),
                        "q75": series.quantile(0.75),
                        "skewness": series.skew(),
                        "kurtosis": series.kurtosis()
                    }
                else:
                    summary[col] = {"error": "No valid data"}
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary statistics: {e}")
        return {"error": str(e)}


def create_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create additional date-based features.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with additional date features
    """
    try:
        df_with_features = df.copy()
        
        if date_col in df.columns:
            # Ensure datetime type
            df_with_features[date_col] = pd.to_datetime(df_with_features[date_col])
            
            # Extract date components
            df_with_features['year'] = df_with_features[date_col].dt.year
            df_with_features['month'] = df_with_features[date_col].dt.month
            df_with_features['day'] = df_with_features[date_col].dt.day
            df_with_features['day_of_week'] = df_with_features[date_col].dt.dayofweek
            df_with_features['day_of_year'] = df_with_features[date_col].dt.dayofyear
            df_with_features['week_of_year'] = df_with_features[date_col].dt.isocalendar().week
            df_with_features['quarter'] = df_with_features[date_col].dt.quarter
            
            # Binary features
            df_with_features['is_weekend'] = (df_with_features['day_of_week'] >= 5).astype(int)
            df_with_features['is_month_start'] = df_with_features[date_col].dt.is_month_start.astype(int)
            df_with_features['is_month_end'] = df_with_features[date_col].dt.is_month_end.astype(int)
            df_with_features['is_quarter_start'] = df_with_features[date_col].dt.is_quarter_start.astype(int)
            df_with_features['is_quarter_end'] = df_with_features[date_col].dt.is_quarter_end.astype(int)
        
        return df_with_features
        
    except Exception as e:
        logger.error(f"Error creating date features: {e}")
        return df


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = self.metrics.get(operation, []) + [duration]
            logger.info(f"Operation '{operation}' completed in {duration:.2f} seconds")
            return duration
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation in self.metrics:
            return np.mean(self.metrics[operation])
        return 0.0
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        for operation, times in self.metrics.items():
            summary[operation] = {
                "count": len(times),
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "total_time": np.sum(times)
            }
        return summary


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test data validation
    test_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=30),
        'volume': np.random.rand(30) * 1000,
        'price': np.random.rand(30) * 10
    })
    
    is_valid = validate_dataframe(test_df, ['date', 'volume'], min_rows=10)
    print(f"DataFrame validation: {is_valid}")
    
    # Test summary statistics
    stats = generate_summary_stats(test_df, ['volume', 'price'])
    print(f"Summary stats keys: {list(stats.keys())}")
    
    # Test date features
    df_with_dates = create_date_features(test_df)
    print(f"Date features added: {[col for col in df_with_dates.columns if col not in test_df.columns]}")
    
    # Test performance monitoring
    performance_monitor.start_timer("test_operation")
    time.sleep(0.1)  # Simulate work
    duration = performance_monitor.end_timer("test_operation")
    print(f"Performance monitoring works: {duration > 0}")
    
    print("Utility functions test completed!")
