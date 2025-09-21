"""
Data processing module for Full Sail Finance liquidity pool volume prediction.
Handles data cleaning, aggregation, correlation analysis, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Handles data cleaning, aggregation, and feature engineering."""
    
    def __init__(self):
        """Initialize DataProcessor."""
        self.processed_data = {}
        self.correlation_matrix = None
        self.feature_importance = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values, outliers, and data types.
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Convert date column to datetime if it exists
        if 'date' in cleaned_df.columns:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
            cleaned_df = cleaned_df.sort_values('date').reset_index(drop=True)
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Fill missing values with forward fill, then backward fill
            cleaned_df[col] = cleaned_df[col].ffill().bfill()
            
            # Replace any remaining NaN with column median
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove outliers using IQR method and ensure no negative values
        for col in numeric_columns:
            if col not in ['date']:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing to preserve data
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Ensure no negative values for financial data
                if col in ['volume_24h', 'tvl', 'fee_revenue', 'liquidity']:
                    cleaned_df[col] = cleaned_df[col].abs().clip(lower=0)
        
        return cleaned_df
    
    def aggregate_by_epoch(self, df: pd.DataFrame, epoch: str = 'W') -> pd.DataFrame:
        """
        Aggregate data by time epoch (weekly, daily, etc.).
        
        Args:
            df: DataFrame with date column
            epoch: Aggregation period ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            Aggregated DataFrame
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain 'date' column")
        
        # Set date as index for resampling
        df_indexed = df.set_index('date')
        
        # Define aggregation functions for different column types
        agg_functions = {}
        for col in df_indexed.columns:
            if col in ['volume_24h', 'fee_revenue', 'transaction_count']:
                agg_functions[col] = 'sum'  # Sum volumes and transactions
            elif col in ['tvl', 'liquidity', 'price_usd', 'market_cap']:
                agg_functions[col] = 'mean'  # Average for state variables
            elif col in ['active_addresses']:
                agg_functions[col] = 'max'  # Max for address counts
            elif col == 'pool':
                continue  # Skip non-numeric columns
            else:
                agg_functions[col] = 'mean'  # Default to mean
        
        # Aggregate by epoch
        if 'pool' in df.columns:
            # Group by pool first, then resample
            aggregated = df_indexed.groupby('pool').resample(epoch).agg(agg_functions).reset_index()
        else:
            aggregated = df_indexed.resample(epoch).agg(agg_functions).reset_index()
        
        return aggregated
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute basic statistics for numeric columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        return stats
    
    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix for numeric columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Correlation matrix DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def add_technical_indicators(self, df: pd.DataFrame, volume_col: str = 'volume_24h') -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            df: DataFrame with volume data
            volume_col: Name of volume column
            
        Returns:
            DataFrame with technical indicators
        """
        if volume_col not in df.columns:
            return df
        
        df_with_indicators = df.copy()
        
        # Moving averages
        df_with_indicators[f'{volume_col}_ma7'] = df[volume_col].rolling(window=7, min_periods=1).mean()
        df_with_indicators[f'{volume_col}_ma30'] = df[volume_col].rolling(window=30, min_periods=1).mean()
        
        # Exponential moving averages
        df_with_indicators[f'{volume_col}_ema12'] = df[volume_col].ewm(span=12).mean()
        df_with_indicators[f'{volume_col}_ema26'] = df[volume_col].ewm(span=26).mean()
        
        # MACD
        df_with_indicators[f'{volume_col}_macd'] = (
            df_with_indicators[f'{volume_col}_ema12'] - df_with_indicators[f'{volume_col}_ema26']
        )
        
        # RSI (Relative Strength Index)
        delta = df[volume_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df_with_indicators[f'{volume_col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = df[volume_col].rolling(window=20, min_periods=1).mean()
        rolling_std = df[volume_col].rolling(window=20, min_periods=1).std()
        df_with_indicators[f'{volume_col}_bb_upper'] = rolling_mean + (rolling_std * 2)
        df_with_indicators[f'{volume_col}_bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # Volume trend
        df_with_indicators[f'{volume_col}_trend'] = df[volume_col].pct_change()
        df_with_indicators[f'{volume_col}_volatility'] = df[volume_col].rolling(window=7, min_periods=1).std()
        
        return df_with_indicators
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for machine learning models.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Time-based features
        if 'date' in features_df.columns:
            features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
            features_df['month'] = pd.to_datetime(features_df['date']).dt.month
            features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lagged features
        if 'volume_24h' in features_df.columns:
            for lag in [1, 2, 3, 7]:
                features_df[f'volume_lag_{lag}'] = features_df['volume_24h'].shift(lag)
        
        # Ratio features
        if 'volume_24h' in features_df.columns and 'tvl' in features_df.columns:
            features_df['volume_tvl_ratio'] = features_df['volume_24h'] / (features_df['tvl'] + 1)
        
        if 'fee_revenue' in features_df.columns and 'volume_24h' in features_df.columns:
            features_df['fee_rate'] = features_df['fee_revenue'] / (features_df['volume_24h'] + 1)
        
        # Market condition indicators
        if 'price_usd' in features_df.columns:
            features_df['price_change'] = features_df['price_usd'].pct_change()
            features_df['price_trend'] = features_df['price_change'].rolling(window=7, min_periods=1).mean()
        
        return features_df
    
    def detect_events(self, df: pd.DataFrame, volume_col: str = 'volume_24h', threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect volume spike events (e.g., buybacks, major announcements).
        
        Args:
            df: DataFrame with volume data
            volume_col: Name of volume column
            threshold: Number of standard deviations for spike detection
            
        Returns:
            DataFrame with event indicators
        """
        df_with_events = df.copy()
        
        if volume_col not in df.columns:
            return df_with_events
        
        # Calculate rolling statistics
        rolling_mean = df[volume_col].rolling(window=14, min_periods=1).mean()
        rolling_std = df[volume_col].rolling(window=14, min_periods=1).std()
        
        # Detect spikes
        z_score = (df[volume_col] - rolling_mean) / rolling_std
        df_with_events['volume_spike'] = (z_score > threshold).astype(int)
        df_with_events['volume_z_score'] = z_score
        
        # Detect significant drops
        df_with_events['volume_drop'] = (z_score < -threshold).astype(int)
        
        return df_with_events
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'volume_24h', 
                            forecast_horizon: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training prediction models.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Create features and targets
        features_df = df.copy()
        
        # Create target values (future values)
        targets_df = pd.DataFrame()
        for h in range(1, forecast_horizon + 1):
            targets_df[f'{target_col}_t+{h}'] = df[target_col].shift(-h)
        
        # Remove rows with missing targets
        valid_rows = ~targets_df.isna().any(axis=1)
        features_df = features_df[valid_rows].reset_index(drop=True)
        targets_df = targets_df[valid_rows].reset_index(drop=True)
        
        return features_df, targets_df
    
    def process_pool_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process data for multiple pools separately.
        
        Args:
            df: DataFrame with pool data
            
        Returns:
            Dictionary with processed data for each pool
        """
        processed_pools = {}
        
        if 'pool' not in df.columns:
            # Single pool data
            processed_pools['default'] = self._process_single_pool(df)
        else:
            # Multiple pools
            for pool in df['pool'].unique():
                pool_data = df[df['pool'] == pool].copy()
                processed_pools[pool] = self._process_single_pool(pool_data)
        
        return processed_pools
    
    def _process_single_pool(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for a single pool."""
        # Clean data
        cleaned_df = self.clean_data(df)
        
        # Add technical indicators
        df_with_indicators = self.add_technical_indicators(cleaned_df)
        
        # Create features
        df_with_features = self.create_features(df_with_indicators)
        
        # Detect events
        df_with_events = self.detect_events(df_with_features)
        
        return df_with_events


# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    # Initialize components
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    print("Testing data processing...")
    
    # Fetch and process sample data
    historical_data = fetcher.fetch_historical_volumes(30)
    print(f"Raw data shape: {historical_data.shape}")
    
    # Process data
    processed_pools = processor.process_pool_data(historical_data)
    
    for pool_name, pool_data in processed_pools.items():
        print(f"\nProcessed {pool_name} data:")
        print(f"Shape: {pool_data.shape}")
        print(f"Columns: {list(pool_data.columns)}")
        
        # Compute statistics
        stats = processor.compute_statistics(pool_data)
        print(f"Volume statistics: {stats.get('volume_24h', {})}")
        
        # Compute correlations
        if len(pool_data) > 1:
            correlations = processor.compute_correlations(pool_data)
            print(f"Correlation matrix shape: {correlations.shape}")
    
    # Aggregate by week
    weekly_data = processor.aggregate_by_epoch(historical_data, 'W')
    print(f"\nWeekly aggregated data shape: {weekly_data.shape}")
    
    print("Data processing test completed!")
