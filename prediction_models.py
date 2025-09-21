"""
Prediction models module for Full Sail Finance liquidity pool volume forecasting.
Implements Prophet and ARIMA time-series forecasting models with confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

from datetime import datetime, timedelta
import json


class VolumePredictor:
    """Time-series forecasting for liquidity pool volumes."""
    
    def __init__(self):
        """Initialize VolumePredictor."""
        self.models = {}
        self.predictions = {}
        self.model_performance = {}
    
    def prepare_prophet_data(self, df: pd.DataFrame, target_col: str = 'volume_24h') -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Args:
            df: DataFrame with date and target columns
            target_col: Name of target column
            
        Returns:
            DataFrame formatted for Prophet (ds, y columns)
        """
        if 'date' not in df.columns or target_col not in df.columns:
            raise ValueError(f"DataFrame must contain 'date' and '{target_col}' columns")
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df[target_col]
        })
        
        # Remove any rows with missing values
        prophet_df = prophet_df.dropna()
        
        return prophet_df
    
    def fit_prophet_model(self, df: pd.DataFrame, target_col: str = 'volume_24h',
                         seasonality_mode: str = 'multiplicative',
                         events: Optional[pd.DataFrame] = None) -> Any:
        """
        Fit Prophet forecasting model.
        
        Args:
            df: DataFrame with historical data
            target_col: Target column name
            seasonality_mode: 'additive' or 'multiplicative'
            events: DataFrame with event data (holidays/special events)
            
        Returns:
            Fitted Prophet model
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed")
        
        # Prepare data
        prophet_data = self.prepare_prophet_data(df, target_col)
        
        # Initialize Prophet model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Not enough data typically
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,  # Strength of seasonality
            interval_width=0.95  # Confidence interval width
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Add events/holidays if provided
        if events is not None and not events.empty:
            model.add_country_holidays(country_name='US')  # Can be customized
        
        # Fit the model
        model.fit(prophet_data)
        
        return model
    
    def prophet_predict(self, model: Any, periods: int = 7) -> pd.DataFrame:
        """
        Generate predictions using Prophet model.
        
        Args:
            model: Fitted Prophet model
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract relevant columns
        prediction_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        prediction_df.columns = ['date', 'predicted', 'lower_bound', 'upper_bound']
        
        # Ensure predictions are non-negative
        prediction_df['predicted'] = prediction_df['predicted'].clip(lower=0)
        prediction_df['lower_bound'] = prediction_df['lower_bound'].clip(lower=0)
        prediction_df['upper_bound'] = prediction_df['upper_bound'].clip(lower=0)
        
        return prediction_df
    
    def prepare_arima_data(self, df: pd.DataFrame, target_col: str = 'volume_24h') -> pd.Series:
        """
        Prepare data for ARIMA model.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            Time series data
        """
        if target_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{target_col}' column")
        
        # Create time series
        if 'date' in df.columns:
            ts_data = df.set_index('date')[target_col]
        else:
            ts_data = df[target_col]
        
        # Remove missing values
        ts_data = ts_data.dropna()
        
        return ts_data
    
    def check_stationarity(self, ts_data: pd.Series) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            # Perform ADF test
            adf_result = adfuller(ts_data.dropna())
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            return {"error": f"Error in stationarity test: {str(e)}"}
    
    def find_arima_order(self, ts_data: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC criterion.
        
        Args:
            ts_data: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Optimal (p, d, q) order
        """
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)  # Default order
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Grid search for best parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def fit_arima_model(self, df: pd.DataFrame, target_col: str = 'volume_24h',
                       order: Optional[Tuple[int, int, int]] = None) -> Any:
        """
        Fit ARIMA forecasting model.
        
        Args:
            df: DataFrame with historical data
            target_col: Target column name
            order: ARIMA order (p, d, q). If None, will be automatically determined
            
        Returns:
            Fitted ARIMA model
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not installed")
        
        # Prepare data
        ts_data = self.prepare_arima_data(df, target_col)
        
        # Find optimal order if not provided
        if order is None:
            order = self.find_arima_order(ts_data)
            print(f"Auto-selected ARIMA order: {order}")
        
        # Fit ARIMA model
        try:
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            # Fallback to simple order
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            return fitted_model
    
    def arima_predict(self, model: Any, periods: int = 7) -> pd.DataFrame:
        """
        Generate predictions using ARIMA model.
        
        Args:
            model: Fitted ARIMA model
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        # Generate forecast
        forecast_result = model.forecast(steps=periods, alpha=0.05)  # 95% confidence
        
        if hasattr(forecast_result, 'predicted_mean'):
            # Newer statsmodels version
            predicted = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]
        else:
            # Older statsmodels version or different return format
            predicted = forecast_result
            # Generate simple confidence intervals
            std_error = np.std(model.resid) if hasattr(model, 'resid') else np.std(predicted) * 0.1
            lower_bound = predicted - 1.96 * std_error
            upper_bound = predicted + 1.96 * std_error
        
        # Create date range for predictions
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else datetime.now()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'date': future_dates,
            'predicted': predicted.values if hasattr(predicted, 'values') else predicted,
            'lower_bound': lower_bound.values if hasattr(lower_bound, 'values') else lower_bound,
            'upper_bound': upper_bound.values if hasattr(upper_bound, 'values') else upper_bound
        })
        
        # Ensure predictions are non-negative
        prediction_df['predicted'] = prediction_df['predicted'].clip(lower=0)
        prediction_df['lower_bound'] = prediction_df['lower_bound'].clip(lower=0)
        prediction_df['upper_bound'] = prediction_df['upper_bound'].clip(lower=0)
        
        return prediction_df
    
    def ensemble_predict(self, df: pd.DataFrame, target_col: str = 'volume_24h',
                        periods: int = 7, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate ensemble predictions combining Prophet and ARIMA.
        
        Args:
            df: DataFrame with historical data
            target_col: Target column name
            periods: Number of periods to forecast
            weights: Model weights for ensemble (default: equal weight)
            
        Returns:
            DataFrame with ensemble predictions
        """
        predictions = {}
        
        # Default weights
        if weights is None:
            weights = {'prophet': 0.6, 'arima': 0.4}
        
        # Prophet predictions
        if PROPHET_AVAILABLE:
            try:
                prophet_model = self.fit_prophet_model(df, target_col)
                prophet_pred = self.prophet_predict(prophet_model, periods)
                predictions['prophet'] = prophet_pred
                self.models['prophet'] = prophet_model
            except Exception as e:
                print(f"Prophet model failed: {e}")
        
        # ARIMA predictions
        if STATSMODELS_AVAILABLE:
            try:
                arima_model = self.fit_arima_model(df, target_col)
                arima_pred = self.arima_predict(arima_model, periods)
                predictions['arima'] = arima_pred
                self.models['arima'] = arima_model
            except Exception as e:
                print(f"ARIMA model failed: {e}")
        
        # Combine predictions
        if not predictions:
            # Fallback to simple prediction
            return self._simple_forecast(df, target_col, periods)
        
        # Ensemble combination
        ensemble_df = None
        total_weight = 0
        
        for model_name, pred_df in predictions.items():
            weight = weights.get(model_name, 1.0 / len(predictions))
            
            if ensemble_df is None:
                ensemble_df = pred_df.copy()
                ensemble_df['predicted'] *= weight
                ensemble_df['lower_bound'] *= weight
                ensemble_df['upper_bound'] *= weight
            else:
                ensemble_df['predicted'] += pred_df['predicted'] * weight
                ensemble_df['lower_bound'] += pred_df['lower_bound'] * weight
                ensemble_df['upper_bound'] += pred_df['upper_bound'] * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_df['predicted'] /= total_weight
            ensemble_df['lower_bound'] /= total_weight
            ensemble_df['upper_bound'] /= total_weight
        
        # Store predictions
        self.predictions[target_col] = predictions
        
        return ensemble_df
    
    def simple_forecast(self, df: pd.DataFrame, target_col: str, periods: int) -> pd.DataFrame:
        """
        Public interface for simple forecasting.
        
        Args:
            df: DataFrame with historical data
            target_col: Column to forecast
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        return self._simple_forecast(df, target_col, periods)
    
    def _simple_forecast(self, df: pd.DataFrame, target_col: str, periods: int) -> pd.DataFrame:
        """
        Simple fallback forecasting method using moving average and trend.
        
        Args:
            df: DataFrame with historical data
            target_col: Target column name
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with simple predictions
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")
        
        # Calculate trend and seasonal components
        recent_data = df[target_col].tail(14)  # Last 2 weeks
        
        # Simple moving average
        ma_7 = recent_data.tail(7).mean()
        
        # Linear trend
        if len(recent_data) >= 7:
            x = np.arange(len(recent_data))
            y = recent_data.values
            trend = np.polyfit(x, y, 1)[0]  # Linear trend slope
        else:
            trend = 0
        
        # Generate future dates
        last_date = df['date'].max() if 'date' in df.columns else datetime.now()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Simple prediction: moving average + trend
        predictions = []
        for i in range(periods):
            pred = ma_7 + trend * (i + 1)
            predictions.append(max(0, pred))  # Ensure non-negative
        
        # Create confidence intervals (±20% of prediction)
        predictions = np.array(predictions)
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted': predictions,
            'lower_bound': predictions * 0.8,
            'upper_bound': predictions * 1.2
        })
    
    def evaluate_model_performance(self, df: pd.DataFrame, target_col: str = 'volume_24h',
                                 test_size: int = 7) -> Dict:
        """
        Evaluate model performance using backtesting.
        
        Args:
            df: DataFrame with historical data
            target_col: Target column name
            test_size: Number of periods for testing
            
        Returns:
            Dictionary with performance metrics
        """
        if len(df) < test_size + 14:  # Need enough data for training
            return {"error": "Insufficient data for evaluation"}
        
        # Split data
        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()
        
        # Generate predictions
        try:
            pred_df = self.ensemble_predict(train_df, target_col, test_size)
            
            # Calculate metrics
            actual = test_df[target_col].values
            predicted = pred_df['predicted'].values
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
            
            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(actual - predicted))
            
            # R-squared
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'test_size': test_size,
                'train_size': len(train_df)
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}


# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from data_processor import DataProcessor
    
    # Initialize components
    fetcher = DataFetcher()
    processor = DataProcessor()
    predictor = VolumePredictor()
    
    print("Testing prediction models...")
    
    # Fetch and process data
    historical_data = fetcher.fetch_historical_volumes(60)  # 60 days of data
    processed_data = processor.process_pool_data(historical_data)
    
    # Test predictions for each pool
    for pool_name, pool_data in processed_data.items():
        print(f"\nTesting predictions for {pool_name}:")
        
        if len(pool_data) < 20:  # Need sufficient data
            print("Insufficient data for prediction")
            continue
        
        try:
            # Generate 7-day forecast
            predictions = predictor.ensemble_predict(pool_data, 'volume_24h', 7)
            print(f"Generated {len(predictions)} predictions")
            print(predictions.head())
            
            # Evaluate model performance
            performance = predictor.evaluate_model_performance(pool_data, 'volume_24h')
            print(f"Model performance: {performance}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    print("Prediction model testing completed!")
