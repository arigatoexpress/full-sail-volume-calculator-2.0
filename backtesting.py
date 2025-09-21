"""
Backtesting module for Full Sail Finance volume prediction models.
Provides comprehensive model evaluation and comparison across different time frames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from prediction_models import VolumePredictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelBacktester:
    """Comprehensive backtesting framework for prediction models."""
    
    def __init__(self):
        """Initialize backtester."""
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.predictor = VolumePredictor()
        self.backtest_results = {}
        
    def generate_extended_historical_data(self, days: int = 365) -> pd.DataFrame:
        """Generate extended historical data for backtesting."""
        print(f"Generating {days} days of historical data for backtesting...")
        
        # Generate more comprehensive historical data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        pools = ['SAIL/USDC', 'SAIL/SUI', 'USDC/SUI', 'WETH/USDC', 'IKA/SUI', 
                'WAL/USDC', 'DEEP/SUI', 'CETUS/SUI', 'BUCK/USDC', 'NAVX/SUI']
        
        records = []
        
        # Add market cycles and trends for realistic backtesting
        for i, date in enumerate(dates):
            # Market cycle simulation (bull/bear phases)
            cycle_position = (i / len(dates)) * 4 * np.pi  # 2 full cycles
            market_trend = np.sin(cycle_position) * 0.3 + 1.0
            
            # Weekly seasonality
            day_of_week = date.weekday()
            weekly_factor = 1.0 + 0.2 * np.sin(day_of_week * 2 * np.pi / 7)
            
            # Random market events (spikes)
            event_probability = 0.02  # 2% chance per day
            event_multiplier = np.random.choice([1.0, 2.5], p=[1-event_probability, event_probability])
            
            for pool in pools:
                base_volume = {
                    'SAIL/USDC': 50000, 'SAIL/SUI': 30000, 'USDC/SUI': 80000, 'WETH/USDC': 40000,
                    'IKA/SUI': 25000, 'WAL/USDC': 35000, 'DEEP/SUI': 45000, 'CETUS/SUI': 60000,
                    'BUCK/USDC': 55000, 'NAVX/SUI': 20000
                }.get(pool, 30000)
                
                # Combine all factors
                volume = (base_volume * market_trend * weekly_factor * event_multiplier * 
                         (1 + 0.15 * np.random.randn()))
                
                # Ensure positive volume
                volume = max(0, volume)
                
                records.append({
                    'date': date,
                    'pool': pool,
                    'volume_24h': volume,
                    'liquidity': volume * 5,
                    'fee_revenue': volume * 0.003,
                    'market_trend': market_trend,
                    'weekly_factor': weekly_factor,
                    'event_multiplier': event_multiplier
                })
        
        return pd.DataFrame(records)
    
    def run_walk_forward_backtest(self, df: pd.DataFrame, pool: str, 
                                 model_types: List[str] = None,
                                 forecast_horizons: List[int] = None,
                                 train_window: int = 60) -> Dict:
        """
        Run walk-forward backtesting for multiple models and horizons.
        
        Args:
            df: Historical data
            pool: Pool to backtest
            model_types: List of models to test
            forecast_horizons: List of forecast periods
            train_window: Training window size in days
            
        Returns:
            Dictionary with backtest results
        """
        if model_types is None:
            model_types = ['simple', 'prophet', 'arima', 'ensemble']
        
        if forecast_horizons is None:
            forecast_horizons = [1, 3, 7, 14]
        
        # Filter data for specific pool
        pool_data = df[df['pool'] == pool].copy().sort_values('date').reset_index(drop=True)
        
        if len(pool_data) < train_window + max(forecast_horizons):
            return {"error": f"Insufficient data for pool {pool}"}
        
        results = {
            'pool': pool,
            'models': {},
            'summary': {}
        }
        
        # Test each model type and horizon combination
        for model_type in model_types:
            results['models'][model_type] = {}
            
            for horizon in forecast_horizons:
                print(f"Backtesting {pool} - {model_type} - {horizon}d horizon...")
                
                model_results = self._backtest_single_model(
                    pool_data, model_type, horizon, train_window
                )
                
                results['models'][model_type][f'{horizon}d'] = model_results
        
        # Generate summary statistics
        results['summary'] = self._generate_backtest_summary(results['models'])
        
        return results
    
    def _backtest_single_model(self, data: pd.DataFrame, model_type: str, 
                              horizon: int, train_window: int) -> Dict:
        """Backtest a single model configuration."""
        predictions = []
        actuals = []
        dates = []
        errors = []
        
        # Walk-forward validation
        start_idx = train_window
        end_idx = len(data) - horizon
        
        for i in range(start_idx, end_idx, horizon):  # Step by horizon to avoid overlap
            # Training data
            train_data = data.iloc[i-train_window:i].copy()
            
            # Test data (actual future values)
            test_data = data.iloc[i:i+horizon].copy()
            
            if len(train_data) < 14 or len(test_data) < horizon:
                continue
            
            try:
                # Generate prediction
                if model_type == 'simple':
                    pred_df = self.predictor._simple_forecast(train_data, 'volume_24h', horizon)
                elif model_type == 'prophet':
                    try:
                        model = self.predictor.fit_prophet_model(train_data)
                        pred_df = self.predictor.prophet_predict(model, horizon)
                    except:
                        pred_df = self.predictor._simple_forecast(train_data, 'volume_24h', horizon)
                elif model_type == 'arima':
                    try:
                        model = self.predictor.fit_arima_model(train_data)
                        pred_df = self.predictor.arima_predict(model, horizon)
                    except:
                        pred_df = self.predictor._simple_forecast(train_data, 'volume_24h', horizon)
                else:  # ensemble
                    pred_df = self.predictor.ensemble_predict(train_data, 'volume_24h', horizon)
                
                # Store results
                for j in range(min(len(pred_df), len(test_data))):
                    predictions.append(pred_df['predicted'].iloc[j])
                    actuals.append(test_data['volume_24h'].iloc[j])
                    dates.append(test_data['date'].iloc[j])
                    
                    # Calculate error
                    error = abs(pred_df['predicted'].iloc[j] - test_data['volume_24h'].iloc[j])
                    errors.append(error)
            
            except Exception as e:
                print(f"Error in backtest iteration: {e}")
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = self._calculate_backtest_metrics(np.array(actuals), np.array(predictions))
            
            return {
                'predictions': predictions,
                'actuals': actuals,
                'dates': dates,
                'errors': errors,
                'metrics': metrics,
                'n_predictions': len(predictions)
            }
        else:
            return {
                'error': 'No successful predictions generated',
                'n_predictions': 0
            }
    
    def _calculate_backtest_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate comprehensive backtest metrics."""
        try:
            # Basic metrics
            mae = np.mean(np.abs(actuals - predictions))
            mse = np.mean((actuals - predictions) ** 2)
            rmse = np.sqrt(mse)
            
            # Percentage metrics
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
            
            # Directional accuracy
            actual_direction = np.diff(actuals) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # R-squared
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Theil's U statistic (forecast accuracy)
            naive_forecast = actuals[:-1]  # Lag-1 forecast
            naive_mse = np.mean((actuals[1:] - naive_forecast) ** 2)
            theil_u = rmse / np.sqrt(naive_mse) if naive_mse > 0 else float('inf')
            
            # Maximum error
            max_error = np.max(np.abs(actuals - predictions))
            
            # Error distribution
            error_std = np.std(actuals - predictions)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'theil_u': theil_u,
                'max_error': max_error,
                'error_std': error_std,
                'mean_actual': np.mean(actuals),
                'mean_predicted': np.mean(predictions)
            }
        
        except Exception as e:
            return {'error': f'Metrics calculation failed: {str(e)}'}
    
    def _generate_backtest_summary(self, model_results: Dict) -> Dict:
        """Generate summary statistics across all models and horizons."""
        summary = {
            'best_model_by_metric': {},
            'model_rankings': {},
            'horizon_analysis': {}
        }
        
        metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
        
        # Find best model for each metric
        for metric in metrics:
            best_score = float('inf') if metric in ['mae', 'rmse', 'mape'] else float('-inf')
            best_model = None
            best_horizon = None
            
            for model_type, horizons in model_results.items():
                for horizon, results in horizons.items():
                    if 'metrics' in results and metric in results['metrics']:
                        score = results['metrics'][metric]
                        
                        if metric in ['mae', 'rmse', 'mape']:
                            if score < best_score:
                                best_score = score
                                best_model = model_type
                                best_horizon = horizon
                        else:
                            if score > best_score:
                                best_score = score
                                best_model = model_type
                                best_horizon = horizon
            
            summary['best_model_by_metric'][metric] = {
                'model': best_model,
                'horizon': best_horizon,
                'score': best_score
            }
        
        return summary
    
    def create_backtest_visualization(self, backtest_results: Dict) -> go.Figure:
        """Create comprehensive backtest visualization."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Performance Comparison (MAPE)',
                'Directional Accuracy by Horizon',
                'Prediction vs Actual (Best Model)',
                'Error Distribution',
                'Model Stability (RMSE)',
                'Forecast Horizon Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(backtest_results['models'].keys())
        horizons = ['1d', '3d', '7d', '14d']
        
        # Colors for models
        model_colors = {
            'simple': '#FF6B35',
            'prophet': '#00D4FF', 
            'arima': '#00E676',
            'ensemble': '#BB86FC'
        }
        
        # 1. Model Performance Comparison (MAPE)
        for model in models:
            mapes = []
            for horizon in horizons:
                if (horizon in backtest_results['models'][model] and 
                    'metrics' in backtest_results['models'][model][horizon]):
                    mapes.append(backtest_results['models'][model][horizon]['metrics']['mape'])
                else:
                    mapes.append(None)
            
            fig.add_trace(
                go.Scatter(
                    x=horizons, y=mapes, name=f'{model} MAPE',
                    line=dict(color=model_colors.get(model, '#666666')),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # 2. Directional Accuracy
        for model in models:
            accuracies = []
            for horizon in horizons:
                if (horizon in backtest_results['models'][model] and 
                    'metrics' in backtest_results['models'][model][horizon]):
                    accuracies.append(backtest_results['models'][model][horizon]['metrics']['directional_accuracy'])
                else:
                    accuracies.append(None)
            
            fig.add_trace(
                go.Scatter(
                    x=horizons, y=accuracies, name=f'{model} Direction',
                    line=dict(color=model_colors.get(model, '#666666'), dash='dash'),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # 3. Best model prediction vs actual (if available)
        best_model_info = backtest_results['summary']['best_model_by_metric'].get('mape', {})
        if best_model_info.get('model') and best_model_info.get('horizon'):
            model_name = best_model_info['model']
            horizon_name = best_model_info['horizon']
            
            if (model_name in backtest_results['models'] and 
                horizon_name in backtest_results['models'][model_name]):
                
                results = backtest_results['models'][model_name][horizon_name]
                if 'predictions' in results and 'actuals' in results:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(results['actuals']))),
                            y=results['actuals'],
                            name='Actual',
                            line=dict(color='#00E676')
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(results['predictions']))),
                            y=results['predictions'],
                            name='Predicted',
                            line=dict(color='#FF6B35', dash='dash')
                        ),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Backtest Results for {backtest_results['pool']}",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def run_comprehensive_backtest(self, pools: List[str] = None, days: int = 180) -> Dict:
        """Run comprehensive backtesting across multiple pools."""
        if pools is None:
            pools = ['SAIL/USDC', 'SAIL/SUI', 'USDC/SUI', 'IKA/SUI', 'DEEP/SUI']
        
        print("🔄 Generating extended historical data...")
        historical_data = self.generate_extended_historical_data(days)
        
        print("🧪 Running comprehensive backtesting...")
        all_results = {}
        
        for pool in pools:
            print(f"\n📊 Backtesting {pool}...")
            pool_results = self.run_walk_forward_backtest(
                historical_data, pool, 
                model_types=['simple', 'ensemble'],  # Focus on working models
                forecast_horizons=[1, 3, 7],
                train_window=60
            )
            all_results[pool] = pool_results
        
        # Generate cross-pool summary
        all_results['cross_pool_summary'] = self._generate_cross_pool_summary(all_results)
        
        return all_results
    
    def _generate_cross_pool_summary(self, all_results: Dict) -> Dict:
        """Generate summary statistics across all pools."""
        summary = {
            'best_performing_pools': {},
            'model_consistency': {},
            'average_metrics': {}
        }
        
        # Calculate average metrics across pools
        metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
        
        for metric in metrics:
            metric_values = []
            
            for pool, results in all_results.items():
                if pool == 'cross_pool_summary':
                    continue
                
                if 'models' in results:
                    for model_type, horizons in results['models'].items():
                        for horizon, model_results in horizons.items():
                            if 'metrics' in model_results and metric in model_results['metrics']:
                                metric_values.append(model_results['metrics'][metric])
            
            if metric_values:
                summary['average_metrics'][metric] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values)
                }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing backtesting framework...")
    
    backtester = ModelBacktester()
    
    # Test with limited scope
    test_results = backtester.run_comprehensive_backtest(
        pools=['SAIL/USDC', 'IKA/SUI'], 
        days=90
    )
    
    print("\n📊 Backtest Results Summary:")
    for pool, results in test_results.items():
        if pool != 'cross_pool_summary' and 'summary' in results:
            print(f"\n{pool}:")
            best_models = results['summary']['best_model_by_metric']
            for metric, info in best_models.items():
                print(f"  Best {metric}: {info['model']} ({info['horizon']}) - {info['score']:.2f}")
    
    print("\n🎉 Backtesting framework ready!")
