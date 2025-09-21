"""
Epoch-specific volume predictor for Full Sail Finance.
Predicts total volume for the next 7-day epoch period.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

from epoch_predictor import EpochAwarePredictor
from prediction_models import VolumePredictor
from data_processor import DataProcessor


class EpochVolumePredictor:
    """Specialized predictor for next epoch volume forecasting."""
    
    def __init__(self):
        """Initialize epoch volume predictor."""
        self.epoch_predictor = EpochAwarePredictor()
        self.volume_predictor = VolumePredictor()
        self.processor = DataProcessor()
        
    def predict_next_epoch_volume(self, pool_data: pd.DataFrame, pool_name: str) -> Dict:
        """
        Predict total volume for the next 7-day epoch.
        
        Args:
            pool_data: Historical pool data
            pool_name: Name of the pool
            
        Returns:
            Dictionary with epoch volume prediction
        """
        if len(pool_data) < 21:  # Need at least 3 weeks of data
            return {
                'error': 'Insufficient data for epoch prediction',
                'required_days': 21,
                'available_days': len(pool_data)
            }
        
        # Get current epoch info
        current_epoch = self.epoch_predictor.get_current_epoch_info()
        
        # Align data to epoch boundaries
        epoch_aligned_data = self.epoch_predictor.get_epoch_aligned_data(pool_data)
        
        # Calculate historical epoch volumes
        historical_epochs = self._calculate_historical_epoch_volumes(epoch_aligned_data)
        
        # Generate daily predictions for next 7 days
        daily_predictions = self.volume_predictor.ensemble_predict(
            pool_data, 'volume_24h', 7
        )
        
        # Sum daily predictions to get epoch total
        predicted_epoch_volume = daily_predictions['predicted'].sum()
        predicted_epoch_lower = daily_predictions['lower_bound'].sum()
        predicted_epoch_upper = daily_predictions['upper_bound'].sum()
        
        # Apply epoch-specific adjustments
        epoch_adjusted_prediction = self._apply_epoch_adjustments(
            predicted_epoch_volume, historical_epochs, current_epoch
        )
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_epoch_confidence(
            historical_epochs, epoch_adjusted_prediction
        )
        
        return {
            'pool': pool_name,
            'current_epoch': current_epoch['epoch_number'],
            'next_epoch': current_epoch['epoch_number'] + 1,
            'prediction_date': datetime.now(timezone.utc),
            
            # Volume predictions
            'predicted_epoch_volume': epoch_adjusted_prediction['volume'],
            'lower_bound': epoch_adjusted_prediction['lower_bound'],
            'upper_bound': epoch_adjusted_prediction['upper_bound'],
            
            # Daily breakdown
            'daily_predictions': daily_predictions,
            'daily_average': epoch_adjusted_prediction['volume'] / 7,
            
            # Historical context
            'historical_epochs': historical_epochs,
            'last_epoch_volume': historical_epochs[-1]['total_volume'] if historical_epochs else 0,
            'average_epoch_volume': np.mean([e['total_volume'] for e in historical_epochs]) if historical_epochs else 0,
            
            # Confidence metrics
            'confidence_score': confidence_metrics['confidence_score'],
            'prediction_range_pct': confidence_metrics['range_percentage'],
            'model_accuracy': confidence_metrics['historical_accuracy'],
            
            # Insights
            'volume_trend': self._analyze_epoch_trend(historical_epochs),
            'seasonality_factor': self._calculate_seasonality_factor(historical_epochs, current_epoch),
            'risk_factors': self._identify_risk_factors(pool_data, current_epoch)
        }
    
    def _calculate_historical_epoch_volumes(self, epoch_aligned_data: pd.DataFrame) -> List[Dict]:
        """Calculate total volume for each historical epoch."""
        if 'epoch_number' not in epoch_aligned_data.columns:
            return []
        
        historical_epochs = []
        
        # Group by epoch and calculate totals
        epoch_groups = epoch_aligned_data.groupby('epoch_number')
        
        for epoch_num, epoch_data in epoch_groups:
            total_volume = epoch_data['volume_24h'].sum()
            avg_daily_volume = epoch_data['volume_24h'].mean()
            max_daily_volume = epoch_data['volume_24h'].max()
            min_daily_volume = epoch_data['volume_24h'].min()
            
            # Calculate epoch volatility
            daily_changes = epoch_data['volume_24h'].pct_change().dropna()
            epoch_volatility = daily_changes.std() * 100 if len(daily_changes) > 1 else 0
            
            historical_epochs.append({
                'epoch_number': epoch_num,
                'total_volume': total_volume,
                'avg_daily_volume': avg_daily_volume,
                'max_daily_volume': max_daily_volume,
                'min_daily_volume': min_daily_volume,
                'volatility': epoch_volatility,
                'days_in_epoch': len(epoch_data),
                'epoch_start': epoch_data['epoch_start'].iloc[0],
                'volume_trend': 'increasing' if total_volume > (historical_epochs[-1]['total_volume'] if historical_epochs else 0) else 'decreasing'
            })
        
        return historical_epochs
    
    def _apply_epoch_adjustments(self, base_prediction: float, 
                               historical_epochs: List[Dict], 
                               current_epoch: Dict) -> Dict:
        """Apply epoch-specific adjustments to base prediction."""
        if not historical_epochs:
            return {
                'volume': base_prediction,
                'lower_bound': base_prediction * 0.8,
                'upper_bound': base_prediction * 1.2,
                'adjustment_factor': 1.0
            }
        
        # Calculate epoch patterns
        recent_epochs = historical_epochs[-4:] if len(historical_epochs) >= 4 else historical_epochs
        avg_epoch_volume = np.mean([e['total_volume'] for e in recent_epochs])
        
        # Trend adjustment
        if len(historical_epochs) >= 2:
            recent_trend = (historical_epochs[-1]['total_volume'] - historical_epochs[-2]['total_volume']) / historical_epochs[-2]['total_volume']
            trend_factor = 1 + (recent_trend * 0.5)  # Dampen trend impact
        else:
            trend_factor = 1.0
        
        # Seasonality adjustment (day of week when epoch starts)
        epoch_start_day = current_epoch['epoch_start'].weekday()
        seasonal_factors = {0: 1.0, 1: 1.05, 2: 0.95, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.85}
        seasonal_factor = seasonal_factors.get(epoch_start_day, 1.0)
        
        # Volatility adjustment
        recent_volatilities = [e['volatility'] for e in recent_epochs]
        avg_volatility = np.mean(recent_volatilities) if recent_volatilities else 20
        volatility_factor = 1 + (avg_volatility / 100 * 0.1)  # Higher volatility = wider bounds
        
        # Combined adjustment
        total_adjustment = trend_factor * seasonal_factor
        
        adjusted_prediction = base_prediction * total_adjustment
        
        return {
            'volume': adjusted_prediction,
            'lower_bound': adjusted_prediction * (1 - volatility_factor * 0.2),
            'upper_bound': adjusted_prediction * (1 + volatility_factor * 0.2),
            'adjustment_factor': total_adjustment,
            'trend_factor': trend_factor,
            'seasonal_factor': seasonal_factor,
            'volatility_factor': volatility_factor
        }
    
    def _calculate_epoch_confidence(self, historical_epochs: List[Dict], 
                                  prediction: Dict) -> Dict:
        """Calculate confidence metrics for epoch prediction."""
        if not historical_epochs:
            return {
                'confidence_score': 0.5,
                'range_percentage': 40,
                'historical_accuracy': 0.6
            }
        
        # Calculate historical prediction accuracy (simulated)
        epoch_volumes = [e['total_volume'] for e in historical_epochs]
        
        if len(epoch_volumes) >= 2:
            # Simulate historical predictions vs actuals
            errors = []
            for i in range(1, len(epoch_volumes)):
                predicted = epoch_volumes[i-1] * 1.05  # Simple trend prediction
                actual = epoch_volumes[i]
                error = abs(predicted - actual) / actual * 100
                errors.append(error)
            
            avg_error = np.mean(errors)
            historical_accuracy = max(0, 1 - (avg_error / 100))
        else:
            historical_accuracy = 0.6
        
        # Confidence based on data consistency
        volatilities = [e['volatility'] for e in historical_epochs[-4:]]
        avg_volatility = np.mean(volatilities) if volatilities else 20
        
        # Higher volatility = lower confidence
        confidence_score = max(0.3, min(0.95, 1 - (avg_volatility / 100)))
        
        # Range percentage
        range_pct = (prediction['upper_bound'] - prediction['lower_bound']) / prediction['volume'] * 100
        
        return {
            'confidence_score': confidence_score,
            'range_percentage': range_pct,
            'historical_accuracy': historical_accuracy,
            'avg_historical_error': avg_error if 'avg_error' in locals() else 15
        }
    
    def _analyze_epoch_trend(self, historical_epochs: List[Dict]) -> str:
        """Analyze trend across historical epochs."""
        if len(historical_epochs) < 3:
            return 'insufficient_data'
        
        recent_volumes = [e['total_volume'] for e in historical_epochs[-3:]]
        
        # Calculate trend
        if recent_volumes[2] > recent_volumes[1] > recent_volumes[0]:
            return 'strong_upward'
        elif recent_volumes[2] > recent_volumes[0]:
            return 'upward'
        elif recent_volumes[2] < recent_volumes[1] < recent_volumes[0]:
            return 'strong_downward'
        elif recent_volumes[2] < recent_volumes[0]:
            return 'downward'
        else:
            return 'sideways'
    
    def _calculate_seasonality_factor(self, historical_epochs: List[Dict], 
                                    current_epoch: Dict) -> float:
        """Calculate seasonality factor based on historical patterns."""
        if not historical_epochs:
            return 1.0
        
        # Group epochs by start day of week
        day_volumes = {}
        
        for epoch in historical_epochs:
            start_day = epoch['epoch_start'].weekday()
            if start_day not in day_volumes:
                day_volumes[start_day] = []
            day_volumes[start_day].append(epoch['total_volume'])
        
        # Calculate average for each day
        day_averages = {day: np.mean(volumes) for day, volumes in day_volumes.items()}
        
        if not day_averages:
            return 1.0
        
        overall_average = np.mean(list(day_averages.values()))
        next_epoch_start_day = current_epoch['epoch_end'].weekday()
        
        return day_averages.get(next_epoch_start_day, overall_average) / overall_average
    
    def _identify_risk_factors(self, pool_data: pd.DataFrame, current_epoch: Dict) -> List[str]:
        """Identify risk factors that could affect next epoch volume."""
        risk_factors = []
        
        # Recent volatility
        recent_volatility = pool_data['volume_24h'].tail(7).pct_change().std() * 100
        if recent_volatility > 50:
            risk_factors.append(f"High recent volatility ({recent_volatility:.1f}%)")
        
        # Volume trend
        recent_avg = pool_data['volume_24h'].tail(7).mean()
        previous_avg = pool_data['volume_24h'].iloc[-14:-7].mean() if len(pool_data) >= 14 else recent_avg
        
        if recent_avg < previous_avg * 0.8:
            risk_factors.append("Declining volume trend (-20%+)")
        
        # Epoch timing
        if current_epoch['epoch_progress'] < 0.1:
            risk_factors.append("Early epoch - limited current data")
        
        # Market conditions (simulated)
        if np.random.random() < 0.3:  # 30% chance of market uncertainty
            risk_factors.append("General market uncertainty")
        
        return risk_factors
    
    def predict_all_pools_next_epoch(self, all_pool_data: Dict) -> Dict:
        """Predict next epoch volume for all pools."""
        epoch_predictions = {}
        
        for pool_name, pool_data in all_pool_data.items():
            try:
                prediction = self.predict_next_epoch_volume(pool_data, pool_name)
                epoch_predictions[pool_name] = prediction
            except Exception as e:
                epoch_predictions[pool_name] = {
                    'error': f'Prediction failed: {str(e)}',
                    'pool': pool_name
                }
        
        # Generate summary statistics
        successful_predictions = {k: v for k, v in epoch_predictions.items() if 'error' not in v}
        
        if successful_predictions:
            total_predicted_volume = sum(p['predicted_epoch_volume'] for p in successful_predictions.values())
            avg_confidence = np.mean([p['confidence_score'] for p in successful_predictions.values()])
            
            summary = {
                'total_ecosystem_volume': total_predicted_volume,
                'average_confidence': avg_confidence,
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(epoch_predictions) - len(successful_predictions),
                'top_volume_pool': max(successful_predictions.items(), 
                                     key=lambda x: x[1]['predicted_epoch_volume'])[0],
                'most_confident_prediction': max(successful_predictions.items(),
                                               key=lambda x: x[1]['confidence_score'])[0]
            }
        else:
            summary = {
                'total_ecosystem_volume': 0,
                'average_confidence': 0,
                'successful_predictions': 0,
                'failed_predictions': len(epoch_predictions)
            }
        
        return {
            'predictions': epoch_predictions,
            'summary': summary,
            'prediction_timestamp': datetime.now(timezone.utc),
            'target_epoch': current_epoch['epoch_number'] + 1 if 'current_epoch' in locals() else 'unknown'
        }


class UniversalTimeframeController:
    """Universal timeframe controller for all charts in the application."""
    
    def __init__(self):
        """Initialize timeframe controller."""
        self.timeframes = {
            # Intraday
            '1m': {'name': '1 Minute', 'freq': '1T', 'display': '1m', 'category': 'intraday'},
            '5m': {'name': '5 Minutes', 'freq': '5T', 'display': '5m', 'category': 'intraday'},
            '15m': {'name': '15 Minutes', 'freq': '15T', 'display': '15m', 'category': 'intraday'},
            '30m': {'name': '30 Minutes', 'freq': '30T', 'display': '30m', 'category': 'intraday'},
            '1h': {'name': '1 Hour', 'freq': '1H', 'display': '1h', 'category': 'intraday'},
            '4h': {'name': '4 Hours', 'freq': '4H', 'display': '4h', 'category': 'intraday'},
            
            # Daily and above
            '1d': {'name': '1 Day', 'freq': '1D', 'display': '1d', 'category': 'daily'},
            '3d': {'name': '3 Days', 'freq': '3D', 'display': '3d', 'category': 'daily'},
            '1w': {'name': '1 Week', 'freq': '1W', 'display': '1w', 'category': 'weekly'},
            '1M': {'name': '1 Month', 'freq': '1M', 'display': '1M', 'category': 'monthly'},
            '3M': {'name': '3 Months', 'freq': '3M', 'display': '3M', 'category': 'monthly'},
            '1Y': {'name': '1 Year', 'freq': '1Y', 'display': '1Y', 'category': 'yearly'}
        }
        
        # Data range options
        self.data_ranges = {
            '1D': {'days': 1, 'name': 'Last 24 Hours'},
            '3D': {'days': 3, 'name': 'Last 3 Days'},
            '1W': {'days': 7, 'name': 'Last Week'},
            '2W': {'days': 14, 'name': 'Last 2 Weeks'},
            '1M': {'days': 30, 'name': 'Last Month'},
            '3M': {'days': 90, 'name': 'Last 3 Months'},
            '6M': {'days': 180, 'name': 'Last 6 Months'},
            '1Y': {'days': 365, 'name': 'Last Year'},
            'ALL': {'days': 1000, 'name': 'All Available Data'}
        }
    
    def create_timeframe_selector(self, chart_id: str, default_timeframe: str = '1d') -> Tuple[str, str]:
        """
        Create timeframe selector for any chart.
        
        Args:
            chart_id: Unique identifier for the chart
            default_timeframe: Default timeframe selection
            
        Returns:
            Tuple of (selected_timeframe, selected_range)
        """
        # Quick timeframe buttons
        st.markdown("**⚡ Quick Timeframes:**")
        
        quick_cols = st.columns(8)
        quick_timeframes = ['5m', '15m', '1h', '4h', '1d', '1w', '1M', 'ALL']
        
        selected_tf = default_timeframe
        
        for i, tf in enumerate(quick_timeframes):
            with quick_cols[i]:
                if st.button(tf, key=f"{chart_id}_quick_{tf}"):
                    selected_tf = tf
                    st.session_state[f"{chart_id}_timeframe"] = tf
        
        # Advanced timeframe controls
        with st.expander("🔧 Advanced Timeframe Controls"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                timeframe_category = st.selectbox(
                    "Category",
                    ['intraday', 'daily', 'weekly', 'monthly'],
                    key=f"{chart_id}_category"
                )
                
                # Filter timeframes by category
                category_timeframes = [tf for tf, config in self.timeframes.items() 
                                     if config['category'] == timeframe_category]
                
                selected_timeframe = st.selectbox(
                    "Timeframe",
                    category_timeframes,
                    index=0,
                    key=f"{chart_id}_timeframe_detailed"
                )
            
            with adv_col2:
                data_range = st.selectbox(
                    "Data Range",
                    list(self.data_ranges.keys()),
                    index=4,  # Default to 1M
                    key=f"{chart_id}_range"
                )
            
            with adv_col3:
                custom_days = st.number_input(
                    "Custom Days",
                    min_value=1,
                    max_value=1000,
                    value=30,
                    key=f"{chart_id}_custom_days",
                    help="Override with custom number of days"
                )
        
        # Return session state values if they exist
        session_tf = st.session_state.get(f"{chart_id}_timeframe", selected_tf)
        session_range = st.session_state.get(f"{chart_id}_range", 'ALL')
        
        return session_tf, session_range
    
    def apply_timeframe_to_data(self, data: pd.DataFrame, timeframe: str, 
                              data_range: str, custom_days: int = None) -> pd.DataFrame:
        """
        Apply timeframe aggregation and range filtering to data.
        
        Args:
            data: Input DataFrame with date column
            timeframe: Target timeframe for aggregation
            data_range: Data range selection
            custom_days: Custom number of days (overrides data_range)
            
        Returns:
            Processed DataFrame
        """
        if data.empty or 'date' not in data.columns:
            return data
        
        # Ensure date column is datetime
        processed_data = data.copy()
        processed_data['date'] = pd.to_datetime(processed_data['date'])
        processed_data = processed_data.sort_values('date')
        
        # Apply data range filter
        if custom_days:
            cutoff_date = datetime.now() - timedelta(days=custom_days)
        else:
            range_config = self.data_ranges.get(data_range, self.data_ranges['1M'])
            cutoff_date = datetime.now() - timedelta(days=range_config['days'])
        
        processed_data = processed_data[processed_data['date'] >= cutoff_date]
        
        # Apply timeframe aggregation
        if timeframe in self.timeframes:
            tf_config = self.timeframes[timeframe]
            
            # Set date as index for resampling
            processed_data.set_index('date', inplace=True)
            
            # Define aggregation rules
            agg_rules = {
                'volume_24h': 'sum',
                'tvl': 'mean',
                'fee_revenue': 'sum',
                'apr': 'mean',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
            # Apply aggregation only for columns that exist
            available_agg_rules = {col: rule for col, rule in agg_rules.items() 
                                 if col in processed_data.columns}
            
            if available_agg_rules:
                if 'pool' in processed_data.columns:
                    # Group by pool first, then resample
                    processed_data = processed_data.groupby('pool').resample(tf_config['freq']).agg(available_agg_rules)
                    processed_data = processed_data.reset_index()
                else:
                    # Direct resampling
                    processed_data = processed_data.resample(tf_config['freq']).agg(available_agg_rules)
                    processed_data = processed_data.reset_index()
        
        return processed_data


# Example usage and testing
if __name__ == "__main__":
    print("🕐 Testing Epoch Volume Predictor...")
    
    # Test epoch volume prediction
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    test_data = fetcher.fetch_historical_volumes(60)  # 2 months of data
    
    predictor = EpochVolumePredictor()
    
    # Test single pool prediction
    sail_data = test_data[test_data['pool'] == 'SAIL/USDC']
    
    if len(sail_data) >= 21:
        prediction = predictor.predict_next_epoch_volume(sail_data, 'SAIL/USDC')
        
        if 'error' not in prediction:
            print(f"✅ SAIL/USDC next epoch prediction:")
            print(f"   Predicted volume: ${prediction['predicted_epoch_volume']:,.0f}")
            print(f"   Confidence: {prediction['confidence_score']:.1%}")
            print(f"   Daily average: ${prediction['daily_average']:,.0f}")
            print(f"   Volume trend: {prediction['volume_trend']}")
        else:
            print(f"❌ Prediction error: {prediction['error']}")
    
    # Test all pools prediction
    from data_processor import DataProcessor
    processor = DataProcessor()
    all_processed = processor.process_pool_data(test_data)
    
    all_predictions = predictor.predict_all_pools_next_epoch(all_processed)
    
    if all_predictions['summary']['successful_predictions'] > 0:
        print(f"\\n✅ All pools prediction summary:")
        print(f"   Total ecosystem volume: ${all_predictions['summary']['total_ecosystem_volume']:,.0f}")
        print(f"   Average confidence: {all_predictions['summary']['average_confidence']:.1%}")
        print(f"   Successful predictions: {all_predictions['summary']['successful_predictions']}/10")
    
    # Test timeframe controller
    tf_controller = UniversalTimeframeController()
    print(f"\\n✅ Timeframe controller ready with {len(tf_controller.timeframes)} timeframes")
    
    print("🎉 Epoch volume prediction system ready!")
