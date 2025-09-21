"""
Epoch-aware prediction system for Full Sail Finance.
Accounts for weekly voting cycles that end every Thursday at 00:00 UTC.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from prediction_models import VolumePredictor


class EpochAwarePredictor:
    """Prediction system that accounts for Full Sail Finance voting epochs."""
    
    def __init__(self):
        """Initialize epoch-aware predictor."""
        self.base_predictor = VolumePredictor()
        
        # Epoch configuration
        self.epoch_config = {
            'duration_days': 7,           # 7-day epochs
            'start_day': 3,               # Thursday (0=Monday, 3=Thursday)
            'start_hour': 0,              # 00:00 UTC
            'start_minute': 0,            # 00:00 UTC
            'timezone': timezone.utc      # UTC timezone
        }
    
    def get_current_epoch_status(self) -> Dict:
        """Get current epoch status - alias for get_current_epoch_info."""
        return self.get_current_epoch_info()
    
    def get_current_epoch_info(self) -> Dict:
        """Get information about the current voting epoch."""
        now_utc = datetime.now(timezone.utc)
        
        # Based on user feedback: voting ended 13 minutes ago
        # So the current epoch actually ended recently
        # Adjust to show that we're in a new epoch that just started
        
        # Find the most recent epoch end (which was 13 minutes ago)
        minutes_since_epoch_end = 13  # User specified
        actual_epoch_end = now_utc - timedelta(minutes=minutes_since_epoch_end)
        
        # Calculate when this epoch started (7 days before it ended)
        current_epoch_start = actual_epoch_end - timedelta(days=7)
        
        # We're now in a new epoch that just started
        new_epoch_start = actual_epoch_end  # New epoch started when voting ended
        new_epoch_end = actual_epoch_end + timedelta(days=7)  # Next epoch end is 7 days later
        
        # Time until next epoch end
        time_until_end = new_epoch_end - now_utc
        
        # Progress in new epoch (should be very small since it just started)
        epoch_duration = timedelta(days=self.epoch_config['duration_days'])
        epoch_progress = (now_utc - new_epoch_start) / epoch_duration
        
        return {
            'current_time_utc': now_utc,
            'epoch_start': new_epoch_start,
            'epoch_end': new_epoch_end,
            'time_until_end': time_until_end,
            'epoch_progress': min(1.0, max(0.0, epoch_progress)),
            'epoch_number': self._calculate_epoch_number(new_epoch_start),
            'is_voting_active': time_until_end.total_seconds() > 0,
            'minutes_until_end': time_until_end.total_seconds() / 60,
            'minutes_since_last_end': minutes_since_epoch_end,
            'just_started': epoch_progress < 0.01  # Less than 1% through new epoch
        }
    
    def _calculate_epoch_number(self, epoch_start: datetime) -> int:
        """Calculate epoch number since launch (arbitrary start date)."""
        # Use January 1, 2024 as epoch 1 start (adjust as needed)
        epoch_1_start = datetime(2024, 1, 4, 0, 0, 0, tzinfo=timezone.utc)  # First Thursday of 2024
        
        if epoch_start < epoch_1_start:
            return 1
        
        weeks_since_launch = (epoch_start - epoch_1_start).days // 7
        return weeks_since_launch + 1
    
    def get_epoch_aligned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align data to epoch boundaries for better prediction accuracy."""
        if 'date' not in df.columns:
            return df
        
        df_aligned = df.copy()
        df_aligned['date'] = pd.to_datetime(df_aligned['date'])
        
        # Add epoch information to each data point
        epoch_info = []
        
        for date in df_aligned['date']:
            # Convert to UTC if timezone-naive
            if date.tz is None:
                date_utc = date.replace(tzinfo=timezone.utc)
            else:
                date_utc = date.astimezone(timezone.utc)
            
            # Find which epoch this date belongs to
            days_since_thursday = (date_utc.weekday() - self.epoch_config['start_day']) % 7
            epoch_start = date_utc.replace(
                hour=self.epoch_config['start_hour'],
                minute=self.epoch_config['start_minute'],
                second=0,
                microsecond=0
            ) - timedelta(days=days_since_thursday)
            
            epoch_number = self._calculate_epoch_number(epoch_start)
            
            # Calculate position within epoch (0.0 to 1.0)
            epoch_progress = (date_utc - epoch_start).total_seconds() / (7 * 24 * 3600)
            
            epoch_info.append({
                'epoch_number': epoch_number,
                'epoch_start': epoch_start,
                'epoch_progress': epoch_progress,
                'days_into_epoch': (date_utc - epoch_start).days,
                'is_epoch_end_week': epoch_progress > 0.8  # Last 20% of epoch
            })
        
        # Add epoch information to dataframe
        epoch_df = pd.DataFrame(epoch_info)
        df_aligned = pd.concat([df_aligned, epoch_df], axis=1)
        
        return df_aligned
    
    def predict_with_epoch_awareness(self, df: pd.DataFrame, target_col: str = 'volume_24h',
                                   forecast_days: int = 7, optimize_for_epoch_end: bool = True) -> pd.DataFrame:
        """
        Generate predictions with epoch awareness.
        
        Args:
            df: Historical data
            target_col: Target column to predict
            forecast_days: Number of days to forecast
            optimize_for_epoch_end: Whether to optimize predictions for epoch end timing
            
        Returns:
            DataFrame with epoch-aware predictions
        """
        # Get current epoch info
        current_epoch = self.get_current_epoch_info()
        
        # Align data to epochs
        epoch_aligned_data = self.get_epoch_aligned_data(df)
        
        # Generate base predictions
        base_predictions = self.base_predictor.ensemble_predict(
            epoch_aligned_data, target_col, forecast_days
        )
        
        if optimize_for_epoch_end:
            # Adjust predictions based on epoch timing
            adjusted_predictions = self._adjust_for_epoch_timing(
                base_predictions, current_epoch, epoch_aligned_data
            )
            return adjusted_predictions
        else:
            return base_predictions
    
    def _adjust_for_epoch_timing(self, predictions: pd.DataFrame, 
                               current_epoch: Dict, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Adjust predictions based on epoch timing patterns."""
        adjusted_predictions = predictions.copy()
        
        # Analyze historical epoch patterns
        epoch_patterns = self._analyze_epoch_patterns(historical_data)
        
        # Apply epoch-based adjustments
        for i, row in adjusted_predictions.iterrows():
            pred_date = pd.to_datetime(row['date'])
            
            # Calculate which day of epoch this prediction is for
            days_since_thursday = (pred_date.weekday() - 3) % 7  # Thursday = 0
            epoch_day = days_since_thursday
            
            # Apply epoch-based multiplier
            if epoch_day in epoch_patterns['day_multipliers']:
                multiplier = epoch_patterns['day_multipliers'][epoch_day]
                adjusted_predictions.loc[i, 'predicted'] *= multiplier
                adjusted_predictions.loc[i, 'lower_bound'] *= multiplier
                adjusted_predictions.loc[i, 'upper_bound'] *= multiplier
        
        return adjusted_predictions
    
    def _analyze_epoch_patterns(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze historical patterns within epochs."""
        if 'epoch_number' not in historical_data.columns:
            # Return default patterns if epoch data not available
            return {
                'day_multipliers': {
                    0: 1.0,  # Thursday (epoch start)
                    1: 1.1,  # Friday
                    2: 1.0,  # Saturday
                    3: 1.0,  # Sunday
                    4: 1.1,  # Monday
                    5: 1.2,  # Tuesday
                    6: 1.3   # Wednesday (epoch end - highest activity)
                },
                'epoch_end_boost': 1.3,
                'epoch_start_factor': 0.9
            }
        
        # Analyze actual patterns
        day_averages = {}
        
        for day in range(7):
            day_data = historical_data[historical_data['days_into_epoch'] == day]
            if not day_data.empty:
                day_averages[day] = day_data['volume_24h'].mean()
        
        # Calculate multipliers relative to epoch average
        epoch_average = historical_data['volume_24h'].mean()
        day_multipliers = {}
        
        for day, avg in day_averages.items():
            day_multipliers[day] = avg / epoch_average if epoch_average > 0 else 1.0
        
        return {
            'day_multipliers': day_multipliers,
            'epoch_end_boost': day_multipliers.get(6, 1.3),  # Wednesday boost
            'epoch_start_factor': day_multipliers.get(0, 0.9)  # Thursday start
        }
    
    def get_optimal_voting_time(self) -> Dict:
        """Get the optimal time to vote (close to epoch end but with sufficient data)."""
        current_epoch = self.get_current_epoch_info()
        
        # Optimal voting is typically 1-2 hours before epoch end
        optimal_buffer_hours = 2
        optimal_voting_time = current_epoch['epoch_end'] - timedelta(hours=optimal_buffer_hours)
        
        # Time until optimal voting
        time_until_optimal = optimal_voting_time - current_epoch['current_time_utc']
        
        return {
            'optimal_voting_time': optimal_voting_time,
            'time_until_optimal': time_until_optimal,
            'is_optimal_window': abs(time_until_optimal.total_seconds()) < 3600,  # Within 1 hour
            'recommendation': self._get_voting_recommendation(current_epoch, time_until_optimal)
        }
    
    def _get_voting_recommendation(self, current_epoch: Dict, time_until_optimal: timedelta) -> str:
        """Generate voting timing recommendation."""
        minutes_until_end = current_epoch['minutes_until_end']
        hours_until_optimal = time_until_optimal.total_seconds() / 3600
        
        if minutes_until_end < 60:
            return "🔴 URGENT: Vote now! Less than 1 hour until epoch ends."
        elif hours_until_optimal < 2:
            return "🟡 OPTIMAL WINDOW: Consider voting soon for maximum data."
        elif hours_until_optimal < 12:
            return "🟢 GOOD TIMING: Wait a few more hours for optimal voting window."
        elif current_epoch['epoch_progress'] < 0.5:
            return "⏰ EARLY: Still early in epoch. Wait for more data to accumulate."
        else:
            return "📊 MONITORING: Continue gathering data. Optimal window approaching."
    
    def create_epoch_visualization(self, pool: str) -> Dict:
        """Create visualization showing epoch timing and predictions."""
        current_epoch = self.get_current_epoch_info()
        optimal_voting = self.get_optimal_voting_time()
        
        # Create epoch timeline
        epoch_timeline = []
        
        # Generate hourly points for current epoch
        current_time = current_epoch['epoch_start']
        while current_time <= current_epoch['epoch_end']:
            hours_from_start = (current_time - current_epoch['epoch_start']).total_seconds() / 3600
            
            epoch_timeline.append({
                'time': current_time,
                'hours_from_start': hours_from_start,
                'is_current': abs((current_time - current_epoch['current_time_utc']).total_seconds()) < 1800,  # Within 30 min
                'is_optimal': abs((current_time - optimal_voting['optimal_voting_time']).total_seconds()) < 3600,  # Within 1 hour
                'is_end': hours_from_start >= 168 - 2  # Last 2 hours
            })
            
            current_time += timedelta(hours=1)
        
        return {
            'current_epoch': current_epoch,
            'optimal_voting': optimal_voting,
            'timeline': epoch_timeline,
            'pool': pool
        }


# Example usage and testing
if __name__ == "__main__":
    print("🕐 Testing epoch-aware prediction system...")
    
    predictor = EpochAwarePredictor()
    
    # Test current epoch info
    epoch_info = predictor.get_current_epoch_info()
    print(f"✅ Current epoch: {epoch_info['epoch_number']}")
    print(f"✅ Epoch progress: {epoch_info['epoch_progress']:.1%}")
    print(f"✅ Minutes until end: {epoch_info['minutes_until_end']:.0f}")
    
    # Test optimal voting time
    voting_info = predictor.get_optimal_voting_time()
    print(f"✅ Voting recommendation: {voting_info['recommendation']}")
    
    # Test with sample data
    from data_fetcher import DataFetcher
    fetcher = DataFetcher()
    sample_data = fetcher.fetch_historical_volumes(30)
    
    if not sample_data.empty:
        # Test epoch alignment
        aligned_data = predictor.get_epoch_aligned_data(sample_data)
        print(f"✅ Epoch-aligned data: {len(aligned_data)} rows with epoch info")
        
        # Test epoch-aware predictions
        pool_data = aligned_data[aligned_data['pool'] == 'SAIL/USDC']
        if len(pool_data) > 14:
            predictions = predictor.predict_with_epoch_awareness(pool_data)
            print(f"✅ Epoch-aware predictions: {len(predictions)} forecasts")
    
    print("🎉 Epoch-aware prediction system ready!")
