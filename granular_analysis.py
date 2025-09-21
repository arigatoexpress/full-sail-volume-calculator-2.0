"""
Granular time frame analysis module for Full Sail Finance.
Provides detailed analysis capabilities across multiple time frames and metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from visualization import VolumeVisualizer


class GranularAnalyzer:
    """Provides granular analysis capabilities for Full Sail Finance data."""
    
    def __init__(self):
        """Initialize the granular analyzer."""
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.visualizer = VolumeVisualizer()
        
        # Time frame configurations
        self.time_frames = {
            '1H': {'freq': 'H', 'periods': 24 * 7, 'name': '1 Hour'},
            '4H': {'freq': '4H', 'periods': 6 * 7, 'name': '4 Hours'},
            '1D': {'freq': 'D', 'periods': 30, 'name': '1 Day'},
            '1W': {'freq': 'W', 'periods': 52, 'name': '1 Week'},
            '1M': {'freq': 'M', 'periods': 12, 'name': '1 Month'}
        }
        
        # Analysis metrics
        self.metrics = {
            'volume': {'column': 'volume_24h', 'format': '${:,.0f}', 'title': 'Volume'},
            'tvl': {'column': 'tvl', 'format': '${:,.0f}', 'title': 'TVL'},
            'fees': {'column': 'fee_revenue', 'format': '${:,.2f}', 'title': 'Fees'},
            'apr': {'column': 'apr', 'format': '{:.2f}%', 'title': 'APR'},
            'volume_tvl_ratio': {'column': 'volume_tvl_ratio', 'format': '{:.3f}', 'title': 'Volume/TVL'}
        }
    
    def generate_intraday_data(self, pool: str, days: int = 7) -> pd.DataFrame:
        """Generate realistic intraday data for a specific pool."""
        # Get daily data first
        daily_data = self.fetcher.fetch_historical_volumes(days)
        pool_daily = daily_data[daily_data['pool'] == pool].copy()
        
        if pool_daily.empty:
            return pd.DataFrame()
        
        intraday_records = []
        
        for _, day_row in pool_daily.iterrows():
            base_date = pd.to_datetime(day_row['date'])
            daily_volume = day_row['volume_24h']
            daily_tvl = day_row['tvl']
            daily_fees = day_row['fee_revenue']
            
            # Generate 24 hourly data points
            for hour in range(24):
                timestamp = base_date + timedelta(hours=hour)
                
                # Realistic intraday patterns
                hour_factor = self._get_hourly_trading_factor(hour)
                
                # Add some randomness
                random_factor = np.random.normal(1.0, 0.2)
                
                # Calculate hourly metrics
                hourly_volume = (daily_volume / 24) * hour_factor * random_factor
                hourly_tvl = daily_tvl * np.random.normal(1.0, 0.05)  # TVL varies slightly
                hourly_fees = (daily_fees / 24) * hour_factor * random_factor
                
                intraday_records.append({
                    'timestamp': timestamp,
                    'pool': pool,
                    'volume_1h': max(0, hourly_volume),
                    'tvl': max(0, hourly_tvl),
                    'fee_revenue_1h': max(0, hourly_fees),
                    'apr_1h': (hourly_fees * 24 * 365) / (hourly_tvl + 1) * 100 if hourly_tvl > 0 else 0
                })
        
        return pd.DataFrame(intraday_records)
    
    def _get_hourly_trading_factor(self, hour: int) -> float:
        """Get trading activity factor based on hour of day (UTC)."""
        # Simulate realistic trading patterns
        # Higher activity during US and Asian trading hours
        
        if 0 <= hour <= 6:    # Asian session
            return 1.2
        elif 6 <= hour <= 12: # European morning
            return 0.8
        elif 12 <= hour <= 18: # US session
            return 1.5
        else:                 # Evening/night
            return 0.6
    
    def analyze_time_frame(self, pool: str, time_frame: str, 
                          metric: str = 'volume', periods: int = None) -> Dict:
        """
        Analyze a specific pool across a given time frame.
        
        Args:
            pool: Pool to analyze
            time_frame: Time frame ('1H', '4H', '1D', '1W', '1M')
            metric: Metric to analyze ('volume', 'tvl', 'fees', 'apr')
            periods: Number of periods to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if time_frame not in self.time_frames:
            raise ValueError(f"Invalid time frame. Choose from: {list(self.time_frames.keys())}")
        
        if metric not in self.metrics:
            raise ValueError(f"Invalid metric. Choose from: {list(self.metrics.keys())}")
        
        tf_config = self.time_frames[time_frame]
        metric_config = self.metrics[metric]
        
        if periods is None:
            periods = tf_config['periods']
        
        # Get appropriate data
        if time_frame in ['1H', '4H']:
            # Use intraday data
            data = self.generate_intraday_data(pool, days=periods//24 + 1)
            if data.empty:
                return {'error': f'No data available for pool {pool}'}
            
            # Aggregate to desired time frame
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            if time_frame == '1H':
                agg_data = data.resample('H').agg({
                    'volume_1h': 'sum',
                    'tvl': 'mean',
                    'fee_revenue_1h': 'sum',
                    'apr_1h': 'mean'
                }).tail(periods)
                agg_data['metric_value'] = agg_data['volume_1h'] if metric == 'volume' else agg_data[f'{metric}_1h']
            else:  # 4H
                agg_data = data.resample('4H').agg({
                    'volume_1h': 'sum',
                    'tvl': 'mean', 
                    'fee_revenue_1h': 'sum',
                    'apr_1h': 'mean'
                }).tail(periods)
                agg_data['metric_value'] = agg_data['volume_1h'] if metric == 'volume' else agg_data[f'{metric}_1h']
        
        else:
            # Use daily data
            daily_data = self.fetcher.fetch_historical_volumes(periods * 2)  # Get extra data for safety
            pool_data = daily_data[daily_data['pool'] == pool].copy()
            
            if pool_data.empty:
                return {'error': f'No data available for pool {pool}'}
            
            pool_data['date'] = pd.to_datetime(pool_data['date'])
            pool_data.set_index('date', inplace=True)
            
            # Aggregate to desired time frame
            if time_frame == '1D':
                agg_data = pool_data.tail(periods)
                agg_data['metric_value'] = agg_data[metric_config['column']]
            elif time_frame == '1W':
                agg_data = pool_data.resample('W').agg({
                    'volume_24h': 'sum',
                    'tvl': 'mean',
                    'fee_revenue': 'sum',
                    'apr': 'mean'
                }).tail(periods)
                agg_data['metric_value'] = agg_data[metric_config['column']]
            else:  # 1M
                agg_data = pool_data.resample('M').agg({
                    'volume_24h': 'sum',
                    'tvl': 'mean',
                    'fee_revenue': 'sum',
                    'apr': 'mean'
                }).tail(periods)
                agg_data['metric_value'] = agg_data[metric_config['column']]
        
        # Calculate statistics
        stats = self._calculate_time_frame_stats(agg_data['metric_value'])
        
        # Identify patterns
        patterns = self._identify_patterns(agg_data['metric_value'], time_frame)
        
        return {
            'pool': pool,
            'time_frame': time_frame,
            'metric': metric,
            'data': agg_data,
            'statistics': stats,
            'patterns': patterns,
            'config': {
                'tf_name': tf_config['name'],
                'metric_name': metric_config['title'],
                'format': metric_config['format']
            }
        }
    
    def _calculate_time_frame_stats(self, series: pd.Series) -> Dict:
        """Calculate comprehensive statistics for a time series."""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'cv': series.std() / series.mean() if series.mean() != 0 else 0,  # Coefficient of variation
            'trend': self._calculate_trend(series),
            'volatility': series.pct_change().std() * 100  # Percentage volatility
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction."""
        if len(series) < 2:
            return 'insufficient_data'
        
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        
        change_pct = (second_half - first_half) / first_half * 100 if first_half != 0 else 0
        
        if change_pct > 5:
            return 'strong_upward'
        elif change_pct > 1:
            return 'upward'
        elif change_pct < -5:
            return 'strong_downward'
        elif change_pct < -1:
            return 'downward'
        else:
            return 'sideways'
    
    def _identify_patterns(self, series: pd.Series, time_frame: str) -> Dict:
        """Identify patterns in the time series."""
        patterns = {}
        
        # Cyclical patterns
        if time_frame in ['1H', '4H']:
            patterns['daily_cycle'] = self._detect_daily_cycle(series)
        elif time_frame == '1D':
            patterns['weekly_cycle'] = self._detect_weekly_cycle(series)
        
        # Volatility clustering
        returns = series.pct_change().dropna()
        if len(returns) > 10:
            patterns['volatility_clustering'] = self._detect_volatility_clustering(returns)
        
        # Support and resistance levels
        patterns['support_resistance'] = self._find_support_resistance(series)
        
        return patterns
    
    def _detect_daily_cycle(self, series: pd.Series) -> Dict:
        """Detect daily cyclical patterns in intraday data."""
        if len(series) < 24:
            return {'detected': False}
        
        # Group by hour of day
        hourly_avg = series.groupby(series.index.hour).mean()
        
        peak_hour = hourly_avg.idxmax()
        trough_hour = hourly_avg.idxmin()
        
        return {
            'detected': True,
            'peak_hour': peak_hour,
            'trough_hour': trough_hour,
            'peak_value': hourly_avg.max(),
            'trough_value': hourly_avg.min(),
            'amplitude': (hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean() * 100
        }
    
    def _detect_weekly_cycle(self, series: pd.Series) -> Dict:
        """Detect weekly cyclical patterns in daily data."""
        if len(series) < 7:
            return {'detected': False}
        
        # Group by day of week
        weekly_avg = series.groupby(series.index.dayofweek).mean()
        
        peak_day = weekly_avg.idxmax()
        trough_day = weekly_avg.idxmin()
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'detected': True,
            'peak_day': day_names[peak_day],
            'trough_day': day_names[trough_day],
            'peak_value': weekly_avg.max(),
            'trough_value': weekly_avg.min(),
            'amplitude': (weekly_avg.max() - weekly_avg.min()) / weekly_avg.mean() * 100
        }
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> Dict:
        """Detect volatility clustering in returns."""
        abs_returns = returns.abs()
        
        # Calculate autocorrelation of squared returns (ARCH effect)
        squared_returns = returns ** 2
        autocorr = squared_returns.autocorr(lag=1)
        
        return {
            'detected': autocorr > 0.1,  # Threshold for significance
            'autocorr_lag1': autocorr,
            'avg_volatility': abs_returns.mean() * 100,
            'volatility_persistence': 'high' if autocorr > 0.3 else 'medium' if autocorr > 0.1 else 'low'
        }
    
    def _find_support_resistance(self, series: pd.Series) -> Dict:
        """Find potential support and resistance levels."""
        if len(series) < 10:
            return {'support': None, 'resistance': None}
        
        # Simple approach: use quantiles as support/resistance
        support = series.quantile(0.2)  # 20th percentile as support
        resistance = series.quantile(0.8)  # 80th percentile as resistance
        
        # Count how many times price touched these levels (within 5%)
        support_touches = ((series >= support * 0.95) & (series <= support * 1.05)).sum()
        resistance_touches = ((series >= resistance * 0.95) & (series <= resistance * 1.05)).sum()
        
        return {
            'support': support,
            'resistance': resistance,
            'support_touches': support_touches,
            'resistance_touches': resistance_touches,
            'range_pct': (resistance - support) / series.mean() * 100
        }
    
    def create_granular_chart(self, analysis_result: Dict) -> go.Figure:
        """Create detailed chart for granular analysis."""
        if 'error' in analysis_result:
            fig = go.Figure()
            fig.add_annotation(text=analysis_result['error'], 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        data = analysis_result['data']
        config = analysis_result['config']
        stats = analysis_result['statistics']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"{config['metric_name']} - {config['tf_name']} Chart",
                "Statistical Analysis"
            ),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Main time series chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['metric_value'],
                mode='lines+markers',
                name=config['metric_name'],
                line=dict(color=self.visualizer.colors['primary'], width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{config['metric_name']}</b><br>" +
                            "Time: %{x}<br>" +
                            f"Value: {config['format']}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add trend line
        x_numeric = np.arange(len(data))
        z = np.polyfit(x_numeric, data['metric_value'], 1)
        trend_line = np.poly1d(z)(x_numeric)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color=self.visualizer.colors['warning'], width=1, dash='dash'),
                hovertemplate="Trend: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add support/resistance levels if available
        if 'support_resistance' in analysis_result['patterns']:
            sr = analysis_result['patterns']['support_resistance']
            if sr['support'] is not None:
                fig.add_hline(
                    y=sr['support'], 
                    line_dash="dot", 
                    line_color=self.visualizer.colors['success'],
                    annotation_text="Support",
                    row=1, col=1
                )
            if sr['resistance'] is not None:
                fig.add_hline(
                    y=sr['resistance'], 
                    line_dash="dot", 
                    line_color=self.visualizer.colors['warning'],
                    annotation_text="Resistance", 
                    row=1, col=1
                )
        
        # Statistics visualization (box plot)
        fig.add_trace(
            go.Box(
                y=data['metric_value'],
                name="Distribution",
                boxpoints='outliers',
                marker_color=self.visualizer.colors['info'],
                hovertemplate="<b>Statistics</b><br>" +
                            f"Q1: {config['format']}<br>" +
                            f"Median: {config['format']}<br>" +
                            f"Q3: {config['format']}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Granular Analysis: {analysis_result['pool']} - {config['tf_name']}",
                x=0.5,
                font=dict(size=18, color=self.visualizer.colors['text'])
            ),
            height=700,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.visualizer.colors['text'])
        )
        
        # Add statistics text annotation
        stats_text = (f"Mean: {config['format'].format(stats['mean'])}<br>" +
                     f"Std: {config['format'].format(stats['std'])}<br>" +
                     f"Trend: {stats['trend']}<br>" +
                     f"Volatility: {stats['volatility']:.2f}%")
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color=self.visualizer.colors['text']),
            bgcolor="rgba(0, 212, 255, 0.1)",
            bordercolor=self.visualizer.colors['primary'],
            borderwidth=1
        )
        
        return fig
    
    def compare_pools_granular(self, pools: List[str], time_frame: str, 
                              metric: str = 'volume', periods: int = 30) -> Dict:
        """Compare multiple pools across the same time frame and metric."""
        results = {}
        
        for pool in pools:
            try:
                analysis = self.analyze_time_frame(pool, time_frame, metric, periods)
                if 'error' not in analysis:
                    results[pool] = analysis
            except Exception as e:
                print(f"Error analyzing {pool}: {e}")
        
        if not results:
            return {'error': 'No valid pool data found'}
        
        # Generate comparison insights
        comparison = self._generate_pool_comparison(results, metric)
        
        return {
            'pools': results,
            'comparison': comparison,
            'time_frame': time_frame,
            'metric': metric
        }
    
    def _generate_pool_comparison(self, results: Dict, metric: str) -> Dict:
        """Generate comparison insights across pools."""
        pool_stats = {}
        
        for pool, analysis in results.items():
            stats = analysis['statistics']
            pool_stats[pool] = {
                'mean': stats['mean'],
                'volatility': stats['volatility'],
                'trend': stats['trend'],
                'cv': stats['cv']
            }
        
        # Find best/worst performers
        best_performer = max(pool_stats.items(), key=lambda x: x[1]['mean'])
        worst_performer = min(pool_stats.items(), key=lambda x: x[1]['mean'])
        most_volatile = max(pool_stats.items(), key=lambda x: x[1]['volatility'])
        least_volatile = min(pool_stats.items(), key=lambda x: x[1]['volatility'])
        
        return {
            'best_performer': {'pool': best_performer[0], 'value': best_performer[1]['mean']},
            'worst_performer': {'pool': worst_performer[0], 'value': worst_performer[1]['mean']},
            'most_volatile': {'pool': most_volatile[0], 'volatility': most_volatile[1]['volatility']},
            'least_volatile': {'pool': least_volatile[0], 'volatility': least_volatile[1]['volatility']},
            'pool_count': len(pool_stats)
        }


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Testing granular analysis framework...")
    
    analyzer = GranularAnalyzer()
    
    # Test single pool analysis
    result = analyzer.analyze_time_frame('SAIL/USDC', '1D', 'volume', 30)
    
    if 'error' not in result:
        print(f"✅ Analysis completed for {result['pool']}")
        print(f"   Time frame: {result['config']['tf_name']}")
        print(f"   Metric: {result['config']['metric_name']}")
        print(f"   Mean: {result['statistics']['mean']:,.0f}")
        print(f"   Trend: {result['statistics']['trend']}")
    else:
        print(f"❌ Analysis failed: {result['error']}")
    
    # Test pool comparison
    comparison = analyzer.compare_pools_granular(['SAIL/USDC', 'SUI/USDC', 'IKA/SUI'], '1D', 'volume', 20)
    
    if 'error' not in comparison:
        print(f"\n✅ Pool comparison completed:")
        print(f"   Best performer: {comparison['comparison']['best_performer']['pool']}")
        print(f"   Most volatile: {comparison['comparison']['most_volatile']['pool']}")
    
    print("\n🎉 Granular analysis framework ready!")
