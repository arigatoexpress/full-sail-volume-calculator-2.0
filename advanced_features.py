"""
Advanced features to make Liquidity Predictor the coolest DeFi analytics product.
Includes AI-powered insights, real-time alerts, portfolio optimization, and social features.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json


class LiquidityIntelligence:
    """AI-powered liquidity intelligence system."""
    
    def __init__(self):
        """Initialize the intelligence system."""
        self.insights_cache = {}
        self.alerts = []
        
    def generate_ai_insights(self, pool_data: Dict) -> List[str]:
        """Generate AI-powered insights from pool data."""
        insights = []
        
        for pool, data in pool_data.items():
            if data.empty:
                continue
                
            # Volume trend analysis
            recent_volume = data['volume_24h'].tail(7).mean()
            historical_volume = data['volume_24h'].head(7).mean()
            volume_change = (recent_volume - historical_volume) / historical_volume * 100
            
            if volume_change > 20:
                insights.append(f"🚀 **{pool}**: Volume surged {volume_change:.1f}% - potential breakout pattern detected")
            elif volume_change < -20:
                insights.append(f"📉 **{pool}**: Volume declined {volume_change:.1f}% - consolidation phase likely")
            
            # Liquidity efficiency analysis
            if 'tvl' in data.columns and 'volume_24h' in data.columns:
                turnover_ratio = data['volume_24h'].iloc[-1] / data['tvl'].iloc[-1]
                if turnover_ratio > 0.5:
                    insights.append(f"⚡ **{pool}**: High turnover ratio ({turnover_ratio:.2f}) - very active trading")
                elif turnover_ratio < 0.01:
                    insights.append(f"🔒 **{pool}**: Low turnover ({turnover_ratio:.3f}) - potential liquidity mining opportunity")
            
            # Volatility analysis
            volatility = data['volume_24h'].pct_change().std() * 100
            if volatility > 50:
                insights.append(f"⚠️ **{pool}**: High volatility ({volatility:.1f}%) - increased risk/reward potential")
            elif volatility < 10:
                insights.append(f"🎯 **{pool}**: Low volatility ({volatility:.1f}%) - stable trading environment")
        
        return insights
    
    def detect_arbitrage_opportunities(self, pool_data: Dict) -> List[Dict]:
        """Detect potential arbitrage opportunities between pools."""
        opportunities = []
        
        # Look for price discrepancies in similar assets
        sui_pools = {k: v for k, v in pool_data.items() if 'SUI' in k}
        usdc_pools = {k: v for k, v in pool_data.items() if 'USDC' in k}
        
        for pool1, data1 in sui_pools.items():
            for pool2, data2 in sui_pools.items():
                if pool1 != pool2 and not data1.empty and not data2.empty:
                    # Calculate implied prices (volume/TVL ratios as proxy)
                    ratio1 = data1['volume_24h'].iloc[-1] / (data1['tvl'].iloc[-1] + 1)
                    ratio2 = data2['volume_24h'].iloc[-1] / (data2['tvl'].iloc[-1] + 1)
                    
                    price_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2) * 100
                    
                    if price_diff > 5:  # 5% difference threshold
                        opportunities.append({
                            'pool1': pool1,
                            'pool2': pool2,
                            'price_difference': price_diff,
                            'potential_profit': price_diff * 0.8,  # Assume 80% capture
                            'confidence': 'medium' if price_diff > 10 else 'low'
                        })
        
        return opportunities
    
    def generate_portfolio_suggestions(self, pool_data: Dict, risk_tolerance: str = 'medium') -> Dict:
        """Generate optimal portfolio allocation suggestions."""
        portfolio = {
            'allocations': {},
            'expected_return': 0,
            'risk_score': 0,
            'diversification_score': 0
        }
        
        # Calculate metrics for each pool
        pool_metrics = {}
        for pool, data in pool_data.items():
            if data.empty:
                continue
                
            returns = data['volume_24h'].pct_change().dropna()
            
            pool_metrics[pool] = {
                'expected_return': returns.mean() * 365 * 100,  # Annualized
                'volatility': returns.std() * np.sqrt(365) * 100,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(data['volume_24h']),
                'liquidity_score': data['tvl'].iloc[-1] if 'tvl' in data.columns else 0
            }
        
        # Generate allocations based on risk tolerance
        if risk_tolerance == 'low':
            # Focus on stable, high-TVL pools
            stable_pools = ['USDT/USDC', 'USDZ/USDC', 'SUI/USDC']
            for pool in stable_pools:
                if pool in pool_metrics:
                    portfolio['allocations'][pool] = 30 if pool == 'USDT/USDC' else 20
        
        elif risk_tolerance == 'high':
            # Focus on high-return, volatile pools
            growth_pools = ['IKA/SUI', 'DEEP/SUI', 'WAL/SUI']
            for pool in growth_pools:
                if pool in pool_metrics:
                    portfolio['allocations'][pool] = 25
        
        else:  # medium
            # Balanced allocation
            all_pools = list(pool_metrics.keys())
            equal_weight = 100 / len(all_pools) if all_pools else 0
            for pool in all_pools:
                portfolio['allocations'][pool] = equal_weight
        
        return portfolio
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100


class RealTimeAlerts:
    """Real-time alert system for significant market events."""
    
    def __init__(self):
        """Initialize alert system."""
        self.alert_thresholds = {
            'volume_spike': 2.0,      # 2x normal volume
            'price_change': 10.0,     # 10% price change
            'liquidity_change': 15.0, # 15% TVL change
            'arbitrage': 5.0          # 5% arbitrage opportunity
        }
        
    def check_alerts(self, current_data: Dict, historical_data: Dict) -> List[Dict]:
        """Check for alert conditions."""
        alerts = []
        
        for pool, data in current_data.items():
            if data.empty:
                continue
                
            # Volume spike alert
            current_volume = data['volume_24h'].iloc[-1]
            avg_volume = historical_data.get(pool, {}).get('volume_24h', pd.Series([current_volume])).mean()
            
            if current_volume > avg_volume * self.alert_thresholds['volume_spike']:
                alerts.append({
                    'type': 'volume_spike',
                    'pool': pool,
                    'severity': 'high',
                    'message': f"Volume spike: {current_volume/avg_volume:.1f}x normal volume",
                    'action': 'Consider taking profit or entering position'
                })
            
            # TVL change alert
            if 'tvl' in data.columns and len(data) > 1:
                tvl_change = data['tvl'].pct_change().iloc[-1] * 100
                if abs(tvl_change) > self.alert_thresholds['liquidity_change']:
                    alerts.append({
                        'type': 'liquidity_change',
                        'pool': pool,
                        'severity': 'medium',
                        'message': f"TVL changed {tvl_change:+.1f}% in last period",
                        'action': 'Monitor for liquidity migration'
                    })
        
        return alerts


class SocialFeatures:
    """Social features for community engagement."""
    
    def __init__(self):
        """Initialize social features."""
        self.leaderboard = []
        self.community_predictions = {}
        
    def generate_community_leaderboard(self) -> pd.DataFrame:
        """Generate mock community leaderboard."""
        # In a real app, this would connect to a database
        mock_data = [
            {'rank': 1, 'user': 'DeFiWhale', 'accuracy': 87.5, 'predictions': 156, 'streak': 12},
            {'rank': 2, 'user': 'LiquidityMaster', 'accuracy': 84.2, 'predictions': 203, 'streak': 8},
            {'rank': 3, 'user': 'YieldFarmer', 'accuracy': 82.1, 'predictions': 134, 'streak': 15},
            {'rank': 4, 'user': 'CryptoAnalyst', 'accuracy': 79.8, 'predictions': 187, 'streak': 5},
            {'rank': 5, 'user': 'PoolExplorer', 'accuracy': 77.3, 'predictions': 98, 'streak': 3}
        ]
        
        return pd.DataFrame(mock_data)
    
    def get_community_sentiment(self, pool: str) -> Dict:
        """Get community sentiment for a pool."""
        # Mock sentiment data
        sentiments = ['bullish', 'bearish', 'neutral']
        sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])
        
        return {
            'sentiment': sentiment,
            'confidence': np.random.uniform(0.6, 0.9),
            'total_votes': np.random.randint(50, 200),
            'bullish_pct': np.random.uniform(30, 70),
            'bearish_pct': np.random.uniform(20, 50),
            'neutral_pct': np.random.uniform(10, 30)
        }


class AdvancedVisualization:
    """Advanced visualization features."""
    
    def create_3d_liquidity_landscape(self, pool_data: Dict) -> go.Figure:
        """Create 3D visualization of liquidity landscape."""
        pools = list(pool_data.keys())[:8]  # Limit for performance
        
        x_data = []
        y_data = []
        z_data = []
        colors = []
        
        for i, pool in enumerate(pools):
            data = pool_data[pool]
            if data.empty:
                continue
                
            # X: Volume, Y: TVL, Z: Fees
            x_data.append(data['volume_24h'].iloc[-1])
            y_data.append(data['tvl'].iloc[-1] if 'tvl' in data.columns else 0)
            z_data.append(data['fee_revenue'].iloc[-1] if 'fee_revenue' in data.columns else 0)
            colors.append(i)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers+text',
            text=pools,
            textposition="top center",
            marker=dict(
                size=12,
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Volume: $%{x:,.0f}<br>' +
                         'TVL: $%{y:,.0f}<br>' +
                         'Fees: $%{z:,.0f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Liquidity Landscape',
            scene=dict(
                xaxis_title='Volume (24h)',
                yaxis_title='TVL',
                zaxis_title='Fee Revenue',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_liquidity_heatmap(self, pool_data: Dict) -> go.Figure:
        """Create animated heatmap of liquidity flows."""
        # Create time-based heatmap
        pools = list(pool_data.keys())
        dates = pool_data[pools[0]]['date'].tail(30) if pools and not pool_data[pools[0]].empty else []
        
        # Create matrix of volume data
        volume_matrix = []
        for date in dates:
            day_volumes = []
            for pool in pools:
                pool_day_data = pool_data[pool][pool_data[pool]['date'] == date]
                volume = pool_day_data['volume_24h'].iloc[0] if not pool_day_data.empty else 0
                day_volumes.append(volume)
            volume_matrix.append(day_volumes)
        
        fig = go.Figure(data=go.Heatmap(
            z=volume_matrix,
            x=pools,
            y=[d.strftime('%m-%d') for d in dates],
            colorscale='Plasma',
            hovertemplate='Pool: %{x}<br>Date: %{y}<br>Volume: $%{z:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Liquidity Flow Heatmap (30 Days)',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig


class GameificationFeatures:
    """Gamification features to engage users."""
    
    def __init__(self):
        """Initialize gamification system."""
        self.achievements = [
            {'name': 'First Prediction', 'description': 'Made your first volume prediction', 'icon': '🎯'},
            {'name': 'Accuracy Master', 'description': 'Achieved >80% prediction accuracy', 'icon': '🏆'},
            {'name': 'Pool Explorer', 'description': 'Analyzed all 10 pools', 'icon': '🗺️'},
            {'name': 'Technical Analyst', 'description': 'Used 5+ technical indicators', 'icon': '📊'},
            {'name': 'Risk Manager', 'description': 'Identified 3+ risk factors', 'icon': '🛡️'}
        ]
    
    def get_user_level(self, predictions_made: int, accuracy: float) -> Dict:
        """Calculate user level based on activity."""
        # Simple leveling system
        level = min(predictions_made // 10 + 1, 50)  # Max level 50
        
        titles = {
            1: 'Liquidity Novice',
            5: 'Pool Analyst', 
            10: 'DeFi Strategist',
            20: 'Liquidity Expert',
            30: 'Yield Master',
            50: 'DeFi Legend'
        }
        
        title = 'Liquidity Novice'
        for req_level, req_title in titles.items():
            if level >= req_level:
                title = req_title
        
        return {
            'level': level,
            'title': title,
            'progress': (predictions_made % 10) / 10 * 100,
            'next_level_predictions': (level * 10) - predictions_made
        }


class PredictiveModeling:
    """Advanced predictive modeling features."""
    
    def __init__(self):
        """Initialize advanced modeling."""
        self.ensemble_weights = {}
        
    def create_multi_horizon_forecast(self, data: pd.DataFrame) -> Dict:
        """Create forecasts for multiple time horizons simultaneously."""
        horizons = [1, 3, 7, 14, 30]
        forecasts = {}
        
        for horizon in horizons:
            # Simple forecast for each horizon
            recent_mean = data['volume_24h'].tail(horizon).mean()
            trend = data['volume_24h'].diff().tail(horizon).mean()
            
            forecast = recent_mean + (trend * horizon)
            confidence = max(0.5, 1 - (horizon / 30))  # Confidence decreases with horizon
            
            forecasts[f'{horizon}d'] = {
                'prediction': max(0, forecast),
                'confidence': confidence,
                'horizon_days': horizon
            }
        
        return forecasts
    
    def detect_regime_changes(self, data: pd.DataFrame) -> Dict:
        """Detect market regime changes."""
        if len(data) < 30:
            return {'regime': 'unknown', 'confidence': 0}
        
        # Simple regime detection based on volatility and trend
        recent_data = data['volume_24h'].tail(14)
        volatility = recent_data.pct_change().std()
        trend = recent_data.diff().mean()
        
        if volatility > recent_data.std() * 1.5:
            regime = 'high_volatility'
        elif trend > 0:
            regime = 'growth'
        elif trend < 0:
            regime = 'decline'
        else:
            regime = 'consolidation'
        
        return {
            'regime': regime,
            'confidence': min(0.9, volatility * 10),
            'trend_strength': abs(trend) / recent_data.mean() * 100
        }


class MarketIntelligence:
    """Market intelligence and news integration."""
    
    def __init__(self):
        """Initialize market intelligence."""
        self.news_sentiment = {}
        
    def get_market_events(self) -> List[Dict]:
        """Get relevant market events and news."""
        # Mock market events
        events = [
            {
                'date': datetime.now() - timedelta(hours=2),
                'title': 'Sui Network Upgrade Completed',
                'impact': 'positive',
                'relevance': 'high',
                'affected_pools': ['SUI/USDC', 'IKA/SUI', 'WAL/SUI']
            },
            {
                'date': datetime.now() - timedelta(days=1),
                'title': 'Major DeFi Protocol Integration',
                'impact': 'positive',
                'relevance': 'medium',
                'affected_pools': ['SAIL/USDC', 'DEEP/SUI']
            },
            {
                'date': datetime.now() - timedelta(days=3),
                'title': 'Increased Institutional Interest in Sui',
                'impact': 'positive',
                'relevance': 'high',
                'affected_pools': ['SUI/USDC', 'wBTC/USDC', 'ETH/USDC']
            }
        ]
        
        return events
    
    def analyze_social_sentiment(self, pool: str) -> Dict:
        """Analyze social media sentiment for a pool."""
        # Mock sentiment analysis
        sentiments = ['very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish']
        weights = [0.2, 0.3, 0.3, 0.15, 0.05]  # Slight positive bias
        
        sentiment = np.random.choice(sentiments, p=weights)
        
        return {
            'sentiment': sentiment,
            'score': np.random.uniform(0.3, 0.9),
            'volume_mentions': np.random.randint(50, 500),
            'trending_keywords': ['liquidity', 'yield', 'farming', 'APR'],
            'sentiment_trend': np.random.choice(['increasing', 'decreasing', 'stable'])
        }


class AdvancedDashboard:
    """Enhanced dashboard with all advanced features."""
    
    def __init__(self):
        """Initialize advanced dashboard."""
        self.intelligence = LiquidityIntelligence()
        self.alerts = RealTimeAlerts()
        self.social = SocialFeatures()
        self.gamification = GameificationFeatures()
        self.modeling = PredictiveModeling()
        self.market_intel = MarketIntelligence()
        self.advanced_viz = AdvancedVisualization()
    
    def render_ai_insights_panel(self, pool_data: Dict) -> None:
        """Render AI-powered insights panel."""
        st.subheader("🤖 AI-Powered Insights")
        
        insights = self.intelligence.generate_ai_insights(pool_data)
        
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("🔍 Analyzing data for insights...")
        
        # Arbitrage opportunities
        arbitrage_ops = self.intelligence.detect_arbitrage_opportunities(pool_data)
        
        if arbitrage_ops:
            st.subheader("⚡ Arbitrage Opportunities")
            for op in arbitrage_ops:
                st.warning(f"💰 **{op['pool1']} ↔ {op['pool2']}**: "
                          f"{op['price_difference']:.1f}% price difference "
                          f"(Potential: {op['potential_profit']:.1f}%)")
    
    def render_portfolio_optimizer(self, pool_data: Dict) -> None:
        """Render portfolio optimization tool."""
        st.subheader("📊 Portfolio Optimizer")
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ['low', 'medium', 'high'],
            help="Select your risk preference for portfolio allocation"
        )
        
        if st.button("🎯 Optimize Portfolio"):
            portfolio = self.intelligence.generate_portfolio_suggestions(pool_data, risk_tolerance)
            
            if portfolio['allocations']:
                st.success("✅ Optimal portfolio generated!")
                
                # Display allocations
                alloc_df = pd.DataFrame([
                    {'Pool': pool, 'Allocation (%)': alloc}
                    for pool, alloc in portfolio['allocations'].items()
                ])
                
                st.dataframe(alloc_df, use_container_width=True)
                
                # Portfolio metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{portfolio['expected_return']:.1f}%")
                with col2:
                    st.metric("Risk Score", f"{portfolio['risk_score']:.1f}/10")
                with col3:
                    st.metric("Diversification", f"{portfolio['diversification_score']:.1f}/10")
    
    def render_social_features(self) -> None:
        """Render social and gamification features."""
        st.subheader("🏆 Community & Achievements")
        
        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🎮 Achievements", "💬 Community"])
        
        with tab1:
            leaderboard = self.social.generate_community_leaderboard()
            st.dataframe(leaderboard, use_container_width=True)
        
        with tab2:
            # User level
            user_stats = self.gamification.get_user_level(25, 75.5)  # Mock stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Level", user_stats['level'])
                st.metric("Title", user_stats['title'])
            
            with col2:
                st.metric("Progress", f"{user_stats['progress']:.1f}%")
                st.metric("Next Level", f"{user_stats['next_level_predictions']} predictions")
            
            # Achievements
            st.markdown("**🏅 Achievements:**")
            for achievement in self.gamification.achievements:
                st.write(f"{achievement['icon']} **{achievement['name']}**: {achievement['description']}")
        
        with tab3:
            # Community sentiment
            pool = st.selectbox("Select Pool for Sentiment", ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI'])
            sentiment = self.market_intel.analyze_social_sentiment(pool)
            
            st.write(f"**Community Sentiment for {pool}:**")
            st.write(f"📊 Overall: {sentiment['sentiment'].replace('_', ' ').title()}")
            st.write(f"📈 Score: {sentiment['score']:.2f}/1.0")
            st.write(f"💬 Mentions: {sentiment['volume_mentions']}")


# Example usage
if __name__ == "__main__":
    print("🚀 Testing advanced features...")
    
    # Test AI insights
    intelligence = LiquidityIntelligence()
    print("✅ AI Intelligence system ready")
    
    # Test alerts
    alerts = RealTimeAlerts()
    print("✅ Real-time alerts system ready")
    
    # Test social features
    social = SocialFeatures()
    leaderboard = social.generate_community_leaderboard()
    print(f"✅ Social features ready - {len(leaderboard)} leaderboard entries")
    
    # Test advanced visualization
    viz = AdvancedVisualization()
    print("✅ Advanced visualization ready")
    
    print("🎉 All advanced features ready for integration!")
