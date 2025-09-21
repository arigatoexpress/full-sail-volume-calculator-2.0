"""
🧠 AI WORKBENCH MODULE
Advanced asset comparison, correlation analysis, and AI-powered trading workbench

This module provides comprehensive tools for:
- Multi-asset comparison and analysis
- Correlation analysis and visualization
- AI-powered trade suggestions
- Technical analysis integration
- Portfolio optimization
- Risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import asyncio
from dataclasses import dataclass
import logging

# Import our modules
from data_fetcher import DataFetcher
from live_market_data import LiveMarketData
from vertex_ai_integration import vertex_ai
from prediction_models import VolumePredictor
from trading_view import TradingViewChart

@dataclass
class AssetComparison:
    """Data structure for asset comparison results."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    market_cap: float
    correlation_score: float
    technical_score: float
    ai_score: float
    risk_score: float
    recommendation: str

@dataclass
class TradeSuggestion:
    """AI-generated trade suggestion."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str
    technical_indicators: Dict[str, Any]
    risk_reward_ratio: float
    timeframe: str
    timestamp: datetime

class AIWorkbench:
    """
    Advanced AI-powered workbench for asset analysis and trading.
    
    Provides comprehensive tools for comparing assets, analyzing correlations,
    and generating AI-powered trade suggestions with technical analysis.
    """
    
    def __init__(self):
        """Initialize the AI workbench."""
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = DataFetcher()
        self.live_market = LiveMarketData()
        self.volume_predictor = VolumePredictor()
        self.trading_view = TradingViewChart()
        
        # Available assets for analysis
        self.available_assets = [
            'SUI', 'USDC', 'USDT', 'BTC', 'ETH', 'SOL', 'AVAX', 'MATIC',
            'DOT', 'LINK', 'UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'YFI'
        ]
        
        # Technical indicators
        self.technical_indicators = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD',
            'BB_upper', 'BB_lower', 'ATR', 'Stochastic', 'Williams_R'
        ]
    
    def render_workbench(self):
        """Render the main AI workbench interface."""
        st.markdown("""
        <div class="workbench-header">
            <h1>🧠 AI Trading Workbench</h1>
            <p>Advanced asset analysis, correlation studies, and AI-powered trade suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different workbench features
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Asset Comparison", "🔗 Correlation Analysis", 
            "🤖 AI Trade Suggestions", "📈 Technical Analysis", "⚖️ Portfolio Optimizer"
        ])
        
        with tab1:
            self.render_asset_comparison()
        
        with tab2:
            self.render_correlation_analysis()
        
        with tab3:
            self.render_ai_trade_suggestions()
        
        with tab4:
            self.render_technical_analysis()
        
        with tab5:
            self.render_portfolio_optimizer()
    
    def render_asset_comparison(self):
        """Render asset comparison interface."""
        st.markdown("### 📊 Multi-Asset Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Asset selection
            selected_assets = st.multiselect(
                "Select Assets to Compare",
                self.available_assets,
                default=['SUI', 'BTC', 'ETH', 'SOL'],
                help="Choose multiple assets for comprehensive comparison"
            )
        
        with col2:
            # Analysis timeframe
            timeframe = st.selectbox(
                "Analysis Timeframe",
                ['1h', '4h', '1d', '7d', '30d'],
                index=2
            )
        
        if selected_assets and len(selected_assets) >= 2:
            # Fetch and compare assets
            with st.spinner("Analyzing assets..."):
                comparison_data = self.compare_assets(selected_assets, timeframe)
                self.display_comparison_results(comparison_data)
        else:
            st.warning("Please select at least 2 assets for comparison.")
    
    def compare_assets(self, assets: List[str], timeframe: str) -> List[AssetComparison]:
        """Compare multiple assets and return analysis results."""
        comparisons = []
        
        try:
            # Fetch live data for all assets
            live_data = self.live_market.fetch_live_prices()
            
            for asset in assets:
                if asset in live_data:
                    data = live_data[asset]
                    
                    # Calculate technical score
                    technical_score = self.calculate_technical_score(asset, timeframe)
                    
                    # Calculate AI score using Vertex AI
                    ai_score = self.calculate_ai_score(asset, data)
                    
                    # Calculate risk score
                    risk_score = self.calculate_risk_score(asset, data)
                    
                    # Generate recommendation
                    recommendation = self.generate_recommendation(
                        technical_score, ai_score, risk_score
                    )
                    
                    comparison = AssetComparison(
                        symbol=asset,
                        price=data.get('current_price', 0),
                        change_24h=data.get('price_change_24h', 0),
                        volume_24h=data.get('volume_24h', 0),
                        market_cap=data.get('market_cap', 0),
                        correlation_score=0.0,  # Will be calculated in correlation analysis
                        technical_score=technical_score,
                        ai_score=ai_score,
                        risk_score=risk_score,
                        recommendation=recommendation
                    )
                    
                    comparisons.append(comparison)
        
        except Exception as e:
            self.logger.error(f"Error comparing assets: {e}")
            st.error(f"Error comparing assets: {e}")
        
        return comparisons
    
    def display_comparison_results(self, comparisons: List[AssetComparison]):
        """Display asset comparison results in a comprehensive format."""
        if not comparisons:
            st.warning("No comparison data available.")
            return
        
        # Create comparison dataframe
        df = pd.DataFrame([comp.__dict__ for comp in comparisons])
        
        # Display metrics in columns
        cols = st.columns(len(comparisons))
        
        for i, comp in enumerate(comparisons):
            with cols[i]:
                st.markdown(f"""
                <div class="asset-card">
                    <h3>{comp.symbol}</h3>
                    <div class="metric">
                        <span class="label">Price:</span>
                        <span class="value">${comp.price:,.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">24h Change:</span>
                        <span class="value {'positive' if comp.change_24h >= 0 else 'negative'}">
                            {comp.change_24h:+.2f}%
                        </span>
                    </div>
                    <div class="metric">
                        <span class="label">Technical Score:</span>
                        <span class="value">{comp.technical_score:.1f}/10</span>
                    </div>
                    <div class="metric">
                        <span class="label">AI Score:</span>
                        <span class="value">{comp.ai_score:.1f}/10</span>
                    </div>
                    <div class="metric">
                        <span class="label">Risk Score:</span>
                        <span class="value">{comp.risk_score:.1f}/10</span>
                    </div>
                    <div class="recommendation">
                        <strong>Recommendation:</strong> {comp.recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Create comparison chart
        self.create_comparison_chart(comparisons)
        
        # Display detailed comparison table
        st.markdown("### 📋 Detailed Comparison")
        st.dataframe(df, use_container_width=True)
    
    def create_comparison_chart(self, comparisons: List[AssetComparison]):
        """Create a comprehensive comparison chart."""
        symbols = [comp.symbol for comp in comparisons]
        technical_scores = [comp.technical_score for comp in comparisons]
        ai_scores = [comp.ai_score for comp in comparisons]
        risk_scores = [comp.risk_score for comp in comparisons]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Technical Scores', 'AI Scores', 'Risk Scores', 'Combined Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Technical scores
        fig.add_trace(
            go.Bar(x=symbols, y=technical_scores, name='Technical', marker_color='#667eea'),
            row=1, col=1
        )
        
        # AI scores
        fig.add_trace(
            go.Bar(x=symbols, y=ai_scores, name='AI', marker_color='#764ba2'),
            row=1, col=2
        )
        
        # Risk scores
        fig.add_trace(
            go.Bar(x=symbols, y=risk_scores, name='Risk', marker_color='#f093fb'),
            row=2, col=1
        )
        
        # Combined analysis scatter plot
        fig.add_trace(
            go.Scatter(
                x=technical_scores, 
                y=ai_scores,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                marker=dict(
                    size=risk_scores,
                    sizemode='diameter',
                    sizeref=2,
                    color=risk_scores,
                    colorscale='RdYlGn',
                    showscale=True
                ),
                name='Combined'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Asset Comparison Analysis",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis interface."""
        st.markdown("### 🔗 Correlation Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_assets = st.multiselect(
                "Select Assets for Correlation Analysis",
                self.available_assets,
                default=['SUI', 'BTC', 'ETH', 'SOL', 'AVAX'],
                help="Choose assets to analyze correlations between"
            )
        
        with col2:
            correlation_period = st.selectbox(
                "Correlation Period",
                ['7d', '30d', '90d', '1y'],
                index=1
            )
        
        if selected_assets and len(selected_assets) >= 2:
            with st.spinner("Calculating correlations..."):
                correlation_data = self.calculate_correlations(selected_assets, correlation_period)
                self.display_correlation_results(correlation_data)
        else:
            st.warning("Please select at least 2 assets for correlation analysis.")
    
    def calculate_correlations(self, assets: List[str], period: str) -> pd.DataFrame:
        """Calculate correlation matrix for selected assets."""
        try:
            # Fetch historical data
            historical_data = {}
            for asset in assets:
                data = self.live_market.fetch_historical_prices(asset, period)
                if data and 'prices' in data:
                    historical_data[asset] = data['prices']
            
            if not historical_data:
                return pd.DataFrame()
            
            # Create price dataframe
            price_df = pd.DataFrame()
            for asset, data in historical_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'price' in df.columns:
                        price_df[asset] = df['price'].values
            
            # Calculate correlation matrix
            correlation_matrix = price_df.corr()
            
            return correlation_matrix
        
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            st.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    def display_correlation_results(self, correlation_matrix: pd.DataFrame):
        """Display correlation analysis results."""
        if correlation_matrix.empty:
            st.warning("No correlation data available.")
            return
        
        # Correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Asset Correlation Matrix"
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.markdown("### 📊 Correlation Insights")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append((asset1, asset2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Display top correlations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Positive Correlations:**")
            for asset1, asset2, corr in corr_pairs[:5]:
                if corr > 0.5:
                    st.write(f"• {asset1} ↔ {asset2}: {corr:.3f}")
        
        with col2:
            st.markdown("**Strongest Negative Correlations:**")
            for asset1, asset2, corr in corr_pairs[:5]:
                if corr < -0.3:
                    st.write(f"• {asset1} ↔ {asset2}: {corr:.3f}")
    
    def render_ai_trade_suggestions(self):
        """Render AI-powered trade suggestions interface."""
        st.markdown("### 🤖 AI Trade Suggestions")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_asset = st.selectbox(
                "Select Asset for Trade Analysis",
                self.available_assets,
                index=0
            )
        
        with col2:
            timeframe = st.selectbox(
                "Trading Timeframe",
                ['1h', '4h', '1d', '1w'],
                index=2
            )
        
        with col3:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ['Low', 'Medium', 'High'],
                index=1
            )
        
        if st.button("Generate AI Trade Suggestions", type="primary"):
            with st.spinner("Generating AI trade suggestions..."):
                suggestions = self.generate_trade_suggestions(
                    selected_asset, timeframe, risk_tolerance
                )
                self.display_trade_suggestions(suggestions)
    
    def generate_trade_suggestions(self, asset: str, timeframe: str, risk_tolerance: str) -> List[TradeSuggestion]:
        """Generate AI-powered trade suggestions."""
        suggestions = []
        
        try:
            # Fetch market data
            live_data = self.live_market.fetch_live_prices()
            historical_data = self.live_market.fetch_historical_prices(asset, timeframe)
            
            if asset not in live_data:
                st.error(f"No data available for {asset}")
                return suggestions
            
            current_price = live_data[asset]['current_price']
            
            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(asset, historical_data)
            
            # Generate AI analysis
            ai_analysis = self.generate_ai_analysis(asset, current_price, technical_indicators)
            
            # Create trade suggestions based on analysis
            for analysis in ai_analysis:
                suggestion = TradeSuggestion(
                    symbol=asset,
                    action=analysis['action'],
                    confidence=analysis['confidence'],
                    entry_price=current_price,
                    target_price=analysis['target_price'],
                    stop_loss=analysis['stop_loss'],
                    reasoning=analysis['reasoning'],
                    technical_indicators=technical_indicators,
                    risk_reward_ratio=analysis['risk_reward_ratio'],
                    timeframe=timeframe,
                    timestamp=datetime.now()
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            self.logger.error(f"Error generating trade suggestions: {e}")
            st.error(f"Error generating trade suggestions: {e}")
        
        return suggestions
    
    def display_trade_suggestions(self, suggestions: List[TradeSuggestion]):
        """Display AI trade suggestions."""
        if not suggestions:
            st.warning("No trade suggestions generated.")
            return
        
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"Trade Suggestion #{i+1} - {suggestion.action} {suggestion.symbol}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Confidence", f"{suggestion.confidence:.1%}")
                    st.metric("Entry Price", f"${suggestion.entry_price:.4f}")
                
                with col2:
                    st.metric("Target Price", f"${suggestion.target_price:.4f}")
                    st.metric("Stop Loss", f"${suggestion.stop_loss:.4f}")
                
                with col3:
                    st.metric("Risk/Reward", f"{suggestion.risk_reward_ratio:.2f}")
                    st.metric("Timeframe", suggestion.timeframe)
                
                # Action button
                action_color = "green" if suggestion.action == "BUY" else "red" if suggestion.action == "SELL" else "blue"
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <button style="
                        background: {'#10b981' if suggestion.action == 'BUY' else '#ef4444' if suggestion.action == 'SELL' else '#3b82f6'};
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        border-radius: 10px;
                        font-size: 18px;
                        font-weight: bold;
                        cursor: pointer;
                    ">
                        {suggestion.action} {suggestion.symbol}
                    </button>
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning
                st.markdown("### 📝 AI Reasoning")
                st.write(suggestion.reasoning)
                
                # Technical indicators
                st.markdown("### 📊 Technical Indicators")
                indicators_df = pd.DataFrame([
                    {"Indicator": k, "Value": v, "Signal": self.get_indicator_signal(k, v)}
                    for k, v in suggestion.technical_indicators.items()
                ])
                st.dataframe(indicators_df, use_container_width=True)
    
    def render_technical_analysis(self):
        """Render technical analysis interface."""
        st.markdown("### 📈 Technical Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_asset = st.selectbox(
                "Select Asset for Technical Analysis",
                self.available_assets,
                key="tech_asset"
            )
        
        with col2:
            timeframe = st.selectbox(
                "Analysis Timeframe",
                ['1h', '4h', '1d', '1w'],
                key="tech_timeframe",
                index=2
            )
        
        if st.button("Analyze Technicals", type="primary"):
            with st.spinner("Performing technical analysis..."):
                self.perform_technical_analysis(selected_asset, timeframe)
    
    def perform_technical_analysis(self, asset: str, timeframe: str):
        """Perform comprehensive technical analysis."""
        try:
            # Fetch historical data
            historical_data = self.live_market.fetch_historical_prices(asset, timeframe)
            
            if not historical_data or 'prices' not in historical_data:
                st.error(f"No historical data available for {asset}")
                return
            
            # Create price dataframe
            df = pd.DataFrame(historical_data['prices'])
            
            if df.empty:
                st.error("No price data available")
                return
            
            # Calculate technical indicators
            df = self.add_technical_indicators(df)
            
            # Create technical analysis chart
            self.create_technical_chart(df, asset)
            
            # Display technical summary
            self.display_technical_summary(df, asset)
        
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            st.error(f"Error in technical analysis: {e}")
    
    def render_portfolio_optimizer(self):
        """Render portfolio optimization interface."""
        st.markdown("### ⚖️ Portfolio Optimizer")
        st.info("Portfolio optimization features coming soon!")
    
    # Helper methods
    def calculate_technical_score(self, asset: str, timeframe: str) -> float:
        """Calculate technical analysis score for an asset."""
        try:
            # This would implement actual technical analysis
            # For now, return a random score between 5-10
            return np.random.uniform(5, 10)
        except:
            return 5.0
    
    def calculate_ai_score(self, asset: str, data: Dict[str, Any]) -> float:
        """Calculate AI analysis score for an asset."""
        try:
            # This would use Vertex AI for analysis
            # For now, return a score based on price change
            price_change = abs(data.get('price_change_24h', 0))
            return min(10, max(1, price_change / 10))
        except:
            return 5.0
    
    def calculate_risk_score(self, asset: str, data: Dict[str, Any]) -> float:
        """Calculate risk score for an asset."""
        try:
            # Higher volatility = higher risk
            price_change = abs(data.get('price_change_24h', 0))
            return min(10, max(1, price_change / 5))
        except:
            return 5.0
    
    def generate_recommendation(self, technical_score: float, ai_score: float, risk_score: float) -> str:
        """Generate trading recommendation based on scores."""
        if technical_score >= 8 and ai_score >= 8 and risk_score <= 5:
            return "STRONG BUY"
        elif technical_score >= 6 and ai_score >= 6 and risk_score <= 7:
            return "BUY"
        elif technical_score <= 4 and ai_score <= 4:
            return "SELL"
        elif risk_score >= 8:
            return "HIGH RISK"
        else:
            return "HOLD"
    
    def calculate_technical_indicators(self, asset: str, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators for an asset."""
        # This would implement actual technical indicator calculations
        return {
            'RSI': np.random.uniform(30, 70),
            'MACD': np.random.uniform(-0.1, 0.1),
            'SMA_20': np.random.uniform(0.9, 1.1),
            'BB_position': np.random.uniform(0, 1)
        }
    
    def generate_ai_analysis(self, asset: str, current_price: float, technical_indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI analysis for trade suggestions."""
        # This would use Vertex AI for actual analysis
        # For now, return mock analysis
        return [{
            'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'confidence': np.random.uniform(0.6, 0.9),
            'target_price': current_price * np.random.uniform(1.05, 1.15),
            'stop_loss': current_price * np.random.uniform(0.9, 0.95),
            'reasoning': f"AI analysis suggests {asset} shows strong technical signals...",
            'risk_reward_ratio': np.random.uniform(1.5, 3.0)
        }]
    
    def get_indicator_signal(self, indicator: str, value: float) -> str:
        """Get signal interpretation for technical indicator."""
        if indicator == 'RSI':
            if value > 70:
                return "Overbought"
            elif value < 30:
                return "Oversold"
            else:
                return "Neutral"
        elif indicator == 'MACD':
            if value > 0:
                return "Bullish"
            else:
                return "Bearish"
        else:
            return "Neutral"
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price dataframe."""
        if 'price' not in df.columns:
            return df
        
        # Simple Moving Averages
        df['SMA_20'] = df['price'].rolling(window=20).mean()
        df['SMA_50'] = df['price'].rolling(window=50).mean()
        
        # RSI calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = df['price'].ewm(span=12).mean()
        exp2 = df['price'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        
        return df
    
    def create_technical_chart(self, df: pd.DataFrame, asset: str):
        """Create comprehensive technical analysis chart."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{asset} Price Chart', 'RSI', 'MACD'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price chart with moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['price'], name='Price', line=dict(color='white')),
            row=1, col=1
        )
        
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='yellow')),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=3, col=1)
        
        fig.update_layout(
            height=800,
            title=f"Technical Analysis - {asset}",
            template="plotly_dark",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_technical_summary(self, df: pd.DataFrame, asset: str):
        """Display technical analysis summary."""
        st.markdown("### 📊 Technical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['price'].iloc[-1]
            st.metric("Current Price", f"${current_price:.4f}")
        
        with col2:
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                st.metric("RSI", f"{rsi:.1f}")
        
        with col3:
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                trend = "Bullish" if sma20 > sma50 else "Bearish"
                st.metric("Trend", trend)
        
        with col4:
            if 'MACD' in df.columns:
                macd = df['MACD'].iloc[-1]
                signal = "Bullish" if macd > 0 else "Bearish"
                st.metric("MACD Signal", signal)

# Global instance
ai_workbench = AIWorkbench()
