"""
🚀 ENHANCED FULL SAIL VOLUME CALCULATOR 2.0
Enhanced version with simplified navigation, onboarding, and AI chat assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import altair as alt
import json
import os
from typing import Dict, List, Optional

# Import custom modules
from data_fetcher import DataFetcher
from data_processor import DataProcessor
from prediction_models import VolumePredictor
from visualization import VolumeVisualizer
from granular_analysis import GranularAnalyzer
from backtesting import ModelBacktester
from macro_analysis import MacroAssetAnalyzer
from trading_view import TradingViewInterface
from epoch_predictor import EpochAwarePredictor
from advanced_features import (
    LiquidityIntelligence, RealTimeAlerts, SocialFeatures, 
    GameificationFeatures, AdvancedVisualization, AdvancedDashboard
)
from multi_asset_analyzer import MultiAssetDataFetcher, UniversalChartInterface, UniversalTechnicalAnalysis
from epoch_volume_predictor import EpochVolumePredictor, UniversalTimeframeController
from live_market_data import LiveMarketData
from robust_live_data import RobustLiveDataStreamer
from comprehensive_data_aggregator import ComprehensiveDataAggregator
from onchain_analytics import OnChainAnalytics, LegitimateForensicsAnalyzer
from premium_ui import PremiumUIManager
from performance_optimizer import smart_cache
from arbitrage_engine import RealTimeArbitrageEngine, ActionableAIInsights
from multi_source_data_agent import LegitimateDataAgent
from advanced_visualizations import AdvancedPoolVisualizer, PoolAnalyticsEngine
from sui_auth import SuiAuthenticator, DataSourceTracker
from historical_data_manager import HistoricalDataManager
from sui_yield_optimizer import SuiYieldOptimizer
from ai_workbench import AIWorkbench
from vertex_ai_integration import VertexAIIntegration, get_ai_market_insights, get_ai_predictions

# Import enhanced components
from enhanced_navigation import EnhancedNavigation
from ai_chat_assistant import AIChatAssistant
from enhanced_styles import get_enhanced_css
from voting_prep import VotingPrep

# Page configuration
st.set_page_config(
    page_title="Full Sail Volume Calculator 2.0",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply enhanced styles
st.markdown(get_enhanced_css(), unsafe_allow_html=True)

# Apply original styles (keep existing design)
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global theme with animated background */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 107, 53, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 230, 118, 0.1) 0%, transparent 50%),
            linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Epic header with glow effects */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00D4FF 0%, #FF6B35 25%, #00E676 50%, #BB86FC 75%, #FFD600 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 4s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5));
        position: relative;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        opacity: 0.9;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ---- Session state defaults ----
def _init_session_state_defaults():
    defaults = {
        'data_loaded': False,
        'historical_data': pd.DataFrame(columns=['date', 'pool', 'volume_24h', 'price']),
        'show_prediction_modal': False,
        'active_tab': 'dashboard',
        'jump_to_voting_prep': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class EnhancedLiquidityPredictorDashboard:
    """
    🚀 ENHANCED LIQUIDITY PREDICTOR DASHBOARD
    
    Enhanced version with simplified navigation, onboarding flow, and AI chat assistant.
    """
    
    def __init__(self):
        """Initialize enhanced dashboard."""
        # Initialize enhanced components
        self.navigation = EnhancedNavigation()
        self.ai_chat = AIChatAssistant()
        
        # Initialize original components
        self.fetcher = DataFetcher()
        self.processor = DataProcessor()
        self.predictor = VolumePredictor()
        self.visualizer = VolumeVisualizer()
        self.granular_analyzer = GranularAnalyzer()
        self.backtester = ModelBacktester()
        self.macro_analyzer = MacroAssetAnalyzer()
        self.trading_interface = TradingViewInterface()
        self.epoch_predictor = EpochAwarePredictor()
        
        # Advanced features
        self.advanced_dashboard = AdvancedDashboard()
        self.liquidity_intelligence = LiquidityIntelligence()
        self.real_time_alerts = RealTimeAlerts()
        self.social_features = SocialFeatures()
        self.gameification = GameificationFeatures()
        self.advanced_visualization = AdvancedVisualization()
        
        # Multi-asset components
        self.multi_asset_fetcher = MultiAssetDataFetcher()
        self.universal_charts = UniversalChartInterface()
        self.universal_ta = UniversalTechnicalAnalysis()
        self.epoch_volume_predictor = EpochVolumePredictor()
        self.timeframe_controller = UniversalTimeframeController()
        
        # Live data components
        self.live_market = LiveMarketData()
        self.robust_streamer = RobustLiveDataStreamer()
        self.comprehensive_aggregator = ComprehensiveDataAggregator()
        
        # Analytics components
        self.onchain_analytics = OnChainAnalytics()
        self.forensics_analyzer = LegitimateForensicsAnalyzer()
        self.premium_ui = PremiumUIManager()
        self.arbitrage_engine = RealTimeArbitrageEngine()
        self.ai_insights = ActionableAIInsights()
        self.data_agent = LegitimateDataAgent()
        self.advanced_pool_visualizer = AdvancedPoolVisualizer()
        self.pool_analytics_engine = PoolAnalyticsEngine()
        self.sui_authenticator = SuiAuthenticator()
        self.data_source_tracker = DataSourceTracker()
        self.historical_data_manager = HistoricalDataManager()
        self.sui_yield_optimizer = SuiYieldOptimizer()
        self.ai_workbench = AIWorkbench()
        self.vertex_ai = VertexAIIntegration()
        self.voting_prep = VotingPrep()
    
    def render_header(self):
        """Render enhanced header with onboarding check."""
        st.markdown("""
        <div class="main-header">
            🚢 Full Sail Volume Calculator 2.0
        </div>
        <div class="sub-header">
            AI-Powered DeFi Analytics Platform
        </div>
        """, unsafe_allow_html=True)
    
    def render_enhanced_dashboard(self):
        """Render enhanced dashboard with simplified navigation."""
        # Check for onboarding
        if self.navigation.render_onboarding_flow():
            return
        
        # Render enhanced sidebar
        settings = self.navigation.render_enhanced_sidebar()
        
        # Render simplified navigation tabs
        main_tabs = self.navigation.render_simplified_navigation()
        
        # Dashboard Tab
        with main_tabs[0]:
            self.render_enhanced_dashboard_content(settings)
        
        # Analytics Tab
        with main_tabs[1]:
            self.render_enhanced_analytics_content(settings)
        
        # AI Assistant Tab
        with main_tabs[2]:
            self.render_ai_assistant_content()
        
        # Tools & Settings Tab
        with main_tabs[3]:
            self.render_tools_and_settings_content(settings)
    
    def render_enhanced_dashboard_content(self, settings: Dict):
        """Render enhanced dashboard content."""
        st.markdown("### 🏠 Dashboard Overview")

        # Weekly predictions status banner
        weekly = st.session_state.get('weekly_predictions', {})
        epoch_keys = list(weekly.keys())
        if epoch_keys:
            latest_epoch = epoch_keys[-1]
            num_pools = len(weekly[latest_epoch])
            st.success(f"✅ Weekly predictions ready for epoch {latest_epoch}: {num_pools} pools. View in Analytics → 🗳 Voting Prep.")
        else:
            st.info("ℹ️ Weekly predictions not generated yet. Use Analytics → 🗳 Voting Prep to run them.")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Volume (24h)",
                "$2.4M",
                "↗️ 12%",
                help="Total trading volume across all pools"
            )
        
        with col2:
            st.metric(
                "Active Pools",
                "8",
                "↗️ 2",
                help="Number of active liquidity pools"
            )
        
        with col3:
            st.metric(
                "AI Accuracy",
                "94%",
                "↗️ 3%",
                help="Average prediction accuracy"
            )
        
        with col4:
            st.metric(
                "Predictions Today",
                "156",
                "↗️ 23",
                help="Number of predictions generated today"
            )
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                st.session_state.show_prediction_modal = True
                st.rerun()
        
        with action_cols[1]:
            if st.button("📊 View Charts", use_container_width=True):
                st.session_state.active_tab = 'analytics'
                st.rerun()
        
        with action_cols[2]:
            # Voting prep quick jump
            if st.button("🗳 View Weekly Predictions", use_container_width=True):
                # Switch to Analytics → Voting Prep
                st.session_state.active_tab = 'analytics'
                st.session_state.jump_to_voting_prep = True
                st.rerun()
        
        # Recent activity
        st.markdown("### 📈 Recent Activity")
        
        # Pool performance
        if st.session_state.data_loaded and 'pool' in st.session_state.historical_data.columns:
            st.subheader("🏊 Pool Performance")
            fig_comparison = self.visualizer.create_pool_comparison(
                st.session_state.historical_data,
                'volume_24h',
                "Average Daily Volume by Pool"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Smart suggestions
        self.render_smart_suggestions()
    
    def render_enhanced_analytics_content(self, settings: Dict):
        """Render enhanced analytics content."""
        st.markdown("### 📊 Analytics Hub")
        
        # Analytics tabs
        analytics_tabs = st.tabs([
            "📈 Charts & Analysis",
            "🗳 Voting Prep",
            "💰 Live Markets", 
            "⚡ Arbitrage Scanner",
            "🌾 Yield Optimizer"
        ])
        
        with analytics_tabs[0]:
            self.render_charts_hub(settings)
        
        with analytics_tabs[1]:
            self.voting_prep.render()
        
        with analytics_tabs[2]:
            self.render_live_markets()

        with analytics_tabs[3]:
            self.render_arbitrage_scanner()
        
        with analytics_tabs[4]:
            self.render_yield_optimizer()
    
    def render_ai_assistant_content(self):
        """Render AI assistant content."""
        self.ai_chat.render_chat_interface()
    
    def render_tools_and_settings_content(self, settings: Dict):
        """Render tools and settings content."""
        st.markdown("### ⚙️ Tools & Settings")
        
        # Tools tabs
        tools_tabs = st.tabs([
            "🔍 Deep Analysis",
            "📚 Education Center",
            "🏆 Social Features",
            "⚙️ Settings"
        ])
        
        with tools_tabs[0]:
            self.render_deep_analysis()
        
        with tools_tabs[1]:
            self.render_education_center()
        
        with tools_tabs[2]:
            self.render_social_features()
        
        with tools_tabs[3]:
            self.render_settings()
    
    def render_smart_suggestions(self):
        """Render smart suggestions based on user context."""
        suggestions = self.ai_chat.generate_smart_suggestions()
        
        if suggestions:
            st.markdown("### 💡 Smart Suggestions")
            
            for suggestion in suggestions:
                with st.expander(f"💡 {suggestion['title']}"):
                    st.markdown(suggestion['description'])
                    if st.button(f"Try: {suggestion['action']}", key=suggestion['id']):
                        self.ai_chat.execute_suggestion(suggestion)
    
    def render_charts_hub(self, settings: Dict):
        """Render charts and analysis hub."""
        st.markdown("#### 📈 Charts & Technical Analysis")
        
        # Chart controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pool = st.selectbox(
                "Pool",
                ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI'],
                index=0
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ['1h', '4h', '1d', '1w'],
                index=2
            )
        
        with col3:
            chart_type = st.selectbox(
                "Chart Type",
                ['Candlestick', 'Line', 'Area'],
                index=0
            )
        
        # Generate chart
        if st.button("📊 Generate Chart", type="primary"):
            with st.spinner("Generating chart..."):
                # Use existing Altair visualization method
                chart = self.visualizer.create_altair_volume_chart(
                    st.session_state.historical_data,
                    pool,
                    timeframe
                )
                st.altair_chart(chart, width='stretch')
    
    def render_live_markets(self):
        """Render live markets data."""
        st.markdown("#### 💰 Live Market Data")
        
        # Live data metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("SAIL/USDC", "$0.45", "↗️ 2.3%")
        
        with col2:
            st.metric("SUI/USDC", "$1.23", "↗️ 1.8%")
        
        with col3:
            st.metric("IKA/SUI", "$0.12", "↘️ 0.5%")
        
        # Live chart
        if st.button("🔄 Refresh Live Data"):
            with st.spinner("Fetching live data..."):
                # Simulate live data fetch
                st.success("Live data updated!")
    
    def render_arbitrage_scanner(self):
        """Render arbitrage opportunity scanner."""
        st.markdown("#### ⚡ Arbitrage Scanner")
        
        if st.button("🔍 Scan for Opportunities", type="primary"):
            with st.spinner("Scanning for arbitrage opportunities..."):
                # Simulate arbitrage scan
                st.success("Found 3 arbitrage opportunities!")
                
                # Display opportunities
                opportunities = [
                    {"pool": "SAIL/USDC", "dex": "Full Sail", "price": "$0.45", "profit": "2.3%"},
                    {"pool": "SUI/USDC", "dex": "Cetus", "price": "$1.23", "profit": "1.8%"},
                    {"pool": "IKA/SUI", "dex": "Turbos", "price": "$0.12", "profit": "1.2%"}
                ]
                
                for opp in opportunities:
                    st.info(f"💰 {opp['pool']} on {opp['dex']}: {opp['price']} (Profit: {opp['profit']})")
    
    def render_yield_optimizer(self):
        """Render yield farming optimizer."""
        st.markdown("#### 🌾 Yield Farming Optimizer")
        
        if st.button("🎯 Optimize Yields", type="primary"):
            with st.spinner("Optimizing yield farming strategies..."):
                # Simulate yield optimization
                st.success("Yield optimization complete!")
                
                # Display recommendations
                recommendations = [
                    {"pool": "SAIL/USDC", "apr": "24.5%", "risk": "Low", "recommendation": "High"},
                    {"pool": "SUI/USDC", "apr": "18.2%", "risk": "Medium", "recommendation": "Medium"},
                    {"pool": "IKA/SUI", "apr": "31.8%", "risk": "High", "recommendation": "Low"}
                ]
                
                for rec in recommendations:
                    st.info(f"🌾 {rec['pool']}: {rec['apr']} APR (Risk: {rec['risk']}, Recommendation: {rec['recommendation']})")
    
    def render_deep_analysis(self):
        """Render deep analysis tools."""
        st.markdown("#### 🔍 Deep Analysis Tools")
        
        analysis_options = st.multiselect(
            "Select Analysis Types",
            ["Technical Analysis", "Fundamental Analysis", "Sentiment Analysis", "Risk Analysis"],
            default=["Technical Analysis"]
        )
        
        if st.button("🔬 Run Deep Analysis", type="primary"):
            with st.spinner("Running deep analysis..."):
                st.success("Deep analysis complete!")
                st.info("Analysis results will be displayed here.")
    
    def render_education_center(self):
        """Render education center."""
        st.markdown("#### 📚 Education Center")
        
        education_tabs = st.tabs([
            "🎓 DeFi Basics",
            "📊 Technical Analysis",
            "🤖 AI & Predictions",
            "⚠️ Risk Management"
        ])
        
        with education_tabs[0]:
            st.markdown("""
            ### 🎓 DeFi Basics
            
            **What is DeFi?**
            Decentralized Finance (DeFi) is a financial system built on blockchain technology that operates without traditional intermediaries.
            
            **Key Concepts:**
            - **Liquidity Pools**: Pools of tokens that enable trading
            - **Automated Market Makers (AMMs)**: Algorithms that determine prices
            - **Yield Farming**: Earning rewards by providing liquidity
            - **Impermanent Loss**: Risk of providing liquidity
            """)
        
        with education_tabs[1]:
            st.markdown("""
            ### 📊 Technical Analysis
            
            **Technical Indicators:**
            - **Moving Averages**: Smooth out price data to identify trends
            - **RSI**: Relative Strength Index for overbought/oversold conditions
            - **MACD**: Moving Average Convergence Divergence for trend changes
            - **Bollinger Bands**: Volatility and support/resistance levels
            """)
        
        with education_tabs[2]:
            st.markdown("""
            ### 🤖 AI & Predictions
            
            **Our AI Models:**
            - **Prophet**: Facebook's time series forecasting model
            - **ARIMA**: Classical statistical time series model
            - **Ensemble**: Combines multiple models for robust predictions
            
            **Understanding Predictions:**
            - Confidence intervals show prediction uncertainty
            - Historical accuracy helps assess model reliability
            - Multiple timeframes provide different perspectives
            """)
        
        with education_tabs[3]:
            st.markdown("""
            ### ⚠️ Risk Management
            
            **Important Disclaimers:**
            - Predictions are not financial advice
            - Past performance doesn't guarantee future results
            - Always do your own research (DYOR)
            - Never invest more than you can afford to lose
            
            **Risk Factors:**
            - Market volatility
            - Liquidity risks
            - Smart contract risks
            - Regulatory changes
            """)
    
    def render_social_features(self):
        """Render social features."""
        st.markdown("#### 🏆 Social Features")
        
        social_tabs = st.tabs([
            "👥 Community",
            "🏆 Leaderboard",
            "💬 Discussions",
            "📊 Shared Analysis"
        ])
        
        with social_tabs[0]:
            st.markdown("### 👥 Community")
            st.info("Community features coming soon! Connect with other DeFi traders and share insights.")
        
        with social_tabs[1]:
            st.markdown("### 🏆 Leaderboard")
            st.info("Leaderboard features coming soon! Compete with other users and earn rewards.")
        
        with social_tabs[2]:
            st.markdown("### 💬 Discussions")
            st.info("Discussion features coming soon! Share your analysis and learn from others.")
        
        with social_tabs[3]:
            st.markdown("### 📊 Shared Analysis")
            st.info("Shared analysis features coming soon! Collaborate on market analysis.")
    
    def render_settings(self):
        """Render settings and preferences."""
        st.markdown("#### ⚙️ Settings & Preferences")
        
        # User preferences
        st.markdown("### 👤 User Preferences")
        
        experience_level = st.selectbox(
            "Experience Level",
            ["Beginner", "Intermediate", "Expert"],
            index=1
        )
        
        favorite_pools = st.multiselect(
            "Favorite Pools",
            ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI', 'USDZ/USDC', 'USDT/USDC', 'wBTC/USDC', 'ETH/USDC'],
            default=['SAIL/USDC']
        )
        
        # Display preferences
        st.markdown("### 🎨 Display Preferences")
        
        theme = st.selectbox(
            "Theme",
            ["Dark", "Light", "Auto"],
            index=0
        )
        
        show_advanced = st.checkbox(
            "Show Advanced Features",
            value=False,
            help="Enable advanced features and settings"
        )
        
        # Notification preferences
        st.markdown("### 🔔 Notifications")
        
        email_notifications = st.checkbox("Email Notifications", value=True)
        push_notifications = st.checkbox("Push Notifications", value=False)
        prediction_alerts = st.checkbox("Prediction Alerts", value=True)
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    def load_data(self):
        """Load initial data."""
        with st.spinner("Loading data..."):
            # Simulate data loading
            import time
            time.sleep(2)
            
            # Create sample data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            sample_data = pd.DataFrame({
                'date': dates,
                'pool': ['SAIL/USDC'] * len(dates),
                'volume_24h': np.random.normal(1000000, 200000, len(dates)),
                'price': np.random.normal(0.45, 0.05, len(dates))
            })
            
            st.session_state.historical_data = sample_data
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
    
    def run(self):
        """Run the enhanced dashboard application."""
        # Ensure session state keys exist
        _init_session_state_defaults()
        # Render header
        self.render_header()
        
        # Load data if not already loaded
        if not st.session_state.data_loaded:
            self.load_data()
        
        # Render enhanced dashboard
        self.render_enhanced_dashboard()


# Main application entry point
if __name__ == "__main__":
    dashboard = EnhancedLiquidityPredictorDashboard()
    dashboard.run()
