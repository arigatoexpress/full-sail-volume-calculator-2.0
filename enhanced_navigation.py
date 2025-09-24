"""
🎨 ENHANCED NAVIGATION & ONBOARDING SYSTEM
Simplified navigation structure and user onboarding flow
"""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime
import json

class EnhancedNavigation:
    """Enhanced navigation system with simplified structure and onboarding."""
    
    def __init__(self):
        """Initialize enhanced navigation system."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for enhanced navigation."""
        # Navigation state
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'dashboard'
        
        # Onboarding state
        if 'onboarding_complete' not in st.session_state:
            st.session_state.onboarding_complete = False
        if 'onboarding_step' not in st.session_state:
            st.session_state.onboarding_step = 1
        
        # User preferences
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'experience_level': 'intermediate',
                'favorite_pools': ['SAIL/USDC'],
                'default_timeframe': '30d',
                'preferred_model': 'Ensemble',
                'show_advanced_features': False,
                'theme': 'dark'
            }
        
        # Smart defaults
        if 'smart_defaults' not in st.session_state:
            st.session_state.smart_defaults = self.get_smart_defaults()
    
    def get_smart_defaults(self) -> Dict:
        """Get intelligent default settings based on user behavior."""
        return {
            'pool': 'SAIL/USDC',  # Most popular pool
            'timeframe': '30d',   # Optimal for predictions
            'model': 'Ensemble',  # Best performing model
            'confidence': 0.95,   # Standard confidence level
            'show_volume': True,
            'show_indicators': True,
            'auto_refresh': True,
            'notifications': True
        }
    
    def render_onboarding_flow(self):
        """Render the onboarding flow for new users."""
        if not st.session_state.onboarding_complete:
            self.render_onboarding_step()
            return True
        return False
    
    def render_onboarding_step(self):
        """Render current onboarding step."""
        step = st.session_state.onboarding_step
        
        # Onboarding container
        with st.container():
            st.markdown("""
            <div class="onboarding-container">
                <div class="onboarding-header">
                    <h1>🚢 Welcome to Full Sail Volume Calculator</h1>
                    <p>Let's get you started in just 3 steps!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress indicator
            progress = step / 3
            st.progress(progress)
            st.markdown(f"**Step {step} of 3**")
            
            if step == 1:
                self.render_welcome_step()
            elif step == 2:
                self.render_pool_selection_step()
            elif step == 3:
                self.render_first_prediction_step()
    
    def render_welcome_step(self):
        """Render welcome and role selection step."""
        st.markdown("""
        <div class="onboarding-step">
            <h2>🎯 Choose Your Experience Level</h2>
            <p>This helps us customize the interface for you.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🆕 **Beginner**\n\nSimple interface\nBasic features\nGuided experience", 
                        use_container_width=True, key="beginner"):
                st.session_state.user_preferences['experience_level'] = 'beginner'
                st.session_state.onboarding_step = 2
                st.rerun()
        
        with col2:
            if st.button("⚡ **Intermediate**\n\nProfessional tools\nAdvanced charts\nAI insights", 
                        use_container_width=True, key="intermediate"):
                st.session_state.user_preferences['experience_level'] = 'intermediate'
                st.session_state.onboarding_step = 2
                st.rerun()
        
        with col3:
            if st.button("🚀 **Expert**\n\nFull features\nCustom analysis\nAdvanced AI", 
                        use_container_width=True, key="expert"):
                st.session_state.user_preferences['experience_level'] = 'expert'
                st.session_state.onboarding_step = 2
                st.rerun()
    
    def render_pool_selection_step(self):
        """Render pool selection step."""
        st.markdown("""
        <div class="onboarding-step">
            <h2>🏊 Select Your Favorite Pool</h2>
            <p>Choose the liquidity pool you're most interested in analyzing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pool selection
        pool_options = [
            'SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI', 
            'USDZ/USDC', 'USDT/USDC', 'wBTC/USDC', 'ETH/USDC'
        ]
        
        selected_pool = st.selectbox(
            "Choose your primary pool:",
            pool_options,
            index=0,
            help="You can change this later in settings"
        )
        
        st.session_state.user_preferences['favorite_pools'] = [selected_pool]
        st.session_state.smart_defaults['pool'] = selected_pool
        
        # Quick pool info
        pool_info = {
            'SAIL/USDC': 'Full Sail Finance native token with USDC',
            'SUI/USDC': 'Sui blockchain native token with USDC',
            'IKA/SUI': 'IKA token paired with Sui',
            'ALKIMI/SUI': 'ALKIMI token paired with Sui'
        }
        
        st.info(f"💡 **{selected_pool}**: {pool_info.get(selected_pool, 'Popular trading pair')}")
        
        if st.button("Continue to Predictions", type="primary"):
            st.session_state.onboarding_step = 3
            st.rerun()
    
    def render_first_prediction_step(self):
        """Render first prediction generation step."""
        st.markdown("""
        <div class="onboarding-step">
            <h2>🔮 Generate Your First Prediction</h2>
            <p>Let's create your first volume prediction to see how it works!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction settings
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Forecast Days", 1, 14, 7)
            model = st.selectbox("Prediction Model", 
                               ["Ensemble", "Prophet", "ARIMA"], 
                               index=0)
        
        with col2:
            confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            st.markdown(f"**Selected Pool**: {st.session_state.smart_defaults['pool']}")
        
        # Generate prediction button
        if st.button("🚀 Generate My First Prediction", type="primary", use_container_width=True):
            # Simulate prediction generation
            with st.spinner("Generating your first prediction..."):
                import time
                time.sleep(2)  # Simulate processing
                
                # Show success
                st.success("🎉 Congratulations! Your first prediction is ready!")
                
                # Complete onboarding
                st.session_state.onboarding_complete = True
                st.session_state.onboarding_step = 1
                
                # Show next steps
                st.markdown("""
                <div class="onboarding-success">
                    <h3>🎯 What's Next?</h3>
                    <ul>
                        <li>📊 Explore the Dashboard for key metrics</li>
                        <li>📈 Check out Analytics for detailed charts</li>
                        <li>🤖 Try the AI Assistant for insights</li>
                        <li>⚙️ Customize your experience in Settings</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Start Exploring", type="primary"):
                    st.rerun()
    
    def render_simplified_navigation(self):
        """Render simplified navigation with 4 main sections."""
        # Main navigation tabs
        main_tabs = st.tabs([
            "🏠 Dashboard",
            "📊 Analytics", 
            "🤖 AI Assistant",
            "⚙️ Tools & Settings"
        ])
        
        return main_tabs
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with smart defaults and quick actions."""
        with st.sidebar:
            # User info
            st.markdown("### 👤 User Profile")
            experience_level = st.session_state.user_preferences['experience_level']
            st.markdown(f"**Level**: {experience_level.title()}")
            st.markdown(f"**Favorite Pool**: {st.session_state.smart_defaults['pool']}")
            
            # Quick actions
            st.markdown("### 🚀 Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.data_loaded = False
                st.rerun()
            
            if st.button("🔮 Generate Prediction", use_container_width=True):
                st.session_state.show_prediction_modal = True
                st.rerun()
            
            if st.button("📊 View Charts", use_container_width=True):
                st.session_state.active_tab = 'analytics'
                st.rerun()
            
            # Smart settings
            st.markdown("### ⚙️ Smart Settings")
            
            # Pool selection with smart default
            selected_pool = st.selectbox(
                "🏊 Pool",
                ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI', 
                 'USDZ/USDC', 'USDT/USDC', 'wBTC/USDC', 'ETH/USDC'],
                index=0,  # Default to SAIL/USDC
                help="Select pool to analyze"
            )
            
            # Timeframe with smart default
            timeframe = st.selectbox(
                "⏰ Timeframe",
                ['7d', '14d', '30d', '60d', '90d'],
                index=2,  # Default to 30d
                help="Historical data period"
            )
            
            # Model selection with smart default
            model = st.selectbox(
                "🤖 Model",
                ['Ensemble', 'Prophet', 'ARIMA'],
                index=0,  # Default to Ensemble
                help="Prediction model to use"
            )
            
            # Advanced features toggle
            if st.session_state.user_preferences['experience_level'] in ['intermediate', 'expert']:
                st.markdown("### 🔧 Advanced Features")
                
                show_advanced = st.checkbox(
                    "Show Advanced Options",
                    value=st.session_state.user_preferences['show_advanced_features'],
                    help="Enable advanced features and settings"
                )
                st.session_state.user_preferences['show_advanced_features'] = show_advanced
                
                if show_advanced:
                    confidence = st.slider(
                        "Confidence Level",
                        0.80, 0.99, 0.95, 0.01,
                        help="Prediction confidence interval"
                    )
                    
                    auto_refresh = st.checkbox(
                        "Auto Refresh",
                        value=st.session_state.smart_defaults['auto_refresh'],
                        help="Automatically refresh data"
                    )
                    st.session_state.smart_defaults['auto_refresh'] = auto_refresh
            
            # Help and support
            st.markdown("### 🆘 Help & Support")
            
            if st.button("❓ Quick Help", use_container_width=True):
                st.session_state.show_help_modal = True
                st.rerun()
            
            if st.button("🔄 Reset Onboarding", use_container_width=True):
                st.session_state.onboarding_complete = False
                st.session_state.onboarding_step = 1
                st.rerun()
            
            # Return settings
            return {
                'pool': selected_pool,
                'timeframe': timeframe,
                'model': model,
                'confidence': confidence if 'confidence' in locals() else 0.95,
                'show_advanced': show_advanced if 'show_advanced' in locals() else False
            }
    
    def render_contextual_help(self, feature_name: str, help_text: str):
        """Render contextual help tooltip."""
        st.markdown(f"""
        <div class="help-tooltip">
            <span class="help-icon" title="{help_text}">❓</span>
            <span class="help-text">{feature_name}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def get_user_experience_level(self) -> str:
        """Get user experience level for adaptive interface."""
        return st.session_state.user_preferences.get('experience_level', 'intermediate')
    
    def should_show_advanced_features(self) -> bool:
        """Check if advanced features should be shown."""
        level = self.get_user_experience_level()
        return level in ['expert'] or st.session_state.user_preferences.get('show_advanced_features', False)
    
    def get_adaptive_layout(self) -> Dict:
        """Get adaptive layout based on user experience level."""
        level = self.get_user_experience_level()
        
        if level == 'beginner':
            return {
                'show_technical_indicators': False,
                'show_advanced_charts': False,
                'show_ai_insights': True,
                'show_educational_content': True,
                'max_tabs': 3
            }
        elif level == 'intermediate':
            return {
                'show_technical_indicators': True,
                'show_advanced_charts': True,
                'show_ai_insights': True,
                'show_educational_content': True,
                'max_tabs': 4
            }
        else:  # expert
            return {
                'show_technical_indicators': True,
                'show_advanced_charts': True,
                'show_ai_insights': True,
                'show_educational_content': False,
                'max_tabs': 4
            }
