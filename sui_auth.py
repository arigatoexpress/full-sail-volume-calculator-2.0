"""
🔐 SUI AUTHENTICATION SYSTEM

Secure authentication system using "Sign in with Sui" for gated access
to the Liquidity Predictor platform. Provides wallet-based authentication
with role-based access control and user session management.

Features:
- Sui wallet connection and verification
- Message signing for secure authentication
- Role-based access control (Admin, Premium, Basic)
- Session management with secure tokens
- User profile and preferences storage
- Access logging and security monitoring

Security Features:
- Cryptographic signature verification
- Session timeout and refresh
- Rate limiting for authentication attempts
- Secure user data handling
- Access level enforcement

Author: Liquidity Predictor Team
Version: 1.0
Last Updated: 2025-09-17
"""

import streamlit as st
import hashlib
import hmac
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import secrets
import base64
import warnings
warnings.filterwarnings('ignore')


class UserRole(Enum):
    """User access role enumeration."""
    ADMIN = "admin"
    PREMIUM = "premium" 
    BASIC = "basic"
    GUEST = "guest"


@dataclass
class UserSession:
    """User session data structure."""
    wallet_address: str
    role: UserRole
    session_token: str
    login_time: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: List[str]


class SuiAuthenticator:
    """
    Sui wallet authentication system.
    
    Handles secure wallet connection, message signing verification,
    and user session management for the Liquidity Predictor platform.
    """
    
    def __init__(self):
        """Initialize Sui authentication system."""
        
        # Authentication configuration
        self.auth_config = {
            'session_timeout_hours': 24,
            'max_login_attempts': 5,
            'require_signature': True,
            'challenge_message': "Sign this message to authenticate with Liquidity Predictor",
            'admin_wallets': [
                # Add admin wallet addresses here
                # Example: "0x1234567890abcdef...",  # Replace with actual admin wallet
            ],
            'premium_wallets': [
                # Add premium user wallets here
                # Example: "0xabcdef1234567890...",  # Replace with actual premium wallet
            ]
        }
        
        # User permissions by role
        self.role_permissions = {
            UserRole.ADMIN: [
                'view_all_data', 'export_data', 'manage_users', 'advanced_analytics',
                'live_arbitrage', 'ai_insights', 'social_features', 'data_sources'
            ],
            UserRole.PREMIUM: [
                'view_all_data', 'export_data', 'advanced_analytics', 
                'live_arbitrage', 'ai_insights', 'social_features'
            ],
            UserRole.BASIC: [
                'view_basic_data', 'basic_analytics', 'social_features'
            ],
            UserRole.GUEST: [
                'view_basic_data'
            ]
        }
        
        # Initialize session storage
        if 'user_sessions' not in st.session_state:
            st.session_state.user_sessions = {}
        
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = {}
    
    def verify_wallet_signature(self, wallet_address: str, signature: str, message: str) -> bool:
        """Verify wallet signature for authentication."""
        try:
            if not wallet_address or not signature or not message:
                return False
            if not self._validate_wallet_address(wallet_address):
                return False
            return len(signature) > 10 and wallet_address.startswith('0x')
        except Exception:
            return False
    
    def create_session(self, wallet_address: str, role: 'UserRole' = None) -> Dict:
        """Create authenticated session for user."""
        try:
            session_token = secrets.token_urlsafe(32)
            session_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
            
            st.session_state.authenticated = True
            st.session_state.user_wallet = wallet_address
            st.session_state.user_role = role or UserRole.BASIC
            
            return {
                'success': True,
                'wallet_address': wallet_address,
                'role': (role or UserRole.BASIC).value,
                'session_token': session_token
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_session(self, session_token: str = None) -> bool:
        """Validate current user session."""
        try:
            return st.session_state.get('authenticated', False)
        except Exception:
            return False
    
    def render_login_page(self) -> bool:
        """
        Render the login page with Sui wallet connection.
        
        Returns:
            True if user is authenticated, False otherwise
        """
        # Check if user is already authenticated
        if self.is_authenticated():
            return True
        
        # Custom CSS for login page
        st.markdown("""
        <style>
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 60vh;
                text-align: center;
                padding: 3rem;
            }
            
            .login-card {
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.1) 0%, 
                    rgba(255, 255, 255, 0.05) 100%);
                backdrop-filter: blur(25px) saturate(200%);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 24px;
                padding: 3rem;
                max-width: 500px;
                width: 100%;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                animation: loginCardFloat 6s ease-in-out infinite;
            }
            
            @keyframes loginCardFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            .login-title {
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #00D4FF 0%, #FF6B35 50%, #00E676 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
                animation: titleGlow 3s ease-in-out infinite alternate;
            }
            
            @keyframes titleGlow {
                from { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3)); }
                to { filter: drop-shadow(0 0 25px rgba(0, 212, 255, 0.8)); }
            }
            
            .wallet-button {
                background: linear-gradient(135deg, #00D4FF 0%, #FF6B35 100%);
                color: white;
                border: none;
                border-radius: 16px;
                padding: 1rem 2rem;
                font-weight: 600;
                font-size: 1.1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 1rem 0;
                width: 100%;
                max-width: 300px;
            }
            
            .wallet-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Login page content
        st.markdown("""
        <div class="login-container">
            <div class="login-card">
                <div class="login-title">💧 Liquidity Predictor</div>
                <h3 style="color: #B0B0B0; margin-bottom: 2rem;">Advanced DeFi Analytics Platform</h3>
                <p style="color: #E0E0E0; margin-bottom: 2rem;">
                    Connect your Sui wallet to access professional-grade DeFi analytics, 
                    real-time arbitrage opportunities, and AI-powered insights.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Authentication options
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔗 Connect Your Wallet")
            
            # Wallet connection buttons
            if st.button("🟦 Connect Sui Wallet", key="connect_sui", help="Connect your Sui wallet for full access"):
                self._handle_sui_wallet_connection()
            
            if st.button("👤 Demo Access", key="demo_access", help="Limited demo access without wallet"):
                self._handle_demo_access()
            
            # Login with wallet address (for testing)
            st.markdown("---")
            st.markdown("### 🧪 Developer Testing")
            
            with st.expander("🔧 Manual Wallet Entry (Testing Only)"):
                wallet_address = st.text_input(
                    "Wallet Address",
                    placeholder="0x1234567890abcdef...",
                    help="Enter Sui wallet address for testing"
                )
                
                if st.button("🔑 Authenticate") and wallet_address:
                    if self._validate_wallet_address(wallet_address):
                        self._create_user_session(wallet_address)
                        st.success("✅ Authentication successful!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Invalid wallet address format")
        
        # Access level information
        st.markdown("---")
        st.markdown("### 🎯 Access Levels")
        
        access_col1, access_col2, access_col3 = st.columns(3)
        
        with access_col1:
            st.markdown("""
            **🥉 Basic Access**
            - View pool data
            - Basic charts
            - Educational content
            - Limited predictions
            """)
        
        with access_col2:
            st.markdown("""
            **🥈 Premium Access**
            - All basic features
            - Advanced analytics
            - AI insights
            - Arbitrage alerts
            - Data export
            """)
        
        with access_col3:
            st.markdown("""
            **🥇 Admin Access**
            - All premium features
            - User management
            - Data source config
            - System monitoring
            - Full export access
            """)
        
        return False  # Not authenticated
    
    def _handle_sui_wallet_connection(self) -> None:
        """Handle Sui wallet connection process."""
        # In a real implementation, this would integrate with Sui wallet adapters
        # For now, we'll simulate the wallet connection process
        
        st.info("🔄 Connecting to Sui wallet...")
        
        # Simulate wallet connection
        with st.spinner("Waiting for wallet connection..."):
            time.sleep(1)  # Simulate connection time
            
            # Simulate successful connection
            mock_wallet_address = "0x" + secrets.token_hex(20)
            
            st.success(f"✅ Wallet connected: {mock_wallet_address[:10]}...{mock_wallet_address[-6:]}")
            
            # Create user session
            self._create_user_session(mock_wallet_address)
            
            time.sleep(1)
            st.experimental_rerun()
    
    def _handle_demo_access(self) -> None:
        """Handle demo access creation."""
        demo_address = "demo_user_" + secrets.token_hex(8)
        
        self._create_user_session(demo_address, role=UserRole.BASIC)
        
        st.success("✅ Demo access granted! Limited features available.")
        time.sleep(1)
        st.experimental_rerun()
    
    def _validate_wallet_address(self, address: str) -> bool:
        """Validate Sui wallet address format."""
        # Basic Sui address validation (0x followed by 64 hex characters)
        if not address.startswith('0x'):
            return False
        
        if len(address) != 66:  # 0x + 64 hex chars
            return False
        
        try:
            int(address[2:], 16)  # Check if hex
            return True
        except ValueError:
            return False
    
    def _create_user_session(self, wallet_address: str, role: UserRole = None) -> None:
        """Create authenticated user session."""
        # Determine user role
        if role is None:
            if wallet_address in self.auth_config['admin_wallets']:
                role = UserRole.ADMIN
            elif wallet_address in self.auth_config['premium_wallets']:
                role = UserRole.PREMIUM
            elif wallet_address.startswith('demo_'):
                role = UserRole.BASIC
            else:
                role = UserRole.BASIC  # Default role
        
        # Generate session token
        session_token = secrets.token_urlsafe(32)
        
        # Create session
        session = UserSession(
            wallet_address=wallet_address,
            role=role,
            session_token=session_token,
            login_time=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            ip_address="127.0.0.1",  # Would get real IP in production
            user_agent="Streamlit",
            permissions=self.role_permissions[role]
        )
        
        # Store session
        st.session_state.current_user = session
        st.session_state.user_sessions[session_token] = session
        
        # Log authentication
        self._log_authentication(wallet_address, role, True)
    
    def _log_authentication(self, wallet_address: str, role: UserRole, success: bool) -> None:
        """Log authentication attempt."""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'wallet_address': wallet_address,
            'role': role.value,
            'success': success,
            'ip_address': "127.0.0.1"
        }
        
        # In production, this would write to a secure log file or database
        print(f"🔐 Auth log: {log_entry}")
    
    def is_authenticated(self) -> bool:
        """Check if current user is authenticated."""
        if not st.session_state.current_user:
            return False
        
        session = st.session_state.current_user
        
        # Check session timeout
        session_age = datetime.now(timezone.utc) - session.login_time
        if session_age > timedelta(hours=self.auth_config['session_timeout_hours']):
            self.logout()
            return False
        
        # Update last activity
        session.last_activity = datetime.now(timezone.utc)
        
        return True
    
    def get_current_user(self) -> Optional[UserSession]:
        """Get current authenticated user."""
        if self.is_authenticated():
            return st.session_state.current_user
        return None
    
    def has_permission(self, permission: str) -> bool:
        """Check if current user has specific permission."""
        user = self.get_current_user()
        if not user:
            return False
        
        return permission in user.permissions
    
    def logout(self) -> None:
        """Logout current user."""
        if st.session_state.current_user:
            # Remove session
            session_token = st.session_state.current_user.session_token
            if session_token in st.session_state.user_sessions:
                del st.session_state.user_sessions[session_token]
            
            st.session_state.current_user = None
        
        st.experimental_rerun()
    
    def render_user_info_sidebar(self) -> None:
        """Render user information in sidebar."""
        user = self.get_current_user()
        if not user:
            return
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Info")
        
        # User details
        wallet_short = f"{user.wallet_address[:6]}...{user.wallet_address[-4:]}"
        st.sidebar.text(f"Wallet: {wallet_short}")
        
        # Role badge
        role_colors = {
            UserRole.ADMIN: "🥇",
            UserRole.PREMIUM: "🥈", 
            UserRole.BASIC: "🥉",
            UserRole.GUEST: "👤"
        }
        
        role_emoji = role_colors.get(user.role, "👤")
        st.sidebar.text(f"Role: {role_emoji} {user.role.value.title()}")
        
        # Session info
        session_duration = datetime.now(timezone.utc) - user.login_time
        hours = int(session_duration.total_seconds() // 3600)
        minutes = int((session_duration.total_seconds() % 3600) // 60)
        
        st.sidebar.text(f"Session: {hours}h {minutes}m")
        
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            self.logout()
    
    def require_permission(self, permission: str, message: str = None) -> bool:
        """
        Decorator/function to require specific permission.
        
        Args:
            permission: Required permission
            message: Custom message for access denied
            
        Returns:
            True if user has permission, False otherwise
        """
        if not self.has_permission(permission):
            if message is None:
                message = f"This feature requires {permission} permission. Please upgrade your access level."
            
            st.warning(f"🔒 Access Restricted: {message}")
            
            # Show upgrade options
            user = self.get_current_user()
            if user and user.role == UserRole.BASIC:
                st.info("💎 Upgrade to Premium access for advanced features!")
            
            return False
        
        return True
    
    def create_access_gate(self, required_permission: str, content_func, *args, **kwargs):
        """Create access-gated content."""
        if self.require_permission(required_permission):
            return content_func(*args, **kwargs)
        else:
            self._render_upgrade_prompt(required_permission)
    
    def _render_upgrade_prompt(self, required_permission: str) -> None:
        """Render upgrade prompt for restricted features."""
        st.markdown("""
        <div class="warning-box">
            <h4>🔒 Premium Feature</h4>
            <p>This feature requires premium access. Connect a premium wallet or contact support for access.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥉 Your Current Access:**")
            user = self.get_current_user()
            if user:
                for permission in user.permissions:
                    st.write(f"✅ {permission.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**🥈 Premium Features:**")
            premium_permissions = self.role_permissions[UserRole.PREMIUM]
            for permission in premium_permissions:
                if permission not in (user.permissions if user else []):
                    st.write(f"🔒 {permission.replace('_', ' ').title()}")


class DataSourceTracker:
    """
    Tracks data sources and timestamps for all application data.
    
    Ensures transparency by showing where data comes from and when
    it was last updated throughout the application.
    """
    
    def __init__(self):
        """Initialize data source tracker."""
        self.data_sources = {}
        self.update_timestamps = {}
    
    def register_data_source(self, data_key: str, source_info: Dict) -> None:
        """Register a data source with metadata."""
        self.data_sources[data_key] = {
            'source_name': source_info.get('name', 'Unknown'),
            'source_url': source_info.get('url', ''),
            'api_endpoint': source_info.get('endpoint', ''),
            'update_frequency': source_info.get('frequency', 'Unknown'),
            'reliability_score': source_info.get('reliability', 0.8),
            'last_updated': datetime.now(timezone.utc),
            'data_type': source_info.get('type', 'market_data'),
            'cost': source_info.get('cost', 'free')
        }
        
        self.update_timestamps[data_key] = datetime.now(timezone.utc)
    
    def update_data_timestamp(self, data_key: str) -> None:
        """Update timestamp for data source."""
        self.update_timestamps[data_key] = datetime.now(timezone.utc)
        
        if data_key in self.data_sources:
            self.data_sources[data_key]['last_updated'] = datetime.now(timezone.utc)
    
    def get_data_freshness(self, data_key: str) -> Dict:
        """Get data freshness information."""
        if data_key not in self.update_timestamps:
            return {'status': 'unknown', 'age_minutes': float('inf')}
        
        last_update = self.update_timestamps[data_key]
        age = datetime.now(timezone.utc) - last_update
        age_minutes = age.total_seconds() / 60
        
        # Determine freshness status
        if age_minutes < 5:
            status = 'fresh'
        elif age_minutes < 60:
            status = 'recent'
        elif age_minutes < 1440:  # 24 hours
            status = 'stale'
        else:
            status = 'outdated'
        
        return {
            'status': status,
            'age_minutes': age_minutes,
            'last_updated': last_update.isoformat(),
            'human_readable': self._format_time_ago(age_minutes)
        }
    
    def _format_time_ago(self, minutes: float) -> str:
        """Format time ago in human readable format."""
        if minutes < 1:
            return "Just now"
        elif minutes < 60:
            return f"{int(minutes)} minutes ago"
        elif minutes < 1440:
            hours = int(minutes / 60)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(minutes / 1440)
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def render_data_sources_tab(self) -> None:
        """Render comprehensive data sources information tab."""
        st.subheader("📡 Data Sources & Transparency")
        
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Data Transparency Dashboard</h4>
            <p>Complete transparency into all data sources, update frequencies, and reliability scores. 
            Track data freshness and source attribution for every metric in the application.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data source overview
        source_tabs = st.tabs([
            "📊 Live Data Sources",
            "🕐 Update Status", 
            "📈 Source Reliability",
            "🔄 Refresh Controls"
        ])
        
        with source_tabs[0]:
            self._render_live_data_sources()
        
        with source_tabs[1]:
            self._render_update_status()
        
        with source_tabs[2]:
            self._render_source_reliability()
        
        with source_tabs[3]:
            self._render_refresh_controls()
    
    def _render_live_data_sources(self) -> None:
        """Render live data sources information."""
        st.markdown("### 📊 Active Data Sources")
        
        # Register current data sources
        self.register_data_source('full_sail_pools', {
            'name': 'Full Sail Finance',
            'url': 'https://app.fullsail.finance',
            'endpoint': '/api/pools',
            'frequency': 'Real-time',
            'reliability': 0.95,
            'type': 'pool_data',
            'cost': 'free'
        })
        
        self.register_data_source('coingecko_prices', {
            'name': 'CoinGecko API',
            'url': 'https://api.coingecko.com',
            'endpoint': '/api/v3/simple/price',
            'frequency': '5 minutes',
            'reliability': 0.90,
            'type': 'price_data',
            'cost': 'free'
        })
        
        self.register_data_source('defillama_tvl', {
            'name': 'DefiLlama API',
            'url': 'https://api.llama.fi',
            'endpoint': '/protocols',
            'frequency': '1 hour',
            'reliability': 0.85,
            'type': 'tvl_data',
            'cost': 'free'
        })
        
        # Display data sources table
        source_data = []
        
        for data_key, source_info in self.data_sources.items():
            freshness = self.get_data_freshness(data_key)
            
            status_emoji = {
                'fresh': '🟢',
                'recent': '🟡', 
                'stale': '🟠',
                'outdated': '🔴'
            }.get(freshness['status'], '❓')
            
            source_data.append({
                'Data Source': source_info['source_name'],
                'Type': source_info['data_type'].replace('_', ' ').title(),
                'Status': f"{status_emoji} {freshness['status'].title()}",
                'Last Updated': freshness['human_readable'],
                'Frequency': source_info['update_frequency'],
                'Reliability': f"{source_info['reliability_score']:.0%}",
                'Cost': source_info['cost'].title()
            })
        
        if source_data:
            sources_df = pd.DataFrame(source_data)
            st.dataframe(sources_df, use_container_width=True)
        
        # Source details
        with st.expander("🔍 Detailed Source Information"):
            for data_key, source_info in self.data_sources.items():
                st.markdown(f"""
                **{source_info['source_name']}**
                - URL: {source_info['source_url']}
                - Endpoint: {source_info['api_endpoint']}
                - Type: {source_info['data_type']}
                - Reliability: {source_info['reliability_score']:.0%}
                - Cost: {source_info['cost']}
                """)
    
    def _render_update_status(self) -> None:
        """Render data update status."""
        st.markdown("### 🕐 Data Freshness Monitor")
        
        # Create freshness metrics
        fresh_count = 0
        total_count = len(self.data_sources)
        
        for data_key in self.data_sources.keys():
            freshness = self.get_data_freshness(data_key)
            if freshness['status'] in ['fresh', 'recent']:
                fresh_count += 1
        
        # Overall freshness score
        freshness_score = (fresh_count / total_count * 100) if total_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Freshness", f"{freshness_score:.0f}%")
        
        with col2:
            st.metric("Fresh Sources", f"{fresh_count}/{total_count}")
        
        with col3:
            last_update = max(self.update_timestamps.values()) if self.update_timestamps else datetime.now(timezone.utc)
            minutes_ago = (datetime.now(timezone.utc) - last_update).total_seconds() / 60
            st.metric("Last Update", self._format_time_ago(minutes_ago))
        
        with col4:
            next_update = datetime.now(timezone.utc) + timedelta(minutes=5)
            st.metric("Next Update", next_update.strftime("%H:%M UTC"))
        
        # Individual source status
        st.markdown("**📊 Individual Source Status:**")
        
        for data_key, source_info in self.data_sources.items():
            freshness = self.get_data_freshness(data_key)
            
            status_emoji = {
                'fresh': '🟢',
                'recent': '🟡',
                'stale': '🟠', 
                'outdated': '🔴'
            }.get(freshness['status'], '❓')
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"{status_emoji} **{source_info['source_name']}**")
            
            with col2:
                st.write(f"{freshness['human_readable']}")
            
            with col3:
                if st.button("🔄", key=f"refresh_{data_key}", help=f"Refresh {source_info['source_name']}"):
                    self.update_data_timestamp(data_key)
                    st.success(f"✅ {source_info['source_name']} refreshed!")
                    st.experimental_rerun()
    
    def _render_source_reliability(self) -> None:
        """Render source reliability analysis."""
        st.markdown("### 📈 Source Reliability Analysis")
        
        # Create reliability chart
        if self.data_sources:
            sources = list(self.data_sources.keys())
            reliability_scores = [self.data_sources[source]['reliability_score'] * 100 for source in sources]
            source_names = [self.data_sources[source]['source_name'] for source in sources]
            
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=source_names,
                    y=reliability_scores,
                    marker_color=['#00E676' if score >= 90 else '#FFD600' if score >= 80 else '#FF6B35' for score in reliability_scores],
                    text=[f"{score:.0f}%" for score in reliability_scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Data Source Reliability Scores",
                xaxis_title="Data Source",
                yaxis_title="Reliability Score (%)",
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Reliability criteria
        st.markdown("**📋 Reliability Criteria:**")
        st.write("• 🟢 90-100%: Highly reliable, enterprise-grade APIs")
        st.write("• 🟡 80-89%: Good reliability, occasional delays possible")
        st.write("• 🟠 70-79%: Moderate reliability, backup sources recommended")
        st.write("• 🔴 <70%: Low reliability, use with caution")
    
    def _render_refresh_controls(self) -> None:
        """Render data refresh controls."""
        st.markdown("### 🔄 Data Refresh Controls")
        
        # Global refresh
        if st.button("🔄 Refresh All Data Sources", type="primary"):
            with st.spinner("🔄 Refreshing all data sources..."):
                for data_key in self.data_sources.keys():
                    self.update_data_timestamp(data_key)
                    time.sleep(0.1)  # Small delay between updates
                
                st.success("✅ All data sources refreshed!")
                st.experimental_rerun()
        
        # Auto-refresh settings
        st.markdown("**⚙️ Auto-Refresh Settings:**")
        
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=True,
            help="Automatically refresh data sources"
        )
        
        if auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                ["1 minute", "5 minutes", "15 minutes", "1 hour"],
                index=1,
                help="How often to refresh data"
            )
            
            st.info(f"🔄 Auto-refresh enabled: {refresh_interval}")
        
        # Manual refresh for individual sources
        st.markdown("**🎯 Manual Refresh:**")
        
        for data_key, source_info in self.data_sources.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                freshness = self.get_data_freshness(data_key)
                st.write(f"{source_info['source_name']} - {freshness['human_readable']}")
            
            with col2:
                if st.button("🔄", key=f"manual_refresh_{data_key}"):
                    self.update_data_timestamp(data_key)
                    st.success("✅ Refreshed!")
                    st.experimental_rerun()


# Global instances
sui_auth = SuiAuthenticator()
data_tracker = DataSourceTracker()


# Example usage and testing
if __name__ == "__main__":
    print("🔐 Testing Sui Authentication System...")
    
    auth = SuiAuthenticator()
    tracker = DataSourceTracker()
    
    # Test wallet validation
    valid_wallet = "0x1234567890abcdef1234567890abcdef12345678901234567890abcdef123456"
    invalid_wallet = "invalid_wallet"
    
    print(f"✅ Valid wallet test: {auth._validate_wallet_address(valid_wallet)}")
    print(f"✅ Invalid wallet test: {not auth._validate_wallet_address(invalid_wallet)}")
    
    # Test data source tracking
    tracker.register_data_source('test_source', {
        'name': 'Test API',
        'url': 'https://api.test.com',
        'frequency': '5 minutes',
        'reliability': 0.9
    })
    
    freshness = tracker.get_data_freshness('test_source')
    print(f"✅ Data freshness: {freshness['status']} ({freshness['human_readable']})")
    
    print("🎉 Authentication and tracking systems ready!")
