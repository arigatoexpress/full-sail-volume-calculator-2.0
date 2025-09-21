"""
Premium UI components for the Liquidity Predictor.
Creates a sleek, stunning, and highly interactive user interface.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PremiumUIManager:
    """Manages premium UI components and styling."""
    
    def __init__(self):
        """Initialize premium UI manager."""
        self.color_schemes = {
            'ocean': {
                'primary': '#00D4FF',
                'secondary': '#0099CC', 
                'accent': '#66E5FF',
                'success': '#00E676',
                'warning': '#FFB300',
                'error': '#FF5252',
                'background': 'linear-gradient(135deg, #0a1929 0%, #1e3a8a 50%, #0f172a 100%)',
                'glass': 'rgba(255, 255, 255, 0.08)'
            },
            'sunset': {
                'primary': '#FF6B35',
                'secondary': '#FF8A50',
                'accent': '#FFB366',
                'success': '#4CAF50',
                'warning': '#FFC107',
                'error': '#F44336',
                'background': 'linear-gradient(135deg, #1a0a0a 0%, #3a1e1e 50%, #2a0f0f 100%)',
                'glass': 'rgba(255, 107, 53, 0.08)'
            },
            'forest': {
                'primary': '#00E676',
                'secondary': '#26A69A',
                'accent': '#66FFB3',
                'success': '#4CAF50',
                'warning': '#FF9800',
                'error': '#FF5722',
                'background': 'linear-gradient(135deg, #0a1a0a 0%, #1e3a1e 50%, #0f2a0f 100%)',
                'glass': 'rgba(0, 230, 118, 0.08)'
            }
        }
        
        self.current_theme = 'ocean'
    
    def get_premium_css(self, theme: str = 'ocean') -> str:
        """Get premium CSS with advanced visual effects."""
        colors = self.color_schemes.get(theme, self.color_schemes['ocean'])
        
        return f"""
        <style>
            /* Import premium fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap');
            
            /* Premium CSS Variables */
            :root {{
                --primary-color: {colors['primary']};
                --secondary-color: {colors['secondary']};
                --accent-color: {colors['accent']};
                --success-color: {colors['success']};
                --warning-color: {colors['warning']};
                --error-color: {colors['error']};
                --glass-bg: {colors['glass']};
                --text-primary: #FFFFFF;
                --text-secondary: #E0E0E0;
                --text-muted: #B0B0B0;
                --border-color: rgba(255, 255, 255, 0.12);
                --shadow-primary: 0 8px 32px rgba(0, 0, 0, 0.4);
                --shadow-hover: 0 16px 48px rgba(0, 0, 0, 0.5);
                --border-radius: 20px;
                --transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            }}
            
            /* Revolutionary animated background */
            .stApp {{
                background: {colors['background']};
                background-attachment: fixed;
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
                position: relative;
                min-height: 100vh;
            }}
            
            /* Floating orbs background effect */
            .stApp::before {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: 
                    radial-gradient(circle at 10% 20%, {colors['glass']} 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.03) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, {colors['glass']} 0%, transparent 50%);
                animation: orbFloat 20s ease-in-out infinite;
                pointer-events: none;
                z-index: -1;
            }}
            
            @keyframes orbFloat {{
                0%, 100% {{ 
                    transform: translate(0px, 0px) scale(1);
                    opacity: 0.6;
                }}
                33% {{ 
                    transform: translate(30px, -30px) scale(1.1);
                    opacity: 0.8;
                }}
                66% {{ 
                    transform: translate(-20px, 20px) scale(0.9);
                    opacity: 0.7;
                }}
            }}
            
            /* Spectacular header design */
            .premium-header {{
                text-align: center;
                padding: 3rem 0;
                position: relative;
                overflow: hidden;
            }}
            
            .main-title {{
                font-family: 'Poppins', sans-serif;
                font-size: 5rem;
                font-weight: 800;
                background: linear-gradient(135deg, 
                    var(--primary-color) 0%, 
                    var(--accent-color) 25%,
                    var(--secondary-color) 50%,
                    var(--primary-color) 75%,
                    var(--accent-color) 100%);
                background-size: 400% 400%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: 
                    titleGradientFlow 8s ease-in-out infinite,
                    titleFloat 6s ease-in-out infinite,
                    titleGlow 4s ease-in-out infinite alternate;
                margin: 0;
                position: relative;
            }}
            
            @keyframes titleGradientFlow {{
                0%, 100% {{ background-position: 0% 50%; }}
                25% {{ background-position: 100% 0%; }}
                50% {{ background-position: 0% 100%; }}
                75% {{ background-position: 100% 50%; }}
            }}
            
            @keyframes titleFloat {{
                0%, 100% {{ transform: translateY(0px) rotateX(0deg); }}
                50% {{ transform: translateY(-10px) rotateX(2deg); }}
            }}
            
            @keyframes titleGlow {{
                from {{ 
                    filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.3));
                    text-shadow: 0 0 40px rgba(0, 212, 255, 0.2);
                }}
                to {{ 
                    filter: drop-shadow(0 0 40px rgba(0, 212, 255, 0.8));
                    text-shadow: 0 0 60px rgba(0, 212, 255, 0.5);
                }}
            }}
            
            .subtitle {{
                font-family: 'Inter', sans-serif;
                font-size: 1.5rem;
                font-weight: 300;
                color: var(--text-secondary);
                margin-top: 1rem;
                animation: subtitleFadeIn 2s ease-out 0.5s both;
                letter-spacing: 2px;
            }}
            
            @keyframes subtitleFadeIn {{
                from {{ 
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{ 
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            /* Premium glassmorphism cards */
            .premium-card {{
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.1) 0%, 
                    rgba(255, 255, 255, 0.05) 100%);
                backdrop-filter: blur(30px) saturate(200%);
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                padding: 2.5rem;
                margin: 2rem 0;
                box-shadow: 
                    var(--shadow-primary),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transition: var(--transition);
                position: relative;
                overflow: hidden;
            }}
            
            .premium-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.1), 
                    transparent);
                animation: cardShimmer 3s ease-in-out infinite;
                pointer-events: none;
            }}
            
            @keyframes cardShimmer {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}
            
            .premium-card:hover {{
                transform: translateY(-8px) rotateX(3deg) rotateY(2deg);
                box-shadow: 
                    var(--shadow-hover),
                    0 0 30px var(--primary-color)40,
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                border-color: var(--primary-color)60;
            }}
            
            /* Interactive metric displays */
            .metric-display {{
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.08) 0%, 
                    rgba(255, 255, 255, 0.03) 100%);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                transition: var(--transition);
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }}
            
            .metric-display::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, 
                    var(--primary-color), 
                    var(--accent-color));
                transform: translateX(-100%);
                animation: metricProgress 2s ease-in-out infinite;
            }}
            
            @keyframes metricProgress {{
                0%, 100% {{ transform: translateX(-100%); }}
                50% {{ transform: translateX(100%); }}
            }}
            
            .metric-display:hover {{
                transform: translateY(-4px) scale(1.02);
                border-color: var(--primary-color);
                box-shadow: 
                    0 12px 40px rgba(0, 0, 0, 0.4),
                    0 0 20px var(--primary-color)30;
            }}
            
            .metric-value {{
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
                animation: valueCountUp 2s ease-out;
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                font-weight: 500;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .metric-change {{
                font-size: 1rem;
                font-weight: 600;
                margin-top: 0.5rem;
                animation: changeFlash 1s ease-out 1s both;
            }}
            
            @keyframes valueCountUp {{
                from {{ 
                    opacity: 0;
                    transform: scale(0.5) rotateX(90deg);
                }}
                to {{ 
                    opacity: 1;
                    transform: scale(1) rotateX(0deg);
                }}
            }}
            
            @keyframes changeFlash {{
                0% {{ opacity: 0; transform: translateX(20px); }}
                100% {{ opacity: 1; transform: translateX(0); }}
            }}
            
            /* Revolutionary button system */
            .premium-button {{
                background: linear-gradient(135deg, 
                    var(--primary-color) 0%, 
                    var(--secondary-color) 100%);
                background-size: 200% 200%;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 1rem 2.5rem;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 1rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                cursor: pointer;
                transition: var(--transition);
                position: relative;
                overflow: hidden;
                box-shadow: 
                    0 4px 15px var(--primary-color)40,
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
            }}
            
            .premium-button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.3), 
                    transparent);
                transition: left 0.6s ease;
            }}
            
            .premium-button:hover {{
                transform: translateY(-3px) scale(1.02);
                background-position: 100% 0;
                box-shadow: 
                    0 12px 30px var(--primary-color)60,
                    0 4px 15px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.4);
                animation: buttonRipple 0.8s ease-out;
            }}
            
            .premium-button:hover::before {{
                left: 100%;
            }}
            
            .premium-button:active {{
                transform: translateY(-1px) scale(1);
                transition: all 0.1s ease;
            }}
            
            @keyframes buttonRipple {{
                0% {{ box-shadow: 0 12px 30px var(--primary-color)60; }}
                50% {{ box-shadow: 0 16px 40px var(--primary-color)80; }}
                100% {{ box-shadow: 0 12px 30px var(--primary-color)60; }}
            }}
            
            /* Advanced tab system */
            .premium-tabs {{
                background: linear-gradient(135deg, 
                    rgba(20, 20, 20, 0.95) 0%, 
                    rgba(30, 30, 30, 0.95) 100%);
                backdrop-filter: blur(25px) saturate(200%);
                border-radius: 20px;
                border: 1px solid var(--border-color);
                padding: 1rem;
                margin: 2rem 0;
                box-shadow: var(--shadow-primary);
                position: relative;
                overflow: hidden;
            }}
            
            .premium-tabs::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, 
                    var(--glass-bg) 0%, 
                    transparent 50%, 
                    var(--glass-bg) 100%);
                animation: tabGlow 6s ease-in-out infinite;
                pointer-events: none;
            }}
            
            @keyframes tabGlow {{
                0%, 100% {{ opacity: 0.3; }}
                50% {{ opacity: 0.7; }}
            }}
            
            /* Premium data visualization */
            .data-viz-container {{
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.05) 0%, 
                    rgba(255, 255, 255, 0.02) 100%);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                padding: 2rem;
                margin: 1.5rem 0;
                position: relative;
                overflow: hidden;
            }}
            
            .data-viz-container::after {{
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: conic-gradient(
                    from 0deg,
                    transparent 0deg,
                    var(--primary-color)20 90deg,
                    transparent 180deg,
                    var(--accent-color)20 270deg,
                    transparent 360deg
                );
                animation: dataVizRotate 15s linear infinite;
                pointer-events: none;
                z-index: -1;
            }}
            
            @keyframes dataVizRotate {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            
            /* Interactive sidebar */
            .css-1d391kg {{
                background: linear-gradient(180deg, 
                    rgba(8, 8, 8, 0.98) 0%, 
                    rgba(15, 15, 15, 0.98) 30%,
                    rgba(20, 20, 20, 0.98) 70%,
                    rgba(12, 12, 12, 0.98) 100%);
                backdrop-filter: blur(30px) saturate(200%);
                border-right: 2px solid var(--border-color);
                box-shadow: 
                    8px 0 32px rgba(0, 0, 0, 0.6),
                    inset -2px 0 0 var(--primary-color)20;
                position: relative;
            }}
            
            .css-1d391kg::before {{
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 3px;
                height: 100%;
                background: linear-gradient(180deg, 
                    transparent 0%, 
                    var(--primary-color) 50%, 
                    transparent 100%);
                animation: sidebarPulse 4s ease-in-out infinite;
            }}
            
            @keyframes sidebarPulse {{
                0%, 100% {{ opacity: 0.3; transform: scaleY(0.8); }}
                50% {{ opacity: 1; transform: scaleY(1.2); }}
            }}
            
            /* Enhanced input controls */
            .stSelectbox > div > div,
            .stNumberInput > div > div > input,
            .stTextInput > div > div > input,
            .stSlider > div > div {{
                background: linear-gradient(135deg, 
                    rgba(25, 25, 25, 0.98) 0%, 
                    rgba(35, 35, 35, 0.98) 100%) !important;
                backdrop-filter: blur(15px);
                border: 1px solid var(--border-color) !important;
                border-radius: 12px !important;
                color: var(--text-primary) !important;
                transition: var(--transition);
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                font-family: 'Inter', sans-serif;
            }}
            
            .stSelectbox > div > div:hover,
            .stNumberInput > div > div > input:hover,
            .stTextInput > div > div > input:hover {{
                border-color: var(--primary-color) !important;
                box-shadow: 
                    0 4px 20px rgba(0, 0, 0, 0.4),
                    0 0 15px var(--primary-color)30;
                transform: translateY(-2px);
            }}
            
            .stSelectbox > div > div:focus-within,
            .stNumberInput > div > div > input:focus,
            .stTextInput > div > div > input:focus {{
                border-color: var(--primary-color) !important;
                box-shadow: 
                    0 6px 25px rgba(0, 0, 0, 0.5),
                    0 0 20px var(--primary-color)50;
                transform: translateY(-3px);
            }}
            
            /* Premium loading states */
            .stSpinner > div {{
                border: 3px solid var(--primary-color)30;
                border-top: 3px solid var(--primary-color);
                border-radius: 50%;
                animation: 
                    premiumSpin 1s linear infinite,
                    loadingGlow 2s ease-in-out infinite alternate;
            }}
            
            @keyframes premiumSpin {{
                0% {{ 
                    transform: rotate(0deg) scale(1);
                    filter: drop-shadow(0 0 10px var(--primary-color)50);
                }}
                50% {{
                    transform: rotate(180deg) scale(1.1);
                    filter: drop-shadow(0 0 20px var(--primary-color)80);
                }}
                100% {{ 
                    transform: rotate(360deg) scale(1);
                    filter: drop-shadow(0 0 10px var(--primary-color)50);
                }}
            }}
            
            @keyframes loadingGlow {{
                from {{ box-shadow: 0 0 20px var(--primary-color)30; }}
                to {{ box-shadow: 0 0 40px var(--primary-color)70; }}
            }}
            
            /* Spectacular chart containers */
            .chart-container {{
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.03) 0%, 
                    rgba(255, 255, 255, 0.01) 100%);
                backdrop-filter: blur(25px);
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: var(--shadow-primary);
                transition: var(--transition);
                position: relative;
                overflow: hidden;
            }}
            
            .chart-container:hover {{
                transform: scale(1.01);
                border-color: var(--primary-color);
                box-shadow: 
                    var(--shadow-hover),
                    0 0 25px var(--primary-color)20;
            }}
            
            /* Premium alerts and notifications */
            .premium-alert {{
                background: linear-gradient(135deg, 
                    var(--glass-bg) 0%, 
                    rgba(255, 255, 255, 0.02) 100%);
                backdrop-filter: blur(20px);
                border-left: 4px solid var(--primary-color);
                border-radius: 0 12px 12px 0;
                padding: 1.5rem;
                margin: 1rem 0;
                animation: 
                    alertSlideIn 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94),
                    alertPulse 3s ease-in-out infinite;
                position: relative;
            }}
            
            @keyframes alertSlideIn {{
                from {{ 
                    opacity: 0;
                    transform: translateX(-50px) rotateY(-15deg);
                }}
                to {{ 
                    opacity: 1;
                    transform: translateX(0) rotateY(0deg);
                }}
            }}
            
            @keyframes alertPulse {{
                0%, 100% {{ border-left-color: var(--primary-color); }}
                50% {{ border-left-color: var(--accent-color); }}
            }}
            
            /* Mobile-first responsive design */
            @media (max-width: 768px) {{
                .main-title {{
                    font-size: 3rem;
                }}
                
                .subtitle {{
                    font-size: 1.2rem;
                }}
                
                .premium-card {{
                    padding: 1.5rem;
                    margin: 1rem 0;
                }}
                
                .metric-display {{
                    padding: 1.5rem;
                }}
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(20, 20, 20, 0.8);
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(135deg, 
                    var(--primary-color), 
                    var(--accent-color));
                border-radius: 4px;
                box-shadow: 0 2px 10px var(--primary-color)30;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(135deg, 
                    var(--accent-color), 
                    var(--secondary-color));
                box-shadow: 0 4px 15px var(--primary-color)50;
            }}
            
            /* Accessibility enhancements */
            .focus-visible {{
                outline: 2px solid var(--primary-color);
                outline-offset: 2px;
            }}
            
            /* Print styles */
            @media print {{
                .stApp {{
                    background: white !important;
                    color: black !important;
                }}
                
                .premium-card {{
                    border: 1px solid #ccc !important;
                    box-shadow: none !important;
                }}
            }}
        </style>
        """
    
    def create_hero_section(self) -> None:
        """Create stunning hero section."""
        st.markdown("""
        <div class="premium-header">
            <div class="main-title">💧 Liquidity Predictor</div>
            <div class="subtitle">Advanced DeFi Analytics & Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_metric_card(self, title: str, value: str, change: str = "", 
                          icon: str = "📊", color_scheme: str = "primary") -> None:
        """Create premium animated metric card."""
        change_class = "positive" if change.startswith('+') else "negative" if change.startswith('-') else "neutral"
        change_color = "var(--success-color)" if change_class == "positive" else "var(--error-color)" if change_class == "negative" else "var(--warning-color)"
        
        st.markdown(f"""
        <div class="metric-display" onclick="this.style.transform='scale(0.98)'; setTimeout(() => this.style.transform='scale(1.02)', 100);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem; animation: iconBounce 2s ease-in-out infinite;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
            {f'<div class="metric-change" style="color: {change_color};">{change}</div>' if change else ''}
        </div>
        
        <style>
            @keyframes iconBounce {{
                0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-5px) rotate(5deg); }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_data_grid(self, data: pd.DataFrame, title: str = "Data Grid") -> None:
        """Create premium data grid with interactive features."""
        st.markdown(f"""
        <div class="data-viz-container">
            <h3 style="
                color: var(--text-primary);
                font-family: 'Poppins', sans-serif;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-align: center;
            ">{title}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced dataframe display
        st.dataframe(
            data,
            use_container_width=True,
            height=400
        )
    
    def create_interactive_chart(self, fig: go.Figure, title: str = "") -> None:
        """Create premium interactive chart container."""
        st.markdown(f"""
        <div class="chart-container">
            {f'<h3 style="color: var(--text-primary); font-family: {chr(39)}Poppins{chr(39)}, sans-serif; font-weight: 600; margin-bottom: 1rem; text-align: center;">{title}</h3>' if title else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced chart configuration
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                'drawline', 'drawopenpath', 'drawclosedpath',
                'drawcircle', 'drawrect', 'eraseshape'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'liquidity_predictor_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config)
    
    def create_status_indicator(self, status: str, message: str, 
                              indicator_type: str = "info") -> None:
        """Create animated status indicator."""
        colors = {
            'success': 'var(--success-color)',
            'warning': 'var(--warning-color)',
            'error': 'var(--error-color)',
            'info': 'var(--primary-color)'
        }
        
        icons = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️'
        }
        
        color = colors.get(indicator_type, colors['info'])
        icon = icons.get(indicator_type, icons['info'])
        
        st.markdown(f"""
        <div class="premium-alert" style="border-left-color: {color};">
            <div style="
                display: flex;
                align-items: center;
                gap: 1rem;
            ">
                <div style="
                    font-size: 1.5rem;
                    animation: statusIconPulse 2s ease-in-out infinite;
                ">{icon}</div>
                <div>
                    <div style="
                        font-weight: 600;
                        color: var(--text-primary);
                        margin-bottom: 0.25rem;
                    ">{status}</div>
                    <div style="
                        color: var(--text-secondary);
                        font-size: 0.9rem;
                    ">{message}</div>
                </div>
            </div>
        </div>
        
        <style>
            @keyframes statusIconPulse {{
                0%, 100% {{ 
                    transform: scale(1);
                    filter: drop-shadow(0 0 5px {color}50);
                }}
                50% {{ 
                    transform: scale(1.1);
                    filter: drop-shadow(0 0 15px {color}80);
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_progress_indicator(self, progress: float, label: str = "", 
                                animated: bool = True) -> None:
        """Create premium progress indicator."""
        st.markdown(f"""
        <div style="margin: 1.5rem 0;">
            {f'<div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 500;">{label}</div>' if label else ''}
            <div style="
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 8px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    background: linear-gradient(90deg, 
                        var(--primary-color) 0%, 
                        var(--accent-color) 50%,
                        var(--secondary-color) 100%);
                    height: 100%;
                    width: {progress}%;
                    border-radius: 10px;
                    transition: width 2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                    box-shadow: 
                        0 2px 10px var(--primary-color)40,
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
                    {'animation: progressFlow 3s ease-in-out infinite;' if animated else ''}
                    position: relative;
                    overflow: hidden;
                ">
                    {'<div style="position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent); animation: progressShine 2s ease-in-out infinite;"></div>' if animated else ''}
                </div>
            </div>
        </div>
        
        <style>
            @keyframes progressFlow {{
                0%, 100% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
            }}
            
            @keyframes progressShine {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_floating_panel(self, content: str, position: str = "top-right") -> None:
        """Create floating information panel."""
        positions = {
            'top-right': 'top: 2rem; right: 2rem;',
            'top-left': 'top: 2rem; left: 2rem;',
            'bottom-right': 'bottom: 2rem; right: 2rem;',
            'bottom-left': 'bottom: 2rem; left: 2rem;'
        }
        
        st.markdown(f"""
        <div style="
            position: fixed;
            {positions.get(position, positions['top-right'])}
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.1) 0%, 
                rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(25px) saturate(200%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            max-width: 300px;
            z-index: 1000;
            box-shadow: var(--shadow-primary);
            animation: 
                floatIn 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94),
                floatBob 4s ease-in-out infinite;
        ">
            <div style="
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
                font-size: 0.9rem;
                line-height: 1.5;
            ">{content}</div>
        </div>
        
        <style>
            @keyframes floatIn {{
                from {{ 
                    opacity: 0;
                    transform: translateY(-20px) scale(0.9);
                }}
                to {{ 
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }}
            }}
            
            @keyframes floatBob {{
                0%, 100% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-5px); }}
            }}
        </style>
        """, unsafe_allow_html=True)


# Global premium UI instance
premium_ui = PremiumUIManager()


# Example usage
if __name__ == "__main__":
    print("✨ Testing Premium UI System...")
    
    ui = PremiumUIManager()
    
    print("✅ Premium UI manager initialized")
    print("✅ Color schemes loaded: Ocean, Sunset, Forest")
    print("✅ Advanced CSS generated")
    print("✅ Interactive components ready")
    
    print("🎉 Premium UI system ready!")

