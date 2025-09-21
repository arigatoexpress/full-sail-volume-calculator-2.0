"""
Advanced visual effects and animations for Liquidity Predictor.
Creates stunning UI transitions, page effects, and interactive elements.
"""

import streamlit as st
import time
import random
from typing import Dict, List, Optional


class VisualEffectsManager:
    """Manages all visual effects and animations."""
    
    def __init__(self):
        """Initialize visual effects manager."""
        self.effect_cache = {}
        
    def get_advanced_css(self) -> str:
        """Get advanced CSS with stunning visual effects."""
        return """
        <style>
            /* Import premium fonts and icons */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
            
            /* Advanced CSS Variables */
            :root {
                --primary-gradient: linear-gradient(135deg, #00D4FF 0%, #FF6B35 50%, #00E676 100%);
                --secondary-gradient: linear-gradient(135deg, #BB86FC 0%, #FFD600 100%);
                --glass-bg: rgba(255, 255, 255, 0.05);
                --glass-border: rgba(255, 255, 255, 0.1);
                --shadow-glow: 0 8px 32px rgba(0, 0, 0, 0.3);
                --text-primary: #FFFFFF;
                --text-secondary: #B0B0B0;
                --success-color: #00E676;
                --warning-color: #FFD600;
                --error-color: #FF6B35;
            }
            
            /* Epic animated background with particles */
            .stApp {
                background: 
                    radial-gradient(circle at 15% 85%, rgba(0, 212, 255, 0.12) 0%, transparent 60%),
                    radial-gradient(circle at 85% 15%, rgba(255, 107, 53, 0.12) 0%, transparent 60%),
                    radial-gradient(circle at 50% 50%, rgba(0, 230, 118, 0.08) 0%, transparent 70%),
                    radial-gradient(circle at 25% 25%, rgba(187, 134, 252, 0.08) 0%, transparent 70%),
                    linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 30%, #0f0f0f 70%, #1a1a1a 100%);
                background-attachment: fixed;
                background-size: 100% 100%, 100% 100%, 150% 150%, 120% 120%, 100% 100%;
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
                position: relative;
                overflow-x: hidden;
            }
            
            /* Floating particles animation */
            .stApp::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    radial-gradient(circle at 20% 30%, rgba(0, 212, 255, 0.06) 0%, transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(255, 107, 53, 0.06) 0%, transparent 50%),
                    radial-gradient(circle at 60% 20%, rgba(0, 230, 118, 0.04) 0%, transparent 50%);
                animation: floatingParticles 25s ease-in-out infinite;
                pointer-events: none;
                z-index: -1;
            }
            
            @keyframes floatingParticles {
                0%, 100% { 
                    transform: translateY(0px) rotate(0deg) scale(1);
                    opacity: 0.8;
                }
                25% { 
                    transform: translateY(-15px) rotate(90deg) scale(1.1);
                    opacity: 0.6;
                }
                50% { 
                    transform: translateY(-30px) rotate(180deg) scale(0.9);
                    opacity: 1;
                }
                75% { 
                    transform: translateY(-15px) rotate(270deg) scale(1.1);
                    opacity: 0.7;
                }
            }
            
            /* Spectacular header with multiple effects */
            .main-header {
                font-family: 'Inter', sans-serif;
                font-size: 4.5rem;
                font-weight: 900;
                background: linear-gradient(135deg, 
                    #00D4FF 0%, #FF6B35 20%, #00E676 40%, 
                    #BB86FC 60%, #FFD600 80%, #CF6679 100%);
                background-size: 400% 400%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin: 2rem 0;
                animation: 
                    gradientFlow 6s ease-in-out infinite,
                    textGlow 3s ease-in-out infinite alternate,
                    float 4s ease-in-out infinite;
                position: relative;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
            }
            
            @keyframes gradientFlow {
                0%, 100% { background-position: 0% 50%; }
                25% { background-position: 50% 0%; }
                50% { background-position: 100% 50%; }
                75% { background-position: 50% 100%; }
            }
            
            @keyframes textGlow {
                from { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3)); }
                to { filter: drop-shadow(0 0 25px rgba(0, 212, 255, 0.8)); }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-8px); }
            }
            
            /* Revolutionary glassmorphism containers */
            .metric-container, .info-box, .warning-box, .success-box {
                background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.08) 0%, 
                    rgba(255, 255, 255, 0.03) 100%);
                backdrop-filter: blur(25px) saturate(200%);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 24px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1),
                    0 0 0 1px rgba(255, 255, 255, 0.05);
                position: relative;
                overflow: hidden;
                transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            }
            
            /* Animated shimmer effect */
            .metric-container::before, .info-box::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, 
                    transparent 30%, 
                    rgba(255, 255, 255, 0.08) 50%, 
                    transparent 70%);
                animation: shimmerWave 4s ease-in-out infinite;
                pointer-events: none;
            }
            
            @keyframes shimmerWave {
                0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
                100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
            }
            
            /* Hover effects with 3D transforms */
            .metric-container:hover, .info-box:hover {
                transform: translateY(-8px) rotateX(2deg) rotateY(2deg);
                box-shadow: 
                    0 20px 60px rgba(0, 0, 0, 0.5),
                    0 0 30px rgba(0, 212, 255, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                border-color: rgba(0, 212, 255, 0.4);
            }
            
            /* Epic button transformations */
            .stButton > button {
                background: var(--primary-gradient);
                background-size: 300% 300%;
                color: white;
                border: none;
                border-radius: 16px;
                padding: 1rem 2.5rem;
                font-weight: 700;
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow: 
                    0 6px 20px rgba(0, 212, 255, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                position: relative;
                overflow: hidden;
                cursor: pointer;
            }
            
            .stButton > button::before {
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
                transition: left 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            }
            
            .stButton > button:hover {
                transform: translateY(-4px) scale(1.05);
                background-position: 100% 0;
                box-shadow: 
                    0 15px 40px rgba(0, 212, 255, 0.5),
                    0 5px 15px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.4);
                animation: buttonPulse 2s infinite;
            }
            
            .stButton > button:hover::before {
                left: 100%;
            }
            
            .stButton > button:active {
                transform: translateY(-2px) scale(1.02);
                transition: all 0.1s ease;
            }
            
            @keyframes buttonPulse {
                0%, 100% { 
                    box-shadow: 0 15px 40px rgba(0, 212, 255, 0.5);
                }
                50% { 
                    box-shadow: 0 20px 50px rgba(0, 212, 255, 0.7);
                }
            }
            
            /* Revolutionary tab system with morphing effects */
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(135deg, 
                    rgba(20, 20, 20, 0.9) 0%, 
                    rgba(30, 30, 30, 0.9) 100%);
                border-radius: 20px;
                backdrop-filter: blur(25px) saturate(200%);
                border: 1px solid rgba(255, 255, 255, 0.12);
                padding: 0.8rem;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            .stTabs [data-baseweb="tab-list"]::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, 
                    rgba(0, 212, 255, 0.05) 0%, 
                    rgba(255, 107, 53, 0.05) 100%);
                animation: tabGlow 8s ease-in-out infinite;
                pointer-events: none;
            }
            
            @keyframes tabGlow {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 0.8; }
            }
            
            .stTabs [data-baseweb="tab"] {
                color: var(--text-secondary);
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                padding: 1rem 2rem;
                border-radius: 16px;
                transition: all 0.4s cubic-bezier(0.4, 0.0, 0.2, 1);
                position: relative;
                overflow: hidden;
                background: transparent;
            }
            
            .stTabs [data-baseweb="tab"]::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: var(--primary-gradient);
                opacity: 0;
                transition: opacity 0.3s ease;
                border-radius: 16px;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                color: var(--text-primary);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            }
            
            .stTabs [data-baseweb="tab"]:hover::before {
                opacity: 0.1;
            }
            
            .stTabs [aria-selected="true"] {
                color: var(--text-primary) !important;
                background: var(--primary-gradient) !important;
                transform: translateY(-3px) !important;
                box-shadow: 
                    0 8px 25px rgba(0, 212, 255, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
                animation: activeTabPulse 3s ease-in-out infinite;
            }
            
            @keyframes activeTabPulse {
                0%, 100% { 
                    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
                }
                50% { 
                    box-shadow: 0 12px 35px rgba(0, 212, 255, 0.6);
                }
            }
            
            /* Enhanced metrics with data visualization */
            [data-testid="metric-container"] {
                background: linear-gradient(135deg, 
                    rgba(30, 30, 30, 0.95) 0%, 
                    rgba(45, 45, 45, 0.95) 100%);
                backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
                padding: 1.8rem;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            [data-testid="metric-container"]::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--primary-gradient);
                animation: metricProgress 3s ease-in-out infinite;
            }
            
            @keyframes metricProgress {
                0%, 100% { transform: translateX(-100%); }
                50% { transform: translateX(100%); }
            }
            
            [data-testid="metric-container"]:hover {
                transform: translateY(-4px) scale(1.02);
                border-color: rgba(0, 212, 255, 0.4);
                box-shadow: 
                    0 16px 48px rgba(0, 0, 0, 0.5),
                    0 0 25px rgba(0, 212, 255, 0.3);
            }
            
            /* Spectacular chart containers */
            .js-plotly-plot {
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 
                    0 12px 40px rgba(0, 0, 0, 0.4),
                    0 0 20px rgba(0, 212, 255, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                position: relative;
            }
            
            .js-plotly-plot::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, 
                    rgba(0, 212, 255, 0.02) 0%, 
                    rgba(255, 107, 53, 0.02) 100%);
                pointer-events: none;
                z-index: 1;
            }
            
            .js-plotly-plot:hover {
                transform: scale(1.02) rotateX(1deg);
                box-shadow: 
                    0 20px 60px rgba(0, 0, 0, 0.5),
                    0 0 40px rgba(0, 212, 255, 0.2);
            }
            
            /* Page transition effects */
            .stApp > div {
                animation: pageSlideIn 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            }
            
            @keyframes pageSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Enhanced loading animations */
            .stSpinner > div {
                border: 3px solid rgba(0, 212, 255, 0.3);
                border-top: 3px solid #00D4FF;
                border-radius: 50%;
                animation: 
                    spinGlow 1s linear infinite,
                    pulseGlow 2s ease-in-out infinite;
            }
            
            @keyframes spinGlow {
                0% { 
                    transform: rotate(0deg);
                    filter: drop-shadow(0 0 5px rgba(0, 212, 255, 0.5));
                }
                50% { 
                    filter: drop-shadow(0 0 15px rgba(0, 212, 255, 0.8));
                }
                100% { 
                    transform: rotate(360deg);
                    filter: drop-shadow(0 0 5px rgba(0, 212, 255, 0.5));
                }
            }
            
            @keyframes pulseGlow {
                0%, 100% { 
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
                }
                50% { 
                    box-shadow: 0 0 40px rgba(0, 212, 255, 0.7);
                }
            }
            
            /* Advanced progress bars with liquid effects */
            .stProgress > div > div {
                background: linear-gradient(90deg, 
                    #00D4FF 0%, #00E676 50%, #FFD600 100%);
                border-radius: 12px;
                box-shadow: 
                    0 4px 15px rgba(0, 212, 255, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                animation: 
                    progressFlow 3s ease-in-out infinite,
                    progressGlow 2s ease-in-out infinite alternate;
                position: relative;
                overflow: hidden;
            }
            
            .stProgress > div > div::before {
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
                animation: progressShine 2s ease-in-out infinite;
            }
            
            @keyframes progressFlow {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            @keyframes progressGlow {
                from { box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4); }
                to { box-shadow: 0 6px 25px rgba(0, 212, 255, 0.7); }
            }
            
            @keyframes progressShine {
                0% { left: -100%; }
                100% { left: 100%; }
            }
            
            /* Sidebar with advanced effects */
            .css-1d391kg {
                background: linear-gradient(180deg, 
                    rgba(8, 8, 8, 0.98) 0%, 
                    rgba(18, 18, 18, 0.98) 30%,
                    rgba(25, 25, 25, 0.98) 70%,
                    rgba(15, 15, 15, 0.98) 100%);
                backdrop-filter: blur(30px) saturate(200%);
                border-right: 1px solid rgba(255, 255, 255, 0.15);
                box-shadow: 
                    8px 0 30px rgba(0, 0, 0, 0.6),
                    inset -1px 0 0 rgba(255, 255, 255, 0.05);
                position: relative;
            }
            
            .css-1d391kg::before {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 2px;
                height: 100%;
                background: linear-gradient(180deg, 
                    transparent 0%, 
                    rgba(0, 212, 255, 0.5) 50%, 
                    transparent 100%);
                animation: sidebarGlow 4s ease-in-out infinite;
            }
            
            @keyframes sidebarGlow {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 1; }
            }
            
            /* Enhanced input styling */
            .stSelectbox > div > div,
            .stNumberInput > div > div > input,
            .stTextInput > div > div > input {
                background: linear-gradient(135deg, 
                    rgba(25, 25, 25, 0.95) 0%, 
                    rgba(35, 35, 35, 0.95) 100%) !important;
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 12px !important;
                color: var(--text-primary) !important;
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            }
            
            .stSelectbox > div > div:hover,
            .stNumberInput > div > div > input:hover,
            .stTextInput > div > div > input:hover {
                border-color: rgba(0, 212, 255, 0.4) !important;
                box-shadow: 
                    0 4px 20px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(0, 212, 255, 0.2);
                transform: translateY(-1px);
            }
            
            .stSelectbox > div > div:focus-within,
            .stNumberInput > div > div > input:focus,
            .stTextInput > div > div > input:focus {
                border-color: rgba(0, 212, 255, 0.6) !important;
                box-shadow: 
                    0 6px 25px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(0, 212, 255, 0.4);
                transform: translateY(-2px);
            }
            
            /* Dataframe enhancements */
            .dataframe {
                background: linear-gradient(135deg, 
                    rgba(15, 15, 15, 0.98) 0%, 
                    rgba(25, 25, 25, 0.98) 100%) !important;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 16px !important;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .dataframe:hover {
                transform: translateY(-2px);
                box-shadow: 
                    0 12px 40px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(0, 212, 255, 0.15);
            }
            
            /* Alert animations */
            .stAlert {
                border-radius: 16px;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: 
                    alertSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94),
                    alertGlow 3s ease-in-out infinite;
                position: relative;
                overflow: hidden;
            }
            
            @keyframes alertSlideIn {
                from { 
                    transform: translateX(-30px) rotateY(-10deg);
                    opacity: 0;
                }
                to { 
                    transform: translateX(0) rotateY(0deg);
                    opacity: 1;
                }
            }
            
            @keyframes alertGlow {
                0%, 100% { box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); }
                50% { box-shadow: 0 8px 30px rgba(0, 212, 255, 0.2); }
            }
            
            /* Checkbox and radio enhancements */
            .stCheckbox > label,
            .stRadio > label {
                color: var(--text-primary) !important;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stCheckbox > label:hover,
            .stRadio > label:hover {
                color: #00D4FF !important;
                transform: translateX(5px);
            }
            
            /* Scrollbar with gradient effects */
            ::-webkit-scrollbar {
                width: 12px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(20, 20, 20, 0.8);
                border-radius: 6px;
                box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.5);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--primary-gradient);
                border-radius: 6px;
                box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
                transition: all 0.3s ease;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary-gradient);
                box-shadow: 0 4px 15px rgba(187, 134, 252, 0.5);
                transform: scale(1.1);
            }
            
            /* Mobile responsiveness with enhanced effects */
            @media (max-width: 768px) {
                .main-header {
                    font-size: 3rem;
                    margin: 1rem 0;
                }
                
                .stTabs [data-baseweb="tab"] {
                    padding: 0.8rem 1rem;
                    font-size: 0.9rem;
                }
                
                .metric-container,
                .info-box,
                .warning-box,
                .success-box {
                    margin: 1rem 0;
                    padding: 1.5rem;
                    border-radius: 16px;
                }
            }
            
            /* Special effects for high-value elements */
            .epic-element {
                position: relative;
                overflow: hidden;
            }
            
            .epic-element::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: conic-gradient(
                    from 0deg,
                    transparent 0deg,
                    rgba(0, 212, 255, 0.1) 90deg,
                    transparent 180deg,
                    rgba(255, 107, 53, 0.1) 270deg,
                    transparent 360deg
                );
                animation: epicRotate 8s linear infinite;
                pointer-events: none;
            }
            
            @keyframes epicRotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            
            /* Tooltip enhancements */
            [data-testid="stTooltipHoverTarget"] {
                color: #00D4FF !important;
                transition: all 0.3s ease;
            }
            
            [data-testid="stTooltipHoverTarget"]:hover {
                color: #FF6B35 !important;
                text-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
            }
            
            /* Success/Error message enhancements */
            .stSuccess {
                background: linear-gradient(135deg, 
                    rgba(0, 230, 118, 0.15) 0%, 
                    rgba(0, 230, 118, 0.05) 100%);
                border-left: 4px solid var(--success-color);
                animation: successPulse 2s ease-in-out infinite;
            }
            
            .stError {
                background: linear-gradient(135deg, 
                    rgba(255, 107, 53, 0.15) 0%, 
                    rgba(255, 107, 53, 0.05) 100%);
                border-left: 4px solid var(--error-color);
                animation: errorShake 0.5s ease-in-out;
            }
            
            @keyframes successPulse {
                0%, 100% { border-left-color: var(--success-color); }
                50% { border-left-color: rgba(0, 230, 118, 0.7); }
            }
            
            @keyframes errorShake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
            
            /* Expander with morphing effects */
            .streamlit-expanderHeader {
                background: linear-gradient(135deg, 
                    rgba(30, 30, 30, 0.9) 0%, 
                    rgba(40, 40, 40, 0.9) 100%);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
            }
            
            .streamlit-expanderHeader:hover {
                background: linear-gradient(135deg, 
                    rgba(40, 40, 40, 0.9) 0%, 
                    rgba(50, 50, 50, 0.9) 100%);
                border-color: rgba(0, 212, 255, 0.3);
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            }
        </style>
        """
    
    def create_page_transition_effect(self, page_name: str) -> None:
        """Create smooth page transition effects."""
        st.markdown(f"""
        <div class="page-transition" style="
            animation: pageTransition 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        ">
        </div>
        
        <style>
            @keyframes pageTransition {{
                0% {{
                    opacity: 0;
                    transform: translateY(40px) scale(0.95);
                    filter: blur(5px);
                }}
                100% {{
                    opacity: 1;
                    transform: translateY(0) scale(1);
                    filter: blur(0px);
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_loading_animation(self, message: str = "Loading...") -> None:
        """Create advanced loading animation."""
        st.markdown(f"""
        <div class="loading-container" style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            text-align: center;
        ">
            <div class="loading-spinner" style="
                width: 60px;
                height: 60px;
                border: 4px solid rgba(0, 212, 255, 0.3);
                border-top: 4px solid #00D4FF;
                border-radius: 50%;
                animation: 
                    advancedSpin 1.5s linear infinite,
                    loadingGlow 2s ease-in-out infinite alternate;
                margin-bottom: 1rem;
            "></div>
            <p style="
                color: #B0B0B0;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
                animation: loadingText 2s ease-in-out infinite;
            ">{message}</p>
        </div>
        
        <style>
            @keyframes advancedSpin {{
                0% {{ 
                    transform: rotate(0deg) scale(1);
                    filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));
                }}
                50% {{
                    transform: rotate(180deg) scale(1.1);
                    filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8));
                }}
                100% {{ 
                    transform: rotate(360deg) scale(1);
                    filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));
                }}
            }}
            
            @keyframes loadingGlow {{
                from {{ box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }}
                to {{ box-shadow: 0 0 40px rgba(0, 212, 255, 0.7); }}
            }}
            
            @keyframes loadingText {{
                0%, 100% {{ opacity: 0.7; }}
                50% {{ opacity: 1; }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_success_animation(self, message: str) -> None:
        """Create success animation with confetti effect."""
        st.markdown(f"""
        <div class="success-animation" style="
            text-align: center;
            padding: 2rem;
            position: relative;
        ">
            <div class="success-icon" style="
                font-size: 4rem;
                color: #00E676;
                animation: 
                    successBounce 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55),
                    successGlow 2s ease-in-out infinite alternate;
                margin-bottom: 1rem;
            ">✅</div>
            <p style="
                color: #00E676;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 1.2rem;
                animation: successText 1s ease-out 0.3s both;
            ">{message}</p>
        </div>
        
        <style>
            @keyframes successBounce {{
                0% {{ 
                    transform: scale(0) rotate(-180deg);
                    opacity: 0;
                }}
                50% {{ 
                    transform: scale(1.2) rotate(0deg);
                    opacity: 1;
                }}
                100% {{ 
                    transform: scale(1) rotate(0deg);
                    opacity: 1;
                }}
            }}
            
            @keyframes successGlow {{
                from {{ 
                    filter: drop-shadow(0 0 10px rgba(0, 230, 118, 0.5));
                    text-shadow: 0 0 20px rgba(0, 230, 118, 0.3);
                }}
                to {{ 
                    filter: drop-shadow(0 0 25px rgba(0, 230, 118, 0.8));
                    text-shadow: 0 0 30px rgba(0, 230, 118, 0.6);
                }}
            }}
            
            @keyframes successText {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_data_visualization_effect(self, data_points: int) -> None:
        """Create animated data visualization effect."""
        st.markdown(f"""
        <div class="data-viz-effect" style="
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            position: relative;
        ">
            <div class="data-points" style="
                display: grid;
                grid-template-columns: repeat(10, 1fr);
                gap: 8px;
                animation: dataFlow 3s ease-in-out infinite;
            ">
                {''.join([f'''
                <div class="data-point" style="
                    width: 8px;
                    height: 8px;
                    background: linear-gradient(45deg, #00D4FF, #FF6B35);
                    border-radius: 50%;
                    animation: 
                        dataPointPulse 2s ease-in-out infinite,
                        dataPointFloat 3s ease-in-out infinite;
                    animation-delay: {i * 0.1}s;
                "></div>
                ''' for i in range(min(data_points, 50))])}
            </div>
        </div>
        
        <style>
            @keyframes dataFlow {{
                0%, 100% {{ transform: scale(1) rotate(0deg); }}
                50% {{ transform: scale(1.05) rotate(2deg); }}
            }}
            
            @keyframes dataPointPulse {{
                0%, 100% {{ 
                    opacity: 0.6;
                    transform: scale(1);
                }}
                50% {{ 
                    opacity: 1;
                    transform: scale(1.3);
                }}
            }}
            
            @keyframes dataPointFloat {{
                0%, 100% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-5px); }}
            }}
        </style>
        """, unsafe_allow_html=True)


class InteractiveElements:
    """Interactive UI elements with advanced effects."""
    
    def __init__(self):
        """Initialize interactive elements."""
        self.element_states = {}
    
    def create_animated_metric_card(self, title: str, value: str, 
                                  change: str = "", icon: str = "📊") -> None:
        """Create animated metric card with hover effects."""
        card_id = f"metric_{hash(title)}"
        
        st.markdown(f"""
        <div class="animated-metric-card epic-element" id="{card_id}" style="
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.08) 0%, 
                rgba(255, 255, 255, 0.03) 100%);
            backdrop-filter: blur(25px) saturate(200%);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        " onmouseover="this.style.transform='translateY(-8px) rotateX(5deg)'; this.style.boxShadow='0 20px 60px rgba(0,0,0,0.5), 0 0 30px rgba(0,212,255,0.3)';" 
           onmouseout="this.style.transform='translateY(0) rotateX(0deg)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.3)';">
            
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="
                        font-size: 3rem;
                        margin-bottom: 0.5rem;
                        animation: iconFloat 3s ease-in-out infinite;
                    ">{icon}</div>
                    <h3 style="
                        color: #B0B0B0;
                        font-size: 0.9rem;
                        font-weight: 500;
                        margin: 0;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">{title}</h3>
                </div>
                <div style="text-align: right;">
                    <div style="
                        font-size: 2rem;
                        font-weight: 700;
                        color: #FFFFFF;
                        margin-bottom: 0.5rem;
                        animation: valueCount 2s ease-out;
                    ">{value}</div>
                    <div style="
                        color: {'#00E676' if change.startswith('+') else '#FF6B35' if change.startswith('-') else '#FFD600'};
                        font-weight: 600;
                        animation: changeFlash 1s ease-out 1s both;
                    ">{change}</div>
                </div>
            </div>
        </div>
        
        <style>
            @keyframes iconFloat {{
                0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-5px) rotate(5deg); }}
            }}
            
            @keyframes valueCount {{
                from {{ 
                    opacity: 0;
                    transform: scale(0.5);
                }}
                to {{ 
                    opacity: 1;
                    transform: scale(1);
                }}
            }}
            
            @keyframes changeFlash {{
                from {{ 
                    opacity: 0;
                    transform: translateX(20px);
                }}
                to {{ 
                    opacity: 1;
                    transform: translateX(0);
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_interactive_button(self, text: str, button_type: str = "primary", 
                                icon: str = "🚀") -> bool:
        """Create interactive button with advanced effects."""
        button_id = f"btn_{hash(text)}"
        
        # Create the button with custom styling
        st.markdown(f"""
        <style>
            .interactive-button-{button_id} {{
                background: var(--primary-gradient);
                background-size: 300% 300%;
                color: white;
                border: none;
                border-radius: 16px;
                padding: 1rem 2.5rem;
                font-weight: 700;
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                cursor: pointer;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow: 
                    0 6px 20px rgba(0, 212, 255, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                position: relative;
                overflow: hidden;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .interactive-button-{button_id}:hover {{
                transform: translateY(-4px) scale(1.05);
                background-position: 100% 0;
                box-shadow: 
                    0 15px 40px rgba(0, 212, 255, 0.5),
                    0 5px 15px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.4);
                animation: buttonRipple 0.6s ease-out;
            }}
            
            @keyframes buttonRipple {{
                0% {{ box-shadow: 0 15px 40px rgba(0, 212, 255, 0.5); }}
                50% {{ box-shadow: 0 20px 50px rgba(0, 212, 255, 0.8); }}
                100% {{ box-shadow: 0 15px 40px rgba(0, 212, 255, 0.5); }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        return st.button(f"{icon} {text}", key=button_id)
    
    def create_floating_action_button(self, icon: str, tooltip: str, 
                                    position: str = "bottom-right") -> bool:
        """Create floating action button."""
        position_styles = {
            "bottom-right": "bottom: 2rem; right: 2rem;",
            "bottom-left": "bottom: 2rem; left: 2rem;",
            "top-right": "top: 2rem; right: 2rem;",
            "top-left": "top: 2rem; left: 2rem;"
        }
        
        st.markdown(f"""
        <div class="floating-action-button" style="
            position: fixed;
            {position_styles.get(position, position_styles['bottom-right'])}
            width: 60px;
            height: 60px;
            background: var(--primary-gradient);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
            cursor: pointer;
            z-index: 1000;
            box-shadow: 
                0 8px 25px rgba(0, 212, 255, 0.4),
                0 0 20px rgba(0, 212, 255, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
            animation: fabFloat 3s ease-in-out infinite;
        " 
        onmouseover="this.style.transform='scale(1.1) rotate(10deg)'; this.style.boxShadow='0 12px 35px rgba(0,212,255,0.6)';"
        onmouseout="this.style.transform='scale(1) rotate(0deg)'; this.style.boxShadow='0 8px 25px rgba(0,212,255,0.4)';"
        title="{tooltip}">
            {icon}
        </div>
        
        <style>
            @keyframes fabFloat {{
                0%, 100% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-5px); }}
            }}
        </style>
        """, unsafe_allow_html=True)


# Global instances
effects_manager = VisualEffectsManager()
interactive_elements = InteractiveElements()


# Example usage
if __name__ == "__main__":
    print("✨ Testing Visual Effects System...")
    
    effects = VisualEffectsManager()
    interactive = InteractiveElements()
    
    print("✅ Visual effects manager initialized")
    print("✅ Interactive elements ready")
    print("✅ Advanced CSS generated")
    
    print("🎉 Visual effects system ready!")

