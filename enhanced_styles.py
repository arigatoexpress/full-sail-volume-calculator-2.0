"""
🎨 ENHANCED STYLES
Additional CSS for onboarding, chat interface, and enhanced navigation
"""

ENHANCED_CSS = """
<style>
    /* Onboarding Styles */
    .onboarding-container {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(0, 212, 255, 0.05) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 212, 255, 0.2),
            inset 0 1px 0 rgba(0, 212, 255, 0.1);
        text-align: center;
    }
    
    .onboarding-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00D4FF 0%, #FF6B35 50%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .onboarding-step {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .onboarding-step:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .onboarding-success {
        background: linear-gradient(135deg, 
            rgba(0, 230, 118, 0.15) 0%, 
            rgba(0, 230, 118, 0.05) 100%);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .onboarding-success h3 {
        color: #00E676;
        margin-bottom: 1rem;
    }
    
    .onboarding-success ul {
        text-align: left;
        margin: 1rem 0;
    }
    
    .onboarding-success li {
        margin: 0.5rem 0;
        color: #e0e0e0;
    }
    
    /* Chat Interface Styles */
    .chat-header {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(0, 212, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .chat-header h2 {
        color: #00D4FF;
        margin-bottom: 0.5rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.2) 0%, 
            rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        margin-left: 2rem;
        position: relative;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        left: -10px;
        top: 50%;
        transform: translateY(-50%);
        width: 0;
        height: 0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        border-right: 10px solid rgba(0, 212, 255, 0.3);
    }
    
    .ai-message {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        margin-right: 2rem;
        position: relative;
    }
    
    .ai-message::before {
        content: '';
        position: absolute;
        right: -10px;
        top: 50%;
        transform: translateY(-50%);
        width: 0;
        height: 0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        border-left: 10px solid rgba(255, 255, 255, 0.2);
    }
    
    .user-message strong,
    .ai-message strong {
        color: #00D4FF;
    }
    
    .user-message small,
    .ai-message small {
        color: #888;
        font-size: 0.8rem;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* Help Tooltip Styles */
    .help-tooltip {
        position: relative;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .help-icon {
        color: #00D4FF;
        cursor: help;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .help-icon:hover {
        color: #FF6B35;
        transform: scale(1.2);
    }
    
    .help-tooltip:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Smart Suggestions Styles */
    .suggestion-card {
        background: linear-gradient(135deg, 
            rgba(255, 215, 0, 0.1) 0%, 
            rgba(255, 215, 0, 0.05) 100%);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .suggestion-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
        border-color: rgba(255, 215, 0, 0.5);
    }
    
    .suggestion-title {
        color: #FFD700;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .suggestion-description {
        color: #e0e0e0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Enhanced Button Styles */
    .stButton > button[type="primary"] {
        background: linear-gradient(135deg, #00D4FF 0%, #FF6B35 50%, #00E676 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease-in-out infinite;
    }
    
    .stButton > button[type="primary"]:hover {
        background-position: 100% 0;
        transform: translateY(-2px) scale(1.02);
    }
    
    /* Progress Bar Enhancements */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00D4FF 0%, #FF6B35 50%, #00E676 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Enhanced Sidebar Styles */
    .sidebar-section {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.05) 0%, 
            rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-section h3 {
        color: #00D4FF;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Quick Action Buttons */
    .quick-action-button {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.2) 0%, 
            rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
        color: #ffffff;
        font-weight: 600;
    }
    
    .quick-action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        border-color: rgba(0, 212, 255, 0.5);
    }
    
    /* Enhanced Metric Cards */
    .enhanced-metric {
        background: linear-gradient(135deg, 
            rgba(30, 30, 30, 0.9) 0%, 
            rgba(40, 40, 40, 0.9) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .enhanced-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .enhanced-metric:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Loading States */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(0, 212, 255, 0.05) 100%);
        border-radius: 16px;
        margin: 2rem 0;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(0, 212, 255, 0.3);
        border-top: 4px solid #00D4FF;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .onboarding-container {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .onboarding-header h1 {
            font-size: 2rem;
        }
        
        .chat-header {
            padding: 1rem;
        }
        
        .user-message,
        .ai-message {
            margin-left: 0.5rem;
            margin-right: 0.5rem;
        }
        
        .enhanced-metric {
            padding: 1rem;
        }
    }
    
    /* Animation Enhancements */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Success States */
    .success-indicator {
        background: linear-gradient(135deg, 
            rgba(0, 230, 118, 0.2) 0%, 
            rgba(0, 230, 118, 0.1) 100%);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #00E676;
        text-align: center;
    }
    
    .success-indicator::before {
        content: '✅ ';
        font-size: 1.2rem;
    }
    
    /* Warning States */
    .warning-indicator {
        background: linear-gradient(135deg, 
            rgba(255, 107, 53, 0.2) 0%, 
            rgba(255, 107, 53, 0.1) 100%);
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #FF6B35;
        text-align: center;
    }
    
    .warning-indicator::before {
        content: '⚠️ ';
        font-size: 1.2rem;
    }
    
    /* Info States */
    .info-indicator {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.2) 0%, 
            rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #00D4FF;
        text-align: center;
    }
    
    .info-indicator::before {
        content: 'ℹ️ ';
        font-size: 1.2rem;
    }
</style>
"""

def get_enhanced_css():
    """Get enhanced CSS styles."""
    return ENHANCED_CSS
