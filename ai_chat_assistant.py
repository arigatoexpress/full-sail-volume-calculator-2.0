"""
🤖 AI CHAT ASSISTANT
Centralized conversational AI interface for the Full Sail Volume Calculator
"""

import streamlit as st
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

class AIChatAssistant:
    """AI Chat Assistant for natural language interaction with the platform."""
    
    def __init__(self):
        """Initialize AI Chat Assistant."""
        self.initialize_chat_state()
        self.setup_knowledge_base()
    
    def initialize_chat_state(self):
        """Initialize chat session state."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = {
                'current_pool': 'SAIL/USDC',
                'current_timeframe': '30d',
                'user_intent': 'general',
                'last_action': None
            }
        
        if 'ai_suggestions' not in st.session_state:
            st.session_state.ai_suggestions = []
    
    def setup_knowledge_base(self):
        """Setup knowledge base for AI responses."""
        self.knowledge_base = {
            'pools': {
                'SAIL/USDC': {
                    'description': 'Full Sail Finance native token paired with USDC',
                    'category': 'DeFi',
                    'popularity': 'high',
                    'volatility': 'medium'
                },
                'SUI/USDC': {
                    'description': 'Sui blockchain native token paired with USDC',
                    'category': 'Layer 1',
                    'popularity': 'high',
                    'volatility': 'medium'
                },
                'IKA/SUI': {
                    'description': 'IKA token paired with Sui',
                    'category': 'DeFi',
                    'popularity': 'medium',
                    'volatility': 'high'
                }
            },
            'models': {
                'Ensemble': {
                    'description': 'Combines Prophet and ARIMA for robust predictions',
                    'accuracy': 'high',
                    'best_for': 'general predictions'
                },
                'Prophet': {
                    'description': 'Facebook\'s time series forecasting model',
                    'accuracy': 'high',
                    'best_for': 'seasonal patterns'
                },
                'ARIMA': {
                    'description': 'Classical statistical time series model',
                    'accuracy': 'medium',
                    'best_for': 'stationary data'
                }
            },
            'features': {
                'predictions': 'Generate volume predictions using AI models',
                'charts': 'View interactive charts and technical analysis',
                'arbitrage': 'Find arbitrage opportunities across DEXs',
                'yield_farming': 'Optimize yield farming strategies',
                'ai_insights': 'Get AI-powered market insights'
            }
        }
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.markdown("""
        <div class="chat-header">
            <h2>🤖 AI Assistant</h2>
            <p>Ask me anything about DeFi, predictions, or analysis!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                self.render_chat_message(message)
            
            # Chat input
            self.render_chat_input()
            
            # Smart suggestions
            self.render_smart_suggestions()
    
    def render_chat_message(self, message: Dict):
        """Render a chat message."""
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message['content']}
                <small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <strong>AI Assistant:</strong> {message['content']}
                <small>{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Render any actions or suggestions
            if 'actions' in message:
                self.render_message_actions(message['actions'])
    
    def render_message_actions(self, actions: List[Dict]):
        """Render actionable buttons from AI responses."""
        for action in actions:
            if action['type'] == 'button':
                if st.button(action['text'], key=action['id']):
                    self.execute_action(action)
            elif action['type'] == 'chart':
                st.plotly_chart(action['chart'], width='stretch')
    
    def render_chat_input(self):
        """Render chat input area."""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything...",
                placeholder="e.g., 'Show me SAIL/USDC predictions' or 'What's the best pool to analyze?'",
                key="chat_input"
            )
        
        with col2:
            if st.button("Send", type="primary", width='stretch'):
                if user_input:
                    self.process_user_input(user_input)
                    st.rerun()
    
    def render_smart_suggestions(self):
        """Render smart suggestions based on context."""
        if st.session_state.ai_suggestions:
            st.markdown("### 💡 Smart Suggestions")
            
            for idx, suggestion in enumerate(st.session_state.ai_suggestions):
                with st.expander(f"💡 {suggestion['title']}", expanded=False):
                    st.markdown(suggestion['description'])
                    unique_key = f"{suggestion['id']}_{idx}"
                    if st.button(f"Try: {suggestion['action']}", key=unique_key):
                        self.execute_suggestion(suggestion)
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response."""
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime('%H:%M')
        }
        st.session_state.chat_history.append(user_message)
        
        # Process the input
        response = self.generate_ai_response(user_input)
        
        # Add AI response to history
        ai_message = {
            'role': 'assistant',
            'content': response['content'],
            'timestamp': datetime.now().strftime('%H:%M'),
            'actions': response.get('actions', [])
        }
        st.session_state.chat_history.append(ai_message)
        
        # Update context
        self.update_chat_context(user_input, response)
        
        # Generate new suggestions
        st.session_state.ai_suggestions = self.generate_smart_suggestions()
    
    def generate_ai_response(self, user_input: str) -> Dict:
        """Generate AI response based on user input."""
        # Parse user intent
        intent = self.parse_user_intent(user_input)
        
        # Generate response based on intent
        if intent['type'] == 'prediction_request':
            return self.handle_prediction_request(intent)
        elif intent['type'] == 'pool_analysis':
            return self.handle_pool_analysis_request(intent)
        elif intent['type'] == 'feature_explanation':
            return self.handle_feature_explanation(intent)
        elif intent['type'] == 'help_request':
            return self.handle_help_request(intent)
        else:
            return self.handle_general_query(user_input)
    
    def parse_user_intent(self, user_input: str) -> Dict:
        """Parse user input to determine intent."""
        user_input_lower = user_input.lower()
        
        # Prediction-related keywords
        prediction_keywords = ['predict', 'forecast', 'future', 'next', 'will', 'expect']
        if any(keyword in user_input_lower for keyword in prediction_keywords):
            return {
                'type': 'prediction_request',
                'pool': self.extract_pool(user_input),
                'timeframe': self.extract_timeframe(user_input),
                'model': self.extract_model(user_input)
            }
        
        # Pool analysis keywords
        pool_keywords = ['analyze', 'analysis', 'chart', 'volume', 'price', 'performance']
        if any(keyword in user_input_lower for keyword in pool_keywords):
            return {
                'type': 'pool_analysis',
                'pool': self.extract_pool(user_input),
                'metric': self.extract_metric(user_input)
            }
        
        # Feature explanation keywords
        feature_keywords = ['what is', 'how does', 'explain', 'tell me about']
        if any(keyword in user_input_lower for keyword in feature_keywords):
            return {
                'type': 'feature_explanation',
                'feature': self.extract_feature(user_input)
            }
        
        # Help keywords
        help_keywords = ['help', 'how to', 'guide', 'tutorial', 'start']
        if any(keyword in user_input_lower for keyword in help_keywords):
            return {'type': 'help_request'}
        
        return {'type': 'general_query'}
    
    def extract_pool(self, user_input: str) -> str:
        """Extract pool from user input."""
        pools = ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI', 'USDZ/USDC', 'USDT/USDC', 'wBTC/USDC', 'ETH/USDC']
        
        for pool in pools:
            if pool.lower() in user_input.lower():
                return pool
        
        return st.session_state.chat_context.get('current_pool', 'SAIL/USDC')
    
    def extract_timeframe(self, user_input: str) -> str:
        """Extract timeframe from user input."""
        timeframes = ['7d', '14d', '30d', '60d', '90d']
        
        for timeframe in timeframes:
            if timeframe in user_input:
                return timeframe
        
        return '30d'
    
    def extract_model(self, user_input: str) -> str:
        """Extract model from user input."""
        models = ['Ensemble', 'Prophet', 'ARIMA']
        
        for model in models:
            if model.lower() in user_input.lower():
                return model
        
        return 'Ensemble'
    
    def extract_metric(self, user_input: str) -> str:
        """Extract metric from user input."""
        metrics = ['volume', 'price', 'liquidity', 'fees', 'apr']
        
        for metric in metrics:
            if metric in user_input.lower():
                return metric
        
        return 'volume'
    
    def extract_feature(self, user_input: str) -> str:
        """Extract feature from user input."""
        features = ['predictions', 'charts', 'arbitrage', 'yield farming', 'ai insights']
        
        for feature in features:
            if feature in user_input.lower():
                return feature
        
        return 'general'
    
    def handle_prediction_request(self, intent: Dict) -> Dict:
        """Handle prediction request."""
        pool = intent.get('pool', 'SAIL/USDC')
        timeframe = intent.get('timeframe', '30d')
        model = intent.get('model', 'Ensemble')
        
        response = f"""
        I'll help you generate a prediction for {pool} using the {model} model with {timeframe} of historical data.
        
        **Here's what I'll do:**
        1. 📊 Analyze {timeframe} of historical volume data for {pool}
        2. 🤖 Apply the {model} model for forecasting
        3. 📈 Generate predictions with confidence intervals
        4. 💡 Provide insights and recommendations
        
        The {model} model is excellent for this type of analysis because it combines multiple forecasting approaches for robust predictions.
        """
        
        actions = [
            {
                'type': 'button',
                'text': f'🚀 Generate {pool} Prediction',
                'id': f'predict_{pool}_{model}',
                'action': 'generate_prediction',
                'params': {'pool': pool, 'timeframe': timeframe, 'model': model}
            }
        ]
        
        return {
            'content': response,
            'actions': actions
        }
    
    def handle_pool_analysis_request(self, intent: Dict) -> Dict:
        """Handle pool analysis request."""
        pool = intent.get('pool', 'SAIL/USDC')
        metric = intent.get('metric', 'volume')
        
        pool_info = self.knowledge_base['pools'].get(pool, {})
        
        response = f"""
        Let me analyze {pool} for you!
        
        **Pool Information:**
        - **Description**: {pool_info.get('description', 'Popular trading pair')}
        - **Category**: {pool_info.get('category', 'DeFi')}
        - **Popularity**: {pool_info.get('popularity', 'medium')}
        - **Volatility**: {pool_info.get('volatility', 'medium')}
        
        **Analysis Focus**: {metric.title()} analysis
        """
        
        actions = [
            {
                'type': 'button',
                'text': f'📊 View {pool} Charts',
                'id': f'chart_{pool}',
                'action': 'show_charts',
                'params': {'pool': pool, 'metric': metric}
            },
            {
                'type': 'button',
                'text': f'🔍 Deep Analysis',
                'id': f'analyze_{pool}',
                'action': 'deep_analysis',
                'params': {'pool': pool}
            }
        ]
        
        return {
            'content': response,
            'actions': actions
        }
    
    def handle_feature_explanation(self, intent: Dict) -> Dict:
        """Handle feature explanation request."""
        feature = intent.get('feature', 'general')
        
        if feature in self.knowledge_base['features']:
            description = self.knowledge_base['features'][feature]
            response = f"""
            **{feature.title()}** - {description}
            
            This feature helps you understand and analyze DeFi markets more effectively. Would you like me to show you how to use it?
            """
        else:
            response = """
            I can help explain various features of the platform:
            
            - **Predictions**: AI-powered volume forecasting
            - **Charts**: Interactive technical analysis
            - **Arbitrage**: Cross-DEX opportunity detection
            - **Yield Farming**: Optimization strategies
            - **AI Insights**: Market intelligence
            
            What specific feature would you like to learn about?
            """
        
        return {'content': response}
    
    def handle_help_request(self, intent: Dict) -> Dict:
        """Handle help request."""
        response = """
        I'm here to help! Here are some things I can do:
        
        **🔮 Predictions**
        - "Generate a prediction for SAIL/USDC"
        - "What will the volume be next week?"
        - "Show me a 7-day forecast"
        
        **📊 Analysis**
        - "Analyze SUI/USDC performance"
        - "Show me volume charts"
        - "Compare different pools"
        
        **🤖 AI Features**
        - "Explain how Prophet model works"
        - "What's the best model for predictions?"
        - "Help me understand confidence intervals"
        
        **🆘 Getting Started**
        - "How do I use this platform?"
        - "What should I do first?"
        - "Show me the main features"
        
        Just ask me anything in natural language!
        """
        
        return {'content': response}
    
    def handle_general_query(self, user_input: str) -> Dict:
        """Handle general queries."""
        response = f"""
        I understand you're asking about: "{user_input}"
        
        I can help you with:
        - 🔮 Generating volume predictions
        - 📊 Analyzing pool performance
        - 🤖 Explaining AI models and features
        - 🆘 Getting started with the platform
        
        Could you be more specific about what you'd like to do? For example:
        - "Generate a prediction for SAIL/USDC"
        - "Show me how to analyze volume"
        - "Explain the Prophet model"
        """
        
        return {'content': response}
    
    def execute_action(self, action: Dict):
        """Execute an action from AI response."""
        action_type = action['action']
        params = action.get('params', {})
        
        if action_type == 'generate_prediction':
            self.generate_prediction_action(params)
        elif action_type == 'show_charts':
            self.show_charts_action(params)
        elif action_type == 'deep_analysis':
            self.deep_analysis_action(params)
    
    def generate_prediction_action(self, params: Dict):
        """Execute prediction generation action."""
        pool = params.get('pool', 'SAIL/USDC')
        model = params.get('model', 'Ensemble')
        
        # Set session state for prediction
        st.session_state.smart_defaults['pool'] = pool
        st.session_state.smart_defaults['model'] = model
        st.session_state.show_prediction_modal = True
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"🚀 Generating {model} prediction for {pool}...",
            'timestamp': datetime.now().strftime('%H:%M')
        })
    
    def show_charts_action(self, params: Dict):
        """Execute show charts action."""
        pool = params.get('pool', 'SAIL/USDC')
        
        # Set session state for charts
        st.session_state.smart_defaults['pool'] = pool
        st.session_state.active_tab = 'analytics'
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"📊 Opening charts for {pool}...",
            'timestamp': datetime.now().strftime('%H:%M')
        })
    
    def deep_analysis_action(self, params: Dict):
        """Execute deep analysis action."""
        pool = params.get('pool', 'SAIL/USDC')
        
        # Set session state for analysis
        st.session_state.smart_defaults['pool'] = pool
        st.session_state.active_tab = 'analytics'
        st.session_state.show_deep_analysis = True
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"🔍 Starting deep analysis for {pool}...",
            'timestamp': datetime.now().strftime('%H:%M')
        })
    
    def execute_suggestion(self, suggestion: Dict):
        """Execute a smart suggestion."""
        # Process suggestion similar to user input
        self.process_user_input(suggestion['action'])
    
    def generate_smart_suggestions(self) -> List[Dict]:
        """Generate smart suggestions based on current context."""
        suggestions = []
        
        current_pool = st.session_state.chat_context.get('current_pool', 'SAIL/USDC')
        
        # Context-based suggestions
        if len(st.session_state.chat_history) < 3:
            suggestions.extend([
                {
                    'id': 'suggest_prediction',
                    'title': 'Generate Your First Prediction',
                    'description': f'Create a volume prediction for {current_pool} using AI models',
                    'action': f'Generate a prediction for {current_pool}'
                },
                {
                    'id': 'suggest_analysis',
                    'title': 'Analyze Pool Performance',
                    'description': f'View detailed charts and analysis for {current_pool}',
                    'action': f'Show me {current_pool} analysis'
                }
            ])
        
        # Feature suggestions
        suggestions.extend([
            {
                'id': 'suggest_arbitrage',
                'title': 'Find Arbitrage Opportunities',
                'description': 'Discover cross-DEX arbitrage opportunities',
                'action': 'Show me arbitrage opportunities'
            },
            {
                'id': 'suggest_yield',
                'title': 'Optimize Yield Farming',
                'description': 'Find the best yield farming strategies',
                'action': 'Help me optimize yield farming'
            }
        ])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def update_chat_context(self, user_input: str, response: Dict):
        """Update chat context based on interaction."""
        # Update current pool if mentioned
        pool = self.extract_pool(user_input)
        if pool != st.session_state.chat_context['current_pool']:
            st.session_state.chat_context['current_pool'] = pool
        
        # Update user intent
        intent = self.parse_user_intent(user_input)
        st.session_state.chat_context['user_intent'] = intent['type']
        
        # Update last action
        if 'actions' in response and response['actions']:
            st.session_state.chat_context['last_action'] = response['actions'][0]['action']
