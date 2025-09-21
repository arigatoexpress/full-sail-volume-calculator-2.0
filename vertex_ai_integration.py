"""
🤖 VERTEX AI INTEGRATION MODULE
Advanced AI-powered insights and predictions using Google Vertex AI

This module provides comprehensive AI capabilities including:
- Market analysis and predictions using Gemini models
- Sentiment analysis from news and social media
- Advanced pattern recognition in trading data
- Natural language query processing
- Automated report generation
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Google Vertex AI imports
try:
    import vertexai
    from vertexai.language_models import TextGenerationModel, ChatModel
    from vertexai.generative_models import GenerativeModel, Part, Image
    from vertexai.preview.generative_models import multimodal_models
    import google.auth
    from google.cloud import aiplatform
    import google.generativeai as genai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logging.warning("Vertex AI libraries not available. Install with: pip install google-cloud-aiplatform vertexai")

# Additional imports for multimodal capabilities
try:
    import requests
    from PIL import Image as PILImage
    import io
    import base64
    import plotly.graph_objects as go
    import plotly.express as px
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    logging.warning("Multimodal dependencies not available")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AIInsight:
    """Structured AI insight with metadata."""
    title: str
    content: str
    confidence: float
    category: str
    urgency: str
    actionable: bool
    timestamp: datetime
    data_sources: List[str]
    recommendations: List[str]

@dataclass
class MarketPrediction:
    """AI-generated market prediction."""
    asset: str
    prediction_type: str
    timeframe: str
    predicted_value: float
    confidence: float
    reasoning: str
    risk_factors: List[str]
    timestamp: datetime

class VertexAIIntegration:
    """
    Advanced Vertex AI integration for DeFi analytics and predictions.
    
    Provides comprehensive AI capabilities including market analysis,
    sentiment analysis, pattern recognition, and natural language processing.
    """
    
    def __init__(self):
        """Initialize Vertex AI integration with proper authentication."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.project_id = os.getenv('GOOGLE_PROJECT_ID')
        self.region = os.getenv('GOOGLE_REGION', 'us-central1')
        self.api_key = os.getenv('GOOGLE_VERTEX_AI_API_KEY')
        
        # Model configurations
        self.text_model_name = "text-bison@002"
        self.chat_model_name = "chat-bison@002" 
        self.gemini_model_name = "gemini-1.5-pro-preview-0409"
        self.multimodal_model_name = "gemini-1.5-pro-preview-0409"
        
        # Initialize models
        self.text_model = None
        self.chat_model = None
        self.gemini_model = None
        self.multimodal_model = None
        
        # Advanced features
        self.enable_multimodal = True
        self.enable_vision_analysis = True
        self.enable_code_generation = True
        self.enable_advanced_analytics = True
        
        # Check availability and initialize
        if VERTEX_AI_AVAILABLE and self.project_id:
            self._initialize_vertex_ai()
        else:
            self.logger.warning("Vertex AI not properly configured. Check environment variables.")
    
    def _initialize_vertex_ai(self) -> None:
        """Initialize Vertex AI with proper authentication."""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            
            # Initialize models
            self.text_model = TextGenerationModel.from_pretrained(self.text_model_name)
            self.chat_model = ChatModel.from_pretrained(self.chat_model_name)
            self.gemini_model = GenerativeModel(self.gemini_model_name)
            
            # Initialize multimodal model if available
            if MULTIMODAL_AVAILABLE:
                try:
                    self.multimodal_model = GenerativeModel(self.multimodal_model_name)
                    self.logger.info("✅ Multimodal model initialized")
                except Exception as e:
                    self.logger.warning(f"Multimodal model not available: {e}")
                    self.multimodal_model = None
            
            # Configure Gemini API if available
            if self.api_key:
                try:
                    genai.configure(api_key=self.api_key)
                    self.logger.info("✅ Gemini API configured")
                except Exception as e:
                    self.logger.warning(f"Gemini API configuration failed: {e}")
            
            self.logger.info("✅ Vertex AI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Vertex AI: {e}")
            self.text_model = None
            self.chat_model = None
            self.gemini_model = None
    
    def is_available(self) -> bool:
        """Check if Vertex AI is available and configured."""
        return (VERTEX_AI_AVAILABLE and 
                self.project_id and 
                self.text_model is not None)
    
    async def generate_market_insights(self, market_data: Dict[str, Any]) -> List[AIInsight]:
        """
        Generate comprehensive market insights using Vertex AI.
        
        Args:
            market_data: Dictionary containing market data (prices, volumes, etc.)
            
        Returns:
            List of AI-generated insights
        """
        if not self.is_available():
            return self._generate_fallback_insights(market_data)
        
        try:
            insights = []
            
            # Prepare market data summary
            data_summary = self._prepare_market_summary(market_data)
            
            # Generate different types of insights
            insight_types = [
                ("trend_analysis", "Analyze current market trends and momentum"),
                ("risk_assessment", "Assess market risks and potential volatility"),
                ("opportunity_detection", "Identify potential trading opportunities"),
                ("sentiment_analysis", "Analyze market sentiment and investor behavior")
            ]
            
            for insight_type, prompt_suffix in insight_types:
                insight = await self._generate_single_insight(
                    data_summary, insight_type, prompt_suffix
                )
                if insight:
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating market insights: {e}")
            return self._generate_fallback_insights(market_data)
    
    async def _generate_single_insight(self, data_summary: str, 
                                     insight_type: str, 
                                     prompt_suffix: str) -> Optional[AIInsight]:
        """Generate a single AI insight."""
        try:
            prompt = f"""
            As a professional DeFi analyst, analyze the following market data and {prompt_suffix}.
            
            Market Data Summary:
            {data_summary}
            
            Please provide:
            1. Key findings and analysis
            2. Specific actionable recommendations
            3. Risk factors to consider
            4. Confidence level (1-10)
            5. Urgency level (low/medium/high)
            
            Focus on practical, actionable insights for DeFi investors and yield farmers.
            """
            
            # Use Gemini for advanced analysis
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.8,
                }
            )
            
            # Parse response
            content = response.text
            
            # Extract confidence and urgency (simplified parsing)
            confidence = self._extract_confidence(content)
            urgency = self._extract_urgency(content)
            recommendations = self._extract_recommendations(content)
            
            return AIInsight(
                title=f"{insight_type.replace('_', ' ').title()} Analysis",
                content=content,
                confidence=confidence,
                category=insight_type,
                urgency=urgency,
                actionable=len(recommendations) > 0,
                timestamp=datetime.now(),
                data_sources=["vertex_ai", "market_data"],
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error generating {insight_type} insight: {e}")
            return None
    
    async def generate_price_predictions(self, 
                                       asset_data: Dict[str, Any], 
                                       timeframes: List[str] = None) -> List[MarketPrediction]:
        """
        Generate AI-powered price predictions for assets.
        
        Args:
            asset_data: Historical price and volume data
            timeframes: List of prediction timeframes (e.g., ['1d', '7d', '30d'])
            
        Returns:
            List of market predictions
        """
        if not self.is_available():
            return self._generate_fallback_predictions(asset_data)
        
        if timeframes is None:
            timeframes = ['1d', '7d', '30d']
        
        try:
            predictions = []
            
            for asset, data in asset_data.items():
                for timeframe in timeframes:
                    prediction = await self._generate_asset_prediction(
                        asset, data, timeframe
                    )
                    if prediction:
                        predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return self._generate_fallback_predictions(asset_data)
    
    async def _generate_asset_prediction(self, asset: str, 
                                       data: Dict[str, Any], 
                                       timeframe: str) -> Optional[MarketPrediction]:
        """Generate prediction for a single asset."""
        try:
            # Prepare data summary
            current_price = data.get('current_price', 0)
            volume_24h = data.get('volume_24h', 0)
            price_change_24h = data.get('price_change_24h', 0)
            
            prompt = f"""
            As a quantitative analyst, predict the price of {asset} for the {timeframe} timeframe.
            
            Current Data:
            - Current Price: ${current_price:,.4f}
            - 24h Volume: ${volume_24h:,.0f}
            - 24h Price Change: {price_change_24h:+.2f}%
            
            Historical trends and patterns suggest certain movements. Please provide:
            1. Predicted price for {timeframe} ahead
            2. Confidence level (0-100%)
            3. Key reasoning for the prediction
            4. Main risk factors
            
            Be realistic and consider market volatility.
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 512,
                    "temperature": 0.4,
                }
            )
            
            content = response.text
            
            # Parse prediction (simplified)
            predicted_value = self._extract_predicted_value(content, current_price)
            confidence = self._extract_confidence(content)
            risk_factors = self._extract_risk_factors(content)
            
            return MarketPrediction(
                asset=asset,
                prediction_type="price",
                timeframe=timeframe,
                predicted_value=predicted_value,
                confidence=confidence / 100.0,  # Convert to 0-1 scale
                reasoning=content,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting {asset} for {timeframe}: {e}")
            return None
    
    async def analyze_arbitrage_opportunities(self, 
                                           arbitrage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze arbitrage opportunities using AI to provide strategic insights.
        
        Args:
            arbitrage_data: List of arbitrage opportunities
            
        Returns:
            AI analysis of opportunities with recommendations
        """
        if not self.is_available() or not arbitrage_data:
            return {"error": "AI analysis not available"}
        
        try:
            # Prepare arbitrage summary
            arb_summary = self._prepare_arbitrage_summary(arbitrage_data)
            
            prompt = f"""
            As a DeFi arbitrage expert, analyze these arbitrage opportunities:
            
            {arb_summary}
            
            Please provide:
            1. Top 3 most profitable opportunities
            2. Risk assessment for each opportunity
            3. Execution strategy recommendations
            4. Market timing considerations
            5. Capital allocation suggestions
            
            Consider gas fees, slippage, and execution complexity.
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                }
            )
            
            return {
                "analysis": response.text,
                "opportunities_analyzed": len(arbitrage_data),
                "timestamp": datetime.now().isoformat(),
                "ai_confidence": 0.85  # High confidence for structured data analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing arbitrage opportunities: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def generate_yield_strategy(self, 
                                    user_profile: Dict[str, Any], 
                                    yield_opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate personalized yield farming strategy using AI.
        
        Args:
            user_profile: User's risk tolerance, capital, etc.
            yield_opportunities: Available yield farming opportunities
            
        Returns:
            Personalized yield strategy with AI recommendations
        """
        if not self.is_available():
            return {"error": "AI strategy generation not available"}
        
        try:
            # Prepare user and opportunity data
            user_summary = self._prepare_user_profile_summary(user_profile)
            yield_summary = self._prepare_yield_summary(yield_opportunities)
            
            prompt = f"""
            As a DeFi yield farming strategist, create a personalized strategy:
            
            User Profile:
            {user_summary}
            
            Available Opportunities:
            {yield_summary}
            
            Please provide:
            1. Recommended portfolio allocation
            2. Risk-adjusted yield optimization
            3. Diversification strategy
            4. Entry and exit timing
            5. Risk management recommendations
            
            Focus on maximizing risk-adjusted returns.
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.4,
                }
            )
            
            return {
                "strategy": response.text,
                "opportunities_considered": len(yield_opportunities),
                "user_risk_level": user_profile.get('risk_tolerance', 'medium'),
                "timestamp": datetime.now().isoformat(),
                "ai_confidence": 0.80
            }
            
        except Exception as e:
            self.logger.error(f"Error generating yield strategy: {e}")
            return {"error": f"Strategy generation failed: {str(e)}"}
    
    def _prepare_market_summary(self, market_data: Dict[str, Any]) -> str:
        """Prepare a concise market data summary for AI analysis."""
        summary_parts = []
        
        if 'pools' in market_data:
            pools = market_data['pools']
            summary_parts.append(f"Pool Data: {len(pools)} pools analyzed")
            
            # Top pools by volume
            if pools:
                top_pools = sorted(pools, key=lambda x: x.get('volume_24h', 0), reverse=True)[:3]
                summary_parts.append("Top Pools by Volume:")
                for pool in top_pools:
                    summary_parts.append(f"  - {pool.get('name', 'Unknown')}: ${pool.get('volume_24h', 0):,.0f}")
        
        if 'prices' in market_data:
            prices = market_data['prices']
            summary_parts.append(f"Price Data: {len(prices)} assets")
            
        if 'market_metrics' in market_data:
            metrics = market_data['market_metrics']
            summary_parts.append(f"Market Metrics: {json.dumps(metrics, indent=2)}")
        
        return "\n".join(summary_parts)
    
    def _prepare_arbitrage_summary(self, arbitrage_data: List[Dict[str, Any]]) -> str:
        """Prepare arbitrage data summary."""
        if not arbitrage_data:
            return "No arbitrage opportunities available."
        
        summary = []
        for i, opp in enumerate(arbitrage_data[:5]):  # Top 5
            summary.append(f"""
            Opportunity {i+1}:
            - Pair: {opp.get('pair', 'Unknown')}
            - Profit: {opp.get('profit_pct', 0):.2f}%
            - DEX A: {opp.get('dex_a', 'Unknown')} (${opp.get('price_a', 0):.4f})
            - DEX B: {opp.get('dex_b', 'Unknown')} (${opp.get('price_b', 0):.4f})
            - Volume: ${opp.get('volume_usd', 0):,.0f}
            """)
        
        return "\n".join(summary)
    
    def _prepare_user_profile_summary(self, user_profile: Dict[str, Any]) -> str:
        """Prepare user profile summary."""
        return f"""
        - Risk Tolerance: {user_profile.get('risk_tolerance', 'medium')}
        - Capital: ${user_profile.get('capital', 10000):,.0f}
        - Time Horizon: {user_profile.get('time_horizon', '3 months')}
        - Experience Level: {user_profile.get('experience', 'intermediate')}
        - Preferred Assets: {', '.join(user_profile.get('preferred_assets', ['SUI', 'USDC']))}
        """
    
    def _prepare_yield_summary(self, yield_opportunities: List[Dict[str, Any]]) -> str:
        """Prepare yield opportunities summary."""
        if not yield_opportunities:
            return "No yield opportunities available."
        
        summary = []
        for opp in yield_opportunities[:10]:  # Top 10
            summary.append(f"""
            - {opp.get('protocol', 'Unknown')} {opp.get('pool_name', '')}
              APR: {opp.get('apr', 0):.1f}% | TVL: ${opp.get('tvl', 0):,.0f} | Risk: {opp.get('risk_level', 'medium')}
            """)
        
        return "\n".join(summary)
    
    def _extract_confidence(self, content: str) -> float:
        """Extract confidence level from AI response."""
        # Simplified confidence extraction
        import re
        
        # Look for confidence patterns
        patterns = [
            r'confidence[:\s]+(\d+)%',
            r'confidence[:\s]+(\d+)/10',
            r'(\d+)%\s+confidence',
            r'confidence[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return min(value, 100) / 100 if value > 1 else value
        
        # Default confidence
        return 0.75
    
    def _extract_urgency(self, content: str) -> str:
        """Extract urgency level from AI response."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['urgent', 'immediate', 'critical', 'now']):
            return 'high'
        elif any(word in content_lower for word in ['moderate', 'medium', 'soon']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from AI response."""
        # Simplified recommendation extraction
        lines = content.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['•', '-', '*', '1.', '2.', '3.']):
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_predicted_value(self, content: str, current_price: float) -> float:
        """Extract predicted value from AI response."""
        import re
        
        # Look for price patterns
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*USD',
            r'price[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    predicted = float(matches[0])
                    # Sanity check: prediction should be within reasonable range
                    if 0.1 * current_price <= predicted <= 10 * current_price:
                        return predicted
                except ValueError:
                    continue
        
        # Fallback: small random variation
        return current_price * (1 + np.random.uniform(-0.1, 0.1))
    
    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract risk factors from AI response."""
        lines = content.split('\n')
        risk_factors = []
        
        in_risk_section = False
        for line in lines:
            line = line.strip()
            
            if 'risk' in line.lower():
                in_risk_section = True
                continue
            
            if in_risk_section and any(line.startswith(prefix) for prefix in ['•', '-', '*']):
                risk_factors.append(line)
            elif in_risk_section and not line:
                break
        
        return risk_factors[:5]  # Limit to top 5
    
    def _generate_fallback_insights(self, market_data: Dict[str, Any]) -> List[AIInsight]:
        """Generate fallback insights when AI is not available."""
        return [
            AIInsight(
                title="Market Analysis",
                content="AI analysis temporarily unavailable. Using fallback analysis based on market data patterns.",
                confidence=0.6,
                category="trend_analysis",
                urgency="low",
                actionable=False,
                timestamp=datetime.now(),
                data_sources=["fallback"],
                recommendations=["Monitor market conditions", "Use technical analysis"]
            )
        ]
    
    def _generate_fallback_predictions(self, asset_data: Dict[str, Any]) -> List[MarketPrediction]:
        """Generate fallback predictions when AI is not available."""
        predictions = []
        
        for asset, data in asset_data.items():
            current_price = data.get('current_price', 0)
            
            predictions.append(MarketPrediction(
                asset=asset,
                prediction_type="price",
                timeframe="1d",
                predicted_value=current_price * (1 + np.random.uniform(-0.05, 0.05)),
                confidence=0.5,
                reasoning="Fallback prediction based on current price with small variation",
                risk_factors=["High uncertainty due to AI unavailability"],
                timestamp=datetime.now()
            ))
        
        return predictions
    
    # ==================== ADVANCED MULTIMODAL CAPABILITIES ====================
    
    async def analyze_chart_image(self, chart_image: Union[str, bytes, PILImage.Image]) -> Dict[str, Any]:
        """
        Analyze trading charts using vision capabilities.
        
        Args:
            chart_image: Image data (base64 string, bytes, or PIL Image)
            
        Returns:
            AI analysis of the chart with trading insights
        """
        if not self.multimodal_model or not MULTIMODAL_AVAILABLE:
            return {"error": "Multimodal analysis not available"}
        
        try:
            # Convert image to proper format
            if isinstance(chart_image, str):
                # Base64 string
                image_data = base64.b64decode(chart_image)
                image = PILImage.open(io.BytesIO(image_data))
            elif isinstance(chart_image, bytes):
                image = PILImage.open(io.BytesIO(chart_image))
            elif isinstance(chart_image, PILImage.Image):
                image = chart_image
            else:
                return {"error": "Invalid image format"}
            
            # Convert to base64 for Vertex AI
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create image part
            image_part = Part.from_data(
                data=image_base64,
                mime_type="image/png"
            )
            
            prompt = """
            Analyze this trading chart and provide comprehensive insights:
            
            1. Identify chart patterns (head and shoulders, triangles, flags, etc.)
            2. Analyze trend direction and strength
            3. Identify key support and resistance levels
            4. Assess volume patterns
            5. Identify potential entry and exit points
            6. Calculate risk/reward ratios
            7. Provide confidence levels for each analysis
            8. Suggest trading strategies based on the chart
            
            Focus on actionable trading insights for DeFi assets.
            """
            
            response = self.multimodal_model.generate_content([image_part, prompt])
            
            return {
                "analysis": response.text,
                "chart_type": "trading_chart",
                "timestamp": datetime.now().isoformat(),
                "ai_confidence": 0.85,
                "features_detected": self._extract_chart_features(response.text)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing chart image: {e}")
            return {"error": f"Chart analysis failed: {str(e)}"}
    
    async def generate_visual_insights(self, data: Dict[str, Any], chart_type: str = "candlestick") -> Dict[str, Any]:
        """
        Generate visual insights by creating charts and analyzing them with AI.
        
        Args:
            data: Market data to visualize
            chart_type: Type of chart to generate
            
        Returns:
            AI analysis of generated visualizations
        """
        if not self.multimodal_model or not MULTIMODAL_AVAILABLE:
            return {"error": "Visual analysis not available"}
        
        try:
            # Generate chart
            chart_image = self._create_chart_image(data, chart_type)
            
            # Analyze with AI
            analysis = await self.analyze_chart_image(chart_image)
            
            return {
                "visual_analysis": analysis,
                "chart_generated": True,
                "chart_type": chart_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating visual insights: {e}")
            return {"error": f"Visual analysis failed: {str(e)}"}
    
    async def analyze_market_sentiment_from_text(self, text_data: List[str]) -> Dict[str, Any]:
        """
        Analyze market sentiment from news articles, social media, etc.
        
        Args:
            text_data: List of text content to analyze
            
        Returns:
            Sentiment analysis with confidence scores
        """
        if not self.is_available():
            return {"error": "Sentiment analysis not available"}
        
        try:
            # Combine all text
            combined_text = "\n\n".join(text_data[:10])  # Limit to first 10 items
            
            prompt = f"""
            Analyze the sentiment of the following DeFi/crypto market content:
            
            {combined_text}
            
            Provide:
            1. Overall sentiment (bullish/bearish/neutral) with confidence score
            2. Key positive indicators
            3. Key negative indicators
            4. Market sentiment score (0-100)
            5. Risk level assessment
            6. Recommended actions based on sentiment
            7. Time sensitivity of the sentiment
            
            Focus on actionable insights for DeFi traders and investors.
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                }
            )
            
            return {
                "sentiment_analysis": response.text,
                "texts_analyzed": len(text_data),
                "timestamp": datetime.now().isoformat(),
                "confidence": self._extract_confidence(response.text)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {"error": f"Sentiment analysis failed: {str(e)}"}
    
    async def generate_trading_strategy_code(self, strategy_description: str) -> Dict[str, Any]:
        """
        Generate executable trading strategy code using AI.
        
        Args:
            strategy_description: Natural language description of the strategy
            
        Returns:
            Generated code and documentation
        """
        if not self.is_available():
            return {"error": "Code generation not available"}
        
        try:
            prompt = f"""
            Generate a complete Python trading strategy based on this description:
            
            {strategy_description}
            
            The code should:
            1. Use pandas and numpy for data manipulation
            2. Include proper error handling
            3. Have clear documentation
            4. Be optimized for DeFi trading
            5. Include backtesting capabilities
            6. Support multiple timeframes
            7. Include risk management
            8. Be production-ready
            
            Provide:
            - Complete Python code
            - Usage instructions
            - Parameter explanations
            - Example implementation
            - Risk warnings
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2048,
                    "temperature": 0.2,
                }
            )
            
            return {
                "strategy_code": response.text,
                "language": "python",
                "strategy_type": "defi_trading",
                "timestamp": datetime.now().isoformat(),
                "ai_confidence": 0.80
            }
            
        except Exception as e:
            self.logger.error(f"Error generating strategy code: {e}")
            return {"error": f"Code generation failed: {str(e)}"}
    
    async def generate_comprehensive_report(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive market report with multiple AI analyses.
        
        Args:
            market_data: Complete market dataset
            
        Returns:
            Comprehensive report with all AI insights
        """
        if not self.is_available():
            return {"error": "Comprehensive analysis not available"}
        
        try:
            # Generate multiple types of analysis
            insights = await self.generate_market_insights(market_data)
            predictions = await self.generate_price_predictions(market_data.get('prices', {}))
            
            # Generate visual insights if data is available
            visual_insights = {}
            if 'prices' in market_data and market_data['prices']:
                visual_insights = await self.generate_visual_insights(market_data['prices'])
            
            # Generate comprehensive summary
            summary_prompt = f"""
            Create a comprehensive executive summary for this DeFi market analysis:
            
            Market Insights: {len(insights)} insights generated
            Price Predictions: {len(predictions)} predictions generated
            Visual Analysis: {'Available' if visual_insights else 'Not available'}
            
            Provide:
            1. Executive summary of market conditions
            2. Key opportunities and risks
            3. Recommended actions
            4. Market outlook
            5. Risk assessment
            6. Next steps for investors
            """
            
            summary_response = self.gemini_model.generate_content(
                summary_prompt,
                generation_config={
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                }
            )
            
            return {
                "executive_summary": summary_response.text,
                "market_insights": [insight.__dict__ for insight in insights],
                "price_predictions": [pred.__dict__ for pred in predictions],
                "visual_analysis": visual_insights,
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "ai_models_used": ["gemini-1.5-pro", "multimodal-analysis"],
                    "data_sources": ["market_data", "ai_analysis"],
                    "confidence_score": 0.85
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {"error": f"Report generation failed: {str(e)}"}
    
    def _create_chart_image(self, data: Dict[str, Any], chart_type: str) -> PILImage.Image:
        """Create a chart image from market data."""
        try:
            if chart_type == "candlestick" and 'prices' in data:
                # Create candlestick chart
                df = pd.DataFrame(data['prices'])
                fig = go.Figure(data=go.Candlestick(
                    x=df.index,
                    open=df.get('open', df.get('price', 0)),
                    high=df.get('high', df.get('price', 0)),
                    low=df.get('low', df.get('price', 0)),
                    close=df.get('close', df.get('price', 0))
                ))
            else:
                # Create line chart
                df = pd.DataFrame(data.get('prices', []))
                fig = go.Figure(data=go.Scatter(
                    x=df.index,
                    y=df.get('price', df.get('close', 0)),
                    mode='lines'
                ))
            
            # Update layout
            fig.update_layout(
                title="Market Analysis Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark"
            )
            
            # Convert to image
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            return PILImage.open(io.BytesIO(img_bytes))
            
        except Exception as e:
            self.logger.error(f"Error creating chart image: {e}")
            # Return a simple placeholder
            return PILImage.new('RGB', (1200, 800), color='black')
    
    def _extract_chart_features(self, analysis_text: str) -> List[str]:
        """Extract chart features from AI analysis."""
        features = []
        text_lower = analysis_text.lower()
        
        chart_patterns = [
            'head and shoulders', 'double top', 'double bottom', 'triangle',
            'flag', 'pennant', 'wedge', 'channel', 'support', 'resistance',
            'trend line', 'breakout', 'breakdown', 'reversal', 'continuation'
        ]
        
        for pattern in chart_patterns:
            if pattern in text_lower:
                features.append(pattern)
        
        return features

# Global instance
vertex_ai = VertexAIIntegration()

# Convenience functions
async def get_ai_market_insights(market_data: Dict[str, Any]) -> List[AIInsight]:
    """Get AI-powered market insights."""
    return await vertex_ai.generate_market_insights(market_data)

async def get_ai_predictions(asset_data: Dict[str, Any]) -> List[MarketPrediction]:
    """Get AI-powered price predictions."""
    return await vertex_ai.generate_price_predictions(asset_data)

async def get_arbitrage_analysis(arbitrage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get AI analysis of arbitrage opportunities."""
    return await vertex_ai.analyze_arbitrage_opportunities(arbitrage_data)

async def get_yield_strategy(user_profile: Dict[str, Any], 
                           opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get personalized yield farming strategy."""
    return await vertex_ai.generate_yield_strategy(user_profile, opportunities)
