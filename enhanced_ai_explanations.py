"""
🤖 ENHANCED AI EXPLANATIONS
Improved AI model explanations and interpretability features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class EnhancedAIExplanations:
    """Enhanced AI explanations and model interpretability."""
    
    def __init__(self):
        """Initialize enhanced AI explanations."""
        self.model_descriptions = self.get_model_descriptions()
        self.explanation_templates = self.get_explanation_templates()
    
    def get_model_descriptions(self) -> Dict:
        """Get detailed model descriptions."""
        return {
            'Prophet': {
                'name': 'Prophet',
                'description': 'Facebook\'s time series forecasting model',
                'strengths': [
                    'Handles seasonal patterns well',
                    'Robust to missing data',
                    'Automatic holiday detection',
                    'Good for business time series'
                ],
                'weaknesses': [
                    'May overfit on small datasets',
                    'Less effective for non-seasonal data',
                    'Can be slow on large datasets'
                ],
                'best_for': [
                    'Volume predictions with weekly/monthly patterns',
                    'Data with clear seasonality',
                    'Business time series with holidays'
                ],
                'confidence_factors': [
                    'Data quality and completeness',
                    'Seasonal pattern strength',
                    'Historical volatility',
                    'Recent trend stability'
                ]
            },
            'ARIMA': {
                'name': 'ARIMA',
                'description': 'AutoRegressive Integrated Moving Average model',
                'strengths': [
                    'Classical statistical approach',
                    'Good for stationary data',
                    'Interpretable parameters',
                    'Fast computation'
                ],
                'weaknesses': [
                    'Requires stationary data',
                    'Limited to linear relationships',
                    'Sensitive to outliers',
                    'Manual parameter tuning'
                ],
                'best_for': [
                    'Short-term predictions',
                    'Stationary time series',
                    'Linear trend analysis',
                    'Quick forecasts'
                ],
                'confidence_factors': [
                    'Data stationarity',
                    'Model fit quality',
                    'Residual analysis',
                    'Parameter significance'
                ]
            },
            'Ensemble': {
                'name': 'Ensemble',
                'description': 'Combines Prophet and ARIMA for robust predictions',
                'strengths': [
                    'Reduces individual model bias',
                    'More robust predictions',
                    'Better error handling',
                    'Combines different approaches'
                ],
                'weaknesses': [
                    'More complex to interpret',
                    'Requires more computation',
                    'May mask individual model insights'
                ],
                'best_for': [
                    'General-purpose predictions',
                    'When uncertainty is high',
                    'Production environments',
                    'Balanced accuracy needs'
                ],
                'confidence_factors': [
                    'Individual model agreement',
                    'Historical ensemble performance',
                    'Data quality across models',
                    'Market stability'
                ]
            }
        }
    
    def get_explanation_templates(self) -> Dict:
        """Get explanation templates for different scenarios."""
        return {
            'high_confidence': {
                'title': 'High Confidence Prediction',
                'explanation': 'This prediction has high confidence due to strong historical patterns and stable market conditions.',
                'color': '#00E676',
                'icon': '✅'
            },
            'medium_confidence': {
                'title': 'Medium Confidence Prediction',
                'explanation': 'This prediction has moderate confidence. Consider recent market volatility and external factors.',
                'color': '#FFD600',
                'icon': '⚠️'
            },
            'low_confidence': {
                'title': 'Low Confidence Prediction',
                'explanation': 'This prediction has low confidence. Market conditions are uncertain or data is limited.',
                'color': '#FF6B35',
                'icon': '⚠️'
            }
        }
    
    def render_model_explanation(self, model_name: str, prediction_data: Dict):
        """Render detailed model explanation."""
        model_info = self.model_descriptions.get(model_name, {})
        
        st.markdown(f"### 🤖 {model_info.get('name', model_name)} Model Explanation")
        
        # Model overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Description**: {model_info.get('description', 'AI prediction model')}")
            
            # Strengths
            st.markdown("**Strengths:**")
            for strength in model_info.get('strengths', []):
                st.markdown(f"✅ {strength}")
            
            # Best for
            st.markdown("**Best For:**")
            for use_case in model_info.get('best_for', []):
                st.markdown(f"🎯 {use_case}")
        
        with col2:
            # Model performance metrics
            st.markdown("**Performance Metrics**")
            
            if 'accuracy' in prediction_data:
                st.metric("Accuracy", f"{prediction_data['accuracy']:.1%}")
            
            if 'mape' in prediction_data:
                st.metric("MAPE", f"{prediction_data['mape']:.2f}%")
            
            if 'confidence' in prediction_data:
                confidence = prediction_data['confidence']
                confidence_level = self.get_confidence_level(confidence)
                st.metric("Confidence", f"{confidence:.1%}", help=confidence_level['explanation'])
    
    def render_prediction_explanation(self, prediction_data: Dict):
        """Render detailed prediction explanation."""
        st.markdown("### 🔍 Prediction Explanation")
        
        # Confidence level
        confidence = prediction_data.get('confidence', 0.5)
        confidence_level = self.get_confidence_level(confidence)
        
        # Display confidence indicator
        st.markdown(f"""
        <div class="confidence-indicator" style="
            background: linear-gradient(135deg, {confidence_level['color']}20, {confidence_level['color']}10);
            border: 1px solid {confidence_level['color']}40;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        ">
            <h4>{confidence_level['icon']} {confidence_level['title']}</h4>
            <p>{confidence_level['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key factors
        st.markdown("#### 📊 Key Factors")
        
        factors = prediction_data.get('key_factors', [])
        if factors:
            for factor in factors:
                st.markdown(f"• **{factor['name']}**: {factor['impact']} ({factor['description']})")
        else:
            st.info("No specific factors identified for this prediction.")
        
        # Historical context
        st.markdown("#### 📈 Historical Context")
        
        if 'historical_accuracy' in prediction_data:
            accuracy = prediction_data['historical_accuracy']
            st.markdown(f"**Historical Accuracy**: {accuracy:.1%}")
            
            if accuracy > 0.8:
                st.success("This model has shown strong historical performance.")
            elif accuracy > 0.6:
                st.warning("This model has moderate historical performance.")
            else:
                st.error("This model has shown limited historical performance.")
        
        # Risk factors
        st.markdown("#### ⚠️ Risk Factors")
        
        risk_factors = prediction_data.get('risk_factors', [])
        if risk_factors:
            for risk in risk_factors:
                st.markdown(f"• **{risk['name']}**: {risk['description']}")
        else:
            st.info("No significant risk factors identified.")
    
    def render_confidence_breakdown(self, prediction_data: Dict):
        """Render confidence breakdown visualization."""
        st.markdown("### 📊 Confidence Breakdown")
        
        # Create confidence breakdown chart
        factors = prediction_data.get('confidence_factors', {})
        
        if factors:
            # Prepare data for visualization
            factor_names = list(factors.keys())
            factor_values = list(factors.values())
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=factor_names,
                    x=factor_values,
                    orientation='h',
                    marker=dict(
                        color=factor_values,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Confidence Score")
                    ),
                    text=[f"{v:.1%}" for v in factor_values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Confidence Factor Analysis",
                xaxis_title="Confidence Score",
                yaxis_title="Factors",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence factors available for this prediction.")
    
    def render_model_comparison(self, predictions: Dict):
        """Render model comparison explanation."""
        st.markdown("### 🔄 Model Comparison")
        
        if len(predictions) < 2:
            st.info("Multiple model predictions needed for comparison.")
            return
        
        # Create comparison table
        comparison_data = []
        
        for model_name, pred_data in predictions.items():
            comparison_data.append({
                'Model': model_name,
                'Prediction': f"${pred_data.get('prediction', 0):,.0f}",
                'Confidence': f"{pred_data.get('confidence', 0):.1%}",
                'Accuracy': f"{pred_data.get('accuracy', 0):.1%}",
                'MAPE': f"{pred_data.get('mape', 0):.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Model agreement analysis
        st.markdown("#### 🤝 Model Agreement")
        
        predictions_list = [pred_data.get('prediction', 0) for pred_data in predictions.values()]
        if predictions_list:
            mean_prediction = np.mean(predictions_list)
            std_prediction = np.std(predictions_list)
            coefficient_of_variation = std_prediction / mean_prediction if mean_prediction != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Prediction", f"${mean_prediction:,.0f}")
            
            with col2:
                st.metric("Standard Deviation", f"${std_prediction:,.0f}")
            
            with col3:
                st.metric("Coefficient of Variation", f"{coefficient_of_variation:.2%}")
            
            # Agreement interpretation
            if coefficient_of_variation < 0.1:
                st.success("High model agreement - predictions are very consistent.")
            elif coefficient_of_variation < 0.2:
                st.warning("Moderate model agreement - some variation in predictions.")
            else:
                st.error("Low model agreement - significant variation in predictions.")
    
    def render_uncertainty_analysis(self, prediction_data: Dict):
        """Render uncertainty analysis."""
        st.markdown("### 🎯 Uncertainty Analysis")
        
        # Confidence intervals
        if 'confidence_intervals' in prediction_data:
            intervals = prediction_data['confidence_intervals']
            
            st.markdown("#### 📊 Confidence Intervals")
            
            for level, interval in intervals.items():
                lower, upper = interval
                st.markdown(f"**{level}% Confidence**: ${lower:,.0f} - ${upper:,.0f}")
            
            # Visualize confidence intervals
            self.render_confidence_interval_chart(intervals)
        
        # Uncertainty sources
        st.markdown("#### 🔍 Uncertainty Sources")
        
        uncertainty_sources = prediction_data.get('uncertainty_sources', [])
        if uncertainty_sources:
            for source in uncertainty_sources:
                st.markdown(f"• **{source['name']}**: {source['description']} (Impact: {source['impact']})")
        else:
            st.info("No specific uncertainty sources identified.")
    
    def render_confidence_interval_chart(self, intervals: Dict):
        """Render confidence interval visualization."""
        # Prepare data
        levels = list(intervals.keys())
        lower_bounds = [interval[0] for interval in intervals.values()]
        upper_bounds = [interval[1] for interval in intervals.values()]
        
        # Create chart
        fig = go.Figure()
        
        # Add confidence intervals
        for i, level in enumerate(levels):
            fig.add_trace(go.Scatter(
                x=[lower_bounds[i], upper_bounds[i]],
                y=[level, level],
                mode='lines+markers',
                name=f'{level}% CI',
                line=dict(width=6),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Confidence Intervals",
            xaxis_title="Prediction Value",
            yaxis_title="Confidence Level (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_confidence_level(self, confidence: float) -> Dict:
        """Get confidence level description."""
        if confidence >= 0.8:
            return self.explanation_templates['high_confidence']
        elif confidence >= 0.6:
            return self.explanation_templates['medium_confidence']
        else:
            return self.explanation_templates['low_confidence']
    
    def render_interactive_explanation(self, prediction_data: Dict):
        """Render interactive explanation interface."""
        st.markdown("### 🎮 Interactive Explanation")
        
        # Explanation tabs
        explanation_tabs = st.tabs([
            "🤖 Model Details",
            "🔍 Prediction Analysis", 
            "📊 Confidence Breakdown",
            "🔄 Model Comparison",
            "🎯 Uncertainty Analysis"
        ])
        
        with explanation_tabs[0]:
            model_name = prediction_data.get('model', 'Ensemble')
            self.render_model_explanation(model_name, prediction_data)
        
        with explanation_tabs[1]:
            self.render_prediction_explanation(prediction_data)
        
        with explanation_tabs[2]:
            self.render_confidence_breakdown(prediction_data)
        
        with explanation_tabs[3]:
            # For comparison, we'd need multiple predictions
            st.info("Model comparison requires multiple model predictions.")
        
        with explanation_tabs[4]:
            self.render_uncertainty_analysis(prediction_data)
    
    def generate_explanation_summary(self, prediction_data: Dict) -> str:
        """Generate a summary explanation of the prediction."""
        model_name = prediction_data.get('model', 'Ensemble')
        confidence = prediction_data.get('confidence', 0.5)
        prediction = prediction_data.get('prediction', 0)
        
        confidence_level = self.get_confidence_level(confidence)
        
        summary = f"""
        **Prediction Summary:**
        
        The {model_name} model predicts a volume of ${prediction:,.0f} with {confidence:.1%} confidence.
        
        {confidence_level['explanation']}
        
        **Key Insights:**
        - Model: {model_name}
        - Confidence Level: {confidence_level['title']}
        - Prediction: ${prediction:,.0f}
        - Historical Accuracy: {prediction_data.get('accuracy', 0):.1%}
        """
        
        return summary
    
    def render_educational_content(self):
        """Render educational content about AI predictions."""
        st.markdown("### 📚 Understanding AI Predictions")
        
        st.markdown("""
        **How AI Predictions Work:**
        
        1. **Data Analysis**: The AI analyzes historical volume data to identify patterns
        2. **Model Training**: Machine learning models learn from past data
        3. **Pattern Recognition**: Models identify trends, seasonality, and relationships
        4. **Prediction Generation**: Models forecast future values based on learned patterns
        5. **Confidence Assessment**: Models estimate prediction reliability
        
        **Understanding Confidence Levels:**
        
        - **High Confidence (80%+)**: Strong historical patterns, stable conditions
        - **Medium Confidence (60-80%)**: Moderate patterns, some uncertainty
        - **Low Confidence (<60%)**: Weak patterns, high uncertainty
        
        **Important Notes:**
        
        - Predictions are based on historical data and patterns
        - External factors (news, market events) can affect actual outcomes
        - Confidence intervals show the range of likely outcomes
        - Multiple models provide different perspectives on the same data
        """)
        
        # Interactive confidence calculator
        st.markdown("#### 🧮 Confidence Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_quality = st.slider("Data Quality", 0.0, 1.0, 0.8, 0.1)
            pattern_strength = st.slider("Pattern Strength", 0.0, 1.0, 0.7, 0.1)
        
        with col2:
            market_stability = st.slider("Market Stability", 0.0, 1.0, 0.6, 0.1)
            historical_accuracy = st.slider("Historical Accuracy", 0.0, 1.0, 0.75, 0.1)
        
        # Calculate confidence
        confidence = (data_quality + pattern_strength + market_stability + historical_accuracy) / 4
        
        st.metric("Estimated Confidence", f"{confidence:.1%}")
        
        confidence_level = self.get_confidence_level(confidence)
        st.markdown(f"**{confidence_level['icon']} {confidence_level['title']}**")
        st.markdown(confidence_level['explanation'])
