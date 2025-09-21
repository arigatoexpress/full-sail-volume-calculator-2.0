"""
Fixed visualization module with proper Plotly syntax.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from typing import Dict, List, Optional, Tuple
import streamlit as st


class VolumeVisualizer:
    """Creates interactive visualizations for volume data and predictions."""
    
    def __init__(self):
        """Initialize VolumeVisualizer."""
        # Configure Altair
        alt.data_transformers.disable_max_rows()
        
        # Dark mode compatible color schemes
        self.colors = {
            'primary': '#00D4FF',      # Bright cyan
            'secondary': '#FF6B35',    # Bright orange  
            'success': '#00E676',      # Bright green
            'warning': '#FFD600',      # Bright yellow
            'info': '#BB86FC',         # Purple
            'light': '#03DAC6',        # Teal
            'dark': '#CF6679',         # Pink
            'background': '#121212',   # Dark background
            'surface': '#1E1E1E',      # Card background
            'text': '#FFFFFF'          # White text
        }
        
        self.pool_colors = {
            'SAIL/USDC': '#00D4FF',    # Bright cyan
            'SAIL/SUI': '#FF6B35',     # Bright orange
            'USDC/SUI': '#00E676',     # Bright green
            'WETH/USDC': '#FFD600',    # Bright yellow
            'IKA/SUI': '#BB86FC',      # Purple
            'WAL/USDC': '#03DAC6',     # Teal
            'DEEP/SUI': '#CF6679',     # Pink
            'CETUS/SUI': '#FF5722',    # Deep orange
            'BUCK/USDC': '#4CAF50',    # Green
            'NAVX/SUI': '#9C27B0'      # Deep purple
        }

    def get_dark_layout(self, title: str, height: int = 500) -> dict:
        """Get standard dark theme layout."""
        return dict(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Inter, sans-serif", color=self.colors['text'])
            ),
            height=height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=self.colors['text'])
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.colors['text'])
        )

    def create_volume_timeseries(self, df: pd.DataFrame, title: str = "Liquidity Pool Volume Over Time",
                               height: int = 500) -> go.Figure:
        """Create interactive time series plot for volume data."""
        # Ensure date column is datetime
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        fig = go.Figure()
        
        if 'pool' in df.columns:
            # Multiple pools
            for pool in df['pool'].unique():
                pool_data = df[df['pool'] == pool]
                color = self.pool_colors.get(pool, self.colors['primary'])
                
                fig.add_trace(go.Scatter(
                    x=pool_data['date'],
                    y=pool_data['volume_24h'],
                    mode='lines+markers',
                    name=pool,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=(
                        f'<b>{pool}</b><br>'
                        'Date: %{x}<br>'
                        'Volume: $%{y:,.0f}<br>'
                        '<extra></extra>'
                    )
                ))
        else:
            # Single pool
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['volume_24h'],
                mode='lines+markers',
                name='Volume',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=4),
                hovertemplate=(
                    'Date: %{x}<br>'
                    'Volume: $%{y:,.0f}<br>'
                    '<extra></extra>'
                )
            ))
        
        # Update layout with simplified syntax
        layout = self.get_dark_layout(title, height)
        layout.update({
            'xaxis': dict(
                title="Date",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text']
            ),
            'yaxis': dict(
                title="Volume (USD)",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text']
            )
        })
        
        fig.update_layout(layout)
        return fig

    def create_prediction_chart(self, historical_df: pd.DataFrame, prediction_df: pd.DataFrame,
                              title: str = "Volume Prediction with Confidence Intervals") -> go.Figure:
        """Create prediction chart with historical data and forecasts."""
        # Ensure date columns are datetime
        historical_df = historical_df.copy()
        prediction_df = prediction_df.copy()
        
        if 'date' in historical_df.columns:
            historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
        if 'date' in prediction_df.columns:
            prediction_df['date'] = pd.to_datetime(prediction_df['date'], errors='coerce')
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['volume_24h'],
            mode='lines+markers',
            name='Historical Volume',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=4),
            hovertemplate=(
                'Date: %{x}<br>'
                'Volume: $%{y:,.0f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['predicted'],
            mode='lines+markers',
            name='Predicted Volume',
            line=dict(color=self.colors['success'], width=3),
            marker=dict(size=6),
            hovertemplate=(
                'Date: %{x}<br>'
                'Predicted: $%{y:,.0f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(44, 160, 44, 0.2)',
            fill='tonexty',
            name='95% Confidence Interval',
            hovertemplate=(
                'Date: %{x}<br>'
                'Lower Bound: $%{y:,.0f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Add vertical line using shape (more compatible)
        if len(historical_df) > 0 and len(prediction_df) > 0:
            prediction_start = prediction_df['date'].iloc[0]
            
            fig.add_shape(
                type="line",
                x0=prediction_start,
                x1=prediction_start,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color=self.colors['warning'],
                    width=2,
                    dash="dash"
                )
            )
            
            # Add annotation
            fig.add_annotation(
                x=prediction_start,
                y=1.02,
                yref="paper",
                text="Prediction Start",
                showarrow=False,
                font=dict(color=self.colors['warning'], size=12),
                xanchor="center"
            )
        
        # Update layout
        layout = self.get_dark_layout(title, 500)
        layout.update({
            'xaxis': dict(
                title="Date",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text']
            ),
            'yaxis': dict(
                title="Volume (USD)",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text']
            )
        })
        
        fig.update_layout(layout)
        return fig

    def create_pool_comparison(self, df: pd.DataFrame, metric: str = 'volume_24h',
                             title: str = "Pool Volume Comparison") -> go.Figure:
        """Create bar chart comparing metrics across pools."""
        if 'pool' not in df.columns or metric not in df.columns:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No pool comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Aggregate by pool
        pool_stats = df.groupby('pool')[metric].agg(['mean', 'sum', 'std']).reset_index()
        
        # Create bar chart
        fig = go.Figure()
        
        colors = [self.pool_colors.get(pool, self.colors['primary']) for pool in pool_stats['pool']]
        
        fig.add_trace(go.Bar(
            x=pool_stats['pool'],
            y=pool_stats['mean'],
            name='Average',
            marker_color=colors,
            hovertemplate=(
                'Pool: %{x}<br>'
                'Average: $%{y:,.0f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Update layout
        layout = self.get_dark_layout(title, 400)
        layout.update({
            'xaxis': dict(
                title="Pool",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text']
            ),
            'yaxis': dict(
                title=f"Average {metric.replace('_', ' ').title()}",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text']
            )
        })
        
        fig.update_layout(layout)
        return fig

    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                 title: str = "Feature Correlation Heatmap") -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                'X: %{x}<br>'
                'Y: %{y}<br>'
                'Correlation: %{z:.2f}<br>'
                '<extra></extra>'
            )
        ))
        
        layout = self.get_dark_layout(title, 500)
        layout.update({
            'xaxis': dict(color=self.colors['text']),
            'yaxis': dict(color=self.colors['text'])
        })
        
        fig.update_layout(layout)
        return fig

    def create_volume_distribution(self, df: pd.DataFrame, target_col: str = 'volume_24h') -> go.Figure:
        """Create volume distribution plot with statistics."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Volume Distribution', 'Box Plot'),
            vertical_spacing=0.15
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df[target_col],
                nbinsx=30,
                name='Distribution',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        if 'pool' in df.columns:
            for i, pool in enumerate(df['pool'].unique()):
                pool_data = df[df['pool'] == pool]
                color = self.pool_colors.get(pool, self.colors['primary'])
                
                fig.add_trace(
                    go.Box(
                        y=pool_data[target_col],
                        name=pool,
                        marker_color=color
                    ),
                    row=2, col=1
                )
        else:
            fig.add_trace(
                go.Box(
                    y=df[target_col],
                    name='Volume',
                    marker_color=self.colors['primary']
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Volume Distribution Analysis",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.colors['text'])
        )
        
        return fig

    def create_educational_annotations(self) -> Dict[str, str]:
        """Create educational tooltips and explanations."""
        return {
            'volume_spike': 'Volume spikes often indicate market events like buybacks, major announcements, or whale transactions.',
            'moving_average': 'Moving averages smooth out price action to help identify trends. When price is above MA, it suggests upward momentum.',
            'confidence_interval': 'The shaded area shows the range where we expect 95% of actual values to fall. Wider bands indicate higher uncertainty.',
            'correlation': 'Correlation measures how two variables move together. Values near +1 mean they move in the same direction, near -1 means opposite directions.',
            'seasonal_patterns': 'Trading volumes often show patterns based on time of day, day of week, or market cycles.',
            'prediction_accuracy': 'Model accuracy depends on data quality and market conditions. Use predictions as guidance, not absolute truth.',
            'liquidity_pools': 'Higher volume pools typically offer better prices and lower slippage for traders.',
            'fee_revenue': 'Fee revenue is generated from trading activity. Higher volume = higher fees for liquidity providers.'
        }

    def add_educational_features(self, fig: go.Figure, chart_type: str) -> go.Figure:
        """Add educational annotations to charts."""
        annotations = self.create_educational_annotations()
        
        if chart_type == 'volume_timeseries':
            fig.add_annotation(
                text="💡 " + annotations['volume_spike'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255, 255, 0, 0.1)",
                bordercolor=self.colors['warning'],
                borderwidth=1
            )
        elif chart_type == 'prediction':
            fig.add_annotation(
                text="📊 " + annotations['confidence_interval'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(0, 212, 255, 0.1)",
                bordercolor=self.colors['primary'],
                borderwidth=1
            )
        elif chart_type == 'correlation':
            fig.add_annotation(
                text="🔗 " + annotations['correlation'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(0, 230, 118, 0.1)",
                bordercolor=self.colors['success'],
                borderwidth=1
            )
        
        return fig
