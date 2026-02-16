"""
Visualization module for Full Sail Finance liquidity pool volume prediction.
Creates interactive charts using Plotly and Altair with educational features.
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
    
    def create_volume_timeseries(self, df: pd.DataFrame, title: str = "Liquidity Pool Volume Over Time",
                               height: int = 500) -> go.Figure:
        """
        Create interactive time series plot for volume data.
        
        Args:
            df: DataFrame with date and volume columns
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Plotly figure
        """
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
        
        # Add moving averages if available
        if 'volume_24h_ma7' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['volume_24h_ma7'],
                mode='lines',
                name='7-day MA',
                line=dict(color=self.colors['warning'], width=1, dash='dash'),
                hovertemplate='7-day MA: $%{y:,.0f}<extra></extra>'
            ))
        
        if 'volume_24h_ma30' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['volume_24h_ma30'],
                mode='lines',
                name='30-day MA',
                line=dict(color=self.colors['info'], width=1, dash='dot'),
                hovertemplate='30-day MA: $%{y:,.0f}<extra></extra>'
            ))
        
        # Update layout with dark mode styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Inter, sans-serif", color=self.colors['text'])
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text'],
            ),
            yaxis=dict(
                title="Volume (USD)",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text'],
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
        
        return fig
    
    def create_prediction_chart(self, historical_df: pd.DataFrame, prediction_df: pd.DataFrame,
                              title: str = "Volume Prediction with Confidence Intervals") -> go.Figure:
        """
        Create prediction chart with historical data and forecasts.
        
        Args:
            historical_df: Historical volume data
            prediction_df: Prediction data with confidence intervals
            title: Chart title
            
        Returns:
            Plotly figure
        """
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
        
        # Add vertical line at prediction start using shapes (more compatible)
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
        
        # Update layout with dark theme
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Inter, sans-serif", color=self.colors['text'])
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text'],
            ),
            yaxis=dict(
                title="Volume (USD)",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text'],
            ),
            height=500,
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
        
        return fig
    
    def create_pool_comparison(self, df: pd.DataFrame, metric: str = 'volume_24h',
                             title: str = "Pool Volume Comparison") -> go.Figure:
        """
        Create bar chart comparing metrics across pools.
        
        Args:
            df: DataFrame with pool data
            metric: Metric to compare
            title: Chart title
            
        Returns:
            Plotly figure
        """
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
        
        # Update layout with dark theme
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Inter, sans-serif", color=self.colors['text'])
            ),
            xaxis=dict(
                title="Pool",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                color=self.colors['text'],
            ),
            yaxis=dict(
                title=f"Average {metric.replace('_', ' ').title()}",
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                tickformat='$,.0f',
                color=self.colors['text'],
            ),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                 title: str = "Feature Correlation Heatmap") -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            
        Returns:
            Plotly figure
        """
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
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Inter, sans-serif", color=self.colors['text'])
            ),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.colors['text']),
            xaxis=dict(color=self.colors['text']),
            yaxis=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_volume_distribution(self, df: pd.DataFrame, target_col: str = 'volume_24h') -> go.Figure:
        """
        Create volume distribution plot with statistics.
        
        Args:
            df: DataFrame with volume data
            target_col: Volume column name
            
        Returns:
            Plotly figure
        """
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
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_altair_volume_chart(self, df: pd.DataFrame) -> alt.Chart:
        """
        Create Altair volume chart with brush selection.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Altair chart
        """
        # Create selection
        brush = alt.selection_interval(encodings=['x'])
        
        # Base chart
        base = alt.Chart(df).add_selection(brush)
        
        # Main chart
        main_chart = base.mark_line(
            point=True,
            strokeWidth=2
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('volume_24h:Q', title='Volume (USD)', scale=alt.Scale(zero=False)),
            color=alt.Color('pool:N', scale=alt.Scale(range=list(self.pool_colors.values()))) if 'pool' in df.columns else alt.value(self.colors['primary']),
            tooltip=['date:T', 'volume_24h:Q', 'pool:N'] if 'pool' in df.columns else ['date:T', 'volume_24h:Q']
        ).properties(
            width=700,
            height=300,
            title="Interactive Volume Chart (Brush to select time range)"
        )
        
        # Detail chart
        detail_chart = base.mark_line().encode(
            x=alt.X('date:T', title='Date', scale=alt.Scale(domain=brush)),
            y=alt.Y('volume_24h:Q', title='Volume (USD)'),
            color=alt.Color('pool:N') if 'pool' in df.columns else alt.value(self.colors['primary'])
        ).properties(
            width=700,
            height=150,
            title="Detailed View"
        )
        
        return alt.vconcat(main_chart, detail_chart)
    
    def create_altair_correlation_chart(self, df: pd.DataFrame) -> alt.Chart:
        """
        Create Altair correlation scatter plot matrix.
        
        Args:
            df: DataFrame with numeric features
            
        Returns:
            Altair chart
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return alt.Chart().mark_text(text="Insufficient numeric data for correlation")
        
        # Take first 4 numeric columns for readability
        cols_to_plot = numeric_cols[:4]
        
        return alt.Chart(df).mark_circle(size=60).encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color=alt.Color('pool:N') if 'pool' in df.columns else alt.value(self.colors['primary']),
            tooltip=cols_to_plot + (['pool'] if 'pool' in df.columns else [])
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=cols_to_plot,
            column=cols_to_plot
        ).resolve_scale(
            color='independent'
        )
    
    def create_educational_annotations(self) -> Dict[str, str]:
        """
        Create educational tooltips and explanations.
        
        Returns:
            Dictionary with educational content
        """
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
        """
        Add educational annotations to charts.
        
        Args:
            fig: Plotly figure
            chart_type: Type of chart for specific annotations
            
        Returns:
            Enhanced figure with educational features
        """
        annotations = self.create_educational_annotations()
        
        if chart_type == 'volume_timeseries':
            fig.add_annotation(
                text="💡 " + annotations['volume_spike'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="lightyellow",
                bordercolor="orange",
                borderwidth=1
            )
        elif chart_type == 'prediction':
            fig.add_annotation(
                text="📊 " + annotations['confidence_interval'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=1
            )
        elif chart_type == 'correlation':
            fig.add_annotation(
                text="🔗 " + annotations['correlation'],
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from data_processor import DataProcessor
    
    # Initialize components
    fetcher = DataFetcher()
    processor = DataProcessor()
    visualizer = VolumeVisualizer()
    
    print("Testing visualization components...")
    
    # Fetch and process data
    historical_data = fetcher.fetch_historical_volumes(30)
    processed_data = processor.process_pool_data(historical_data)
    
    # Test visualizations
    for pool_name, pool_data in list(processed_data.items())[:1]:  # Test first pool
        print(f"Testing visualizations for {pool_name}:")
        
        # Volume timeseries
        fig_timeseries = visualizer.create_volume_timeseries(pool_data)
        print("✓ Created volume timeseries chart")
        
        # Pool comparison (if multiple pools)
        if 'pool' in historical_data.columns:
            fig_comparison = visualizer.create_pool_comparison(historical_data)
            print("✓ Created pool comparison chart")
        
        # Correlation heatmap
        correlations = processor.compute_correlations(pool_data)
        if not correlations.empty:
            fig_heatmap = visualizer.create_correlation_heatmap(correlations)
            print("✓ Created correlation heatmap")
        
        # Volume distribution
        fig_distribution = visualizer.create_volume_distribution(pool_data)
        print("✓ Created volume distribution chart")
        
        # Altair charts
        try:
            altair_chart = visualizer.create_altair_volume_chart(pool_data)
            print("✓ Created Altair volume chart")
        except Exception as e:
            print(f"✗ Altair chart failed: {e}")
    
    print("Visualization testing completed!")
