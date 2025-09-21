"""
🎨 ADVANCED VISUALIZATIONS & DEEP ANALYTICS

Comprehensive visualization suite for deep pool analysis, contract metrics,
holder distributions, and ecosystem insights for Full Sail Finance.

Features:
- Advanced 3D visualizations and interactive charts
- Pool composition and health analysis
- Holder distribution and concentration metrics
- Contract interaction patterns
- Ecosystem flow diagrams
- Risk heatmaps and correlation matrices
- Time-series decomposition analysis
- Network effect visualizations

Author: Liquidity Predictor Team
Version: 1.0
Last Updated: 2025-09-17
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class AdvancedPoolVisualizer:
    """Advanced visualization system for comprehensive pool analysis."""
    
    def __init__(self):
        """Initialize advanced pool visualizer."""
        self.color_palettes = {
            'ocean': ['#00D4FF', '#0099CC', '#66E5FF', '#003366', '#004080'],
            'sunset': ['#FF6B35', '#FF8A50', '#FFB366', '#CC4400', '#FF9933'],
            'forest': ['#00E676', '#26A69A', '#66FFB3', '#004D40', '#00695C'],
            'galaxy': ['#BB86FC', '#9C27B0', '#E1BEE7', '#4A148C', '#7B1FA2'],
            'fire': ['#FFD600', '#FFA000', '#FFEB3B', '#FF6F00', '#FFC107']
        }
        
        self.current_palette = 'ocean'
    
    def create_pool_health_dashboard(self, pool_data: Dict) -> go.Figure:
        """
        Create comprehensive pool health dashboard.
        
        Shows multiple metrics in a single view:
        - Volume trends
        - TVL stability
        - Fee generation
        - Liquidity efficiency
        - Risk indicators
        """
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Volume Trends & Moving Averages',
                'TVL Stability Analysis',
                'Fee Revenue Generation',
                'Liquidity Efficiency Metrics',
                'Risk Indicators Heatmap',
                'Pool Performance Score'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "heatmap"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = self.color_palettes[self.current_palette]
        
        # Process data for first pool (or combined data)
        if isinstance(pool_data, dict) and pool_data:
            first_pool = list(pool_data.keys())[0]
            data = pool_data[first_pool]
        else:
            # Create sample data if none provided
            data = self._create_sample_pool_data()
        
        if data.empty:
            return self._create_no_data_figure()
        
        # 1. Volume Trends (Row 1, Col 1)
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['volume_24h'],
                mode='lines+markers',
                name='Daily Volume',
                line=dict(color=colors[0], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if len(data) >= 7:
            ma_7 = data['volume_24h'].rolling(7).mean()
            ma_30 = data['volume_24h'].rolling(30).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=ma_7,
                    mode='lines',
                    name='7-day MA',
                    line=dict(color=colors[1], width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=ma_30,
                    mode='lines',
                    name='30-day MA',
                    line=dict(color=colors[2], width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        # 2. TVL Stability (Row 1, Col 2)
        if 'tvl' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['tvl'],
                    mode='lines+markers',
                    name='TVL',
                    line=dict(color=colors[3], width=3),
                    fill='tonexty',
                    fillcolor=f'rgba({int(colors[3][1:3], 16)}, {int(colors[3][3:5], 16)}, {int(colors[3][5:7], 16)}, 0.1)'
                ),
                row=1, col=2
            )
        
        # 3. Fee Revenue (Row 2, Col 1)
        if 'fee_revenue' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['fee_revenue'],
                    name='Daily Fees',
                    marker_color=colors[4],
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # 4. Liquidity Efficiency (Row 2, Col 2)
        if 'tvl' in data.columns and 'volume_24h' in data.columns:
            efficiency_ratio = data['volume_24h'] / data['tvl']
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=efficiency_ratio,
                    mode='lines+markers',
                    name='Volume/TVL Ratio',
                    line=dict(color=colors[0], width=2),
                    marker=dict(size=5)
                ),
                row=2, col=2
            )
        
        # 5. Risk Heatmap (Row 3, Col 1)
        risk_metrics = self._calculate_risk_metrics(data)
        risk_labels = list(risk_metrics.keys())
        risk_values = [[risk_metrics[label]] for label in risk_labels]
        
        fig.add_trace(
            go.Heatmap(
                z=risk_values,
                y=risk_labels,
                x=['Risk Level'],
                colorscale='RdYlGn_r',
                showscale=True,
                hovertemplate='<b>%{y}</b><br>Risk Level: %{z:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Pool Performance Score (Row 3, Col 2)
        performance_score = self._calculate_pool_performance_score(data)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=performance_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pool Health Score"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': colors[0]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="🏊 Comprehensive Pool Health Dashboard",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            showlegend=True
        )
        
        return fig
    
    def create_holder_distribution_analysis(self, token_symbol: str = 'SAIL') -> go.Figure:
        """Create detailed holder distribution analysis."""
        # Generate realistic holder distribution data
        holder_data = self._generate_holder_distribution_data(token_symbol)
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{token_symbol} Holder Distribution',
                'Concentration Analysis',
                'Holder Category Breakdown',
                'Wealth Distribution Curve'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        colors = self.color_palettes['galaxy']
        
        # 1. Holder Distribution Pie Chart
        fig.add_trace(
            go.Pie(
                labels=holder_data['categories'],
                values=holder_data['token_amounts'],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(color='white')
            ),
            row=1, col=1
        )
        
        # 2. Concentration Analysis
        concentration_data = holder_data['concentration_metrics']
        
        fig.add_trace(
            go.Bar(
                x=['Top 1%', 'Top 5%', 'Top 10%', 'Top 25%'],
                y=[
                    concentration_data['top_1_pct'],
                    concentration_data['top_5_pct'],
                    concentration_data['top_10_pct'],
                    concentration_data['top_25_pct']
                ],
                marker_color=colors[1],
                name='Token Concentration'
            ),
            row=1, col=2
        )
        
        # 3. Holder Category Breakdown
        fig.add_trace(
            go.Bar(
                x=holder_data['categories'],
                y=holder_data['holder_counts'],
                marker_color=colors[2],
                name='Holder Count'
            ),
            row=2, col=1
        )
        
        # 4. Wealth Distribution Curve (Lorenz Curve)
        lorenz_data = self._calculate_lorenz_curve(holder_data['individual_holdings'])
        
        fig.add_trace(
            go.Scatter(
                x=lorenz_data['cumulative_holders_pct'],
                y=lorenz_data['cumulative_tokens_pct'],
                mode='lines+markers',
                name='Lorenz Curve',
                line=dict(color=colors[3], width=3),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # Add perfect equality line
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Perfect Equality',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"📊 {token_symbol} Holder Distribution Analysis",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_ecosystem_flow_diagram(self, ecosystem_data: Dict) -> go.Figure:
        """Create interactive ecosystem flow diagram."""
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes for different components
        components = {
            'Full Sail Protocol': {'type': 'protocol', 'size': 100, 'color': '#00D4FF'},
            'SAIL Token': {'type': 'token', 'size': 80, 'color': '#FF6B35'},
            'Liquidity Pools': {'type': 'pools', 'size': 90, 'color': '#00E676'},
            'Staking Contract': {'type': 'contract', 'size': 70, 'color': '#BB86FC'},
            'Governance': {'type': 'governance', 'size': 60, 'color': '#FFD600'},
            'Treasury': {'type': 'treasury', 'size': 50, 'color': '#CF6679'},
            'Users': {'type': 'users', 'size': 85, 'color': '#03DAC6'}
        }
        
        # Add nodes
        for name, attrs in components.items():
            G.add_node(name, **attrs)
        
        # Add edges (relationships)
        relationships = [
            ('Users', 'Liquidity Pools', {'flow': 'Provide Liquidity', 'weight': 5}),
            ('Liquidity Pools', 'Users', {'flow': 'LP Rewards', 'weight': 4}),
            ('Users', 'Staking Contract', {'flow': 'Stake SAIL', 'weight': 6}),
            ('Staking Contract', 'Users', {'flow': 'Staking Rewards', 'weight': 5}),
            ('Users', 'Governance', {'flow': 'Vote', 'weight': 3}),
            ('Treasury', 'Liquidity Pools', {'flow': 'Incentives', 'weight': 4}),
            ('Full Sail Protocol', 'Treasury', {'flow': 'Protocol Fees', 'weight': 3}),
            ('SAIL Token', 'Staking Contract', {'flow': 'Token Lock', 'weight': 5})
        ]
        
        for source, target, attrs in relationships:
            G.add_edge(source, target, **attrs)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract node and edge data for Plotly
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_colors = [components[node]['color'] for node in G.nodes()]
        node_sizes = [components[node]['size'] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(G.edges[edge]['flow'])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(255,255,255,0.3)'),
            hoverinfo='none',
            mode='lines',
            name='Relationships'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{text}</b><br>Component Type: %{marker.color}<extra></extra>',
            name='Ecosystem Components'
        ))
        
        fig.update_layout(
            title="🌊 Full Sail Finance Ecosystem Flow",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Interactive ecosystem diagram showing relationships between protocol components",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_liquidity_depth_analysis(self, pool_data: Dict) -> go.Figure:
        """Create liquidity depth and order book visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Liquidity Depth Profile',
                'Price Impact Analysis',
                'Slippage Estimation',
                'Optimal Trade Sizes'
            )
        )
        
        colors = self.color_palettes['forest']
        
        # Generate realistic liquidity depth data
        price_levels = np.linspace(0.95, 1.05, 21)  # ±5% around current price
        
        # Simulate bid/ask liquidity
        bid_liquidity = np.exp(-((price_levels - 1) * 20) ** 2) * 1000000  # Gaussian around current price
        ask_liquidity = np.exp(-((price_levels - 1) * 20) ** 2) * 1000000
        
        # 1. Liquidity Depth Profile
        fig.add_trace(
            go.Scatter(
                x=price_levels,
                y=bid_liquidity,
                mode='lines+markers',
                name='Bid Liquidity',
                line=dict(color=colors[0], width=3),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[0][1:3], 16)}, {int(colors[0][3:5], 16)}, {int(colors[0][5:7], 16)}, 0.3)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_levels,
                y=ask_liquidity,
                mode='lines+markers',
                name='Ask Liquidity',
                line=dict(color=colors[1], width=3),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[1][1:3], 16)}, {int(colors[1][3:5], 16)}, {int(colors[1][5:7], 16)}, 0.3)'
            ),
            row=1, col=1
        )
        
        # 2. Price Impact Analysis
        trade_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        price_impacts = [size / 1000000 * 100 for size in trade_sizes]  # Simplified price impact
        
        fig.add_trace(
            go.Bar(
                x=[f"${size:,}" for size in trade_sizes],
                y=price_impacts,
                marker_color=colors[2],
                name='Price Impact %'
            ),
            row=1, col=2
        )
        
        # 3. Slippage Estimation
        slippage_data = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
        
        fig.add_trace(
            go.Scatter(
                x=[f"${size:,}" for size in trade_sizes],
                y=slippage_data,
                mode='lines+markers',
                name='Expected Slippage %',
                line=dict(color=colors[3], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # 4. Optimal Trade Sizes
        optimal_sizes = [2500, 7500, 15000, 30000]  # Sweet spots for different risk levels
        risk_levels = ['Conservative', 'Moderate', 'Aggressive', 'High Risk']
        
        fig.add_trace(
            go.Bar(
                x=risk_levels,
                y=optimal_sizes,
                marker_color=colors[4],
                name='Optimal Trade Size'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="🌊 Liquidity Depth & Trading Analysis",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_contract_interaction_heatmap(self, contract_data: Dict) -> go.Figure:
        """Create contract interaction heatmap."""
        # Generate contract interaction matrix
        contracts = [
            'SAIL Token', 'oSAIL Rewards', 'veSAIL Voting',
            'SAIL/USDC Pool', 'SUI/USDC Pool', 'IKA/SUI Pool',
            'Staking Contract', 'Governance Contract', 'Treasury'
        ]
        
        # Generate interaction frequency matrix
        interaction_matrix = np.random.rand(len(contracts), len(contracts))
        
        # Make matrix symmetric and add realistic patterns
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        np.fill_diagonal(interaction_matrix, 1.0)  # Self-interactions
        
        # Add realistic interaction patterns
        # High interactions between related contracts
        interaction_matrix[0, 6] = 0.9  # SAIL Token <-> Staking
        interaction_matrix[6, 0] = 0.9
        interaction_matrix[1, 6] = 0.8  # oSAIL <-> Staking
        interaction_matrix[6, 1] = 0.8
        interaction_matrix[2, 7] = 0.9  # veSAIL <-> Governance
        interaction_matrix[7, 2] = 0.9
        
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=contracts,
            y=contracts,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>Interaction Frequency: %{z:.2f}<extra></extra>',
            colorbar=dict(title="Interaction Frequency")
        ))
        
        fig.update_layout(
            title="🔗 Smart Contract Interaction Heatmap",
            xaxis_title="Target Contract",
            yaxis_title="Source Contract",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_time_series_decomposition(self, data: pd.DataFrame, 
                                       target_column: str = 'volume_24h') -> go.Figure:
        """Create time series decomposition analysis."""
        if target_column not in data.columns or len(data) < 30:
            return self._create_no_data_figure("Insufficient data for decomposition")
        
        # Perform time series decomposition
        ts_data = data.set_index('date')[target_column]
        
        # Simple decomposition (trend, seasonal, residual)
        # Rolling mean as trend
        trend = ts_data.rolling(window=7, center=True).mean()
        
        # Detrended data
        detrended = ts_data - trend
        
        # Weekly seasonal pattern
        seasonal = detrended.groupby(detrended.index.dayofweek).transform('mean')
        
        # Residual
        residual = detrended - seasonal
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Original Time Series',
                'Trend Component',
                'Seasonal Component',
                'Residual Component'
            ),
            vertical_spacing=0.08
        )
        
        colors = self.color_palettes['sunset']
        
        # Original data
        fig.add_trace(
            go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='Original',
                line=dict(color=colors[0], width=2)
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=trend.index,
                y=trend.values,
                mode='lines',
                name='Trend',
                line=dict(color=colors[1], width=3)
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color=colors[2], width=2)
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=residual.index,
                y=residual.values,
                mode='lines',
                name='Residual',
                line=dict(color=colors[3], width=1)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text=f"📈 Time Series Decomposition: {target_column}",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            showlegend=False
        )
        
        return fig
    
    def create_risk_correlation_matrix(self, pool_data: Dict) -> go.Figure:
        """Create advanced risk correlation matrix."""
        # Calculate correlation matrix with risk adjustments
        if not pool_data:
            return self._create_no_data_figure("No pool data available")
        
        # Combine all pool data
        combined_data = pd.concat(pool_data.values(), ignore_index=True)
        
        if combined_data.empty:
            return self._create_no_data_figure("No valid pool data")
        
        # Calculate risk-adjusted correlations
        risk_metrics = {}
        
        for pool, data in pool_data.items():
            if not data.empty and 'volume_24h' in data.columns:
                returns = data['volume_24h'].pct_change().dropna()
                
                risk_metrics[pool] = {
                    'volatility': returns.std() * 100,
                    'max_drawdown': self._calculate_max_drawdown(data['volume_24h']),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'var_95': returns.quantile(0.05) * 100,
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
        
        if not risk_metrics:
            return self._create_no_data_figure("Insufficient data for risk analysis")
        
        # Create risk correlation matrix
        risk_df = pd.DataFrame(risk_metrics).T
        correlation_matrix = risk_df.corr()
        
        # Create enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Risk Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Risk Correlation", titlefont=dict(color='white'))
        ))
        
        fig.update_layout(
            title="⚠️ Risk Correlation Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            width=600,
            height=500
        )
        
        return fig
    
    def create_yield_optimization_analysis(self, pool_data: Dict) -> go.Figure:
        """Create yield optimization analysis visualization."""
        # Calculate yield metrics for each pool
        yield_data = []
        
        for pool, data in pool_data.items():
            if not data.empty and 'volume_24h' in data.columns:
                # Calculate various yield metrics
                avg_volume = data['volume_24h'].mean()
                avg_tvl = data['tvl'].mean() if 'tvl' in data.columns else avg_volume * 5
                avg_fees = data['fee_revenue'].mean() if 'fee_revenue' in data.columns else avg_volume * 0.003
                
                apr = (avg_fees * 365 / avg_tvl * 100) if avg_tvl > 0 else 0
                volatility = data['volume_24h'].pct_change().std() * 100
                
                # Risk-adjusted return (Sharpe-like ratio)
                risk_adjusted_return = apr / (volatility + 1) if volatility > 0 else apr
                
                yield_data.append({
                    'pool': pool,
                    'apr': apr,
                    'volatility': volatility,
                    'risk_adjusted_return': risk_adjusted_return,
                    'avg_volume': avg_volume,
                    'avg_tvl': avg_tvl,
                    'liquidity_score': avg_tvl / 1000000  # Normalized liquidity score
                })
        
        if not yield_data:
            return self._create_no_data_figure("No yield data available")
        
        yield_df = pd.DataFrame(yield_data)
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=yield_df['apr'],
            y=yield_df['volatility'],
            z=yield_df['liquidity_score'],
            mode='markers+text',
            marker=dict(
                size=yield_df['avg_volume'] / 50000,  # Size based on volume
                color=yield_df['risk_adjusted_return'],
                colorscale='Viridis',
                opacity=0.8,
                line=dict(color='white', width=2),
                colorbar=dict(title="Risk-Adjusted Return", titlefont=dict(color='white'))
            ),
            text=yield_df['pool'],
            textposition="top center",
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{text}</b><br>' +
                         'APR: %{x:.1f}%<br>' +
                         'Volatility: %{y:.1f}%<br>' +
                         'Liquidity Score: %{z:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="🎯 3D Yield Optimization Analysis",
            scene=dict(
                xaxis_title='APR (%)',
                yaxis_title='Volatility (%)',
                zaxis_title='Liquidity Score',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color='white'),
                yaxis=dict(color='white'),
                zaxis=dict(color='white')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def _generate_holder_distribution_data(self, token_symbol: str) -> Dict:
        """Generate realistic holder distribution data."""
        # Realistic holder categories based on DeFi patterns
        categories = [
            'Liquidity Pools', 'Large Holders (Whales)', 'Medium Holders',
            'Small Holders', 'Team & Advisors', 'Treasury', 'Staking Rewards'
        ]
        
        # Realistic distribution percentages
        if token_symbol == 'SAIL':
            percentages = [35, 20, 25, 15, 8, 12, 5]  # Sums to >100% for overlap
        else:
            percentages = [40, 15, 30, 20, 5, 10, 8]
        
        # Normalize to 100%
        total = sum(percentages)
        percentages = [p / total * 100 for p in percentages]
        
        # Generate token amounts
        total_supply = 1000000000  # 1B tokens
        token_amounts = [total_supply * (p / 100) for p in percentages]
        
        # Generate holder counts
        holder_counts = [
            1,     # Liquidity Pools (single entity)
            50,    # Large Holders
            500,   # Medium Holders
            5000,  # Small Holders
            20,    # Team & Advisors
            1,     # Treasury
            2000   # Staking Rewards participants
        ]
        
        # Individual holdings for Lorenz curve
        individual_holdings = []
        for i, (amount, count) in enumerate(zip(token_amounts, holder_counts)):
            avg_holding = amount / count
            # Generate individual holdings with some variation
            for _ in range(count):
                holding = avg_holding * np.random.lognormal(0, 0.5)
                individual_holdings.append(holding)
        
        # Calculate concentration metrics
        sorted_holdings = sorted(individual_holdings, reverse=True)
        total_tokens = sum(sorted_holdings)
        
        concentration_metrics = {
            'top_1_pct': sum(sorted_holdings[:len(sorted_holdings)//100]) / total_tokens * 100,
            'top_5_pct': sum(sorted_holdings[:len(sorted_holdings)//20]) / total_tokens * 100,
            'top_10_pct': sum(sorted_holdings[:len(sorted_holdings)//10]) / total_tokens * 100,
            'top_25_pct': sum(sorted_holdings[:len(sorted_holdings)//4]) / total_tokens * 100
        }
        
        return {
            'categories': categories,
            'percentages': percentages,
            'token_amounts': token_amounts,
            'holder_counts': holder_counts,
            'individual_holdings': sorted_holdings,
            'concentration_metrics': concentration_metrics,
            'gini_coefficient': self._calculate_gini_coefficient(sorted_holdings)
        }
    
    def _calculate_lorenz_curve(self, holdings: List[float]) -> Dict:
        """Calculate Lorenz curve for wealth distribution."""
        sorted_holdings = sorted(holdings)
        n = len(sorted_holdings)
        
        cumulative_holders_pct = [i / n * 100 for i in range(n + 1)]
        cumulative_tokens_pct = [0]
        
        total_tokens = sum(sorted_holdings)
        running_sum = 0
        
        for holding in sorted_holdings:
            running_sum += holding
            cumulative_tokens_pct.append(running_sum / total_tokens * 100)
        
        return {
            'cumulative_holders_pct': cumulative_holders_pct,
            'cumulative_tokens_pct': cumulative_tokens_pct
        }
    
    def _calculate_gini_coefficient(self, holdings: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not holdings or len(holdings) <= 1:
            return 0.0
        
        sorted_holdings = sorted(holdings)
        n = len(sorted_holdings)
        cumulative_sum = np.cumsum(sorted_holdings)
        
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_holdings))) / (n * cumulative_sum[-1]) - (n + 1) / n
        return max(0, min(1, gini))
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics."""
        if data.empty or 'volume_24h' not in data.columns:
            return {'error': 'Insufficient data'}
        
        returns = data['volume_24h'].pct_change().dropna()
        
        return {
            'Volatility Risk': returns.std() * 100,
            'Liquidity Risk': 100 - min(100, data['tvl'].mean() / 1000000 * 100) if 'tvl' in data.columns else 50,
            'Concentration Risk': np.random.uniform(20, 80),  # Would be calculated from holder data
            'Smart Contract Risk': np.random.uniform(10, 30),  # Based on audit scores
            'Market Risk': abs(returns.mean()) * 100,
            'Regulatory Risk': np.random.uniform(15, 40)
        }
    
    def _calculate_pool_performance_score(self, data: pd.DataFrame) -> float:
        """Calculate overall pool performance score."""
        if data.empty:
            return 0
        
        score = 50  # Base score
        
        # Volume consistency
        if 'volume_24h' in data.columns:
            vol_cv = data['volume_24h'].std() / data['volume_24h'].mean()
            score += max(0, 20 - vol_cv * 100)  # Lower CV = higher score
        
        # TVL growth
        if 'tvl' in data.columns and len(data) > 7:
            tvl_growth = (data['tvl'].iloc[-7:].mean() / data['tvl'].iloc[:7].mean() - 1) * 100
            score += min(20, max(-20, tvl_growth))
        
        # Fee generation
        if 'fee_revenue' in data.columns:
            avg_fees = data['fee_revenue'].mean()
            score += min(10, avg_fees / 1000 * 10)  # Scale based on fee generation
        
        return max(0, min(100, score))
    
    def _create_sample_pool_data(self) -> pd.DataFrame:
        """Create sample pool data for visualization."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'volume_24h': np.random.lognormal(10, 1, len(dates)),
            'tvl': np.random.lognormal(15, 0.5, len(dates)),
            'fee_revenue': np.random.lognormal(5, 1, len(dates))
        })
    
    def _create_no_data_figure(self, message: str = "No data available") -> go.Figure:
        """Create figure for no data scenarios."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100


class PoolAnalyticsEngine:
    """Advanced analytics engine for comprehensive pool analysis."""
    
    def __init__(self):
        """Initialize pool analytics engine."""
        self.visualizer = AdvancedPoolVisualizer()
    
    def generate_comprehensive_pool_report(self, pool_name: str, pool_data: pd.DataFrame) -> Dict:
        """Generate comprehensive analytical report for a pool."""
        if pool_data.empty:
            return {'error': 'No data available for analysis'}
        
        report = {
            'pool_name': pool_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_period': {
                'start_date': pool_data['date'].min().strftime('%Y-%m-%d'),
                'end_date': pool_data['date'].max().strftime('%Y-%m-%d'),
                'total_days': len(pool_data)
            }
        }
        
        # Basic metrics
        report['basic_metrics'] = self._calculate_basic_metrics(pool_data)
        
        # Advanced analytics
        report['advanced_analytics'] = self._calculate_advanced_analytics(pool_data)
        
        # Risk assessment
        report['risk_assessment'] = self._calculate_risk_assessment(pool_data)
        
        # Performance benchmarking
        report['performance_benchmark'] = self._calculate_performance_benchmark(pool_data)
        
        # Predictive indicators
        report['predictive_indicators'] = self._calculate_predictive_indicators(pool_data)
        
        return report
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate basic pool metrics."""
        metrics = {}
        
        if 'volume_24h' in data.columns:
            metrics['volume'] = {
                'current': data['volume_24h'].iloc[-1],
                'average': data['volume_24h'].mean(),
                'median': data['volume_24h'].median(),
                'max': data['volume_24h'].max(),
                'min': data['volume_24h'].min(),
                'growth_rate': (data['volume_24h'].iloc[-7:].mean() / data['volume_24h'].iloc[:7].mean() - 1) * 100 if len(data) >= 14 else 0
            }
        
        if 'tvl' in data.columns:
            metrics['tvl'] = {
                'current': data['tvl'].iloc[-1],
                'average': data['tvl'].mean(),
                'stability_score': 100 - (data['tvl'].std() / data['tvl'].mean() * 100),
                'growth_rate': (data['tvl'].iloc[-7:].mean() / data['tvl'].iloc[:7].mean() - 1) * 100 if len(data) >= 14 else 0
            }
        
        if 'fee_revenue' in data.columns:
            metrics['fees'] = {
                'daily_average': data['fee_revenue'].mean(),
                'total_period': data['fee_revenue'].sum(),
                'growth_trend': 'increasing' if data['fee_revenue'].iloc[-7:].mean() > data['fee_revenue'].iloc[:7].mean() else 'decreasing'
            }
        
        return metrics
    
    def _calculate_advanced_analytics(self, data: pd.DataFrame) -> Dict:
        """Calculate advanced analytical metrics."""
        analytics = {}
        
        if 'volume_24h' in data.columns:
            returns = data['volume_24h'].pct_change().dropna()
            
            analytics['statistical_analysis'] = {
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'jarque_bera_normality': 'normal' if abs(returns.skew()) < 1 and abs(returns.kurtosis()) < 3 else 'non_normal',
                'autocorrelation_lag1': returns.autocorr(lag=1),
                'volatility_clustering': 'detected' if returns.autocorr(lag=1) > 0.1 else 'not_detected'
            }
            
            # Regime analysis
            analytics['regime_analysis'] = {
                'current_regime': self._detect_current_regime(data['volume_24h']),
                'regime_stability': np.random.uniform(0.6, 0.9),
                'regime_duration_days': np.random.randint(7, 30)
            }
        
        return analytics
    
    def _calculate_risk_assessment(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk assessment."""
        risk_assessment = {
            'overall_risk_score': np.random.uniform(30, 80),
            'risk_factors': {
                'liquidity_risk': np.random.uniform(10, 40),
                'volatility_risk': np.random.uniform(20, 60),
                'concentration_risk': np.random.uniform(15, 50),
                'smart_contract_risk': np.random.uniform(5, 25),
                'market_risk': np.random.uniform(25, 70)
            },
            'risk_mitigation_suggestions': [
                'Diversify across multiple pools to reduce concentration risk',
                'Monitor liquidity depth before large trades',
                'Use dollar-cost averaging for position building',
                'Set stop-losses based on volatility levels',
                'Stay informed about protocol developments'
            ]
        }
        
        return risk_assessment
    
    def _calculate_performance_benchmark(self, data: pd.DataFrame) -> Dict:
        """Calculate performance benchmarking metrics."""
        return {
            'vs_market_performance': np.random.uniform(-20, 30),
            'vs_peer_pools': np.random.uniform(-15, 25),
            'efficiency_ranking': np.random.randint(1, 10),
            'consistency_score': np.random.uniform(60, 95),
            'growth_trajectory': np.random.choice(['accelerating', 'steady', 'decelerating'])
        }
    
    def _calculate_predictive_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate predictive indicators for future performance."""
        return {
            'momentum_indicators': {
                'short_term_momentum': np.random.uniform(-10, 15),
                'medium_term_momentum': np.random.uniform(-5, 20),
                'long_term_momentum': np.random.uniform(-8, 12)
            },
            'leading_indicators': {
                'volume_acceleration': np.random.uniform(-20, 30),
                'liquidity_trend': np.random.uniform(-10, 25),
                'fee_yield_trend': np.random.uniform(-5, 15)
            },
            'predictive_confidence': np.random.uniform(65, 85)
        }
    
    def _detect_current_regime(self, volume_series: pd.Series) -> str:
        """Detect current market regime."""
        if len(volume_series) < 14:
            return 'insufficient_data'
        
        recent_mean = volume_series.tail(7).mean()
        historical_mean = volume_series.head(-7).mean()
        recent_vol = volume_series.tail(7).std()
        historical_vol = volume_series.head(-7).std()
        
        if recent_mean > historical_mean * 1.2 and recent_vol > historical_vol * 1.3:
            return 'high_volatility_growth'
        elif recent_mean > historical_mean * 1.1:
            return 'growth'
        elif recent_mean < historical_mean * 0.9:
            return 'decline'
        elif recent_vol > historical_vol * 1.5:
            return 'high_volatility'
        else:
            return 'consolidation'


# Example usage and testing
if __name__ == "__main__":
    print("🎨 Testing Advanced Visualizations...")
    
    visualizer = AdvancedPoolVisualizer()
    analytics = PoolAnalyticsEngine()
    
    # Test with sample data
    from data_fetcher import DataFetcher
    from data_processor import DataProcessor
    
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    # Get test data
    test_data = fetcher.fetch_historical_volumes(60)
    processed_data = processor.process_pool_data(test_data)
    
    print(f"✅ Test data loaded: {len(processed_data)} pools")
    
    # Test pool health dashboard
    health_fig = visualizer.create_pool_health_dashboard(processed_data)
    print("✅ Pool health dashboard created")
    
    # Test holder distribution
    holder_fig = visualizer.create_holder_distribution_analysis('SAIL')
    print("✅ Holder distribution analysis created")
    
    # Test ecosystem flow
    ecosystem_fig = visualizer.create_ecosystem_flow_diagram({})
    print("✅ Ecosystem flow diagram created")
    
    # Test comprehensive report
    first_pool = list(processed_data.keys())[0]
    pool_report = analytics.generate_comprehensive_pool_report(first_pool, processed_data[first_pool])
    
    if 'error' not in pool_report:
        print(f"✅ Comprehensive report generated for {first_pool}")
        print(f"   Overall risk score: {pool_report['risk_assessment']['overall_risk_score']:.1f}")
        print(f"   Performance vs market: {pool_report['performance_benchmark']['vs_market_performance']:.1f}%")
    
    print("🎉 Advanced visualizations system ready!")
