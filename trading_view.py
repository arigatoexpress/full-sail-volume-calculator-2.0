"""
TradingView-like interface for Full Sail Finance.
Advanced charting with multiple timeframes, technical indicators, and ruler tools.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from data_processor import DataProcessor


class TechnicalAnalysis:
    """Technical analysis indicators for trading charts."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalAnalysis.sma(data, window)
        std = data.rolling(window=window, min_periods=1).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalAnalysis.ema(data, fast)
        ema_slow = TechnicalAnalysis.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysis.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        return k_percent, d_percent
    
    @staticmethod
    def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume Simple Moving Average."""
        return volume.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window, min_periods=1).mean()


class TradingViewChart:
    """TradingView-like chart implementation."""
    
    def __init__(self):
        """Initialize TradingView chart."""
        self.colors = {
            'background': '#131722',
            'grid': '#2A2E39',
            'text': '#D1D4DC',
            'green': '#26A69A',
            'red': '#EF5350',
            'blue': '#2196F3',
            'orange': '#FF9800',
            'purple': '#9C27B0',
            'yellow': '#FFEB3B'
        }
        
        # Time frame configurations
        self.timeframes = {
            '1m': {'name': '1 Minute', 'freq': '1T', 'periods': 1440},
            '5m': {'name': '5 Minutes', 'freq': '5T', 'periods': 288},
            '15m': {'name': '15 Minutes', 'freq': '15T', 'periods': 96},
            '1h': {'name': '1 Hour', 'freq': '1H', 'periods': 168},
            '4h': {'name': '4 Hours', 'freq': '4H', 'periods': 42},
            '1d': {'name': '1 Day', 'freq': '1D', 'periods': 365},
            '1w': {'name': '1 Week', 'freq': '1W', 'periods': 52},
            '1M': {'name': '1 Month', 'freq': '1M', 'periods': 12}
        }
    
    def generate_ohlcv_data(self, pool: str, timeframe: str, periods: int = None) -> pd.DataFrame:
        """Generate OHLCV data for a specific pool and timeframe."""
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        tf_config = self.timeframes[timeframe]
        if periods is None:
            periods = min(tf_config['periods'], 100)  # Limit for performance
        
        # Generate base volume data
        fetcher = DataFetcher()
        base_data = fetcher.fetch_historical_volumes(periods * 2)  # Get extra data
        pool_data = base_data[base_data['pool'] == pool].copy()
        
        if pool_data.empty:
            return pd.DataFrame()
        
        # Convert to OHLCV format
        ohlcv_records = []
        
        for _, row in pool_data.iterrows():
            # Simulate intraday OHLCV from daily volume
            base_volume = row['volume_24h']
            
            # Generate realistic OHLC prices (normalized around volume)
            price_base = base_volume / 10000  # Normalize to reasonable price range
            
            # Add realistic price movements
            open_price = price_base * (1 + np.random.normal(0, 0.02))
            high_price = open_price * (1 + abs(np.random.normal(0, 0.03)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.03)))
            close_price = open_price + np.random.normal(0, 0.02) * open_price
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            ohlcv_records.append({
                'timestamp': row['date'],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': base_volume,
                'pool': pool
            })
        
        df = pd.DataFrame(ohlcv_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.tail(periods)
    
    def create_tradingview_chart(self, pool: str, timeframe: str, 
                               indicators: List[str] = None,
                               show_volume: bool = True) -> go.Figure:
        """Create TradingView-style chart with technical indicators."""
        if indicators is None:
            indicators = []
        
        # Get OHLCV data
        ohlcv_data = self.generate_ohlcv_data(pool, timeframe)
        
        if ohlcv_data.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for {pool}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Determine subplot structure
        subplot_count = 1  # Main price chart
        if show_volume:
            subplot_count += 1
        if 'RSI' in indicators:
            subplot_count += 1
        if 'MACD' in indicators:
            subplot_count += 1
        
        # Create subplot titles
        subplot_titles = [f"{pool} - {self.timeframes[timeframe]['name']}"]
        if show_volume:
            subplot_titles.append("Volume")
        if 'RSI' in indicators:
            subplot_titles.append("RSI")
        if 'MACD' in indicators:
            subplot_titles.append("MACD")
        
        # Calculate row heights
        if subplot_count == 1:
            row_heights = [1.0]
        elif subplot_count == 2:
            row_heights = [0.7, 0.3]
        elif subplot_count == 3:
            row_heights = [0.6, 0.2, 0.2]
        else:
            row_heights = [0.5, 0.2, 0.15, 0.15]
        
        # Create subplots
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_data['timestamp'],
                open=ohlcv_data['open'],
                high=ohlcv_data['high'],
                low=ohlcv_data['low'],
                close=ohlcv_data['close'],
                name="Price",
                increasing_line_color=self.colors['green'],
                decreasing_line_color=self.colors['red'],
                # Use hoverinfo instead of hovertemplate for candlestick
                hoverinfo='x+y+name'
            ),
            row=1, col=1
        )
        
        # Add technical indicators to main chart
        current_row = 1
        
        # Moving Averages
        if 'SMA_20' in indicators:
            sma_20 = TechnicalAnalysis.sma(ohlcv_data['close'], 20)
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=sma_20,
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=self.colors['blue'], width=1),
                    hovertemplate="SMA 20: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in indicators:
            sma_50 = TechnicalAnalysis.sma(ohlcv_data['close'], 50)
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=sma_50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color=self.colors['orange'], width=1),
                    hovertemplate="SMA 50: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        if 'EMA_20' in indicators:
            ema_20 = TechnicalAnalysis.ema(ohlcv_data['close'], 20)
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=ema_20,
                    mode='lines',
                    name='EMA 20',
                    line=dict(color=self.colors['purple'], width=1, dash='dash'),
                    hovertemplate="EMA 20: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB' in indicators:
            bb_upper, bb_middle, bb_lower = TechnicalAnalysis.bollinger_bands(ohlcv_data['close'])
            
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=bb_upper,
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.colors['blue'], width=1, dash='dot'),
                    hovertemplate="BB Upper: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=bb_lower,
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.colors['blue'], width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(33, 150, 243, 0.1)',
                    hovertemplate="BB Lower: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Middle band
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=bb_middle,
                    mode='lines',
                    name='BB Middle',
                    line=dict(color=self.colors['blue'], width=1),
                    hovertemplate="BB Middle: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Volume chart
        if show_volume:
            current_row += 1
            
            # Volume bars
            volume_colors = ['green' if close >= open else 'red' 
                           for close, open in zip(ohlcv_data['close'], ohlcv_data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=ohlcv_data['timestamp'],
                    y=ohlcv_data['volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7,
                    hovertemplate="Volume: %{y:,.0f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            # Volume SMA
            if 'Volume_SMA' in indicators:
                vol_sma = TechnicalAnalysis.volume_sma(ohlcv_data['volume'])
                fig.add_trace(
                    go.Scatter(
                        x=ohlcv_data['timestamp'],
                        y=vol_sma,
                        mode='lines',
                        name='Volume SMA',
                        line=dict(color=self.colors['yellow'], width=2),
                        hovertemplate="Vol SMA: %{y:,.0f}<extra></extra>"
                    ),
                    row=current_row, col=1
                )
        
        # RSI indicator
        if 'RSI' in indicators:
            current_row += 1
            rsi_values = TechnicalAnalysis.rsi(ohlcv_data['close'])
            
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=rsi_values,
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['purple'], width=2),
                    hovertemplate="RSI: %{y:.1f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            # RSI levels
            for level, color in [(70, 'red'), (50, 'gray'), (30, 'green')]:
                fig.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.5,
                    row=current_row, col=1
                )
        
        # MACD indicator
        if 'MACD' in indicators:
            current_row += 1
            macd_line, signal_line, histogram = TechnicalAnalysis.macd(ohlcv_data['close'])
            
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=macd_line,
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['blue'], width=2),
                    hovertemplate="MACD: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data['timestamp'],
                    y=signal_line,
                    mode='lines',
                    name='Signal',
                    line=dict(color=self.colors['red'], width=2),
                    hovertemplate="Signal: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            # Histogram
            histogram_colors = ['green' if h >= 0 else 'red' for h in histogram]
            fig.add_trace(
                go.Bar(
                    x=ohlcv_data['timestamp'],
                    y=histogram,
                    name='MACD Histogram',
                    marker_color=histogram_colors,
                    opacity=0.6,
                    hovertemplate="Histogram: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
        
        # Update layout for TradingView style
        fig.update_layout(
            title=dict(
                text=f"{pool} - {self.timeframes[timeframe]['name']} Chart",
                x=0.5,
                font=dict(size=18, color=self.colors['text'])
            ),
            xaxis_rangeslider_visible=False,  # Hide default range slider
            height=600 + (subplot_count - 1) * 150,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            hovermode='x unified',
            dragmode='zoom'  # Enable zoom by default
        )
        
        # Update all axes
        for i in range(1, subplot_count + 1):
            fig.update_xaxes(
                gridcolor=self.colors['grid'],
                showgrid=True,
                color=self.colors['text'],
                row=i, col=1
            )
            fig.update_yaxes(
                gridcolor=self.colors['grid'],
                showgrid=True,
                color=self.colors['text'],
                row=i, col=1
            )
        
        # Add crosshair cursor
        fig.update_layout(
            xaxis=dict(
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikemode="across",
                spikethickness=1
            ),
            yaxis=dict(
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikemode="across",
                spikethickness=1
            )
        )
        
        return fig
    
    def add_drawing_tools(self, fig: go.Figure) -> go.Figure:
        """Add drawing tools and ruler functionality."""
        # Add drawing tools configuration
        config = {
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ],
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'full_sail_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'height': 800,
                'width': 1200,
                'scale': 1
            }
        }
        
        # Add ruler/measurement tools
        fig.update_layout(
            dragmode='drawrect',  # Default to rectangle drawing
            newshape=dict(
                line_color='yellow',
                line_width=2,
                opacity=0.8
            ),
            modebar=dict(
                bgcolor='rgba(0,0,0,0.5)',
                color='white',
                activecolor='cyan'
            )
        )
        
        return fig, config


class TradingViewInterface:
    """Streamlit interface for TradingView-like functionality."""
    
    def __init__(self):
        """Initialize TradingView interface."""
        self.chart = TradingViewChart()
        self.ta = TechnicalAnalysis()
    
    def render_trading_interface(self) -> None:
        """Render the main trading interface."""
        st.subheader("📈 TradingView-Style Analysis")
        
        # Chart controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pool = st.selectbox(
                "📊 Pool",
                ['SAIL/USDC', 'SUI/USDC', 'IKA/SUI', 'ALKIMI/SUI', 'USDZ/USDC',
                 'USDT/USDC', 'wBTC/USDC', 'ETH/USDC', 'WAL/SUI', 'DEEP/SUI'],
                help="Select pool to analyze"
            )
        
        with col2:
            timeframe = st.selectbox(
                "⏰ Timeframe",
                ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'],
                index=5,  # Default to 1d
                help="Select chart timeframe"
            )
        
        with col3:
            chart_type = st.selectbox(
                "📊 Chart Type",
                ['Candlestick', 'Line', 'Area'],
                help="Select chart visualization type"
            )
        
        with col4:
            show_volume = st.checkbox(
                "📊 Show Volume",
                value=True,
                help="Display volume bars below price chart"
            )
        
        # Technical indicators panel
        st.markdown("### 🔧 Technical Indicators")
        
        indicator_cols = st.columns(6)
        
        with indicator_cols[0]:
            sma_20 = st.checkbox("SMA 20", help="20-period Simple Moving Average")
            sma_50 = st.checkbox("SMA 50", help="50-period Simple Moving Average")
        
        with indicator_cols[1]:
            ema_20 = st.checkbox("EMA 20", help="20-period Exponential Moving Average")
            bb = st.checkbox("Bollinger Bands", help="Bollinger Bands (20, 2)")
        
        with indicator_cols[2]:
            rsi = st.checkbox("RSI", help="Relative Strength Index (14)")
            macd = st.checkbox("MACD", help="MACD (12, 26, 9)")
        
        with indicator_cols[3]:
            volume_sma = st.checkbox("Volume SMA", help="Volume Simple Moving Average")
            atr = st.checkbox("ATR", help="Average True Range")
        
        with indicator_cols[4]:
            support_resistance = st.checkbox("S/R Lines", help="Support/Resistance Lines")
            fibonacci = st.checkbox("Fibonacci", help="Fibonacci Retracements")
        
        with indicator_cols[5]:
            crosshair = st.checkbox("Crosshair", value=True, help="Enable crosshair cursor")
            ruler = st.checkbox("Ruler", value=True, help="Enable measurement tools")
        
        # Compile selected indicators
        selected_indicators = []
        if sma_20:
            selected_indicators.append('SMA_20')
        if sma_50:
            selected_indicators.append('SMA_50')
        if ema_20:
            selected_indicators.append('EMA_20')
        if bb:
            selected_indicators.append('BB')
        if rsi:
            selected_indicators.append('RSI')
        if macd:
            selected_indicators.append('MACD')
        if volume_sma:
            selected_indicators.append('Volume_SMA')
        
        # Generate and display chart
        if st.button("📈 Generate TradingView Chart", type="primary"):
            try:
                with st.spinner("Generating advanced chart..."):
                    fig = self.chart.create_tradingview_chart(
                        pool, timeframe, selected_indicators, show_volume
                    )
                    
                    # Add drawing tools if ruler is enabled
                    if ruler:
                        fig, config = self.chart.add_drawing_tools(fig)
                        st.plotly_chart(fig, use_container_width=True, config=config)
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display technical analysis summary
                    self._display_technical_summary(pool, timeframe, selected_indicators)
            
            except Exception as e:
                st.error(f"❌ Error generating chart: {str(e)}")
        
        # Quick timeframe buttons
        st.markdown("### ⚡ Quick Timeframes")
        
        quick_cols = st.columns(8)
        timeframe_buttons = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
        
        for i, tf in enumerate(timeframe_buttons):
            with quick_cols[i]:
                if st.button(tf, key=f"tf_{tf}"):
                    st.session_state.selected_timeframe = tf
                    st.experimental_rerun()
    
    def _display_technical_summary(self, pool: str, timeframe: str, indicators: List[str]) -> None:
        """Display technical analysis summary."""
        st.markdown("---")
        st.subheader("📊 Technical Analysis Summary")
        
        # Get data for analysis
        ohlcv_data = self.chart.generate_ohlcv_data(pool, timeframe)
        
        if not ohlcv_data.empty:
            current_price = ohlcv_data['close'].iloc[-1]
            price_change = ohlcv_data['close'].iloc[-1] - ohlcv_data['close'].iloc[-2] if len(ohlcv_data) > 1 else 0
            price_change_pct = (price_change / ohlcv_data['close'].iloc[-2] * 100) if len(ohlcv_data) > 1 and ohlcv_data['close'].iloc[-2] != 0 else 0
            
            # Current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:,.4f}",
                    f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                high_24h = ohlcv_data['high'].tail(24).max() if len(ohlcv_data) >= 24 else ohlcv_data['high'].max()
                low_24h = ohlcv_data['low'].tail(24).min() if len(ohlcv_data) >= 24 else ohlcv_data['low'].min()
                st.metric("24h High", f"${high_24h:,.4f}")
                st.metric("24h Low", f"${low_24h:,.4f}")
            
            with col3:
                volume_24h = ohlcv_data['volume'].tail(24).sum() if len(ohlcv_data) >= 24 else ohlcv_data['volume'].sum()
                avg_volume = ohlcv_data['volume'].mean()
                st.metric("24h Volume", f"${volume_24h:,.0f}")
                st.metric("Avg Volume", f"${avg_volume:,.0f}")
            
            with col4:
                volatility = ohlcv_data['close'].pct_change().std() * 100
                st.metric("Volatility", f"{volatility:.2f}%")
                
                # Price range
                price_range = (ohlcv_data['high'].max() - ohlcv_data['low'].min()) / ohlcv_data['close'].mean() * 100
                st.metric("Price Range", f"{price_range:.1f}%")
            
            # Technical signals
            if indicators:
                st.markdown("### 🎯 Technical Signals")
                
                signals = []
                
                if 'RSI' in indicators:
                    rsi_current = TechnicalAnalysis.rsi(ohlcv_data['close']).iloc[-1]
                    if rsi_current > 70:
                        signals.append("🔴 RSI Overbought (>70)")
                    elif rsi_current < 30:
                        signals.append("🟢 RSI Oversold (<30)")
                    else:
                        signals.append(f"🟡 RSI Neutral ({rsi_current:.1f})")
                
                if 'SMA_20' in indicators:
                    sma_20_current = TechnicalAnalysis.sma(ohlcv_data['close'], 20).iloc[-1]
                    if current_price > sma_20_current:
                        signals.append("🟢 Price Above SMA 20 (Bullish)")
                    else:
                        signals.append("🔴 Price Below SMA 20 (Bearish)")
                
                if 'BB' in indicators:
                    bb_upper, bb_middle, bb_lower = TechnicalAnalysis.bollinger_bands(ohlcv_data['close'])
                    if current_price > bb_upper.iloc[-1]:
                        signals.append("🔴 Price Above Upper Bollinger Band")
                    elif current_price < bb_lower.iloc[-1]:
                        signals.append("🟢 Price Below Lower Bollinger Band")
                    else:
                        signals.append("🟡 Price Within Bollinger Bands")
                
                # Display signals
                for signal in signals:
                    st.write(f"• {signal}")
        
        # Trading insights
        with st.expander("💡 Trading Insights"):
            st.markdown(f"""
            **{pool} Analysis ({self.chart.timeframes[timeframe]['name']}):**
            
            **Price Action:**
            - Current trend: {'Bullish' if price_change_pct > 0 else 'Bearish' if price_change_pct < 0 else 'Neutral'}
            - Volatility level: {'High' if volatility > 5 else 'Medium' if volatility > 2 else 'Low'}
            - Volume profile: {'Above average' if volume_24h > avg_volume else 'Below average'}
            
            **Risk Assessment:**
            - Volatility: {volatility:.2f}% (Daily)
            - Price range: {price_range:.1f}% (Period)
            - Liquidity: {'High' if volume_24h > 100000 else 'Medium' if volume_24h > 10000 else 'Low'}
            
            **Key Levels:**
            - Support: ${low_24h:,.4f}
            - Resistance: ${high_24h:,.4f}
            - Current: ${current_price:,.4f}
            """)


# Example usage and testing
if __name__ == "__main__":
    print("📈 Testing TradingView interface...")
    
    # Test technical analysis functions
    ta = TechnicalAnalysis()
    
    # Generate test data
    test_prices = pd.Series(np.random.randn(100).cumsum() + 100)
    
    # Test indicators
    sma = ta.sma(test_prices, 20)
    ema = ta.ema(test_prices, 20)
    rsi = ta.rsi(test_prices)
    
    print(f"✅ SMA calculated: {len(sma)} points")
    print(f"✅ EMA calculated: {len(ema)} points")
    print(f"✅ RSI calculated: {len(rsi)} points")
    
    # Test chart creation
    chart = TradingViewChart()
    
    try:
        fig = chart.create_tradingview_chart('SAIL/USDC', '1d', ['SMA_20', 'RSI'])
        print("✅ TradingView chart created successfully")
    except Exception as e:
        print(f"❌ Chart creation failed: {e}")
    
    print("🎉 TradingView interface ready!")
