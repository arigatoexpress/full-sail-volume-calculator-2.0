"""
Multi-asset analyzer for comprehensive DeFi ecosystem analysis.
Supports Solana, Ethereum, Sui, and other blockchain ecosystems.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from trading_view import TechnicalAnalysis


class MultiAssetDataFetcher:
    """Fetch data from multiple blockchain ecosystems."""
    
    def __init__(self):
        """Initialize multi-asset data fetcher."""
        self.supported_assets = {
            # Layer 1 Blockchains
            'SOL': {'name': 'Solana', 'coingecko_id': 'solana', 'category': 'layer1'},
            'ETH': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'category': 'layer1'},
            'SUI': {'name': 'Sui', 'coingecko_id': 'sui', 'category': 'layer1'},
            'BTC': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'category': 'layer1'},
            'BNB': {'name': 'BNB', 'coingecko_id': 'binancecoin', 'category': 'layer1'},
            'AVAX': {'name': 'Avalanche', 'coingecko_id': 'avalanche-2', 'category': 'layer1'},
            'MATIC': {'name': 'Polygon', 'coingecko_id': 'matic-network', 'category': 'layer2'},
            'ADA': {'name': 'Cardano', 'coingecko_id': 'cardano', 'category': 'layer1'},
            
            # DeFi Tokens
            'UNI': {'name': 'Uniswap', 'coingecko_id': 'uniswap', 'category': 'defi'},
            'AAVE': {'name': 'Aave', 'coingecko_id': 'aave', 'category': 'defi'},
            'COMP': {'name': 'Compound', 'coingecko_id': 'compound-governance-token', 'category': 'defi'},
            'MKR': {'name': 'Maker', 'coingecko_id': 'maker', 'category': 'defi'},
            'CRV': {'name': 'Curve', 'coingecko_id': 'curve-dao-token', 'category': 'defi'},
            
            # Sui Ecosystem
            'SAIL': {'name': 'Full Sail', 'coingecko_id': 'sui', 'category': 'sui_ecosystem'},  # Placeholder
            'IKA': {'name': 'Ika Protocol', 'coingecko_id': 'sui', 'category': 'sui_ecosystem'},
            'DEEP': {'name': 'Deep Protocol', 'coingecko_id': 'sui', 'category': 'sui_ecosystem'},
            'WAL': {'name': 'Wal Protocol', 'coingecko_id': 'sui', 'category': 'sui_ecosystem'},
            
            # Stablecoins
            'USDC': {'name': 'USD Coin', 'coingecko_id': 'usd-coin', 'category': 'stablecoin'},
            'USDT': {'name': 'Tether', 'coingecko_id': 'tether', 'category': 'stablecoin'},
            'DAI': {'name': 'Dai', 'coingecko_id': 'dai', 'category': 'stablecoin'},
            'FRAX': {'name': 'Frax', 'coingecko_id': 'frax', 'category': 'stablecoin'}
        }
        
        # DEX ecosystem mapping
        self.dex_ecosystems = {
            'solana': {
                'name': 'Solana DEXs',
                'dexs': ['Raydium', 'Orca', 'Jupiter', 'Serum', 'Saber'],
                'top_pairs': ['SOL/USDC', 'RAY/SOL', 'ORCA/SOL', 'SAMO/SOL']
            },
            'ethereum': {
                'name': 'Ethereum DEXs', 
                'dexs': ['Uniswap V3', 'Uniswap V2', 'SushiSwap', 'Curve', 'Balancer'],
                'top_pairs': ['ETH/USDC', 'WBTC/ETH', 'UNI/ETH', 'LINK/ETH']
            },
            'sui': {
                'name': 'Sui DEXs',
                'dexs': ['Full Sail Finance', 'Cetus', 'Turbos', 'Aftermath'],
                'top_pairs': ['SUI/USDC', 'SAIL/USDC', 'IKA/SUI', 'DEEP/SUI']
            }
        }
    
    def fetch_asset_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch comprehensive asset data including price, volume, and market metrics.
        
        Args:
            symbol: Asset symbol (e.g., 'SOL', 'ETH', 'SUI')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV and additional metrics
        """
        if symbol not in self.supported_assets:
            return self._generate_synthetic_asset_data(symbol, days)
        
        asset_info = self.supported_assets[symbol]
        
        try:
            # In production, this would use real API calls
            # For now, generate realistic synthetic data
            return self._generate_realistic_asset_data(symbol, asset_info, days)
            
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            return self._generate_synthetic_asset_data(symbol, days)
    
    def _generate_realistic_asset_data(self, symbol: str, asset_info: Dict, days: int) -> pd.DataFrame:
        """Generate realistic asset data based on asset characteristics."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Asset-specific parameters
        asset_params = {
            'SOL': {'base_price': 140, 'volatility': 0.06, 'trend': 0.0002, 'volume_base': 2000000000},
            'ETH': {'base_price': 2800, 'volatility': 0.05, 'trend': 0.0001, 'volume_base': 8000000000},
            'SUI': {'base_price': 1.2, 'volatility': 0.08, 'trend': 0.0003, 'volume_base': 100000000},
            'BTC': {'base_price': 65000, 'volatility': 0.04, 'trend': 0.0001, 'volume_base': 15000000000},
            'UNI': {'base_price': 8.5, 'volatility': 0.07, 'trend': 0.0001, 'volume_base': 150000000},
            'AAVE': {'base_price': 95, 'volatility': 0.08, 'trend': 0.0002, 'volume_base': 80000000},
            'USDC': {'base_price': 1.0, 'volatility': 0.001, 'trend': 0, 'volume_base': 5000000000},
            'USDT': {'base_price': 1.0, 'volatility': 0.001, 'trend': 0, 'volume_base': 12000000000}
        }
        
        params = asset_params.get(symbol, {
            'base_price': 10, 'volatility': 0.08, 'trend': 0.0002, 'volume_base': 50000000
        })
        
        # Generate correlated price movements
        np.random.seed(hash(symbol) % 2**32)
        
        prices = []
        volumes = []
        current_price = params['base_price']
        
        for i, date in enumerate(dates):
            # Market cycle simulation
            cycle_position = (i / len(dates)) * 4 * np.pi
            market_cycle = np.sin(cycle_position) * 0.2 + 1.0
            
            # Weekly patterns
            day_of_week = date.weekday()
            weekly_factor = 1.0 + 0.1 * np.sin(day_of_week * 2 * np.pi / 7)
            
            # Random events
            event_probability = 0.02
            event_multiplier = np.random.choice([1.0, 1.5, 0.8], p=[0.96, 0.02, 0.02])
            
            # Price movement
            daily_return = (params['trend'] + 
                          np.random.normal(0, params['volatility']) * 
                          market_cycle * event_multiplier)
            
            current_price *= (1 + daily_return)
            prices.append(current_price)
            
            # Volume with realistic patterns
            volume = (params['volume_base'] * market_cycle * weekly_factor * 
                     event_multiplier * np.random.lognormal(0, 0.3))
            volumes.append(volume)
        
        # Generate OHLC from prices
        ohlc_data = []
        for i, (date, price, volume) in enumerate(zip(dates, prices, volumes)):
            # Create realistic OHLC
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # High and low with realistic spreads
            spread = abs(np.random.normal(0, params['volatility'] * 0.5))
            high_price = max(open_price, close_price) * (1 + spread)
            low_price = min(open_price, close_price) * (1 - spread)
            
            ohlc_data.append({
                'date': date,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'market_cap': close_price * np.random.uniform(100000000, 1000000000),
                'category': asset_info['category']
            })
        
        return pd.DataFrame(ohlc_data)
    
    def _generate_synthetic_asset_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate synthetic data for unknown assets."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        base_price = np.random.uniform(1, 100)
        
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.05)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(15, 1, len(dates)),
            'market_cap': [p * np.random.uniform(1e6, 1e9) for p in prices],
            'category': 'unknown'
        })
    
    def fetch_dex_ecosystem_data(self, ecosystem: str) -> Dict:
        """Fetch DEX data for an entire ecosystem."""
        if ecosystem not in self.dex_ecosystems:
            return {}
        
        ecosystem_info = self.dex_ecosystems[ecosystem]
        ecosystem_data = {}
        
        # Fetch data for top pairs in the ecosystem
        for pair in ecosystem_info['top_pairs']:
            # Generate realistic DEX pair data
            ecosystem_data[pair] = self._generate_dex_pair_data(pair, ecosystem)
        
        return ecosystem_data
    
    def _generate_dex_pair_data(self, pair: str, ecosystem: str) -> pd.DataFrame:
        """Generate realistic DEX pair data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=90),
            end=datetime.now(),
            freq='D'
        )
        
        # Ecosystem-specific volume patterns
        ecosystem_multipliers = {
            'solana': 1.5,    # High activity
            'ethereum': 2.0,  # Highest activity
            'sui': 0.8        # Growing ecosystem
        }
        
        base_multiplier = ecosystem_multipliers.get(ecosystem, 1.0)
        
        # Pair-specific volumes
        pair_volumes = {
            'SOL/USDC': 50000000, 'ETH/USDC': 100000000, 'SUI/USDC': 5000000,
            'RAY/SOL': 20000000, 'ORCA/SOL': 15000000, 'WBTC/ETH': 80000000,
            'UNI/ETH': 30000000, 'SAIL/USDC': 500000, 'IKA/SUI': 1000000
        }
        
        base_volume = pair_volumes.get(pair, 1000000) * base_multiplier
        
        records = []
        for date in dates:
            # Add realistic variation
            volume = base_volume * (1 + np.random.normal(0, 0.3))
            tvl = volume * np.random.uniform(5, 15)  # TVL typically 5-15x daily volume
            fees = volume * np.random.uniform(0.001, 0.005)  # 0.1-0.5% fees
            
            records.append({
                'date': date,
                'pair': pair,
                'ecosystem': ecosystem,
                'volume_24h': max(0, volume),
                'tvl': max(0, tvl),
                'fee_revenue': max(0, fees),
                'apr': (fees * 365 / tvl * 100) if tvl > 0 else 0
            })
        
        return pd.DataFrame(records)


class UniversalTechnicalAnalysis:
    """Universal technical analysis for any asset or trading pair."""
    
    def __init__(self):
        """Initialize universal technical analysis."""
        self.ta = TechnicalAnalysis()
        
        # Extended indicator library
        self.indicators = {
            # Trend Indicators
            'SMA_10': {'name': 'SMA 10', 'category': 'trend', 'params': {'window': 10}},
            'SMA_20': {'name': 'SMA 20', 'category': 'trend', 'params': {'window': 20}},
            'SMA_50': {'name': 'SMA 50', 'category': 'trend', 'params': {'window': 50}},
            'SMA_200': {'name': 'SMA 200', 'category': 'trend', 'params': {'window': 200}},
            'EMA_12': {'name': 'EMA 12', 'category': 'trend', 'params': {'window': 12}},
            'EMA_26': {'name': 'EMA 26', 'category': 'trend', 'params': {'window': 26}},
            'EMA_50': {'name': 'EMA 50', 'category': 'trend', 'params': {'window': 50}},
            
            # Momentum Indicators
            'RSI_14': {'name': 'RSI 14', 'category': 'momentum', 'params': {'window': 14}},
            'RSI_21': {'name': 'RSI 21', 'category': 'momentum', 'params': {'window': 21}},
            'MACD': {'name': 'MACD', 'category': 'momentum', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
            'STOCH': {'name': 'Stochastic', 'category': 'momentum', 'params': {'k': 14, 'd': 3}},
            
            # Volatility Indicators
            'BB_20': {'name': 'Bollinger Bands 20', 'category': 'volatility', 'params': {'window': 20, 'std': 2}},
            'ATR_14': {'name': 'ATR 14', 'category': 'volatility', 'params': {'window': 14}},
            
            # Volume Indicators
            'VOL_SMA': {'name': 'Volume SMA', 'category': 'volume', 'params': {'window': 20}},
            'OBV': {'name': 'On Balance Volume', 'category': 'volume', 'params': {}}
        }
    
    def calculate_all_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators."""
        if ohlcv_data.empty:
            return ohlcv_data
        
        enhanced_data = ohlcv_data.copy()
        
        # Trend indicators
        enhanced_data['sma_10'] = self.ta.sma(ohlcv_data['close'], 10)
        enhanced_data['sma_20'] = self.ta.sma(ohlcv_data['close'], 20)
        enhanced_data['sma_50'] = self.ta.sma(ohlcv_data['close'], 50)
        enhanced_data['sma_200'] = self.ta.sma(ohlcv_data['close'], 200)
        
        enhanced_data['ema_12'] = self.ta.ema(ohlcv_data['close'], 12)
        enhanced_data['ema_26'] = self.ta.ema(ohlcv_data['close'], 26)
        enhanced_data['ema_50'] = self.ta.ema(ohlcv_data['close'], 50)
        
        # Momentum indicators
        enhanced_data['rsi_14'] = self.ta.rsi(ohlcv_data['close'], 14)
        enhanced_data['rsi_21'] = self.ta.rsi(ohlcv_data['close'], 21)
        
        macd_line, signal_line, histogram = self.ta.macd(ohlcv_data['close'])
        enhanced_data['macd'] = macd_line
        enhanced_data['macd_signal'] = signal_line
        enhanced_data['macd_histogram'] = histogram
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(ohlcv_data['close'])
        enhanced_data['bb_upper'] = bb_upper
        enhanced_data['bb_middle'] = bb_middle
        enhanced_data['bb_lower'] = bb_lower
        
        if 'high' in ohlcv_data.columns and 'low' in ohlcv_data.columns:
            enhanced_data['atr'] = self.ta.atr(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            
            # Stochastic
            stoch_k, stoch_d = self.ta.stochastic(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])
            enhanced_data['stoch_k'] = stoch_k
            enhanced_data['stoch_d'] = stoch_d
        
        # Volume indicators
        if 'volume' in ohlcv_data.columns:
            enhanced_data['volume_sma'] = self.ta.volume_sma(ohlcv_data['volume'])
            enhanced_data['obv'] = self._calculate_obv(ohlcv_data['close'], ohlcv_data['volume'])
        
        return enhanced_data
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = [0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=close.index)
    
    def generate_trading_signals(self, enhanced_data: pd.DataFrame) -> Dict:
        """Generate comprehensive trading signals."""
        if enhanced_data.empty or len(enhanced_data) < 50:
            return {'signals': [], 'score': 0, 'recommendation': 'insufficient_data'}
        
        signals = []
        score = 0
        
        # RSI signals
        current_rsi = enhanced_data['rsi_14'].iloc[-1]
        if current_rsi > 70:
            signals.append({'type': 'sell', 'indicator': 'RSI', 'strength': 'strong', 'reason': 'Overbought (>70)'})
            score -= 2
        elif current_rsi < 30:
            signals.append({'type': 'buy', 'indicator': 'RSI', 'strength': 'strong', 'reason': 'Oversold (<30)'})
            score += 2
        
        # Moving average signals
        current_price = enhanced_data['close'].iloc[-1]
        sma_20 = enhanced_data['sma_20'].iloc[-1]
        sma_50 = enhanced_data['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            signals.append({'type': 'buy', 'indicator': 'MA', 'strength': 'medium', 'reason': 'Price above rising MAs'})
            score += 1
        elif current_price < sma_20 < sma_50:
            signals.append({'type': 'sell', 'indicator': 'MA', 'strength': 'medium', 'reason': 'Price below falling MAs'})
            score -= 1
        
        # MACD signals
        macd_current = enhanced_data['macd'].iloc[-1]
        macd_signal = enhanced_data['macd_signal'].iloc[-1]
        
        if macd_current > macd_signal and enhanced_data['macd'].iloc[-2] <= enhanced_data['macd_signal'].iloc[-2]:
            signals.append({'type': 'buy', 'indicator': 'MACD', 'strength': 'medium', 'reason': 'MACD bullish crossover'})
            score += 1
        elif macd_current < macd_signal and enhanced_data['macd'].iloc[-2] >= enhanced_data['macd_signal'].iloc[-2]:
            signals.append({'type': 'sell', 'indicator': 'MACD', 'strength': 'medium', 'reason': 'MACD bearish crossover'})
            score -= 1
        
        # Bollinger Bands signals
        bb_upper = enhanced_data['bb_upper'].iloc[-1]
        bb_lower = enhanced_data['bb_lower'].iloc[-1]
        
        if current_price > bb_upper:
            signals.append({'type': 'sell', 'indicator': 'BB', 'strength': 'medium', 'reason': 'Price above upper BB'})
            score -= 1
        elif current_price < bb_lower:
            signals.append({'type': 'buy', 'indicator': 'BB', 'strength': 'medium', 'reason': 'Price below lower BB'})
            score += 1
        
        # Overall recommendation
        if score >= 3:
            recommendation = 'strong_buy'
        elif score >= 1:
            recommendation = 'buy'
        elif score <= -3:
            recommendation = 'strong_sell'
        elif score <= -1:
            recommendation = 'sell'
        else:
            recommendation = 'hold'
        
        return {
            'signals': signals,
            'score': score,
            'recommendation': recommendation,
            'signal_count': len(signals)
        }


class UniversalChartInterface:
    """Universal charting interface for any asset or DEX pair."""
    
    def __init__(self):
        """Initialize universal chart interface."""
        self.fetcher = MultiAssetDataFetcher()
        self.ta = UniversalTechnicalAnalysis()
        
        # Chart themes
        self.themes = {
            'dark_professional': {
                'bg_color': '#131722',
                'grid_color': '#2A2E39',
                'text_color': '#D1D4DC',
                'bull_color': '#26A69A',
                'bear_color': '#EF5350'
            },
            'light_professional': {
                'bg_color': '#FFFFFF',
                'grid_color': '#E1E3E6',
                'text_color': '#2E3440',
                'bull_color': '#089981',
                'bear_color': '#F23645'
            },
            'neon': {
                'bg_color': '#000000',
                'grid_color': '#1a1a1a',
                'text_color': '#00FF00',
                'bull_color': '#00FF00',
                'bear_color': '#FF0040'
            }
        }
    
    def create_universal_chart(self, symbol: str, timeframe: str = '1d', 
                             indicators: List[str] = None, 
                             comparison_assets: List[str] = None,
                             theme: str = 'dark_professional') -> go.Figure:
        """
        Create universal chart for any asset with full technical analysis.
        
        Args:
            symbol: Primary asset symbol
            timeframe: Chart timeframe
            indicators: List of technical indicators to display
            comparison_assets: Additional assets to compare
            theme: Chart theme
            
        Returns:
            Comprehensive multi-asset chart
        """
        if indicators is None:
            indicators = ['SMA_20', 'RSI_14', 'MACD', 'BB_20']
        
        # Fetch primary asset data
        primary_data = self.fetcher.fetch_asset_data(symbol, 365)
        
        if primary_data.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for {symbol}")
            return fig
        
        # Calculate technical indicators
        enhanced_data = self.ta.calculate_all_indicators(primary_data)
        
        # Determine subplot structure
        subplot_count = 1  # Main price chart
        if 'RSI_14' in indicators or 'RSI_21' in indicators:
            subplot_count += 1
        if 'MACD' in indicators:
            subplot_count += 1
        if 'VOL_SMA' in indicators or comparison_assets:
            subplot_count += 1
        
        # Create subplots
        subplot_titles = [f"{symbol} - {timeframe.upper()}"]
        row_heights = [0.6]
        
        if 'RSI_14' in indicators or 'RSI_21' in indicators:
            subplot_titles.append("RSI")
            row_heights.append(0.15)
        if 'MACD' in indicators:
            subplot_titles.append("MACD")
            row_heights.append(0.15)
        if 'VOL_SMA' in indicators or comparison_assets:
            subplot_titles.append("Volume / Comparison")
            row_heights.append(0.1)
        
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        theme_config = self.themes[theme]
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=enhanced_data['date'],
                open=enhanced_data['open'],
                high=enhanced_data['high'],
                low=enhanced_data['low'],
                close=enhanced_data['close'],
                name=symbol,
                increasing_line_color=theme_config['bull_color'],
                decreasing_line_color=theme_config['bear_color']
            ),
            row=1, col=1
        )
        
        # Add trend indicators to main chart
        current_row = 1
        
        if 'SMA_10' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['sma_10'],
                    mode='lines',
                    name='SMA 10',
                    line=dict(color='#FF6B35', width=1),
                    hovertemplate="SMA 10: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        if 'SMA_20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#00D4FF', width=1),
                    hovertemplate="SMA 20: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#00E676', width=2),
                    hovertemplate="SMA 50: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        if 'SMA_200' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['sma_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(color='#BB86FC', width=2),
                    hovertemplate="SMA 200: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB_20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(33, 150, 243, 0.5)', width=1),
                    hovertemplate="BB Upper: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(33, 150, 243, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(33, 150, 243, 0.1)',
                    hovertemplate="BB Lower: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # RSI subplot
        if 'RSI_14' in indicators or 'RSI_21' in indicators:
            current_row += 1
            
            if 'RSI_14' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=enhanced_data['date'],
                        y=enhanced_data['rsi_14'],
                        mode='lines',
                        name='RSI 14',
                        line=dict(color='#BB86FC', width=2),
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
        
        # MACD subplot
        if 'MACD' in indicators:
            current_row += 1
            
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#00D4FF', width=2),
                    hovertemplate="MACD: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=enhanced_data['date'],
                    y=enhanced_data['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#FF6B35', width=2),
                    hovertemplate="Signal: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
            
            # MACD histogram
            histogram_colors = ['green' if h >= 0 else 'red' for h in enhanced_data['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=enhanced_data['date'],
                    y=enhanced_data['macd_histogram'],
                    name='MACD Histogram',
                    marker_color=histogram_colors,
                    opacity=0.6,
                    hovertemplate="Histogram: %{y:.4f}<extra></extra>"
                ),
                row=current_row, col=1
            )
        
        # Volume/Comparison subplot
        if 'VOL_SMA' in indicators or comparison_assets:
            current_row += 1
            
            if 'volume' in enhanced_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=enhanced_data['date'],
                        y=enhanced_data['volume'],
                        name='Volume',
                        marker_color='rgba(0, 212, 255, 0.6)',
                        hovertemplate="Volume: %{y:,.0f}<extra></extra>"
                    ),
                    row=current_row, col=1
                )
                
                if 'volume_sma' in enhanced_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=enhanced_data['date'],
                            y=enhanced_data['volume_sma'],
                            mode='lines',
                            name='Volume SMA',
                            line=dict(color='#FFD600', width=2),
                            hovertemplate="Vol SMA: %{y:,.0f}<extra></extra>"
                        ),
                        row=current_row, col=1
                    )
            
            # Add comparison assets
            if comparison_assets:
                for comp_asset in comparison_assets:
                    comp_data = self.fetcher.fetch_asset_data(comp_asset, 365)
                    if not comp_data.empty:
                        # Normalize prices for comparison
                        comp_normalized = comp_data['close'] / comp_data['close'].iloc[0] * enhanced_data['close'].iloc[0]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=comp_data['date'],
                                y=comp_normalized,
                                mode='lines',
                                name=f"{comp_asset} (Normalized)",
                                line=dict(width=2, dash='dash'),
                                hovertemplate=f"{comp_asset}: %{{y:.2f}}<extra></extra>"
                            ),
                            row=1, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Universal Analysis: {symbol}" + (f" vs {', '.join(comparison_assets)}" if comparison_assets else ""),
                x=0.5,
                font=dict(size=18, color=theme_config['text_color'])
            ),
            xaxis_rangeslider_visible=False,
            height=600 + (subplot_count - 1) * 150,
            plot_bgcolor=theme_config['bg_color'],
            paper_bgcolor=theme_config['bg_color'],
            font=dict(color=theme_config['text_color']),
            hovermode='x unified'
        )
        
        # Update all axes
        for i in range(1, subplot_count + 1):
            fig.update_xaxes(
                gridcolor=theme_config['grid_color'],
                showgrid=True,
                color=theme_config['text_color'],
                row=i, col=1
            )
            fig.update_yaxes(
                gridcolor=theme_config['grid_color'],
                showgrid=True,
                color=theme_config['text_color'],
                row=i, col=1
            )
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("🌐 Testing Universal Multi-Asset Analysis...")
    
    # Test multi-asset fetcher
    fetcher = MultiAssetDataFetcher()
    
    # Test major assets
    test_assets = ['SOL', 'ETH', 'SUI', 'BTC']
    
    for asset in test_assets:
        data = fetcher.fetch_asset_data(asset, 30)
        print(f"✅ {asset} data: {len(data)} days")
    
    # Test DEX ecosystem data
    sui_dex_data = fetcher.fetch_dex_ecosystem_data('sui')
    print(f"✅ Sui DEX data: {len(sui_dex_data)} pairs")
    
    # Test universal technical analysis
    ta = UniversalTechnicalAnalysis()
    
    sol_data = fetcher.fetch_asset_data('SOL', 100)
    enhanced_sol = ta.calculate_all_indicators(sol_data)
    print(f"✅ SOL technical analysis: {len(enhanced_sol.columns)} indicators")
    
    # Test trading signals
    signals = ta.generate_trading_signals(enhanced_sol)
    print(f"✅ Trading signals: {signals['signal_count']} signals, recommendation: {signals['recommendation']}")
    
    # Test universal chart
    chart_interface = UniversalChartInterface()
    fig = chart_interface.create_universal_chart('ETH', '1d', ['SMA_20', 'RSI_14'], ['BTC', 'SOL'])
    print("✅ Universal chart created with comparison assets")
    
    print("🎉 Universal multi-asset analysis ready!")
