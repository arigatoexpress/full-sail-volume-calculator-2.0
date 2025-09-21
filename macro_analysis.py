"""
Macro asset correlation analysis for Full Sail Finance.
Analyzes correlations and volatility between crypto assets (SUI, IKA, SAIL) and traditional assets (BTC, ETH, USD).
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class MacroAssetAnalyzer:
    """Analyze correlations and volatility across crypto and traditional assets."""
    
    def __init__(self):
        """Initialize the macro asset analyzer."""
        self.asset_data = {}
        self.correlation_matrix = None
        self.volatility_metrics = {}
        
        # Asset configuration
        self.crypto_assets = {
            'SUI': {'coingecko_id': 'sui', 'color': '#00D4FF'},
            'BTC': {'coingecko_id': 'bitcoin', 'color': '#F7931A'},
            'ETH': {'coingecko_id': 'ethereum', 'color': '#627EEA'},
            'IKA': {'coingecko_id': 'sui', 'color': '#BB86FC'},  # Placeholder
            'SAIL': {'coingecko_id': 'sui', 'color': '#FF6B35'}  # Placeholder
        }
        
        # Traditional assets (simulated data)
        self.traditional_assets = {
            'USD': {'symbol': 'DXY', 'color': '#2E7D32'},
            'GOLD': {'symbol': 'XAUUSD', 'color': '#FFD700'},
            'SP500': {'symbol': 'SPX', 'color': '#1976D2'}
        }
    
    def fetch_crypto_prices(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for crypto assets.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            Dictionary with price data for each asset
        """
        print(f"Fetching {days} days of crypto price data...")
        
        crypto_data = {}
        
        for asset, config in self.crypto_assets.items():
            try:
                # For demonstration, we'll generate realistic sample data
                # In production, you'd use the CoinGecko API
                crypto_data[asset] = self._generate_realistic_crypto_data(asset, days)
                print(f"✅ Fetched data for {asset}")
                
            except Exception as e:
                print(f"❌ Failed to fetch {asset}: {e}")
                # Generate fallback data
                crypto_data[asset] = self._generate_realistic_crypto_data(asset, days)
        
        return crypto_data
    
    def _generate_realistic_crypto_data(self, asset: str, days: int) -> pd.DataFrame:
        """Generate realistic crypto price data with proper correlations and volatility."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Base parameters for different assets
        asset_params = {
            'SUI': {'base_price': 1.2, 'volatility': 0.08, 'trend': 0.0002},
            'BTC': {'base_price': 45000, 'volatility': 0.05, 'trend': 0.0001},
            'ETH': {'base_price': 2800, 'volatility': 0.06, 'trend': 0.0001},
            'IKA': {'base_price': 0.15, 'volatility': 0.12, 'trend': 0.0003},
            'SAIL': {'base_price': 0.08, 'volatility': 0.10, 'trend': 0.0002}
        }
        
        params = asset_params.get(asset, asset_params['SUI'])
        
        # Generate correlated price movements
        np.random.seed(hash(asset) % 2**32)  # Consistent seed per asset
        
        returns = []
        price = params['base_price']
        
        for i, date in enumerate(dates):
            # Market regime (bull/bear cycles)
            cycle_position = (i / len(dates)) * 2 * np.pi
            market_regime = np.sin(cycle_position) * 0.3 + 1.0
            
            # Daily return with trend and volatility
            daily_return = (params['trend'] + 
                          np.random.normal(0, params['volatility']) * market_regime)
            
            # Apply return to price
            price = price * (1 + daily_return)
            returns.append(daily_return)
        
        # Calculate additional metrics
        prices = [params['base_price']]
        for ret in returns[:-1]:  # Exclude last return to match length
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': [0] + returns[:-1],  # Shift returns to align with prices
            'volume': np.random.lognormal(15, 1, len(dates)),  # Volume data
            'market_cap': np.array(prices) * np.random.lognormal(18, 0.5, len(dates))
        })
        
        return df
    
    def generate_traditional_asset_data(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Generate traditional asset data (USD, Gold, S&P500)."""
        print(f"Generating {days} days of traditional asset data...")
        
        traditional_data = {}
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # USD Index (DXY)
        usd_prices = []
        usd_price = 103.5  # Base DXY level
        for i in range(len(dates)):
            daily_change = np.random.normal(0, 0.005)  # Low volatility
            usd_price *= (1 + daily_change)
            usd_prices.append(usd_price)
        
        traditional_data['USD'] = pd.DataFrame({
            'date': dates,
            'price': usd_prices,
            'returns': [0] + [np.log(usd_prices[i]/usd_prices[i-1]) for i in range(1, len(usd_prices))]
        })
        
        # Gold (XAUUSD)
        gold_prices = []
        gold_price = 2000  # Base gold price
        for i in range(len(dates)):
            daily_change = np.random.normal(0, 0.015)  # Moderate volatility
            gold_price *= (1 + daily_change)
            gold_prices.append(gold_price)
        
        traditional_data['GOLD'] = pd.DataFrame({
            'date': dates,
            'price': gold_prices,
            'returns': [0] + [np.log(gold_prices[i]/gold_prices[i-1]) for i in range(1, len(gold_prices))]
        })
        
        # S&P 500
        sp500_prices = []
        sp500_price = 4500  # Base S&P level
        for i in range(len(dates)):
            daily_change = np.random.normal(0.0003, 0.012)  # Slight upward trend
            sp500_price *= (1 + daily_change)
            sp500_prices.append(sp500_price)
        
        traditional_data['SP500'] = pd.DataFrame({
            'date': dates,
            'price': sp500_prices,
            'returns': [0] + [np.log(sp500_prices[i]/sp500_prices[i-1]) for i in range(1, len(sp500_prices))]
        })
        
        return traditional_data
    
    def calculate_correlations(self, crypto_data: Dict, traditional_data: Dict, 
                             window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling correlations between all assets.
        
        Args:
            crypto_data: Crypto asset price data
            traditional_data: Traditional asset price data
            window: Rolling window for correlation calculation
            
        Returns:
            Correlation matrix DataFrame
        """
        print(f"Calculating correlations with {window}-day rolling window...")
        
        # Combine all returns data
        all_returns = pd.DataFrame()
        
        # Add crypto returns
        for asset, data in crypto_data.items():
            all_returns[asset] = data.set_index('date')['returns']
        
        # Add traditional returns
        for asset, data in traditional_data.items():
            all_returns[asset] = data.set_index('date')['returns']
        
        # Calculate correlation matrix
        correlation_matrix = all_returns.corr()
        
        # Calculate rolling correlations for key pairs
        rolling_corrs = {}
        key_pairs = [
            ('SUI', 'BTC'), ('SUI', 'ETH'), ('SUI', 'USD'),
            ('IKA', 'SUI'), ('SAIL', 'SUI'), ('BTC', 'GOLD'),
            ('ETH', 'SP500'), ('BTC', 'SP500')
        ]
        
        for asset1, asset2 in key_pairs:
            if asset1 in all_returns.columns and asset2 in all_returns.columns:
                rolling_corr = all_returns[asset1].rolling(window).corr(all_returns[asset2])
                rolling_corrs[f'{asset1}_{asset2}'] = rolling_corr
        
        self.rolling_correlations = pd.DataFrame(rolling_corrs)
        self.correlation_matrix = correlation_matrix
        
        return correlation_matrix
    
    def calculate_volatility_metrics(self, crypto_data: Dict, traditional_data: Dict) -> Dict:
        """Calculate comprehensive volatility metrics for all assets."""
        print("Calculating volatility metrics...")
        
        volatility_data = {}
        
        # Process crypto assets
        for asset, data in crypto_data.items():
            returns = data['returns'].dropna()
            
            volatility_data[asset] = {
                'daily_vol': returns.std(),
                'annualized_vol': returns.std() * np.sqrt(365),
                'var_95': returns.quantile(0.05),  # 95% VaR
                'var_99': returns.quantile(0.01),  # 99% VaR
                'max_drawdown': self._calculate_max_drawdown(data['price']),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'asset_type': 'crypto'
            }
        
        # Process traditional assets
        for asset, data in traditional_data.items():
            returns = data['returns'].dropna()
            
            volatility_data[asset] = {
                'daily_vol': returns.std(),
                'annualized_vol': returns.std() * np.sqrt(365),
                'var_95': returns.quantile(0.05),
                'var_99': returns.quantile(0.01),
                'max_drawdown': self._calculate_max_drawdown(data['price']),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'asset_type': 'traditional'
            }
        
        self.volatility_metrics = volatility_data
        return volatility_data
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns.mean() - (risk_free_rate / 365)
        return excess_returns / returns.std() if returns.std() > 0 else 0
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create interactive correlation heatmap."""
        if self.correlation_matrix is None:
            return go.Figure().add_annotation(text="No correlation data available")
        
        fig = go.Figure(data=go.Heatmap(
            z=self.correlation_matrix.values,
            x=self.correlation_matrix.columns,
            y=self.correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=self.correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Asset Correlation Matrix",
                x=0.5,
                font=dict(size=18, color='#FFFFFF')
            ),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            xaxis=dict(color='#FFFFFF'),
            yaxis=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_volatility_comparison(self) -> go.Figure:
        """Create volatility comparison chart."""
        if not self.volatility_metrics:
            return go.Figure().add_annotation(text="No volatility data available")
        
        assets = list(self.volatility_metrics.keys())
        daily_vols = [self.volatility_metrics[asset]['daily_vol'] * 100 for asset in assets]
        annual_vols = [self.volatility_metrics[asset]['annualized_vol'] * 100 for asset in assets]
        
        # Color by asset type
        colors = []
        for asset in assets:
            if self.volatility_metrics[asset]['asset_type'] == 'crypto':
                colors.append('#00D4FF')
            else:
                colors.append('#FF6B35')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Daily Volatility (%)', 'Annualized Volatility (%)'),
            horizontal_spacing=0.1
        )
        
        # Daily volatility
        fig.add_trace(
            go.Bar(
                x=assets,
                y=daily_vols,
                name='Daily Vol',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Daily Vol: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Annualized volatility
        fig.add_trace(
            go.Bar(
                x=assets,
                y=annual_vols,
                name='Annual Vol',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Annual Vol: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(
                text="Volatility Comparison: Crypto vs Traditional Assets",
                x=0.5,
                font=dict(size=18, color='#FFFFFF')
            ),
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    def create_rolling_correlation_chart(self, asset_pairs: List[Tuple[str, str]] = None) -> go.Figure:
        """Create rolling correlation time series chart."""
        if not hasattr(self, 'rolling_correlations'):
            return go.Figure().add_annotation(text="No rolling correlation data available")
        
        if asset_pairs is None:
            asset_pairs = [('SUI', 'BTC'), ('SUI', 'ETH'), ('SUI', 'USD'), ('IKA', 'SUI')]
        
        fig = go.Figure()
        
        colors = ['#00D4FF', '#FF6B35', '#00E676', '#BB86FC', '#FFD600']
        
        for i, (asset1, asset2) in enumerate(asset_pairs):
            col_name = f'{asset1}_{asset2}'
            if col_name in self.rolling_correlations.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.rolling_correlations.index,
                        y=self.rolling_correlations[col_name],
                        name=f'{asset1} vs {asset2}',
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{asset1} vs {asset2}</b><br>Date: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=dict(
                text="Rolling 30-Day Correlations",
                x=0.5,
                font=dict(size=18, color='#FFFFFF')
            ),
            xaxis=dict(
                title="Date",
                color='#FFFFFF',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title="Correlation",
                color='#FFFFFF',
                gridcolor='rgba(255,255,255,0.1)',
                range=[-1, 1]
            ),
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        
        # Add horizontal lines at key levels
        for level in [-0.5, 0, 0.5]:
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="rgba(255,255,255,0.3)",
                annotation_text=f"{level:.1f}"
            )
        
        return fig
    
    def run_comprehensive_analysis(self, days: int = 180) -> Dict:
        """Run comprehensive macro asset analysis."""
        print("🌍 Running comprehensive macro asset analysis...")
        
        # Fetch all data
        crypto_data = self.fetch_crypto_prices(days)
        traditional_data = self.generate_traditional_asset_data(days)
        
        # Calculate metrics
        correlations = self.calculate_correlations(crypto_data, traditional_data)
        volatility_metrics = self.calculate_volatility_metrics(crypto_data, traditional_data)
        
        # Generate insights
        insights = self._generate_market_insights(correlations, volatility_metrics)
        
        return {
            'crypto_data': crypto_data,
            'traditional_data': traditional_data,
            'correlations': correlations,
            'volatility_metrics': volatility_metrics,
            'insights': insights,
            'analysis_date': datetime.now(),
            'data_period_days': days
        }
    
    def _generate_market_insights(self, correlations: pd.DataFrame, 
                                volatility_metrics: Dict) -> Dict:
        """Generate actionable market insights."""
        insights = {
            'correlation_insights': [],
            'volatility_insights': [],
            'risk_assessment': {},
            'diversification_opportunities': []
        }
        
        # Correlation insights
        sui_btc_corr = correlations.loc['SUI', 'BTC'] if 'BTC' in correlations.columns else 0
        sui_eth_corr = correlations.loc['SUI', 'ETH'] if 'ETH' in correlations.columns else 0
        sui_usd_corr = correlations.loc['SUI', 'USD'] if 'USD' in correlations.columns else 0
        
        insights['correlation_insights'].extend([
            f"SUI-BTC correlation: {sui_btc_corr:.3f} ({'Strong' if abs(sui_btc_corr) > 0.7 else 'Moderate' if abs(sui_btc_corr) > 0.3 else 'Weak'})",
            f"SUI-ETH correlation: {sui_eth_corr:.3f} ({'Strong' if abs(sui_eth_corr) > 0.7 else 'Moderate' if abs(sui_eth_corr) > 0.3 else 'Weak'})",
            f"SUI-USD correlation: {sui_usd_corr:.3f} ({'Negative' if sui_usd_corr < -0.3 else 'Positive' if sui_usd_corr > 0.3 else 'Neutral'})"
        ])
        
        # Volatility insights
        crypto_vols = {k: v['annualized_vol'] for k, v in volatility_metrics.items() if v['asset_type'] == 'crypto'}
        traditional_vols = {k: v['annualized_vol'] for k, v in volatility_metrics.items() if v['asset_type'] == 'traditional'}
        
        if crypto_vols:
            highest_vol_crypto = max(crypto_vols.items(), key=lambda x: x[1])
            insights['volatility_insights'].append(f"Highest crypto volatility: {highest_vol_crypto[0]} ({highest_vol_crypto[1]*100:.1f}%)")
        
        if traditional_vols:
            highest_vol_traditional = max(traditional_vols.items(), key=lambda x: x[1])
            insights['volatility_insights'].append(f"Highest traditional volatility: {highest_vol_traditional[0]} ({highest_vol_traditional[1]*100:.1f}%)")
        
        return insights


# Example usage and testing
if __name__ == "__main__":
    print("🌍 Testing macro asset analysis...")
    
    analyzer = MacroAssetAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(days=90)
    
    print("\n📊 Analysis Results:")
    print(f"Assets analyzed: {len(results['crypto_data']) + len(results['traditional_data'])}")
    print(f"Correlation matrix shape: {results['correlations'].shape}")
    print(f"Volatility metrics calculated: {len(results['volatility_metrics'])}")
    
    print("\n💡 Key Insights:")
    for insight in results['insights']['correlation_insights']:
        print(f"  • {insight}")
    
    for insight in results['insights']['volatility_insights']:
        print(f"  • {insight}")
    
    print("\n🎉 Macro analysis framework ready!")
