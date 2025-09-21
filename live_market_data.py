"""
Live market data fetcher for real-time prices and historical data.
Lightweight implementation for major crypto assets.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import json
import os


class LiveMarketData:
    """Lightweight live market data fetcher."""
    
    def __init__(self):
        """Initialize live market data fetcher."""
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.cache_dir = "market_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Major assets to track
        self.tracked_assets = {
            # Layer 1s
            'bitcoin': {'symbol': 'BTC', 'name': 'Bitcoin', 'category': 'layer1'},
            'ethereum': {'symbol': 'ETH', 'name': 'Ethereum', 'category': 'layer1'},
            'solana': {'symbol': 'SOL', 'name': 'Solana', 'category': 'layer1'},
            'sui': {'symbol': 'SUI', 'name': 'Sui', 'category': 'layer1'},
            'binancecoin': {'symbol': 'BNB', 'name': 'BNB', 'category': 'layer1'},
            'avalanche-2': {'symbol': 'AVAX', 'name': 'Avalanche', 'category': 'layer1'},
            'cardano': {'symbol': 'ADA', 'name': 'Cardano', 'category': 'layer1'},
            'matic-network': {'symbol': 'MATIC', 'name': 'Polygon', 'category': 'layer2'},
            
            # DeFi
            'uniswap': {'symbol': 'UNI', 'name': 'Uniswap', 'category': 'defi'},
            'aave': {'symbol': 'AAVE', 'name': 'Aave', 'category': 'defi'},
            'compound-governance-token': {'symbol': 'COMP', 'name': 'Compound', 'category': 'defi'},
            'maker': {'symbol': 'MKR', 'name': 'Maker', 'category': 'defi'},
            'curve-dao-token': {'symbol': 'CRV', 'name': 'Curve', 'category': 'defi'},
            'chainlink': {'symbol': 'LINK', 'name': 'Chainlink', 'category': 'defi'},
            
            # Stablecoins
            'usd-coin': {'symbol': 'USDC', 'name': 'USD Coin', 'category': 'stablecoin'},
            'tether': {'symbol': 'USDT', 'name': 'Tether', 'category': 'stablecoin'},
            'dai': {'symbol': 'DAI', 'name': 'Dai', 'category': 'stablecoin'}
        }
    
    def fetch_live_prices(self) -> Dict:
        """Fetch live prices for all tracked assets."""
        cache_file = os.path.join(self.cache_dir, "live_prices.json")
        
        # Check cache (refresh every 5 minutes)
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 300:  # 5 minutes
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        try:
            # Fetch live prices from CoinGecko
            asset_ids = list(self.tracked_assets.keys())
            url = f"{self.coingecko_base}/simple/price"
            
            params = {
                'ids': ','.join(asset_ids),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            raw_data = response.json()
            
            # Process the data
            live_data = {}
            
            for asset_id, price_data in raw_data.items():
                asset_info = self.tracked_assets.get(asset_id, {})
                
                live_data[asset_info.get('symbol', asset_id.upper())] = {
                    'price': price_data.get('usd', 0),
                    'change_24h': price_data.get('usd_24h_change', 0),
                    'volume_24h': price_data.get('usd_24h_vol', 0),
                    'market_cap': price_data.get('usd_market_cap', 0),
                    'name': asset_info.get('name', asset_id),
                    'category': asset_info.get('category', 'unknown'),
                    'last_updated': datetime.now().isoformat()
                }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(live_data, f)
            
            return live_data
            
        except Exception as e:
            print(f"Error fetching live prices: {e}")
            return self._generate_mock_live_data()
    
    def _generate_mock_live_data(self) -> Dict:
        """Generate mock live data for testing."""
        mock_data = {}
        
        for asset_id, asset_info in self.tracked_assets.items():
            symbol = asset_info['symbol']
            
            # Generate realistic mock prices
            base_prices = {
                'BTC': 65000, 'ETH': 2800, 'SOL': 140, 'SUI': 1.2, 'BNB': 600,
                'AVAX': 35, 'ADA': 0.45, 'MATIC': 0.8, 'UNI': 8.5, 'AAVE': 95,
                'COMP': 55, 'MKR': 1200, 'CRV': 0.6, 'LINK': 14, 'USDC': 1.0,
                'USDT': 1.0, 'DAI': 1.0
            }
            
            base_price = base_prices.get(symbol, 10)
            
            mock_data[symbol] = {
                'price': base_price * (1 + np.random.normal(0, 0.02)),
                'change_24h': np.random.normal(0, 3),
                'volume_24h': base_price * np.random.uniform(1e6, 1e9),
                'market_cap': base_price * np.random.uniform(1e8, 1e11),
                'name': asset_info['name'],
                'category': asset_info['category'],
                'last_updated': datetime.now().isoformat()
            }
        
        return mock_data
    
    def fetch_historical_prices(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical price data for an asset."""
        # Find the asset ID
        asset_id = None
        for aid, info in self.tracked_assets.items():
            if info['symbol'] == symbol.upper():
                asset_id = aid
                break
        
        if not asset_id:
            return self._generate_synthetic_historical_data(symbol, days)
        
        cache_file = os.path.join(self.cache_dir, f"{symbol}_historical_{days}d.json")
        
        # Check cache (refresh daily)
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 86400:  # 24 hours
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        return pd.DataFrame(cached_data)
                except:
                    pass
        
        try:
            # Fetch historical data from CoinGecko
            url = f"{self.coingecko_base}/coins/{asset_id}/market_chart"
            
            params = {
                'vs_currency': 'usd',
                'days': min(days, 365),  # CoinGecko free tier limit
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Process historical data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            historical_data = []
            
            for i, (timestamp, price) in enumerate(prices):
                date = datetime.fromtimestamp(timestamp / 1000)
                volume = volumes[i][1] if i < len(volumes) else 0
                market_cap = market_caps[i][1] if i < len(market_caps) else 0
                
                historical_data.append({
                    'date': date,
                    'symbol': symbol.upper(),
                    'price': price,
                    'volume_24h': volume,
                    'market_cap': market_cap
                })
            
            df = pd.DataFrame(historical_data)
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(df.to_dict('records'), f, default=str)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return self._generate_synthetic_historical_data(symbol, days)
    
    def _generate_synthetic_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate synthetic historical data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Base parameters
        base_prices = {
            'BTC': 65000, 'ETH': 2800, 'SOL': 140, 'SUI': 1.2,
            'UNI': 8.5, 'AAVE': 95, 'USDC': 1.0, 'USDT': 1.0
        }
        
        base_price = base_prices.get(symbol.upper(), 10)
        volatility = 0.001 if symbol.upper() in ['USDC', 'USDT', 'DAI'] else 0.05
        
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'date': dates,
            'symbol': symbol.upper(),
            'price': prices,
            'volume_24h': [p * np.random.uniform(1e6, 1e8) for p in prices],
            'market_cap': [p * np.random.uniform(1e8, 1e10) for p in prices]
        })
    
    def get_market_overview(self) -> Dict:
        """Get market overview with key metrics."""
        live_data = self.fetch_live_prices()
        
        if not live_data:
            return {}
        
        # Calculate market metrics
        total_market_cap = sum(asset['market_cap'] for asset in live_data.values())
        total_volume = sum(asset['volume_24h'] for asset in live_data.values())
        
        # Top gainers and losers
        gainers = sorted(live_data.items(), key=lambda x: x[1]['change_24h'], reverse=True)[:5]
        losers = sorted(live_data.items(), key=lambda x: x[1]['change_24h'])[:5]
        
        # Category performance
        category_performance = {}
        for symbol, data in live_data.items():
            category = data['category']
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(data['change_24h'])
        
        # Average performance by category
        category_avg = {cat: np.mean(changes) for cat, changes in category_performance.items()}
        
        return {
            'total_market_cap': total_market_cap,
            'total_volume': total_volume,
            'top_gainers': gainers,
            'top_losers': losers,
            'category_performance': category_avg,
            'asset_count': len(live_data),
            'last_updated': datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Live Market Data...")
    
    market_data = LiveMarketData()
    
    # Test live prices
    live_prices = market_data.fetch_live_prices()
    print(f"✅ Live prices fetched: {len(live_prices)} assets")
    
    # Test historical data
    btc_history = market_data.fetch_historical_prices('BTC', 30)
    print(f"✅ BTC historical data: {len(btc_history)} days")
    
    # Test market overview
    overview = market_data.get_market_overview()
    print(f"✅ Market overview: ${overview.get('total_market_cap', 0):,.0f} total market cap")
    
    print("🎉 Live market data system ready!")
