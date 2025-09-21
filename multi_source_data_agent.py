"""
🤖 MULTI-SOURCE DATA INTEGRATION AGENT

Legitimate data collection and analysis from official APIs and public sources.
Integrates market data, news sentiment, and social metrics through official channels.

Features:
- Official API integrations (CoinGecko, DefiLlama, etc.)
- News sentiment analysis through official RSS feeds
- Social metrics through official APIs
- Exchange data through public endpoints
- Real-time market data aggregation
- Sentiment scoring and trend analysis

Compliance:
- Uses only official APIs and public data
- Respects rate limits and terms of service
- No unauthorized scraping or data harvesting
- Privacy-compliant data collection

Author: Liquidity Predictor Team
Version: 1.0
Last Updated: 2025-09-17
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta, timezone
import json
import time
import re
import feedparser
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class LegitimateDataAgent:
    """
    Legitimate multi-source data collection agent.
    
    Collects data from official APIs and public sources while respecting
    terms of service and rate limits. Provides comprehensive market intelligence
    through legitimate channels.
    """
    
    def __init__(self):
        """Initialize legitimate data collection agent."""
        
        # Official API configurations
        self.api_sources = {
            'coingecko': {
                'base_url': 'https://api.coingecko.com/api/v3',
                'rate_limit': 1.0,  # 1 second between calls for free tier
                'endpoints': {
                    'prices': '/simple/price',
                    'market_data': '/coins/{id}',
                    'trending': '/search/trending',
                    'global': '/global',
                    'exchanges': '/exchanges'
                }
            },
            
            'defillama': {
                'base_url': 'https://api.llama.fi',
                'rate_limit': 0.3,
                'endpoints': {
                    'protocols': '/protocols',
                    'tvl': '/tvl/{protocol}',
                    'dexs': '/overview/dexs',
                    'yields': '/pools'
                }
            },
            
            'coinpaprika': {
                'base_url': 'https://api.coinpaprika.com/v1',
                'rate_limit': 0.5,
                'endpoints': {
                    'coins': '/coins',
                    'tickers': '/tickers',
                    'exchanges': '/exchanges',
                    'global': '/global'
                }
            }
        }
        
        # News RSS feeds (public and legitimate)
        self.news_sources = {
            'cointelegraph': {
                'rss_url': 'https://cointelegraph.com/rss',
                'category': 'news',
                'reliability_score': 0.8
            },
            'coindesk': {
                'rss_url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'category': 'news', 
                'reliability_score': 0.9
            },
            'decrypt': {
                'rss_url': 'https://decrypt.co/feed',
                'category': 'news',
                'reliability_score': 0.7
            },
            'defipulse': {
                'rss_url': 'https://defipulse.com/blog/feed/',
                'category': 'defi_news',
                'reliability_score': 0.8
            }
        }
        
        # Social metrics (official APIs only)
        self.social_sources = {
            'reddit': {
                'api_url': 'https://www.reddit.com/r/cryptocurrency/.json',
                'rate_limit': 2.0,
                'requires_auth': False
            },
            'github': {
                'api_url': 'https://api.github.com',
                'rate_limit': 1.0,
                'requires_auth': False  # For public repos
            }
        }
        
        # Exchange APIs (public endpoints only)
        self.exchange_sources = {
            'binance': {
                'api_url': 'https://api.binance.com/api/v3',
                'rate_limit': 0.1,
                'endpoints': {
                    'ticker': '/ticker/24hr',
                    'depth': '/depth',
                    'trades': '/trades'
                }
            },
            'coinbase': {
                'api_url': 'https://api.exchange.coinbase.com',
                'rate_limit': 0.2,
                'endpoints': {
                    'products': '/products',
                    'ticker': '/products/{product_id}/ticker',
                    'stats': '/products/{product_id}/stats'
                }
            }
        }
    
    async def collect_comprehensive_market_data(self, target_assets: List[str]) -> Dict:
        """
        Collect comprehensive market data from all legitimate sources.
        
        Args:
            target_assets: List of asset symbols to analyze
            
        Returns:
            Comprehensive market intelligence data
        """
        print(f"🤖 Collecting comprehensive data for {len(target_assets)} assets...")
        
        comprehensive_data = {
            'collection_timestamp': datetime.now(timezone.utc).isoformat(),
            'target_assets': target_assets,
            'market_data': {},
            'news_sentiment': {},
            'social_metrics': {},
            'exchange_data': {},
            'market_intelligence': {}
        }
        
        # Collect data from all sources concurrently
        tasks = [
            self._collect_market_data(target_assets),
            self._collect_news_sentiment(target_assets),
            self._collect_social_metrics(target_assets),
            self._collect_exchange_data(target_assets)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        comprehensive_data['market_data'] = results[0] if not isinstance(results[0], Exception) else {}
        comprehensive_data['news_sentiment'] = results[1] if not isinstance(results[1], Exception) else {}
        comprehensive_data['social_metrics'] = results[2] if not isinstance(results[2], Exception) else {}
        comprehensive_data['exchange_data'] = results[3] if not isinstance(results[3], Exception) else {}
        
        # Generate market intelligence
        comprehensive_data['market_intelligence'] = self._generate_market_intelligence(comprehensive_data)
        
        return comprehensive_data
    
    async def _collect_market_data(self, assets: List[str]) -> Dict:
        """Collect market data from official APIs."""
        market_data = {}
        
        # CoinGecko data
        try:
            await asyncio.sleep(self.api_sources['coingecko']['rate_limit'])
            
            # Map symbols to CoinGecko IDs
            symbol_to_id = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'SUI': 'sui',
                'SAIL': 'sui', 'IKA': 'sui', 'DEEP': 'sui', 'WAL': 'sui'  # Sui ecosystem tokens
            }
            
            for asset in assets:
                coin_id = symbol_to_id.get(asset, asset.lower())
                
                async with aiohttp.ClientSession() as session:
                    # Get detailed market data
                    url = f"{self.api_sources['coingecko']['base_url']}/coins/{coin_id}"
                    params = {
                        'localization': 'false',
                        'tickers': 'true',
                        'market_data': 'true',
                        'community_data': 'true',
                        'developer_data': 'true'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            market_data[asset] = {
                                'price_data': {
                                    'current_price': data.get('market_data', {}).get('current_price', {}).get('usd', 0),
                                    'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                                    'volume_24h': data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                                    'price_change_24h': data.get('market_data', {}).get('price_change_percentage_24h', 0),
                                    'price_change_7d': data.get('market_data', {}).get('price_change_percentage_7d', 0),
                                    'price_change_30d': data.get('market_data', {}).get('price_change_percentage_30d', 0)
                                },
                                'market_metrics': {
                                    'market_cap_rank': data.get('market_cap_rank', 0),
                                    'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0),
                                    'total_supply': data.get('market_data', {}).get('total_supply', 0),
                                    'ath': data.get('market_data', {}).get('ath', {}).get('usd', 0),
                                    'atl': data.get('market_data', {}).get('atl', {}).get('usd', 0)
                                },
                                'community_data': {
                                    'twitter_followers': data.get('community_data', {}).get('twitter_followers', 0),
                                    'reddit_subscribers': data.get('community_data', {}).get('reddit_subscribers', 0),
                                    'telegram_channel_user_count': data.get('community_data', {}).get('telegram_channel_user_count', 0)
                                },
                                'developer_data': {
                                    'github_stars': data.get('developer_data', {}).get('stars', 0),
                                    'github_forks': data.get('developer_data', {}).get('forks', 0),
                                    'commit_count_4_weeks': data.get('developer_data', {}).get('commit_count_4_weeks', 0)
                                }
                            }
                
                await asyncio.sleep(self.api_sources['coingecko']['rate_limit'])
        
        except Exception as e:
            print(f"Error collecting market data: {e}")
        
        return market_data
    
    async def _collect_news_sentiment(self, assets: List[str]) -> Dict:
        """Collect news sentiment from RSS feeds."""
        news_sentiment = {}
        
        try:
            for source_name, source_config in self.news_sources.items():
                print(f"📰 Collecting news from {source_name}...")
                
                # Parse RSS feed
                feed = feedparser.parse(source_config['rss_url'])
                
                relevant_articles = []
                
                for entry in feed.entries[:20]:  # Analyze last 20 articles
                    title = entry.get('title', '')
                    description = entry.get('description', '')
                    content = f"{title} {description}".lower()
                    
                    # Check if article mentions our target assets
                    asset_mentions = []
                    for asset in assets:
                        if asset.lower() in content or f"${asset.lower()}" in content:
                            asset_mentions.append(asset)
                    
                    if asset_mentions:
                        # Analyze sentiment using TextBlob
                        try:
                            blob = TextBlob(f"{title} {description}")
                            sentiment_score = blob.sentiment.polarity  # -1 to 1
                            
                            relevant_articles.append({
                                'title': title,
                                'description': description,
                                'published': entry.get('published', ''),
                                'sentiment_score': sentiment_score,
                                'sentiment_label': self._classify_sentiment(sentiment_score),
                                'mentioned_assets': asset_mentions,
                                'source': source_name,
                                'reliability': source_config['reliability_score']
                            })
                        except:
                            pass  # Skip if sentiment analysis fails
                
                news_sentiment[source_name] = {
                    'articles': relevant_articles,
                    'avg_sentiment': np.mean([a['sentiment_score'] for a in relevant_articles]) if relevant_articles else 0,
                    'total_articles': len(relevant_articles),
                    'source_reliability': source_config['reliability_score']
                }
                
                await asyncio.sleep(1)  # Be respectful with RSS requests
        
        except Exception as e:
            print(f"Error collecting news sentiment: {e}")
        
        return news_sentiment
    
    async def _collect_social_metrics(self, assets: List[str]) -> Dict:
        """Collect social metrics from official APIs."""
        social_metrics = {}
        
        try:
            # Reddit public data (no authentication required for public posts)
            async with aiohttp.ClientSession() as session:
                url = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=25"
                headers = {'User-Agent': 'LiquidityPredictor/1.0'}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        reddit_data = await response.json()
                        
                        relevant_posts = []
                        
                        for post in reddit_data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            title = post_data.get('title', '').lower()
                            
                            # Check for asset mentions
                            mentioned_assets = [asset for asset in assets if asset.lower() in title]
                            
                            if mentioned_assets:
                                relevant_posts.append({
                                    'title': post_data.get('title', ''),
                                    'score': post_data.get('score', 0),
                                    'num_comments': post_data.get('num_comments', 0),
                                    'upvote_ratio': post_data.get('upvote_ratio', 0),
                                    'mentioned_assets': mentioned_assets,
                                    'created_utc': post_data.get('created_utc', 0)
                                })
                        
                        social_metrics['reddit'] = {
                            'relevant_posts': relevant_posts,
                            'avg_sentiment_score': np.mean([p['upvote_ratio'] for p in relevant_posts]) if relevant_posts else 0.5,
                            'total_engagement': sum(p['score'] + p['num_comments'] for p in relevant_posts),
                            'mention_frequency': len(relevant_posts)
                        }
        
        except Exception as e:
            print(f"Error collecting social metrics: {e}")
        
        return social_metrics
    
    async def _collect_exchange_data(self, assets: List[str]) -> Dict:
        """Collect exchange data from public APIs."""
        exchange_data = {}
        
        try:
            # Binance public API (no authentication required)
            async with aiohttp.ClientSession() as session:
                url = f"{self.exchange_sources['binance']['api_url']}/ticker/24hr"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        binance_data = await response.json()
                        
                        relevant_tickers = []
                        
                        for ticker in binance_data:
                            symbol = ticker.get('symbol', '')
                            
                            # Check if symbol contains our target assets
                            for asset in assets:
                                if asset in symbol and ('USDT' in symbol or 'USDC' in symbol):
                                    relevant_tickers.append({
                                        'symbol': symbol,
                                        'price': float(ticker.get('lastPrice', 0)),
                                        'volume': float(ticker.get('volume', 0)),
                                        'price_change': float(ticker.get('priceChangePercent', 0)),
                                        'high_24h': float(ticker.get('highPrice', 0)),
                                        'low_24h': float(ticker.get('lowPrice', 0)),
                                        'asset': asset
                                    })
                        
                        exchange_data['binance'] = {
                            'relevant_pairs': relevant_tickers,
                            'total_pairs_monitored': len(relevant_tickers),
                            'exchange_volume_24h': sum(t['volume'] * t['price'] for t in relevant_tickers)
                        }
        
        except Exception as e:
            print(f"Error collecting exchange data: {e}")
        
        return exchange_data
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into categories."""
        if sentiment_score > 0.3:
            return 'very_positive'
        elif sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score > -0.1:
            return 'neutral'
        elif sentiment_score > -0.3:
            return 'negative'
        else:
            return 'very_negative'
    
    def _generate_market_intelligence(self, comprehensive_data: Dict) -> Dict:
        """Generate actionable market intelligence from collected data."""
        intelligence = {
            'overall_sentiment': 'neutral',
            'sentiment_confidence': 0.5,
            'key_trends': [],
            'risk_factors': [],
            'opportunities': [],
            'market_signals': []
        }
        
        # Analyze news sentiment
        news_data = comprehensive_data.get('news_sentiment', {})
        if news_data:
            sentiment_scores = []
            reliability_weights = []
            
            for source, data in news_data.items():
                if data.get('avg_sentiment') is not None:
                    sentiment_scores.append(data['avg_sentiment'])
                    reliability_weights.append(data.get('source_reliability', 0.5))
            
            if sentiment_scores:
                # Weighted average sentiment
                weighted_sentiment = np.average(sentiment_scores, weights=reliability_weights)
                intelligence['overall_sentiment'] = self._classify_sentiment(weighted_sentiment)
                intelligence['sentiment_confidence'] = min(0.9, sum(reliability_weights) / len(reliability_weights))
        
        # Analyze social metrics
        social_data = comprehensive_data.get('social_metrics', {})
        if social_data:
            for platform, data in social_data.items():
                mention_freq = data.get('mention_frequency', 0)
                if mention_freq > 5:  # High mention frequency
                    intelligence['key_trends'].append(f"High social activity on {platform}")
        
        # Analyze market data for trends
        market_data = comprehensive_data.get('market_data', {})
        if market_data:
            for asset, data in market_data.items():
                price_data = data.get('price_data', {})
                
                # Identify trends
                if price_data.get('price_change_7d', 0) > 20:
                    intelligence['opportunities'].append(f"{asset} showing strong 7-day momentum (+{price_data['price_change_7d']:.1f}%)")
                elif price_data.get('price_change_7d', 0) < -20:
                    intelligence['risk_factors'].append(f"{asset} experiencing significant decline (-{abs(price_data['price_change_7d']):.1f}%)")
                
                # Volume analysis
                if price_data.get('volume_24h', 0) > 100000000:  # $100M+ volume
                    intelligence['market_signals'].append(f"{asset} showing high liquidity with ${price_data['volume_24h']:,.0f} 24h volume")
        
        return intelligence
    
    def create_sentiment_dashboard(self, comprehensive_data: Dict) -> Dict:
        """Create sentiment analysis dashboard data."""
        dashboard_data = {
            'sentiment_overview': {},
            'news_analysis': {},
            'social_analysis': {},
            'trend_analysis': {}
        }
        
        # Process news sentiment
        news_data = comprehensive_data.get('news_sentiment', {})
        if news_data:
            all_articles = []
            for source_data in news_data.values():
                all_articles.extend(source_data.get('articles', []))
            
            if all_articles:
                sentiment_distribution = {}
                for article in all_articles:
                    sentiment = article['sentiment_label']
                    sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
                
                dashboard_data['sentiment_overview'] = {
                    'total_articles': len(all_articles),
                    'sentiment_distribution': sentiment_distribution,
                    'avg_sentiment': np.mean([a['sentiment_score'] for a in all_articles]),
                    'recent_articles': sorted(all_articles, key=lambda x: x.get('published', ''), reverse=True)[:10]
                }
        
        # Process social metrics
        social_data = comprehensive_data.get('social_metrics', {})
        if social_data:
            dashboard_data['social_analysis'] = {
                'reddit_engagement': social_data.get('reddit', {}).get('total_engagement', 0),
                'mention_frequency': social_data.get('reddit', {}).get('mention_frequency', 0),
                'community_sentiment': social_data.get('reddit', {}).get('avg_sentiment_score', 0.5)
            }
        
        # Generate trend analysis
        market_intelligence = comprehensive_data.get('market_intelligence', {})
        dashboard_data['trend_analysis'] = {
            'key_trends': market_intelligence.get('key_trends', []),
            'opportunities': market_intelligence.get('opportunities', []),
            'risk_factors': market_intelligence.get('risk_factors', []),
            'market_signals': market_intelligence.get('market_signals', [])
        }
        
        return dashboard_data
    
    def generate_actionable_recommendations(self, comprehensive_data: Dict) -> List[Dict]:
        """Generate specific actionable recommendations from collected data."""
        recommendations = []
        
        market_intelligence = comprehensive_data.get('market_intelligence', {})
        
        # Sentiment-based recommendations
        overall_sentiment = market_intelligence.get('overall_sentiment', 'neutral')
        
        if overall_sentiment in ['positive', 'very_positive']:
            recommendations.append({
                'category': 'market_timing',
                'title': '📈 Positive Market Sentiment Detected',
                'description': 'News and social sentiment trending positive',
                'action_items': [
                    'Consider increasing DeFi exposure during positive sentiment',
                    'Monitor for potential FOMO-driven price increases',
                    'Set profit-taking levels for existing positions',
                    'Look for quality projects that may benefit from sentiment'
                ],
                'urgency': 'medium',
                'confidence': market_intelligence.get('sentiment_confidence', 0.5),
                'time_horizon': '1-2 weeks'
            })
        
        elif overall_sentiment in ['negative', 'very_negative']:
            recommendations.append({
                'category': 'risk_management',
                'title': '⚠️ Negative Market Sentiment Warning',
                'description': 'News and social sentiment trending negative',
                'action_items': [
                    'Consider reducing risk exposure',
                    'Increase stablecoin allocation',
                    'Set tighter stop-losses on positions',
                    'Look for oversold quality opportunities'
                ],
                'urgency': 'high',
                'confidence': market_intelligence.get('sentiment_confidence', 0.5),
                'time_horizon': 'Immediate'
            })
        
        # Volume-based recommendations
        for trend in market_intelligence.get('key_trends', []):
            if 'high social activity' in trend.lower():
                recommendations.append({
                    'category': 'social_signal',
                    'title': '💬 Increased Social Activity',
                    'description': trend,
                    'action_items': [
                        'Monitor for potential price volatility',
                        'Research underlying catalysts',
                        'Consider position sizing adjustments',
                        'Watch for coordinated movements'
                    ],
                    'urgency': 'medium',
                    'confidence': 0.7,
                    'time_horizon': '24-48 hours'
                })
        
        # Opportunity-based recommendations
        for opportunity in market_intelligence.get('opportunities', []):
            recommendations.append({
                'category': 'trading_opportunity',
                'title': '🚀 Momentum Opportunity',
                'description': opportunity,
                'action_items': [
                    'Analyze technical indicators for entry points',
                    'Set appropriate position size based on volatility',
                    'Define clear profit targets and stop-losses',
                    'Monitor volume for confirmation'
                ],
                'urgency': 'high',
                'confidence': 0.8,
                'time_horizon': '1-3 days'
            })
        
        return recommendations
    
    def create_data_quality_report(self, comprehensive_data: Dict) -> Dict:
        """Create data quality and reliability report."""
        quality_report = {
            'overall_quality_score': 0,
            'source_reliability': {},
            'data_freshness': {},
            'coverage_analysis': {},
            'recommendations': []
        }
        
        # Analyze source reliability
        total_sources = 0
        successful_sources = 0
        
        for data_type, data in comprehensive_data.items():
            if data_type in ['market_data', 'news_sentiment', 'social_metrics', 'exchange_data']:
                total_sources += 1
                if data and not isinstance(data, dict) or (isinstance(data, dict) and data):
                    successful_sources += 1
        
        quality_report['overall_quality_score'] = (successful_sources / total_sources * 100) if total_sources > 0 else 0
        
        # Data freshness analysis
        collection_time = datetime.fromisoformat(comprehensive_data.get('collection_timestamp', datetime.now().isoformat()))
        
        quality_report['data_freshness'] = {
            'collection_timestamp': collection_time.isoformat(),
            'age_minutes': (datetime.now(timezone.utc) - collection_time).total_seconds() / 60,
            'freshness_score': max(0, 100 - ((datetime.now(timezone.utc) - collection_time).total_seconds() / 3600 * 10))  # Decay 10% per hour
        }
        
        return quality_report


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing Multi-Source Data Agent...")
    
    agent = LegitimateDataAgent()
    
    # Test data collection
    test_assets = ['BTC', 'ETH', 'SOL', 'SUI']
    
    async def test_data_collection():
        comprehensive_data = await agent.collect_comprehensive_market_data(test_assets)
        return comprehensive_data
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    test_data = loop.run_until_complete(test_data_collection())
    
    print(f"✅ Market data collected: {len(test_data.get('market_data', {}))}")
    print(f"✅ News sources: {len(test_data.get('news_sentiment', {}))}")
    print(f"✅ Social metrics: {len(test_data.get('social_metrics', {}))}")
    print(f"✅ Exchange data: {len(test_data.get('exchange_data', {}))}")
    
    # Test sentiment analysis
    sentiment_dashboard = agent.create_sentiment_dashboard(test_data)
    print(f"✅ Sentiment dashboard: {sentiment_dashboard['sentiment_overview'].get('total_articles', 0)} articles analyzed")
    
    # Test recommendations
    recommendations = agent.generate_actionable_recommendations(test_data)
    print(f"✅ Generated {len(recommendations)} actionable recommendations")
    
    # Test data quality
    quality_report = agent.create_data_quality_report(test_data)
    print(f"✅ Data quality score: {quality_report['overall_quality_score']:.1f}/100")
    
    loop.close()
    
    print("🎉 Multi-source data agent ready!")
