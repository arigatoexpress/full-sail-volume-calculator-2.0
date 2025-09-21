"""
Performance optimization and caching system for Liquidity Predictor.
Implements advanced caching, lazy loading, and performance monitoring.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any
import functools
import time
import hashlib
import pickle
import os
from datetime import datetime, timedelta
import asyncio
import concurrent.futures


class AdvancedCacheManager:
    """Advanced caching system with intelligent invalidation."""
    
    def __init__(self, cache_dir: str = "advanced_cache"):
        """Initialize advanced cache manager."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function and arguments."""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached(self, key: str, max_age_minutes: int = 60) -> Optional[Any]:
        """Get cached data if valid."""
        # Check memory cache first
        if key in self.memory_cache:
            data, timestamp = self.memory_cache[key]
            if time.time() - timestamp < max_age_minutes * 60:
                self.cache_stats['hits'] += 1
                return data
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < max_age_minutes * 60:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Store in memory cache
                    self.memory_cache[key] = (data, time.time())
                    self.cache_stats['hits'] += 1
                    return data
            except Exception:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def set_cached(self, key: str, data: Any) -> None:
        """Store data in cache."""
        # Store in memory
        self.memory_cache[key] = (data, time.time())
        
        # Store on disk
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests * 100 if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        }


def smart_cache(max_age_minutes: int = 60):
    """Decorator for intelligent caching."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager is None:
                cache_manager = AdvancedCacheManager()
                wrapper._cache_manager = cache_manager
            
            # Generate cache key
            cache_key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # Try to get cached result
            cached_result = cache_manager.get_cached(cache_key, max_age_minutes)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set_cached(cache_key, result)
            
            return result
        return wrapper
    return decorator


class LazyLoader:
    """Lazy loading system for heavy operations."""
    
    def __init__(self):
        """Initialize lazy loader."""
        self.loaded_components = {}
        self.loading_states = {}
    
    def lazy_load_component(self, component_name: str, loader_func, *args, **kwargs):
        """Lazy load a component only when needed."""
        if component_name in self.loaded_components:
            return self.loaded_components[component_name]
        
        if component_name in self.loading_states:
            return None  # Currently loading
        
        try:
            self.loading_states[component_name] = True
            component = loader_func(*args, **kwargs)
            self.loaded_components[component_name] = component
            del self.loading_states[component_name]
            return component
        
        except Exception as e:
            if component_name in self.loading_states:
                del self.loading_states[component_name]
            raise e


class AsyncDataProcessor:
    """Asynchronous data processing for better performance."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize async processor."""
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def process_pools_async(self, pool_data: Dict) -> Dict:
        """Process multiple pools asynchronously."""
        future_to_pool = {}
        
        for pool_name, data in pool_data.items():
            future = self.executor.submit(self._process_single_pool, pool_name, data)
            future_to_pool[future] = pool_name
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_pool):
            pool_name = future_to_pool[future]
            try:
                results[pool_name] = future.result()
            except Exception as e:
                results[pool_name] = {'error': str(e)}
        
        return results
    
    def _process_single_pool(self, pool_name: str, data: pd.DataFrame) -> Dict:
        """Process a single pool's data."""
        # Simulate processing time
        time.sleep(0.1)
        
        return {
            'pool': pool_name,
            'processed_data': data,
            'metrics': {
                'mean_volume': data['volume_24h'].mean() if 'volume_24h' in data.columns else 0,
                'volatility': data['volume_24h'].std() if 'volume_24h' in data.columns else 0
            }
        }


# Global instances
cache_manager = AdvancedCacheManager()
lazy_loader = LazyLoader()
async_processor = AsyncDataProcessor()


# Example usage
if __name__ == "__main__":
    print("⚡ Testing Performance Optimization...")
    
    # Test caching
    @smart_cache(max_age_minutes=5)
    def expensive_operation(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * x
    
    # First call (cache miss)
    start = time.time()
    result1 = expensive_operation(10)
    time1 = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = expensive_operation(10)
    time2 = time.time() - start
    
    print(f"✅ Cache test: First call {time1:.3f}s, Second call {time2:.3f}s")
    print(f"✅ Cache speedup: {time1/time2:.1f}x faster")
    
    # Test lazy loading
    def load_heavy_component():
        time.sleep(0.05)
        return "Heavy component loaded"
    
    component = lazy_loader.lazy_load_component("test_component", load_heavy_component)
    print(f"✅ Lazy loading: {component}")
    
    print("🎉 Performance optimization ready!")
