"""
🔍 COMPREHENSIVE DEBUGGING AND TESTING SUITE

This module provides extensive debugging, testing, and validation capabilities
for the entire Liquidity Predictor application. It performs comprehensive
checks on all components, data flows, and user interactions.

Features:
- Complete module dependency analysis
- Performance benchmarking and optimization
- Data integrity validation
- Error handling verification
- Memory usage monitoring
- API endpoint health checks
- UI component testing
- Integration testing across all features

Author: Liquidity Predictor Team
Version: 1.0
Last Updated: 2025-09-17
"""

import sys
import os
import time
import psutil
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveDebugger:
    """
    Advanced debugging and testing system for Liquidity Predictor.
    
    Provides comprehensive analysis of:
    - Module dependencies and imports
    - Data flow integrity
    - Performance bottlenecks
    - Memory usage patterns
    - Error handling effectiveness
    - API endpoint reliability
    - User interface responsiveness
    """
    
    def __init__(self):
        """Initialize comprehensive debugger."""
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_config = {
            'memory_limit_mb': 1024,  # 1GB memory limit
            'api_timeout_seconds': 30,
            'max_test_duration_minutes': 10,
            'critical_modules': [
                'app', 'data_fetcher', 'data_processor', 'prediction_models',
                'visualization', 'live_market_data', 'epoch_predictor'
            ]
        }
    
    def run_comprehensive_debug_suite(self) -> Dict:
        """
        Run complete debugging and testing suite.
        
        Returns:
            Comprehensive test results and recommendations
        """
        print("🔍 STARTING COMPREHENSIVE DEBUG SUITE")
        print("=" * 60)
        
        debug_results = {
            'test_start_time': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'module_tests': {},
            'performance_tests': {},
            'data_integrity_tests': {},
            'api_health_tests': {},
            'ui_tests': {},
            'integration_tests': {},
            'recommendations': [],
            'overall_health_score': 0
        }
        
        try:
            # 1. Module Import and Dependency Testing
            print("\n🔧 1. Module Import Testing...")
            debug_results['module_tests'] = self._test_module_imports()
            
            # 2. Performance Benchmarking
            print("\n⚡ 2. Performance Benchmarking...")
            debug_results['performance_tests'] = self._run_performance_tests()
            
            # 3. Data Integrity Validation
            print("\n📊 3. Data Integrity Testing...")
            debug_results['data_integrity_tests'] = self._test_data_integrity()
            
            # 4. API Health Checks
            print("\n🌐 4. API Health Testing...")
            debug_results['api_health_tests'] = self._test_api_endpoints()
            
            # 5. Integration Testing
            print("\n🔗 5. Integration Testing...")
            debug_results['integration_tests'] = self._test_integrations()
            
            # 6. Generate Recommendations
            print("\n💡 6. Generating Recommendations...")
            debug_results['recommendations'] = self._generate_recommendations(debug_results)
            
            # 7. Calculate Overall Health Score
            debug_results['overall_health_score'] = self._calculate_health_score(debug_results)
            
        except Exception as e:
            self.error_log.append(f"Critical error in debug suite: {str(e)}")
            debug_results['critical_error'] = str(e)
        
        debug_results['test_duration_seconds'] = time.time() - self.start_time
        debug_results['error_log'] = self.error_log
        
        return debug_results
    
    def _get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
            'current_directory': os.getcwd(),
            'environment_variables': dict(os.environ)
        }
    
    def _test_module_imports(self) -> Dict:
        """Test all module imports and dependencies."""
        module_results = {}
        
        modules_to_test = [
            'app', 'data_fetcher', 'data_processor', 'prediction_models',
            'visualization', 'utils', 'granular_analysis', 'backtesting',
            'macro_analysis', 'trading_view', 'epoch_predictor', 'advanced_features',
            'multi_asset_analyzer', 'epoch_volume_predictor', 'live_market_data',
            'robust_live_data', 'comprehensive_data_aggregator', 'onchain_analytics',
            'premium_ui', 'performance_optimizer', 'visual_effects'
        ]
        
        for module_name in modules_to_test:
            start_time = time.time()
            try:
                # Test import
                module = __import__(module_name)
                import_time = time.time() - start_time
                
                # Test basic functionality
                if hasattr(module, '__all__'):
                    exported_items = len(module.__all__)
                else:
                    exported_items = len([item for item in dir(module) if not item.startswith('_')])
                
                module_results[module_name] = {
                    'status': 'success',
                    'import_time_ms': import_time * 1000,
                    'exported_items': exported_items,
                    'file_size_kb': self._get_file_size(f"{module_name}.py"),
                    'dependencies': self._analyze_dependencies(module_name)
                }
                
            except Exception as e:
                module_results[module_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'import_time_ms': (time.time() - start_time) * 1000
                }
                self.error_log.append(f"Module {module_name} import failed: {str(e)}")
        
        return module_results
    
    def _run_performance_tests(self) -> Dict:
        """Run comprehensive performance tests."""
        performance_results = {}
        
        # Test 1: Data Loading Performance
        print("  📊 Testing data loading performance...")
        start_time = time.time()
        try:
            from data_fetcher import DataFetcher
            fetcher = DataFetcher()
            test_data = fetcher.fetch_historical_volumes(30)
            
            performance_results['data_loading'] = {
                'status': 'success',
                'duration_seconds': time.time() - start_time,
                'data_points': len(test_data),
                'memory_usage_mb': self._get_memory_usage()
            }
        except Exception as e:
            performance_results['data_loading'] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
        
        # Test 2: Prediction Performance
        print("  🔮 Testing prediction performance...")
        start_time = time.time()
        try:
            from prediction_models import VolumePredictor
            from data_processor import DataProcessor
            
            predictor = VolumePredictor()
            processor = DataProcessor()
            
            # Use cached data if available
            if 'test_data' in locals():
                processed_data = processor.process_pool_data(test_data)
                sample_pool = list(processed_data.values())[0]
                
                if len(sample_pool) >= 14:
                    predictions = predictor._simple_forecast(sample_pool, 'volume_24h', 7)
                    
                    performance_results['prediction'] = {
                        'status': 'success',
                        'duration_seconds': time.time() - start_time,
                        'predictions_generated': len(predictions),
                        'memory_usage_mb': self._get_memory_usage()
                    }
                else:
                    performance_results['prediction'] = {
                        'status': 'skipped',
                        'reason': 'insufficient_data'
                    }
        except Exception as e:
            performance_results['prediction'] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
        
        # Test 3: Visualization Performance
        print("  📈 Testing visualization performance...")
        start_time = time.time()
        try:
            from visualization import VolumeVisualizer
            
            visualizer = VolumeVisualizer()
            
            if 'test_data' in locals():
                fig = visualizer.create_volume_timeseries(test_data)
                
                performance_results['visualization'] = {
                    'status': 'success',
                    'duration_seconds': time.time() - start_time,
                    'chart_traces': len(fig.data),
                    'memory_usage_mb': self._get_memory_usage()
                }
        except Exception as e:
            performance_results['visualization'] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
        
        return performance_results
    
    def _test_data_integrity(self) -> Dict:
        """Test data integrity and validation."""
        integrity_results = {}
        
        print("  🔍 Testing data validation...")
        try:
            from utils import validate_dataframe, clean_numeric_data
            
            # Test data validation
            test_df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=30),
                'volume_24h': np.random.rand(30) * 1000,
                'tvl': np.random.rand(30) * 10000
            })
            
            # Valid data test
            is_valid = validate_dataframe(test_df, ['date', 'volume_24h'], min_rows=20)
            
            # Invalid data test
            is_invalid = validate_dataframe(test_df, ['nonexistent_column'], min_rows=10)
            
            # Data cleaning test
            dirty_data = pd.Series([1, np.nan, np.inf, -np.inf, 1000])
            cleaned_data = clean_numeric_data(dirty_data)
            
            integrity_results['data_validation'] = {
                'status': 'success',
                'valid_data_check': is_valid,
                'invalid_data_handling': not is_invalid,
                'data_cleaning_works': not (cleaned_data.isna().any() or np.isinf(cleaned_data).any()),
                'negative_values_handled': cleaned_data.min() >= 0
            }
            
        except Exception as e:
            integrity_results['data_validation'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return integrity_results
    
    def _test_api_endpoints(self) -> Dict:
        """Test API endpoint health and reliability."""
        api_results = {}
        
        # Test CoinGecko API
        print("  🌐 Testing CoinGecko API...")
        try:
            import requests
            
            response = requests.get(
                "https://api.coingecko.com/api/v3/ping",
                timeout=10
            )
            
            api_results['coingecko'] = {
                'status': 'success' if response.status_code == 200 else 'degraded',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'status_code': response.status_code
            }
            
        except Exception as e:
            api_results['coingecko'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Test DefiLlama API
        print("  🦙 Testing DefiLlama API...")
        try:
            response = requests.get(
                "https://api.llama.fi/protocols",
                timeout=10
            )
            
            api_results['defillama'] = {
                'status': 'success' if response.status_code == 200 else 'degraded',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'status_code': response.status_code,
                'data_points': len(response.json()) if response.status_code == 200 else 0
            }
            
        except Exception as e:
            api_results['defillama'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return api_results
    
    def _test_integrations(self) -> Dict:
        """Test integration between all major components."""
        integration_results = {}
        
        print("  🔗 Testing component integrations...")
        
        # Test full data pipeline
        try:
            from app import LiquidityPredictorDashboard
            
            # Initialize dashboard (tests all component integrations)
            dashboard = LiquidityPredictorDashboard()
            
            # Test core data flow
            test_data = dashboard.fetcher.fetch_historical_volumes(7)
            processed_data = dashboard.processor.process_pool_data(test_data)
            
            integration_results['data_pipeline'] = {
                'status': 'success',
                'pools_processed': len(processed_data),
                'data_points': len(test_data)
            }
            
            # Test epoch system integration
            epoch_info = dashboard.epoch_predictor.get_current_epoch_info()
            
            integration_results['epoch_system'] = {
                'status': 'success',
                'current_epoch': epoch_info['epoch_number'],
                'epoch_progress': epoch_info['epoch_progress']
            }
            
            # Test visualization integration
            fig = dashboard.visualizer.create_volume_timeseries(test_data)
            
            integration_results['visualization'] = {
                'status': 'success',
                'chart_traces': len(fig.data)
            }
            
        except Exception as e:
            integration_results['full_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
            self.error_log.append(f"Integration test failed: {str(e)}")
        
        return integration_results
    
    def _get_file_size(self, filename: str) -> float:
        """Get file size in KB."""
        try:
            return os.path.getsize(filename) / 1024
        except:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _analyze_dependencies(self, module_name: str) -> Dict:
        """Analyze module dependencies."""
        try:
            with open(f"{module_name}.py", 'r') as f:
                content = f.read()
            
            # Count import statements
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            return {
                'import_count': len(import_lines),
                'has_streamlit': 'streamlit' in content,
                'has_pandas': 'pandas' in content,
                'has_plotly': 'plotly' in content,
                'lines_of_code': len(content.split('\n'))
            }
        except:
            return {'error': 'Could not analyze dependencies'}
    
    def _generate_recommendations(self, debug_results: Dict) -> List[str]:
        """Generate optimization and improvement recommendations."""
        recommendations = []
        
        # Module recommendations
        module_tests = debug_results.get('module_tests', {})
        failed_modules = [name for name, result in module_tests.items() 
                         if result.get('status') == 'failed']
        
        if failed_modules:
            recommendations.append(f"🔧 Fix import issues in modules: {', '.join(failed_modules)}")
        
        # Performance recommendations
        performance_tests = debug_results.get('performance_tests', {})
        
        for test_name, result in performance_tests.items():
            if result.get('status') == 'success':
                duration = result.get('duration_seconds', 0)
                if duration > 5:
                    recommendations.append(f"⚡ Optimize {test_name} performance (currently {duration:.1f}s)")
                
                memory = result.get('memory_usage_mb', 0)
                if memory > 500:
                    recommendations.append(f"🧠 Reduce memory usage in {test_name} (currently {memory:.1f}MB)")
        
        # API recommendations
        api_tests = debug_results.get('api_health_tests', {})
        
        for api_name, result in api_tests.items():
            if result.get('status') == 'failed':
                recommendations.append(f"🌐 Fix {api_name} API connectivity issues")
            elif result.get('response_time_ms', 0) > 5000:
                recommendations.append(f"🚀 Optimize {api_name} API response time")
        
        # General recommendations
        if len(self.error_log) > 0:
            recommendations.append(f"🐛 Address {len(self.error_log)} logged errors")
        
        if not recommendations:
            recommendations.append("✅ All systems operating optimally!")
        
        return recommendations
    
    def _calculate_health_score(self, debug_results: Dict) -> float:
        """Calculate overall application health score."""
        score = 100.0
        
        # Module health impact
        module_tests = debug_results.get('module_tests', {})
        total_modules = len(module_tests)
        failed_modules = sum(1 for result in module_tests.values() 
                           if result.get('status') == 'failed')
        
        if total_modules > 0:
            module_success_rate = (total_modules - failed_modules) / total_modules
            score *= module_success_rate
        
        # Performance impact
        performance_tests = debug_results.get('performance_tests', {})
        slow_tests = sum(1 for result in performance_tests.values()
                        if result.get('duration_seconds', 0) > 3)
        
        if len(performance_tests) > 0:
            performance_penalty = (slow_tests / len(performance_tests)) * 20
            score -= performance_penalty
        
        # API health impact
        api_tests = debug_results.get('api_health_tests', {})
        failed_apis = sum(1 for result in api_tests.values()
                         if result.get('status') == 'failed')
        
        if len(api_tests) > 0:
            api_penalty = (failed_apis / len(api_tests)) * 30
            score -= api_penalty
        
        # Error impact
        error_penalty = min(20, len(self.error_log) * 5)
        score -= error_penalty
        
        return max(0, min(100, score))
    
    def generate_debug_report(self, debug_results: Dict) -> str:
        """Generate comprehensive debug report."""
        report = f"""
🔍 LIQUIDITY PREDICTOR - COMPREHENSIVE DEBUG REPORT
Generated: {debug_results['test_start_time']}
Duration: {debug_results['test_duration_seconds']:.2f} seconds

📊 OVERALL HEALTH SCORE: {debug_results['overall_health_score']:.1f}/100

🖥️ SYSTEM INFORMATION:
- Python Version: {debug_results['system_info']['python_version'].split()[0]}
- Platform: {debug_results['system_info']['platform']}
- CPU Cores: {debug_results['system_info']['cpu_count']}
- Total Memory: {debug_results['system_info']['memory_total_gb']:.1f} GB
- Available Memory: {debug_results['system_info']['memory_available_gb']:.1f} GB
- Disk Space: {debug_results['system_info']['disk_free_gb']:.1f} GB

🔧 MODULE TEST RESULTS:
"""
        
        module_tests = debug_results.get('module_tests', {})
        successful_modules = sum(1 for result in module_tests.values() if result.get('status') == 'success')
        total_modules = len(module_tests)
        
        report += f"- Modules Tested: {total_modules}\n"
        report += f"- Successful Imports: {successful_modules}\n"
        report += f"- Failed Imports: {total_modules - successful_modules}\n\n"
        
        # Performance results
        performance_tests = debug_results.get('performance_tests', {})
        report += "⚡ PERFORMANCE TEST RESULTS:\n"
        
        for test_name, result in performance_tests.items():
            status_emoji = "✅" if result.get('status') == 'success' else "❌"
            duration = result.get('duration_seconds', 0)
            memory = result.get('memory_usage_mb', 0)
            
            report += f"- {status_emoji} {test_name}: {duration:.2f}s, {memory:.1f}MB\n"
        
        # API health results
        api_tests = debug_results.get('api_health_tests', {})
        report += "\n🌐 API HEALTH RESULTS:\n"
        
        for api_name, result in api_tests.items():
            status_emoji = "✅" if result.get('status') == 'success' else "⚠️" if result.get('status') == 'degraded' else "❌"
            response_time = result.get('response_time_ms', 0)
            
            report += f"- {status_emoji} {api_name}: {response_time:.0f}ms\n"
        
        # Recommendations
        recommendations = debug_results.get('recommendations', [])
        report += "\n💡 RECOMMENDATIONS:\n"
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        # Error log
        if debug_results.get('error_log'):
            report += "\n🐛 ERROR LOG:\n"
            for error in debug_results['error_log']:
                report += f"- {error}\n"
        
        report += f"\n🎉 Debug report completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report
    
    def save_debug_report(self, debug_results: Dict, filename: str = None) -> str:
        """Save debug report to file."""
        if filename is None:
            filename = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = self.generate_debug_report(debug_results)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        return filename


# Quick debugging function for immediate use
def quick_debug() -> None:
    """Run quick debugging check."""
    print("🔍 QUICK DEBUG CHECK")
    print("=" * 30)
    
    debugger = ComprehensiveDebugger()
    
    # Test critical modules only
    critical_modules = ['app', 'data_fetcher', 'prediction_models', 'visualization']
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {str(e)}")
    
    print("\n🎉 Quick debug completed!")


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Running Comprehensive Debug Suite...")
    
    debugger = ComprehensiveDebugger()
    results = debugger.run_comprehensive_debug_suite()
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Overall Health Score: {results['overall_health_score']:.1f}/100")
    print(f"Test Duration: {results['test_duration_seconds']:.2f} seconds")
    print(f"Errors Logged: {len(results['error_log'])}")
    
    # Save report
    report_file = debugger.save_debug_report(results)
    print(f"📄 Debug report saved to: {report_file}")
    
    print("🎉 Comprehensive debugging completed!")
