import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from app.services.lstm_predictor import LSTMPredictor
from app.services.elastic_net_predictor import ElasticNetPredictor
from app.services.capm_calculator import CAPMCalculator
from app.models.schemas import PortfolioOptimizationResponse, ExpectedReturns
from app.utils.arbiter_model import ArbiterModel

class PortfolioArbiter:
    """
    Main portfolio arbiter that combines predictions from all models
    and selects the optimal strategy based on client risk profile
    """
    
    def __init__(self):
        self.lstm_predictor = LSTMPredictor()
        self.elastic_net_predictor = ElasticNetPredictor()
        self.capm_calculator = CAPMCalculator()
        self.arbiter_model = ArbiterModel()
    
    async def optimize_portfolio(
        self, 
        symbols: List[str], 
        risk_score: int,
        lookback_days: int = 252
    ) -> PortfolioOptimizationResponse:
        """
        Generate optimal portfolio using all models and arbiter selection logic
        """
        try:
            # Get predictions from all models
            lstm_returns, lstm_metrics = await self.lstm_predictor.generate_expected_returns(symbols)
            elastic_net_returns, elastic_net_metrics = await self.elastic_net_predictor.generate_expected_returns(symbols)
            capm_returns, capm_metrics = await self.capm_calculator.generate_expected_returns(symbols)
            
            # Get portfolio optimizations from each model
            strategies_weights = {}
            strategies_projections = {}
            strategies_performance = {}
            
            # CAPM Optimization
            capm_result = await self.capm_calculator.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in capm_result:
                strategies_weights['CAPM'] = capm_result["optimal_weights"]
                strategies_projections['CAPM'] = capm_result["expected_return"]
                strategies_performance['CAPM'] = self._extract_performance_metrics(capm_result, capm_metrics)
            
            # Elastic Net Optimization
            elastic_result = await self.elastic_net_predictor.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in elastic_result:
                strategies_weights['ElasticNet'] = elastic_result["optimal_weights"]
                strategies_projections['ElasticNet'] = elastic_result["expected_return"]
                strategies_performance['ElasticNet'] = self._extract_performance_metrics(elastic_result, elastic_net_metrics)
            
            # LSTM Optimization
            lstm_result = await self.lstm_predictor.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in lstm_result:
                strategies_weights['LSTM'] = lstm_result["optimal_weights"]
                strategies_projections['LSTM'] = lstm_result["expected_return"]
                strategies_performance['LSTM'] = self._extract_performance_metrics(lstm_result, lstm_metrics)
            
            # Select best strategy using arbiter
            if strategies_performance:
                best_strategy, best_score, weights_used = self.arbiter_model.select_best_strategy(
                    risk_score, strategies_performance
                )
                
                # Use the best strategy's portfolio
                optimal_weights = strategies_weights[best_strategy]
                selected_result = {
                    'CAPM': capm_result, 
                    'ElasticNet': elastic_result, 
                    'LSTM': lstm_result
                }[best_strategy]
                
                # Update model returns with combined predictions
                model_returns = self._combine_model_returns(
                    lstm_returns, elastic_net_returns, capm_returns
                )
                
                return PortfolioOptimizationResponse(
                    optimal_weights=optimal_weights,
                    expected_return=selected_result["expected_return"],
                    expected_volatility=selected_result["expected_volatility"],
                    sharpe_ratio=selected_result["sharpe_ratio"],
                    efficient_frontier=selected_result.get("efficient_frontier", []),
                    model_returns=model_returns,
                    selected_strategy=best_strategy,
                    arbiter_score=best_score,
                    strategy_weights=weights_used
                )
            else:
                raise Exception("No successful portfolio optimizations")
                
        except Exception as e:
            raise Exception(f"Portfolio optimization failed: {str(e)}")
    
    def _extract_performance_metrics(self, optimization_result: Dict, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract performance metrics for arbiter scoring
        """
        # Basic financial metrics from optimization
        performance = {
            'sharpe': optimization_result.get('sharpe_ratio', 0.0),
            'alpha': optimization_result.get('alpha', 0.0),
            'excess': optimization_result.get('excess_return', 0.0)
        }
        
        # Add ML metrics if available
        if not metrics_df.empty:
            # Get portfolio-level metrics
            portfolio_metrics = metrics_df[metrics_df['ticker'] == 'PORTFOLIO']
            if not portfolio_metrics.empty:
                portfolio_row = portfolio_metrics.iloc[0]
                performance['r2'] = portfolio_row.get('test_r2', 0.0)
                performance['loss'] = portfolio_row.get('test_maae', 0.1)  # Default 10% error
        
        return performance
    
    def _combine_model_returns(
        self, 
        lstm_returns: List[ExpectedReturns],
        elastic_net_returns: List[ExpectedReturns],
        capm_returns: List[ExpectedReturns]
    ) -> Dict[str, ExpectedReturns]:
        """
        Combine predictions from multiple models
        """
        model_returns = {}
        all_symbols = set([er.symbol for er in lstm_returns + elastic_net_returns + capm_returns])
        
        for symbol in all_symbols:
            lstm_er = next((er for er in lstm_returns if er.symbol == symbol), None)
            elastic_net_er = next((er for er in elastic_net_returns if er.symbol == symbol), None)
            capm_er = next((er for er in capm_returns if er.symbol == symbol), None)
            
            # Simple average - replace with your arbiter logic
            final_return = np.mean([
                lstm_er.lstm_expected_return if lstm_er else 0.0,
                elastic_net_er.elastic_net_expected_return if elastic_net_er else 0.0,
                capm_er.capm_expected_return if capm_er else 0.0
            ])
            
            model_returns[symbol] = ExpectedReturns(
                symbol=symbol,
                lstm_expected_return=lstm_er.lstm_expected_return if lstm_er else 0.0,
                elastic_net_expected_return=elastic_net_er.elastic_net_expected_return if elastic_net_er else 0.0,
                capm_expected_return=capm_er.capm_expected_return if capm_er else 0.0,
                final_expected_return=final_return
            )
        
        return model_returns
    
    async def compare_strategies(
        self,
        symbols: List[str],
        risk_score: int,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Compare all strategies and their performance metrics
        """
        try:
            # Get optimizations from all models
            strategies_results = {}
            
            # CAPM
            capm_result = await self.capm_calculator.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in capm_result:
                strategies_results['CAPM'] = capm_result
            
            # Elastic Net
            elastic_result = await self.elastic_net_predictor.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in elastic_result:
                strategies_results['ElasticNet'] = elastic_result
            
            # LSTM
            lstm_result = await self.lstm_predictor.optimize_portfolio(symbols, risk_score, lookback_days)
            if "error" not in lstm_result:
                strategies_results['LSTM'] = lstm_result
            
            # Calculate arbiter scores for each
            comparison = {}
            for strategy_name, result in strategies_results.items():
                # Simulate performance metrics (in real implementation, these would come from backtesting)
                simulated_metrics = self._simulate_performance_metrics(strategy_name, result)
                
                # Calculate arbiter score
                all_metrics = {
                    'sharpe': [r.get('sharpe_ratio', 0) for r in strategies_results.values()],
                    'alpha': [r.get('alpha', 0) for r in strategies_results.values()],
                    'excess': [r.get('excess_return', 0) for r in strategies_results.values()]
                }
                
                all_ml_metrics = {
                    'r2': [0.8, 0.85, 0.9],  # Simulated R² values
                    'loss': [0.05, 0.06, 0.055]  # Simulated MAAE values
                }
                
                arbiter_score = self.arbiter_model.calculate_arbiter_score(
                    risk_score,
                    result.get('sharpe_ratio', 0),
                    result.get('alpha', 0),
                    result.get('excess_return', 0),
                    all_metrics,
                    r2_score=simulated_metrics['r2'],
                    loss_metric=simulated_metrics['loss'],
                    all_ml_metrics=all_ml_metrics
                )
                
                comparison[strategy_name] = {
                    'expected_return': result.get('expected_return', 0),
                    'expected_volatility': result.get('expected_volatility', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'arbiter_score': arbiter_score,
                    'optimal_weights': result.get('optimal_weights', {})
                }
            
            return {
                'risk_score': risk_score,
                'symbols': symbols,
                'comparison': comparison,
                'weights_used': self.arbiter_model.get_weights(risk_score)
            }
            
        except Exception as e:
            raise Exception(f"Strategy comparison failed: {str(e)}")
    
    def _simulate_performance_metrics(self, strategy_name: str, result: Dict) -> Dict[str, float]:
        """
        Simulate performance metrics for demonstration
        In production, these would come from actual backtesting
        """
        # These are placeholder values - replace with actual backtesting results
        metrics_map = {
            'CAPM': {'r2': 0.9, 'loss': 0.05},
            'ElasticNet': {'r2': 0.85, 'loss': 0.06},
            'LSTM': {'r2': 0.88, 'loss': 0.055}
        }
        return metrics_map.get(strategy_name, {'r2': 0.8, 'loss': 0.07})
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all models
        """
        lstm_health = await self.lstm_predictor.health_check()
        capm_health = await self.capm_calculator.health_check()
        elastic_net_health = await self.elastic_net_predictor.health_check()
        
        return {
            "lstm": {"health": lstm_health},
            "elastic_net": {"health": elastic_net_health},
            "capm": {"health": capm_health},
            "arbiter": {"initialized": True}
        }