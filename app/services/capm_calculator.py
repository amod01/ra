import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
from app.core.config import settings
from app.models.schemas import ExpectedReturns, EfficientFrontierPoint
from app.utils.data_loader import get_features_for_ticker

class CAPMCalculator:
    """
    FastAPI service wrapper for CAPM Engine
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
        self.engine = None
        self.optimizer = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize CAPM engine"""
        try:
            from app.utils.capm_engine import CAPMEngine, MPTOptimizer
            self.engine = CAPMEngine(risk_free_rate=self.risk_free_rate)
            self.initialized = True
            print("✅ CAPM Engine initialized successfully")
        except Exception as e:
            print(f"❌ CAPM Engine initialization failed: {e}")
            self.initialized = False
    
    async def calculate_betas(self, tickers: List[str], lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate betas for multiple tickers
        """
        if not self.initialized:
            return {}
        
        betas = {}
        for ticker in tickers:
            beta = self.engine.calculate_beta(ticker, lookback_days)
            betas[ticker] = beta
        
        return betas
    
    async def generate_expected_returns(self, tickers: List[str]) -> Tuple[List[ExpectedReturns], pd.DataFrame]:
        """
        Generate expected returns using CAPM model
        """
        if not self.initialized:
            return [], pd.DataFrame()
        
        try:
            # Prepare portfolio inputs (this calculates betas, expected returns, and metrics)
            expected_returns_array, cov_matrix_array = self.engine.prepare_portfolio_inputs(tickers)
            
            # Get detailed metrics
            metrics_df = self.engine.get_metrics_dataframe()
            
            # Convert to ExpectedReturns schema
            expected_returns_list = []
            for i, ticker in enumerate(tickers):
                if i < len(expected_returns_array):
                    expected_returns_list.append(ExpectedReturns(
                        symbol=ticker,
                        lstm_expected_return=0.0,  # Will be filled by arbiter
                        elastic_net_expected_return=0.0,  # Will be filled by arbiter
                        capm_expected_return=expected_returns_array[i],
                        final_expected_return=expected_returns_array[i]
                    ))
            
            return expected_returns_list, metrics_df
            
        except Exception as e:
            print(f"CAPM expected returns generation failed: {e}")
            return [], pd.DataFrame()
    
    async def compute_covariance_matrix(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """
        Compute covariance matrix for portfolio optimization
        """
        if not self.initialized:
            return pd.DataFrame()
        
        try:
            return self.engine.compute_covariance_matrix(tickers, lookback_days)
        except Exception as e:
            print(f"Covariance matrix computation failed: {e}")
            return pd.DataFrame()
    
    async def optimize_portfolio(
        self, 
        tickers: List[str], 
        risk_score: int = 5,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Generate optimal portfolio using CAPM predictions
        """
        if not self.initialized:
            return {"error": "CAPM engine not initialized"}
        
        try:
            # Generate expected returns and covariance matrix
            expected_returns_list, metrics_df = await self.generate_expected_returns(tickers)
            
            if not expected_returns_list:
                return {"error": "Failed to generate expected returns"}
            
            # Extract expected returns array
            expected_returns_array = np.array([er.capm_expected_return for er in expected_returns_list])
            
            # Compute covariance matrix
            cov_matrix = await self.compute_covariance_matrix(tickers, lookback_days)
            
            if cov_matrix.empty:
                return {"error": "Failed to compute covariance matrix"}
            
            # Initialize MPT optimizer with risk score
            from app.utils.capm_engine import MPTOptimizer
            optimizer = MPTOptimizer(
                expected_returns_array, 
                cov_matrix.values, 
                tickers, 
                risk_score
            )
            
            # Generate efficient frontier
            frontier_df = optimizer.generate_efficient_frontier(num_points=99)
            
            # Get key portfolios
            mvp = optimizer.minimize_volatility()
            max_sharpe = optimizer.maximize_sharpe_ratio()
            
            # Convert frontier to API response format
            efficient_frontier = []
            for _, point in frontier_df.iterrows():
                efficient_frontier.append(EfficientFrontierPoint(
                    volatility=point['volatility'],
                    return_=point['return'],
                    weights=point['weights']
                ))
            
            return {
                "optimal_weights": max_sharpe['weights'],
                "expected_return": max_sharpe['return'],
                "expected_volatility": max_sharpe['volatility'],
                "sharpe_ratio": max_sharpe['sharpe_ratio'],
                "efficient_frontier": efficient_frontier,
                "model_returns": {er.symbol: er for er in expected_returns_list},
                "minimum_variance_portfolio": mvp,
                "maximum_sharpe_portfolio": max_sharpe
            }
            
        except Exception as e:
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    async def get_portfolio_metrics(
        self, 
        weights: Dict[str, float], 
        benchmark_return: float = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        """
        if not self.initialized or not self.optimizer:
            return {}
        
        try:
            # Convert weights to array
            tickers = list(weights.keys())
            weights_array = np.array([weights[ticker] for ticker in tickers])
            
            # Calculate portfolio performance
            portfolio_return, portfolio_volatility = self.optimizer.portfolio_performance(weights_array)
            sharpe_ratio = self.optimizer.portfolio_sharpe_ratio(weights_array)
            
            metrics = {
                "expected_return": portfolio_return,
                "expected_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio
            }
            
            # Calculate alpha if benchmark return is provided
            if benchmark_return is not None:
                alpha = self.optimizer.calculate_portfolio_alpha(weights_array, benchmark_return)
                mea = self.optimizer.calculate_mea(weights_array, benchmark_return)
                metrics.update({
                    "alpha": alpha,
                    "mea": mea
                })
            
            return metrics
            
        except Exception as e:
            print(f"Portfolio metrics calculation failed: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for CAPM service
        """
        return {
            "service": "capm_calculator",
            "initialized": self.initialized,
            "risk_free_rate": self.risk_free_rate,
            "market_ticker": self.engine.market_ticker if self.initialized else None
        }