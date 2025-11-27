import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
from app.core.config import settings
from app.models.schemas import ExpectedReturns, EfficientFrontierPoint
from app.utils.data_loader import get_features_for_ticker

class ElasticNetPredictor:
    """
    FastAPI service wrapper for Elastic Net Engine
    """
    
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.engine = None
        self.initialized = False
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize Elastic Net engine"""
        try:
            from app.utils.elastic_net_engine import ElasticNetEngine
            
            self.engine = ElasticNetEngine(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio
            )
            self.initialized = True
            print("✅ Elastic Net Engine initialized successfully")
        except Exception as e:
            print(f"❌ Elastic Net Engine initialization failed: {e}")
            self.initialized = False
    
    async def train_models(
        self, 
        tickers: List[str], 
        lookback_days: int = 1260,
        n_folds: int = 5,
        forward_days: int = 21
    ) -> Dict[str, Any]:
        """
        Train Elastic Net models for given tickers
        """
        if not self.initialized:
            return {"error": "Elastic Net engine not initialized"}
        
        try:
            results = {}
            for ticker in tickers:
                metrics = self.engine.train_model(
                    ticker, 
                    lookback_days=lookback_days,
                    n_folds=n_folds,
                    forward_days=forward_days
                )
                results[ticker] = metrics
            
            return {
                "status": "success",
                "trained_tickers": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    async def generate_expected_returns(
        self, 
        tickers: List[str],
        train_start_date: str = None,
        train_end_date: str = None,
        val_start_date: str = None,
        val_end_date: str = None
    ) -> Tuple[List[ExpectedReturns], pd.DataFrame]:
        """
        Generate expected returns using Elastic Net model
        """
        if not self.initialized:
            return [], pd.DataFrame()
        
        try:
            # Generate expected returns with date filtering
            expected_returns_array, metrics_df, approved_tickers = self.engine.generate_expected_returns(
                tickers,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date
            )
            
            # Convert to ExpectedReturns schema
            expected_returns_list = []
            for i, ticker in enumerate(tickers):
                if i < len(expected_returns_array):
                    expected_returns_list.append(ExpectedReturns(
                        symbol=ticker,
                        lstm_expected_return=0.0,  # Will be filled by arbiter
                        elastic_net_expected_return=expected_returns_array[i],
                        capm_expected_return=0.0,  # Will be filled by arbiter
                        final_expected_return=expected_returns_array[i]
                    ))
            
            return expected_returns_list, metrics_df
            
        except Exception as e:
            print(f"Elastic Net expected returns generation failed: {e}")
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
        Generate optimal portfolio using Elastic Net predictions
        """
        if not self.initialized:
            return {"error": "Elastic Net engine not initialized"}
        
        try:
            # Generate expected returns
            expected_returns_list, metrics_df = await self.generate_expected_returns(tickers)
            
            if not expected_returns_list:
                return {"error": "Failed to generate expected returns"}
            
            # Extract expected returns array
            expected_returns_array = np.array([er.elastic_net_expected_return for er in expected_returns_list])
            
            # Compute covariance matrix
            cov_matrix = await self.compute_covariance_matrix(tickers, lookback_days)
            
            if cov_matrix.empty:
                return {"error": "Failed to compute covariance matrix"}
            
            # Perform portfolio optimization
            optimization_result = self.engine.optimize_portfolio_mvo(
                expected_returns_array, 
                cov_matrix, 
                tickers, 
                risk_score
            )
            
            # Convert to API response format
            efficient_frontier_all = []
            for point in optimization_result['frontier_all']:
                efficient_frontier_all.append(EfficientFrontierPoint(
                    volatility=point['volatility'],
                    return_=point['return'],
                    weights=point['weights']
                ))
            
            efficient_frontier_filtered = []
            if not optimization_result['frontier_filtered'].empty:
                for point in optimization_result['frontier_filtered']:
                    efficient_frontier_filtered.append(EfficientFrontierPoint(
                        volatility=point['volatility'],
                        return_=point['return'],
                        weights=point['weights']
                    ))
            
            return {
                "optimal_weights": optimization_result['max_sharpe']['weights'],
                "expected_return": optimization_result['max_sharpe']['return'],
                "expected_volatility": optimization_result['max_sharpe']['volatility'],
                "sharpe_ratio": optimization_result['max_sharpe']['sharpe_ratio'],
                "efficient_frontier_all": efficient_frontier_all,
                "efficient_frontier_filtered": efficient_frontier_filtered,
                "model_returns": {er.symbol: er for er in expected_returns_list},
                "minimum_variance_portfolio": optimization_result['mvp'],
                "maximum_sharpe_portfolio": optimization_result['max_sharpe'],
                "approved_tickers": self.engine.approved_tickers if hasattr(self.engine, 'approved_tickers') else []
            }
            
        except Exception as e:
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get Elastic Net model performance metrics
        """
        if not self.initialized or not hasattr(self.engine, 'model_metrics'):
            return {}
        
        return self.engine.model_metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for Elastic Net service
        """
        return {
            "service": "elastic_net_predictor",
            "initialized": self.initialized,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "trained_models": len(self.engine.models) if self.initialized else 0
        }