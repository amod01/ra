import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, date
import os
from app.core.config import settings
from app.models.schemas import ExpectedReturns, EfficientFrontierPoint

class LSTMPredictor:
    """
    FastAPI service wrapper for LSTM Engine
    """
    
    def __init__(self):
        self.engine = None
        self.initialized = False
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize LSTM engine with proper error handling"""
        try:
            # Import your LSTM engine
            from app.utils.lstm_engine import LSTMEngine
            
            self.engine = LSTMEngine(
                sequence_length=30,
                lstm_units=40,
                cache_file='Memory/lstm_cache.json'
            )
            self.initialized = True
            print("✅ LSTM Engine initialized successfully")
        except Exception as e:
            print(f"❌ LSTM Engine initialization failed: {e}")
            self.initialized = False
    
    async def train_models(self, tickers: List[str], lookback_days: int = 1260) -> Dict[str, Any]:
        """
        Train LSTM models for given tickers
        """
        if not self.initialized:
            return {"error": "LSTM engine not initialized"}
        
        try:
            results = {}
            for ticker in tickers:
                metrics = self.engine.train_model(ticker, lookback_days)
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
        force_recalculate: bool = False
    ) -> Tuple[List[ExpectedReturns], pd.DataFrame]:
        """
        Generate expected returns using LSTM engine
        """
        if not self.initialized:
            return [], pd.DataFrame()
        
        try:
            # Generate expected returns
            expected_returns_array, metrics_df, approved_tickers = self.engine.generate_expected_returns(
                tickers, force_recalculate
            )
            
            # Convert to ExpectedReturns schema
            expected_returns_list = []
            for i, ticker in enumerate(tickers):
                if i < len(expected_returns_array):
                    expected_returns_list.append(ExpectedReturns(
                        symbol=ticker,
                        lstm_expected_return=expected_returns_array[i],
                        elastic_net_expected_return=0.0,  # Will be filled by arbiter
                        capm_expected_return=0.0,  # Will be filled by arbiter
                        final_expected_return=expected_returns_array[i]
                    ))
            
            return expected_returns_list, metrics_df
            
        except Exception as e:
            print(f"LSTM expected returns generation failed: {e}")
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
        Generate optimal portfolio using LSTM predictions
        """
        if not self.initialized:
            return {"error": "LSTM engine not initialized"}
        
        try:
            # Generate expected returns
            expected_returns_list, metrics_df = await self.generate_expected_returns(tickers)
            
            if not expected_returns_list:
                return {"error": "Failed to generate expected returns"}
            
            # Extract expected returns array
            expected_returns_array = np.array([er.lstm_expected_return for er in expected_returns_list])
            
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
            efficient_frontier = []
            for point in optimization_result['frontier']:
                efficient_frontier.append(EfficientFrontierPoint(
                    volatility=point['volatility'],
                    return_=point['return'],
                    weights=point['weights']
                ))
            
            return {
                "optimal_weights": optimization_result['max_sharpe']['weights'],
                "expected_return": optimization_result['max_sharpe']['return'],
                "expected_volatility": optimization_result['max_sharpe']['volatility'],
                "sharpe_ratio": optimization_result['max_sharpe']['sharpe_ratio'],
                "efficient_frontier": efficient_frontier,
                "model_returns": {er.symbol: er for er in expected_returns_list}
            }
            
        except Exception as e:
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get LSTM model performance metrics
        """
        if not self.initialized or not hasattr(self.engine, 'model_metrics'):
            return {}
        
        return self.engine.model_metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for LSTM service
        """
        return {
            "service": "lstm_predictor",
            "initialized": self.initialized,
            "tensorflow_available": self.engine.tf_available if self.initialized else False,
            "cached_tickers": len(self.engine.expected_returns_cache) if self.initialized else 0
        }