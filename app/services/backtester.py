import pandas as pd
import numpy as np
from typing import Dict, List, Any
from app.utils.data_loader import get_stock_data

class Backtester:
    """
    Portfolio backtesting service
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    async def backtest_portfolio(
        self,
        weights: Dict[str, float],
        initial_investment: float = 10000,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Backtest a portfolio with given weights
        """
        try:
            # Get historical data for all tickers
            returns_data = {}
            for ticker, weight in weights.items():
                if weight > 0.01:  # Only include significant weights
                    data = get_stock_data(ticker)
                    if not data.empty:
                        returns_data[ticker] = data['Daily_Return'].tail(lookback_days)
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {"error": "No valid return data available"}
            
            # Calculate portfolio returns
            portfolio_returns = (returns_df * list(weights.values())).sum(axis=1)
            
            # Calculate performance metrics
            total_return = (portfolio_returns + 1).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "final_value": initial_investment * (1 + total_return)
            }
            
        except Exception as e:
            return {"error": f"Backtesting failed: {str(e)}"}