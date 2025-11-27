from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.portfolio_arbiter import PortfolioArbiter
from app.services.backtester import Backtester

router = APIRouter()
portfolio_arbiter = PortfolioArbiter()
backtester = Backtester()

@router.post("/portfolio-summary")
async def get_portfolio_summary(request: Dict[str, Any]):
    """
    Get comprehensive portfolio summary for dashboard
    """
    try:
        symbols = request.get("symbols", [])
        risk_score = request.get("risk_score", 5)
        investment_amount = request.get("investment_amount", 10000)
        
        # Get optimized portfolio
        portfolio_result = await portfolio_arbiter.optimize_portfolio(
            symbols, risk_score
        )
        
        # Backtest the portfolio
        backtest_result = await backtester.backtest_portfolio(
            portfolio_result.optimal_weights,
            investment_amount
        )
        
        # Combine results for dashboard
        return {
            "portfolio_allocation": portfolio_result.optimal_weights,
            "performance_metrics": {
                "expected_return": portfolio_result.expected_return,
                "expected_volatility": portfolio_result.expected_volatility,
                "sharpe_ratio": portfolio_result.sharpe_ratio,
                "selected_strategy": portfolio_result.selected_strategy,
                "arbiter_score": portfolio_result.arbiter_score
            },
            "backtest_results": backtest_result,
            "risk_profile": {
                "risk_score": risk_score,
                "volatility_range": f"{portfolio_result.expected_volatility:.1%}",
                "recommended_allocation": get_recommended_allocation(risk_score)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dashboard summary failed: {str(e)}")

def get_recommended_allocation(risk_score: int) -> str:
    """Get recommended allocation based on risk score"""
    allocations = {
        1: "90% Bonds, 10% Stocks",
        2: "80% Bonds, 20% Stocks", 
        3: "70% Bonds, 30% Stocks",
        4: "60% Bonds, 40% Stocks",
        5: "50% Bonds, 50% Stocks",
        6: "40% Bonds, 60% Stocks",
        7: "30% Bonds, 70% Stocks",
        8: "20% Bonds, 80% Stocks",
        9: "10% Bonds, 90% Stocks"
    }
    return allocations.get(risk_score, "50% Bonds, 50% Stocks")