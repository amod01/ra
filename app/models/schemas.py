from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class RiskLevel(str, Enum):
    AVERSION = "averse"
    NEUTRAL = "neutral"
    SEEKING = "seeking"

class RiskProfileResponse(BaseModel):
    risk_score: int = Field(..., ge=1, le=9)
    risk_category: RiskLevel
    volatility_range: tuple[float, float]
    tier1_total: int
    yes_count: int
    magnitude: str

class PortfolioRequest(BaseModel):
    risk_score: int = Field(..., ge=1, le=9)
    investment_amount: float = Field(..., gt=0)
    symbols: List[str] = Field(default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    lookback_days: int = Field(default=252)

class ExpectedReturns(BaseModel):
    symbol: str
    lstm_expected_return: float
    elastic_net_expected_return: float
    capm_expected_return: float
    final_expected_return: float

class EfficientFrontierPoint(BaseModel):
    volatility: float
    return_: float
    weights: Dict[str, float]

class PortfolioOptimizationResponse(BaseModel):
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    efficient_frontier: List[EfficientFrontierPoint]
    model_returns: Dict[str, ExpectedReturns]

class BacktestRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    initial_investment: float
    start_date: str
    end_date: str

class BacktestResult(BaseModel):
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trades: List[Dict[str, Any]]

class TradeExecutionRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    investment_amount: float
    execute_live: bool = False

class TradeExecutionResponse(BaseModel):
    success: bool
    trades: List[Dict[str, Any]]
    total_cost: float
    execution_time: datetime

class DashboardSummary(BaseModel):
    risk_profile: RiskProfileResponse
    portfolio_allocation: Dict[str, float]
    portfolio_performance: Dict[str, float]
    current_value: float
    unrealized_pnl: float
    recent_trades: List[Dict[str, Any]]

class PortfolioOptimizationResponse(BaseModel):
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    efficient_frontier: List[EfficientFrontierPoint]
    model_returns: Dict[str, ExpectedReturns]
    selected_strategy: Optional[str] = None
    arbiter_score: Optional[float] = None
    strategy_weights: Optional[Dict[str, float]] = None

class StrategyComparisonResponse(BaseModel):
    risk_score: int
    symbols: List[str]
    comparison: Dict[str, Any]
    weights_used: Dict[str, float]

class RiskAssessmentRequest(BaseModel):
    tier1_answers: Dict[str, str]
    tier2_answers: Dict[str, str]

class RiskAssessmentResponse(BaseModel):
    risk_score: int
    risk_category: str
    volatility_range: tuple[float, float]
    tier1_total: int
    yes_count: int
    magnitude: str

class BacktestRequest(BaseModel):
    weights: Dict[str, float]
    initial_investment: float = 10000
    lookback_days: int = 252

class BacktestResponse(BaseModel):
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    final_value: float

class DashboardSummary(BaseModel):
    portfolio_allocation: Dict[str, float]
    performance_metrics: Dict[str, Any]
    backtest_results: BacktestResponse
    risk_profile: Dict[str, Any]