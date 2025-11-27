from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from app.services.lstm_predictor import LSTMPredictor
from app.services.elastic_net_predictor import ElasticNetPredictor
from app.services.capm_calculator import CAPMCalculator
from app.services.portfolio_arbiter import PortfolioArbiter
from app.models.schemas import PortfolioRequest, PortfolioOptimizationResponse, StrategyComparisonResponse

router = APIRouter()

# Initialize all predictors
lstm_predictor = LSTMPredictor()
elastic_net_predictor = ElasticNetPredictor()
capm_calculator = CAPMCalculator()
portfolio_arbiter = PortfolioArbiter()

# ============================================================================
# LSTM Endpoints
# ============================================================================

@router.post("/lstm/train")
async def train_lstm_models(
    tickers: List[str] = Query(..., description="List of tickers to train models for"),
    lookback_days: int = Query(1260, description="Lookback period in days")
):
    """
    Train LSTM models for specified tickers
    """
    try:
        result = await lstm_predictor.train_models(tickers, lookback_days)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LSTM training failed: {str(e)}")

@router.get("/lstm/health")
async def lstm_health_check():
    """
    Check LSTM service health
    """
    return await lstm_predictor.health_check()

@router.post("/lstm/expected-returns")
async def get_lstm_expected_returns(
    tickers: List[str] = Query(..., description="List of tickers"),
    force_recalculate: bool = Query(False, description="Force recalculation")
):
    """
    Get LSTM expected returns for tickers
    """
    try:
        expected_returns, metrics = await lstm_predictor.generate_expected_returns(
            tickers, force_recalculate
        )
        return {
            "expected_returns": expected_returns,
            "metrics": metrics.to_dict(orient='records') if not metrics.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LSTM expected returns failed: {str(e)}")

@router.post("/lstm/optimize")
async def optimize_lstm_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio using LSTM model only
    """
    try:
        result = await lstm_predictor.optimize_portfolio(
            request.symbols,
            request.risk_score,
            request.lookback_days
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LSTM portfolio optimization failed: {str(e)}")

# ============================================================================
# Elastic Net Endpoints
# ============================================================================

@router.get("/elastic-net/health")
async def elastic_net_health_check():
    """
    Check Elastic Net service health
    """
    return await elastic_net_predictor.health_check()

@router.post("/elastic-net/train")
async def train_elastic_net_models(
    tickers: List[str] = Query(..., description="List of tickers to train models for"),
    lookback_days: int = Query(1260, description="Lookback period in days"),
    n_folds: int = Query(5, description="Number of K-Fold splits"),
    forward_days: int = Query(21, description="Prediction horizon in days")
):
    """
    Train Elastic Net models for specified tickers
    """
    try:
        result = await elastic_net_predictor.train_models(
            tickers, lookback_days, n_folds, forward_days
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Elastic Net training failed: {str(e)}")

@router.post("/elastic-net/expected-returns")
async def get_elastic_net_expected_returns(
    tickers: List[str] = Query(..., description="List of tickers"),
    train_start_date: str = Query(None, description="Training start date (YYYY-MM-DD)"),
    train_end_date: str = Query(None, description="Training end date (YYYY-MM-DD)"),
    val_start_date: str = Query(None, description="Validation start date (YYYY-MM-DD)"),
    val_end_date: str = Query(None, description="Validation end date (YYYY-MM-DD)")
):
    """
    Get Elastic Net expected returns for tickers
    """
    try:
        expected_returns, metrics = await elastic_net_predictor.generate_expected_returns(
            tickers,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date
        )
        return {
            "expected_returns": expected_returns,
            "metrics": metrics.to_dict(orient='records') if not metrics.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Elastic Net expected returns failed: {str(e)}")

@router.post("/elastic-net/optimize")
async def optimize_elastic_net_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio using Elastic Net model only
    """
    try:
        result = await elastic_net_predictor.optimize_portfolio(
            request.symbols,
            request.risk_score,
            request.lookback_days
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Elastic Net portfolio optimization failed: {str(e)}")

# ============================================================================
# CAPM Endpoints
# ============================================================================

@router.get("/capm/health")
async def capm_health_check():
    """
    Check CAPM service health
    """
    return await capm_calculator.health_check()

@router.post("/capm/expected-returns")
async def get_capm_expected_returns(
    tickers: List[str] = Query(..., description="List of tickers")
):
    """
    Get CAPM expected returns for tickers
    """
    try:
        expected_returns, metrics = await capm_calculator.generate_expected_returns(tickers)
        return {
            "expected_returns": expected_returns,
            "metrics": metrics.to_dict(orient='records') if not metrics.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CAPM expected returns failed: {str(e)}")

@router.post("/capm/betas")
async def get_capm_betas(
    tickers: List[str] = Query(..., description="List of tickers"),
    lookback_days: int = Query(252, description="Lookback period in days")
):
    """
    Get CAPM betas for tickers
    """
    try:
        betas = await capm_calculator.calculate_betas(tickers, lookback_days)
        return {"betas": betas}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CAPM beta calculation failed: {str(e)}")

@router.post("/capm/optimize")
async def optimize_capm_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio using CAPM model only
    """
    try:
        result = await capm_calculator.optimize_portfolio(
            request.symbols,
            request.risk_score,
            request.lookback_days
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CAPM portfolio optimization failed: {str(e)}")

# ============================================================================
# Arbiter Endpoints
# ============================================================================

@router.post("/arbiter/compare-strategies", response_model=StrategyComparisonResponse)
async def compare_strategies(
    symbols: List[str] = Query(..., description="List of tickers"),
    risk_score: int = Query(..., description="Risk score 1-9"),
    lookback_days: int = Query(252, description="Lookback period in days")
):
    """
    Compare all strategies and their arbiter scores
    """
    try:
        result = await portfolio_arbiter.compare_strategies(
            symbols, risk_score, lookback_days
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Strategy comparison failed: {str(e)}")

@router.get("/arbiter/weights/{risk_score}")
async def get_arbiter_weights(risk_score: int):
    """
    Get arbiter weights for a specific risk score
    """
    try:
        weights = portfolio_arbiter.arbiter_model.get_weights(risk_score)
        return {
            "risk_score": risk_score,
            "weights": weights,
            "description": "Dynamic weights for Sharpe, Alpha, and Excess Return"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/arbiter/backtest-strategies")
async def backtest_strategies(
    symbols: List[str] = Query(..., description="List of tickers"),
    risk_score: int = Query(..., description="Risk score 1-9"),
    lookback_days: int = Query(252, description="Lookback period in days")
):
    """
    Backtest all strategies and select the best one (placeholder for future implementation)
    """
    try:
        # This would integrate with your backtesting framework
        return {
            "message": "Backtesting functionality will be implemented in Phase 2",
            "symbols": symbols,
            "risk_score": risk_score,
            "lookback_days": lookback_days
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Backtesting failed: {str(e)}")

# ============================================================================
# Combined Portfolio Optimization Endpoints
# ============================================================================

@router.post("/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """
    Generate optimal portfolio using arbiter model selection
    
    This endpoint uses the arbiter to select the best strategy from:
    - CAPM (Modern Portfolio Theory)
    - Elastic Net (Regularized Regression) 
    - LSTM (Neural Network)
    
    Selection is based on client risk profile and strategy performance metrics.
    """
    try:
        result = await portfolio_arbiter.optimize_portfolio(
            symbols=request.symbols,
            risk_score=request.risk_score,
            lookback_days=request.lookback_days
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Portfolio optimization failed: {str(e)}")

@router.post("/train-all-models")
async def train_all_models(
    background_tasks: BackgroundTasks,
    tickers: List[str] = Query(..., description="List of tickers to train models for"),
    lookback_days: int = Query(1260, description="Lookback period in days")
):
    """
    Train all models (LSTM, Elastic Net) in background
    """
    try:
        # Add training tasks to background
        background_tasks.add_task(
            lstm_predictor.train_models,
            tickers,
            lookback_days
        )
        background_tasks.add_task(
            elastic_net_predictor.train_models,
            tickers,
            lookback_days
        )
        
        return {
            "message": "Model training started in background",
            "tickers": tickers,
            "models": ["LSTM", "Elastic Net"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model training initialization failed: {str(e)}")

@router.get("/model-performance")
async def get_model_performance():
    """
    Get performance metrics for all models
    """
    try:
        performance = await portfolio_arbiter.get_model_performance()
        return performance
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get model performance: {str(e)}")

@router.get("/health")
async def portfolio_health_check():
    """
    Comprehensive health check for all portfolio optimization services
    """
    try:
        lstm_health = await lstm_predictor.health_check()
        elastic_net_health = await elastic_net_predictor.health_check()
        capm_health = await capm_calculator.health_check()
        
        all_healthy = (
            lstm_health.get("initialized", False) and
            elastic_net_health.get("initialized", False) and
            capm_health.get("initialized", False)
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": {
                "lstm": lstm_health,
                "elastic_net": elastic_net_health,
                "capm": capm_health,
                "arbiter": {"initialized": True}
            },
            "timestamp": "2024-01-01T00:00:00Z"  # You might want to use actual timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/available-models")
async def get_available_models():
    """
    Get list of available prediction models and their status
    """
    return {
        "models": [
            {
                "name": "LSTM",
                "description": "Long Short-Term Memory neural network for time series forecasting",
                "type": "machine_learning",
                "endpoint": "/api/v1/portfolio/lstm/optimize"
            },
            {
                "name": "Elastic Net", 
                "description": "Regularized linear regression with L1 + L2 penalties",
                "type": "machine_learning",
                "endpoint": "/api/v1/portfolio/elastic-net/optimize"
            },
            {
                "name": "CAPM",
                "description": "Capital Asset Pricing Model with Modern Portfolio Theory",
                "type": "financial_theory", 
                "endpoint": "/api/v1/portfolio/capm/optimize"
            },
            {
                "name": "Arbiter",
                "description": "Combined model selection using all available predictors",
                "type": "ensemble",
                "endpoint": "/api/v1/portfolio/optimize"
            }
        ]
    }

# ============================================================================
# Utility Endpoints
# ============================================================================

@router.post("/covariance-matrix")
async def get_covariance_matrix(
    tickers: List[str] = Query(..., description="List of tickers"),
    lookback_days: int = Query(252, description="Lookback period in days"),
    model: str = Query("capm", description="Model to use for covariance calculation (lstm, elastic-net, capm)")
):
    """
    Get covariance matrix for portfolio construction
    """
    try:
        if model.lower() == "lstm":
            cov_matrix = await lstm_predictor.compute_covariance_matrix(tickers, lookback_days)
        elif model.lower() == "elastic-net":
            cov_matrix = await elastic_net_predictor.compute_covariance_matrix(tickers, lookback_days)
        elif model.lower() == "capm":
            cov_matrix = await capm_calculator.compute_covariance_matrix(tickers, lookback_days)
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified. Use 'lstm', 'elastic-net', or 'capm'")
        
        if cov_matrix.empty:
            raise HTTPException(status_code=400, detail="Failed to compute covariance matrix")
            
        return {
            "tickers": tickers,
            "lookback_days": lookback_days,
            "model": model,
            "covariance_matrix": cov_matrix.to_dict()
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Covariance matrix computation failed: {str(e)}")

@router.get("/model-comparison")
async def compare_models(
    tickers: List[str] = Query(..., description="List of tickers to compare"),
    lookback_days: int = Query(252, description="Lookback period for comparison")
):
    """
    Compare expected returns from all models for given tickers
    """
    try:
        # Get expected returns from all models
        lstm_returns, _ = await lstm_predictor.generate_expected_returns(tickers)
        elastic_net_returns, _ = await elastic_net_predictor.generate_expected_returns(tickers)
        capm_returns, _ = await capm_calculator.generate_expected_returns(tickers)
        
        comparison = {}
        for ticker in tickers:
            lstm_er = next((er for er in lstm_returns if er.symbol == ticker), None)
            elastic_net_er = next((er for er in elastic_net_returns if er.symbol == ticker), None)
            capm_er = next((er for er in capm_returns if er.symbol == ticker), None)
            
            comparison[ticker] = {
                "lstm": lstm_er.lstm_expected_return if lstm_er else 0.0,
                "elastic_net": elastic_net_er.elastic_net_expected_return if elastic_net_er else 0.0,
                "capm": capm_er.capm_expected_return if capm_er else 0.0,
                "average": (
                    (lstm_er.lstm_expected_return if lstm_er else 0.0) +
                    (elastic_net_er.elastic_net_expected_return if elastic_net_er else 0.0) +
                    (capm_er.capm_expected_return if capm_er else 0.0)
                ) / 3 if any([lstm_er, elastic_net_er, capm_er]) else 0.0
            }
        
        return {
            "tickers": tickers,
            "lookback_days": lookback_days,
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model comparison failed: {str(e)}")

@router.get("/strategy-recommendation/{risk_score}")
async def get_strategy_recommendation(risk_score: int):
    """
    Get strategy recommendation based on risk score
    """
    try:
        weights = portfolio_arbiter.arbiter_model.get_weights(risk_score)
        
        # Determine strategy focus based on weights
        if weights['sharpe'] > 0.5:
            focus = "Risk-Adjusted Returns (Sharpe Ratio)"
            recommended_strategies = ["CAPM", "Elastic Net"]
        elif weights['alpha'] > 0.5:
            focus = "Manager Skill (Alpha Generation)"
            recommended_strategies = ["LSTM", "Elastic Net"]
        else:
            focus = "Absolute Returns (Excess Return)"
            recommended_strategies = ["LSTM", "CAPM"]
        
        return {
            "risk_score": risk_score,
            "focus": focus,
            "recommended_strategies": recommended_strategies,
            "weight_breakdown": weights,
            "description": f"For risk score {risk_score}, the arbiter prioritizes {focus.lower()}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))