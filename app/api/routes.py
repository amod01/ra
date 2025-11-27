from fastapi import APIRouter
from app.api.endpoints import risk_profiler, portfolio_optimizer, dashboard

api_router = APIRouter()

api_router.include_router(
    risk_profiler.router, 
    prefix="/risk", 
    tags=["Risk Profiling"]
)

api_router.include_router(
    portfolio_optimizer.router, 
    prefix="/portfolio", 
    tags=["Portfolio Optimization"]
)

api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["Dashboard"]
)