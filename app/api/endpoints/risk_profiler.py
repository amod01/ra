from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.risk_profiler import RiskProfilerService

router = APIRouter()
risk_service = RiskProfilerService()

@router.post("/assess")
async def assess_risk_profile(answers: Dict[str, Any]):
    """
    Assess user's risk profile based on questionnaire answers
    """
    try:
        tier1_answers = answers.get("tier1_answers", {})
        tier2_answers = answers.get("tier2_answers", {})
        
        result = risk_service.calculate_risk_score(tier1_answers, tier2_answers)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk assessment failed: {str(e)}")

@router.get("/questions")
async def get_risk_questions():
    """
    Get all risk assessment questions
    """
    try:
        return {
            "tier1_questions": risk_service.get_tier1_questions(),
            "tier2_questions": risk_service.get_tier2_questions()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get questions: {str(e)}")

@router.get("/recommendation/{risk_score}")
async def get_risk_recommendation(risk_score: int):
    """
    Get investment recommendation based on risk score
    """
    try:
        if risk_score not in range(1, 10):
            raise HTTPException(status_code=400, detail="Risk score must be 1-9")
        
        recommendations = {
            1: {"allocation": "90% Bonds, 10% Stocks", "volatility": "5-10%", "description": "Very Conservative"},
            2: {"allocation": "80% Bonds, 20% Stocks", "volatility": "8-12%", "description": "Conservative"},
            3: {"allocation": "70% Bonds, 30% Stocks", "volatility": "10-15%", "description": "Moderately Conservative"},
            4: {"allocation": "60% Bonds, 40% Stocks", "volatility": "12-18%", "description": "Balanced Conservative"},
            5: {"allocation": "50% Bonds, 50% Stocks", "volatility": "15-20%", "description": "Balanced"},
            6: {"allocation": "40% Bonds, 60% Stocks", "volatility": "18-25%", "description": "Balanced Aggressive"},
            7: {"allocation": "30% Bonds, 70% Stocks", "volatility": "20-30%", "description": "Aggressive"},
            8: {"allocation": "20% Bonds, 80% Stocks", "volatility": "25-35%", "description": "Very Aggressive"},
            9: {"allocation": "10% Bonds, 90% Stocks", "volatility": "30-40%", "description": "Maximum Growth"}
        }
        
        return {
            "risk_score": risk_score,
            "recommendation": recommendations[risk_score]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))