from typing import Dict, Any
from app.models.schemas import RiskProfileResponse, RiskLevel

class RiskProfilerService:
    def __init__(self):
        self.profiler = self._initialize_profiler()
    
    def _initialize_profiler(self):
        # Your optimized risk profiler code here
        from app.utils.risk_profiler_optimized import OptimizedRiskProfiler
        return OptimizedRiskProfiler(use_mistral_api=False)
    
    def calculate_risk_score(self, tier1_answers: Dict, tier2_answers: Dict) -> RiskProfileResponse:
        result = self.profiler.calculate_risk_score(tier1_answers, tier2_answers)
        
        return RiskProfileResponse(
            risk_score=result['final_score'],
            risk_category=RiskLevel(result['category'].lower()),
            volatility_range=result['volatility_range'],
            tier1_total=result['tier1_total'],
            yes_count=result['yes_count'],
            magnitude=result['magnitude']
        )
    
    def get_tier1_questions(self):
        return self.profiler.tier1_questions
    
    def get_tier2_questions(self):
        return self.profiler.tier2_questions