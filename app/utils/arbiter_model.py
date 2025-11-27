"""
Arbiter Model: Dynamic Strategy Selection Engine
Purpose: Selects the best prediction engine strategy based on client risk profile
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import sys
sys.path.append('.')
from Libraries.Backtester import PortfolioBacktester


class ArbiterModel:
    """
    The Arbiter uses a Dynamic Weighted Score to select the optimal strategy
    based on the client's validated Risk Score (1-9).

    Formula: ArbiterScore = (W_Sharpe × Sharpe) + (W_Alpha × Alpha) + (W_Excess × Excess)
    """

    def __init__(self):
        # Dynamic Weight Lookup Table (Risk Score 1-9)
        # Weights prioritize: Safety (Sharpe) for Averse, Skill (Alpha) for Neutral, Return (Excess) for Seeking
        # Based on documented Arbiter Logic (5) ArbiterLogic_3rdNov.md
        self.weight_lookup = {
            1: {'sharpe': 0.70, 'alpha': 0.25, 'excess': 0.05},  # Averse (Hardline)
            2: {'sharpe': 0.60, 'alpha': 0.35, 'excess': 0.05},  # Averse (Conservative)
            3: {'sharpe': 0.50, 'alpha': 0.45, 'excess': 0.05},  # Averse (Borderline)
            4: {'sharpe': 0.225, 'alpha': 0.55, 'excess': 0.225},  # Neutral (Cautious)
            5: {'sharpe': 0.15, 'alpha': 0.70, 'excess': 0.15},  # Neutral (True) - Max Alpha
            6: {'sharpe': 0.20, 'alpha': 0.60, 'excess': 0.20},  # Neutral (Aggressive)
            7: {'sharpe': 0.05, 'alpha': 0.45, 'excess': 0.50},  # Seeking (Growth)
            8: {'sharpe': 0.05, 'alpha': 0.35, 'excess': 0.60},  # Seeking (Aggressive)
            9: {'sharpe': 0.05, 'alpha': 0.25, 'excess': 0.70},  # Seeking (Hardline)
        }

    def get_weights(self, risk_score: int) -> Dict[str, float]:
        """Retrieve dynamic weights for a given risk score."""
        if risk_score not in range(1, 10):
            raise ValueError(f"Risk score must be 1-9, got {risk_score}")
        return self.weight_lookup[risk_score]

    def scale_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Scale a metric to 0-1 range for normalization."""
        if max_val == min_val:
            return 0.5  # Neutral if no variance
        return (value - min_val) / (max_val - min_val)

    def calculate_arbiter_score(
        self,
        risk_score: int,
        sharpe_ratio: float,
        jensen_alpha: float,
        excess_return: float,
        all_strategies_metrics: Dict[str, List[float]],
        r2_score: float = None,
        loss_metric: float = None,
        all_ml_metrics: Dict[str, List[float]] = None,
        backtest_reliability: float = None
    ) -> float:
        """
        Calculate the theoretical ArbiterScore for a single strategy.

        Formula: ArbiterScore = (0.40 × FinancialScore) + (0.40 × MachineScore) + (0.20 × BacktestScore)

        Args:
            risk_score: Client's validated 1-9 risk score
            sharpe_ratio: Strategy's Sharpe Ratio
            jensen_alpha: Strategy's Jensen's Alpha
            excess_return: Strategy's Excess Return (vs benchmark)
            all_strategies_metrics: Dict with 'sharpe', 'alpha', 'excess' lists for scaling
            r2_score: Model's R² score (optional, for ML models)
            loss_metric: Model's loss metric (optional, for ML models)
            all_ml_metrics: Dict with 'r2', 'loss' lists for scaling (optional)
            backtest_reliability: Backtesting Reliability Score (0-1, optional)

        Returns:
            ArbiterScore (0-1 scale)
        """
        # Get dynamic weights for financial metrics
        weights = self.get_weights(risk_score)

        # === FINANCIAL SCORE (100% weight within FS) ===
        # Scale metrics using min-max normalization across all strategies
        sharpe_scaled = self.scale_metric(
            sharpe_ratio,
            min(all_strategies_metrics['sharpe']),
            max(all_strategies_metrics['sharpe'])
        )
        alpha_scaled = self.scale_metric(
            jensen_alpha,
            min(all_strategies_metrics['alpha']),
            max(all_strategies_metrics['alpha'])
        )
        excess_scaled = self.scale_metric(
            excess_return,
            min(all_strategies_metrics['excess']),
            max(all_strategies_metrics['excess'])
        )

        # Calculate Financial Score (weighted by risk profile)
        financial_score = (
            weights['sharpe'] * sharpe_scaled +
            weights['alpha'] * alpha_scaled +
            weights['excess'] * excess_scaled
        )

        # === MACHINE SCORE (100% weight within MS) ===
        if r2_score is not None and loss_metric is not None and all_ml_metrics is not None:
            # R² score: Higher is better (scale normally, already positive)
            r2_scaled = self.scale_metric(
                r2_score,
                min(all_ml_metrics['r2']),
                max(all_ml_metrics['r2'])
            )

            # Loss metric: Unscaled MAAE (positive value in return units)
            # MAAE (Mean Absolute Asymmetrical Error) is in return units (e.g., 0.10 = 10% average asymmetrical error)
            # Example: If predictions have 50% average weighted error, MAAE = 0.50
            # Higher MAAE = worse performance (more error)
            # We invert the scaling: lower MAAE → higher scaled value (better)
            loss_scaled = self.scale_metric(
                loss_metric,  # Positive MAE value (e.g., 0.50 = 50% error)
                min(all_ml_metrics['loss']),  # Lowest MAE (best)
                max(all_ml_metrics['loss'])   # Highest MAE (worst)
            )
            # Invert: lower MAE should give higher score
            loss_scaled = 1.0 - loss_scaled

            # Machine Score: 50% R² + 50% MAAE
            # This ensures the model outperforms average (R² > 0) AND has low asymmetric error
            # MAAE is inverted (lower is better), so we subtract it
            machine_score = 0.50 * r2_scaled + 0.50 * loss_scaled
        else:
            # For non-ML models (CAPM), assume perfect ML quality
            machine_score = 1.0

        # === BACKTEST SCORE (reliability of historical validation) ===
        if backtest_reliability is not None:
            backtest_score = backtest_reliability
        else:
            # Default to neutral if no multi-year backtest performed
            backtest_score = 0.5

        # === FINAL ARBITER SCORE ===
        # 40% Financial + 40% Machine + 20% Backtest Reliability
        # Balanced approach: each component can be wrong at different intervals
        # Captures nuances across all dimensions without over-weighting any single aspect
        arbiter_score = 0.40 * financial_score + 0.40 * machine_score + 0.20 * backtest_score

        return arbiter_score

    def select_best_strategy(
        self,
        risk_score: int,
        strategies_performance: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Select the best strategy from multiple prediction engines.

        Args:
            risk_score: Client's validated 1-9 risk score
            strategies_performance: Dict of {strategy_name: {sharpe, alpha, excess, r2, loss, backtest_reliability}}

        Returns:
            (best_strategy_name, best_score, weights_used)
        """
        # Collect all metrics for scaling
        all_metrics = {
            'sharpe': [s['sharpe'] for s in strategies_performance.values()],
            'alpha': [s['alpha'] for s in strategies_performance.values()],
            'excess': [s['excess'] for s in strategies_performance.values()]
        }

        # Collect ML metrics (if available)
        all_ml_metrics = {
            'r2': [s.get('r2', 1.0) for s in strategies_performance.values()],
            'loss': [s.get('loss', 0.0) for s in strategies_performance.values()]
        }

        # Calculate scores for each strategy
        scores = {}
        for strategy_name, metrics in strategies_performance.items():
            scores[strategy_name] = self.calculate_arbiter_score(
                risk_score,
                metrics['sharpe'],
                metrics['alpha'],
                metrics['excess'],
                all_metrics,
                r2_score=metrics.get('r2'),
                loss_metric=metrics.get('loss'),
                all_ml_metrics=all_ml_metrics,
                backtest_reliability=metrics.get('backtest_reliability')
            )

        # Select winner
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]
        weights_used = self.get_weights(risk_score)

        return best_strategy, best_score, weights_used

    def calculate_net_score(
        self,
        risk_score: int,
        portfolio_weights: np.ndarray,
        returns_data: pd.DataFrame,
        risk_free_rate: float,
        market_returns: pd.Series,
        frictional_costs: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate ArbiterScore_Net accounting for real-world frictional costs.

        Args:
            risk_score: Client's 1-9 risk score
            portfolio_weights: Optimized portfolio weights
            returns_data: Historical returns DataFrame
            risk_free_rate: Risk-free rate (annualized)
            market_returns: Market benchmark returns
            frictional_costs: Dict with 'management_fee', 'trading_spread', 'tax_drag'

        Returns:
            (arbiter_score_net, net_metrics)
        """
        # Calculate portfolio returns
        portfolio_returns = (returns_data * portfolio_weights).sum(axis=1)

        # Apply frictional costs (annualized)
        total_friction = sum(frictional_costs.values())
        net_returns = portfolio_returns - (total_friction / 252)  # Daily adjustment

        # Recalculate metrics with net returns
        sharpe_net = self._calculate_sharpe(net_returns, risk_free_rate)
        alpha_net = self._calculate_alpha(net_returns, market_returns, risk_free_rate)
        excess_net = (net_returns.mean() * 252) - (market_returns.mean() * 252)

        # Get all strategies metrics for scaling (would need to be passed in real implementation)
        all_metrics = {
            'sharpe': [sharpe_net],  # Simplified - would compare against other strategies
            'alpha': [alpha_net],
            'excess': [excess_net]
        }

        # Calculate net score
        arbiter_score_net = self.calculate_arbiter_score(
            risk_score,
            sharpe_net,
            alpha_net,
            excess_net,
            all_metrics
        )

        net_metrics = {
            'sharpe_net': sharpe_net,
            'alpha_net': alpha_net,
            'excess_net': excess_net,
            'total_friction': total_friction
        }

        return arbiter_score_net, net_metrics

    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - (rf_rate / 252)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    def _calculate_alpha(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        rf_rate: float
    ) -> float:
        """Calculate Jensen's Alpha."""
        excess_portfolio = returns - (rf_rate / 252)
        excess_market = market_returns - (rf_rate / 252)

        # Calculate beta
        covariance = np.cov(excess_portfolio, excess_market)[0, 1]
        market_variance = np.var(excess_market)
        beta = covariance / market_variance if market_variance != 0 else 1.0

        # Calculate alpha (annualized)
        alpha = (excess_portfolio.mean() - beta * excess_market.mean()) * 252
        return alpha

    def backtest_and_score_strategies(
        self,
        risk_score: int,
        strategies_weights: Dict[str, Dict[str, float]],
        strategies_projections: Dict[str, float] = None,
        lookback_days: int = 252,
        include_costs: bool = False
    ) -> Tuple[str, float, Dict[str, float], pd.DataFrame]:
        """
        Backtest multiple strategies and select the best one using real historical data.

        Args:
            risk_score: Client's validated 1-9 risk score
            strategies_weights: Dict of {strategy_name: {ticker: weight}}
            strategies_projections: Dict of {strategy_name: expected_annual_return} (optional)
            lookback_days: Number of days to backtest
            include_costs: Whether to include frictional costs

        Returns:
            (best_strategy_name, best_arbiter_score, weights_used, all_backtest_results)
        """
        backtester = PortfolioBacktester(risk_free_rate=0.03)

        # Backtest all strategies
        backtest_results = []
        strategies_performance = {}

        print("\n" + "="*60)
        print("BACKTESTING STRATEGIES")
        print("="*60)

        for strategy_name, weights in strategies_weights.items():
            print(f"\nBacktesting {strategy_name}...")

            # Define frictional costs
            costs = {
                'management_fee': 0.005,  # 0.5% annual
                'trading_spread': 0.001,  # 0.1% annual
                'tax_drag': 0.002         # 0.2% annual
            } if include_costs else None

            metrics = backtester.backtest_portfolio(
                weights,
                lookback_days=lookback_days,
                include_costs=include_costs,
                frictional_costs=costs
            )

            # Get multi-year reliability score
            projection = strategies_projections.get(strategy_name) if strategies_projections else None
            reliability_metrics = backtester.rolling_window_backtest(
                weights,
                num_years=5,
                window_days=252,
                current_year_projection=projection
            )

            # Generate detailed reliability report with tables and charts
            backtester.generate_backtest_reliability_report(
                weights,
                strategy_name,
                current_year_projection=projection,
                output_dir=f'Results/{strategy_name}'
            )

            if metrics:
                # Store for arbiter scoring
                strategies_performance[strategy_name] = {
                    'sharpe': metrics['sharpe_ratio'],
                    'alpha': metrics['jensens_alpha'],
                    'excess': metrics['excess_return'],
                    'backtest_reliability': reliability_metrics['reliability_score'],
                    'r2': metrics.get('kfold_best_r2'), # Added kfold_best_r2
                    'loss': metrics.get('kfold_best_test_maae') # Added kfold_best_test_maae
                }

                # Store full results
                metrics['strategy'] = strategy_name
                metrics['reliability_score'] = reliability_metrics['reliability_score']
                metrics['consistency'] = reliability_metrics['consistency_assessment']
                backtest_results.append(metrics)

                print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
                print(f"  Alpha: {metrics['jensens_alpha']:.2%}")
                print(f"  Excess Return: {metrics['excess_return']:.2%}")
                print(f"  Reliability: {reliability_metrics['reliability_score']:.2f} ({reliability_metrics['consistency_assessment']})")

        # Select best strategy using arbiter model
        best_strategy, best_score, weights_used = self.select_best_strategy(
            risk_score,
            strategies_performance
        )

        results_df = pd.DataFrame(backtest_results)

        return best_strategy, best_score, weights_used, results_df


# Example usage demonstration
if __name__ == "__main__":
    arbiter = ArbiterModel()

    # Example: Compare CAPM, Elastic Net, LSTM strategies
    # Note: In a real scenario, these metrics would come from backtesting
    # and ML model evaluation results.
    strategies = {
        'CAPM': {'sharpe': 1.2, 'alpha': 0.03, 'excess': 0.05, 'r2': 0.9, 'loss': 0.05},
        'ElasticNet': {'sharpe': 1.5, 'alpha': 0.05, 'excess': 0.07, 'r2': 0.85, 'loss': 0.06},
        'LSTM': {'sharpe': 1.1, 'alpha': 0.06, 'excess': 0.08, 'r2': 0.88, 'loss': 0.055}
    }

    # Test for different risk profiles
    for score in [2, 5, 8]:
        best, score_val, weights = arbiter.select_best_strategy(score, strategies)
        print(f"\nRisk Score {score}: Best Strategy = {best} (Score: {score_val:.3f})")
        print(f"Weights Used: {weights}")