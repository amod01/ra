"""
Week 1 Implementation: CAPM Engine with Modern Portfolio Theory (MPT)
- CAPM-based return estimation
- Beta calculation
- Covariance matrix computation
- Mean-Variance Optimization (MVO)
- Efficient Frontier generation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from app.utils.data_loader import get_features_for_ticker
from matplotlib.lines import Line2D
import os

class CAPMEngine:
    """
    CAPM Engine for calculating expected returns using the Capital Asset Pricing Model.
    Implements Modern Portfolio Theory for portfolio optimization.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize CAPM Engine.

        Args:
            risk_free_rate: Annual risk-free rate (default 3%)
        """
        self.risk_free_rate = risk_free_rate
        self.market_ticker = 'SPY'  # S&P 500 as market proxy
        self.betas = {}
        self.expected_returns = {}
        self.covariance_matrix = None
        self.tickers = []
        self.output_dir = 'Results/CAPM' # Add output_dir attribute for CAPM folder routing

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_returns(self, ticker: str, period_days: int = 252) -> pd.Series:
        """
        Calculate historical returns for a ticker.

        Args:
            ticker: Stock ticker symbol
            period_days: Number of days to look back

        Returns:
            Series of daily returns
        """
        data = get_features_for_ticker(ticker)
        if data.empty or 'Daily_Return' not in data.columns:
            return pd.Series()

        return data['Daily_Return'].tail(period_days)

    def calculate_beta(self, ticker: str, lookback_days: int = 252) -> float:
        """
        Calculate beta for a stock relative to the market.

        Beta = Cov(Stock, Market) / Var(Market)

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days for calculation

        Returns:
            Beta value
        """
        stock_returns = self.calculate_returns(ticker, lookback_days)
        market_returns = self.calculate_returns(self.market_ticker, lookback_days)

        if stock_returns.empty or market_returns.empty:
            print(f"Warning: Could not calculate beta for {ticker}")
            return 1.0  # Default to market beta

        # Align the data by date
        combined = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if len(combined) < 30:
            print(f"Warning: Insufficient data for {ticker}")
            return 1.0

        # Calculate beta using covariance
        covariance = combined['stock'].cov(combined['market'])
        market_variance = combined['market'].var()

        beta = covariance / market_variance if market_variance != 0 else 1.0

        return beta

    def estimate_market_return(self, lookback_days: int = 252) -> float:
        """
        Estimate expected market return based on historical average.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Annualized expected market return
        """
        market_returns = self.calculate_returns(self.market_ticker, lookback_days)

        if market_returns.empty:
            print("Warning: Could not calculate market return, using default")
            return 0.10  # Default 10% annual return

        # Annualize the daily returns
        daily_avg = market_returns.mean()
        annualized_return = (1 + daily_avg) ** 252 - 1

        return annualized_return

    def calculate_expected_return_capm(self, ticker: str, beta: float, market_return: float) -> float:
        """
        Calculate expected return using CAPM formula.

        E[R_i] = R_f + β_i * (E[R_m] - R_f)

        Args:
            ticker: Stock ticker symbol
            beta: Stock's beta
            market_return: Expected market return (annual)

        Returns:
            Expected annual return
        """
        expected_return_annual = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        return expected_return_annual

    def calculate_alpha(self, ticker: str, actual_return: float, beta: float, market_return: float) -> float:
        """
        Calculate Jensen's Alpha for a stock.

        Alpha = Actual Return - Expected Return (CAPM)
        Alpha = R_i - [R_f + β_i * (R_m - R_f)]

        Args:
            ticker: Stock ticker symbol
            actual_return: Actual historical return
            beta: Stock's beta
            market_return: Market return

        Returns:
            Alpha (excess return above CAPM prediction)
        """
        expected_return = self.calculate_expected_return_capm(ticker, beta, market_return)
        alpha = actual_return - expected_return
        return alpha

    def calculate_actual_return(self, ticker: str, lookback_days: int = 252) -> float:
        """
        Calculate actual annualized return for a ticker.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back

        Returns:
            Annualized actual return
        """
        returns = self.calculate_returns(ticker, lookback_days)

        if returns.empty:
            print(f"Warning: Could not calculate actual return for {ticker}")
            return 0.0

        # Annualize the daily returns
        daily_avg = returns.mean()
        annualized_return = (1 + daily_avg) ** 252 - 1

        return annualized_return

    def compute_covariance_matrix(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """
        Compute the covariance matrix for a list of tickers.

        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days for calculation

        Returns:
            Covariance matrix as DataFrame (monthly scale)
        """
        returns_data = {}

        for ticker in tickers:
            returns = self.calculate_returns(ticker, lookback_days)
            if not returns.empty:
                returns_data[ticker] = returns

        if not returns_data:
            print("Warning: No return data available")
            return pd.DataFrame()

        # Create DataFrame and align dates
        returns_df = pd.DataFrame(returns_data).dropna()

        # Calculate annual covariance matrix (252 trading days per year)
        cov_matrix = returns_df.cov() * 252

        return cov_matrix

    def prepare_portfolio_inputs(self, tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare expected returns and covariance matrix for portfolio optimization.

        Args:
            tickers: List of ticker symbols

        Returns:
            Tuple of (expected_returns_array, covariance_matrix_array)
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        self.tickers = tickers
        market_return = self.estimate_market_return()

        # Store detailed metrics
        self.capm_metrics = []

        # Calculate betas and expected returns (ANNUAL)
        for ticker in tickers:
            beta = self.calculate_beta(ticker)
            expected_return = self.calculate_expected_return_capm(ticker, beta, market_return)
            actual_return = self.calculate_actual_return(ticker)
            alpha = self.calculate_alpha(ticker, actual_return, beta, market_return)

            # Calculate prediction accuracy metrics
            stock_returns = self.calculate_returns(ticker)
            market_returns = self.calculate_returns(self.market_ticker)

            if not stock_returns.empty and not market_returns.empty:
                combined = pd.DataFrame({
                    'stock': stock_returns,
                    'market': market_returns
                }).dropna()

                # CAPM predictions
                excess_market = combined['market'] - (self.risk_free_rate / 252)
                predicted_excess = beta * excess_market
                predicted_returns = predicted_excess + (self.risk_free_rate / 252)
                actual_returns = combined['stock']

                # Calculate daily metrics
                mse_daily = mean_squared_error(actual_returns, predicted_returns)
                mae_daily = mean_absolute_error(actual_returns, predicted_returns)
                r2_daily = r2_score(actual_returns, predicted_returns)

                # Annualize metrics
                # MSE scales quadratically: MSE_annual = MSE_daily * 252^2
                mse_annual = mse_daily * (252 ** 2)
                # MAE scales linearly: MAE_annual = MAE_daily * 252
                mae_annual = mae_daily * 252
                # R² is scale-invariant (same for daily and annual)
                r2_annual = r2_daily

                self.capm_metrics.append({
                    'ticker': ticker,
                    'beta': beta,
                    'expected_return_annual': expected_return,
                    'actual_return_annual': actual_return,
                    'alpha': alpha,
                    'mse_daily': mse_daily,
                    'mae_daily': mae_daily,
                    'r2_daily': r2_daily,
                    'mse_annual': mse_annual,
                    'mae_annual': mae_annual,
                    'r2_annual': r2_annual,
                    'n_observations': len(combined),
                    'market_return': market_return,
                    'risk_free_rate': self.risk_free_rate
                })

            self.betas[ticker] = beta
            self.expected_returns[ticker] = expected_return

        # Compute covariance matrix
        self.covariance_matrix = self.compute_covariance_matrix(tickers)

        # Convert to arrays for optimization
        expected_returns_array = np.array([self.expected_returns[t] for t in tickers])
        cov_matrix_array = self.covariance_matrix.values

        # Save detailed CAPM metrics and expected returns to CSV
        metrics_df = self.get_metrics_dataframe()

        # Add portfolio summary row
        # This part requires information about the selected portfolio weights and performance.
        # For now, we'll create a placeholder or assume it's available from optimization results.
        # In a real scenario, you'd pass the selected weights and performance metrics here.
        # For demonstration, let's assume we're adding a summary for the Max Sharpe portfolio.
        if self.tickers:
            # Re-optimize to get max sharpe weights if not already available or to be sure
            optimizer_for_summary = MPTOptimizer(expected_returns_array, cov_matrix_array, self.tickers)
            max_sharpe_portfolio_summary = optimizer_for_summary.maximize_sharpe_ratio(self.risk_free_rate)
            weights_dict_selected = max_sharpe_portfolio_summary['weights']

            # Calculate portfolio metrics from weights
            portfolio_return, portfolio_volatility = optimizer_for_summary.portfolio_performance(
                np.array([weights_dict_selected.get(t, 0) for t in self.tickers])
            )
            portfolio_sharpe_ratio = optimizer_for_summary.portfolio_sharpe_ratio(
                np.array([weights_dict_selected.get(t, 0) for t in self.tickers]), self.risk_free_rate
            )

            portfolio_summary = pd.DataFrame([{
                'Ticker': 'PORTFOLIO',
                'Weight': sum(weights_dict_selected.values()), # Should be 1.0 if weights sum to 1
                'Expected_Return': portfolio_return,
                'Volatility': portfolio_volatility,
                'Sharpe_Ratio': portfolio_sharpe_ratio,
                # Placeholder metrics for portfolio, may need adjustment
                'Train_MAE': np.nan, # Placeholder
                'Val_MAE': np.nan,   # Placeholder
                'Train_R2': 1.0,     # CAPM is analytical, perfect fit for expected returns
                'Val_R2': 1.0,       # Placeholder
                'mse_daily': np.nan, # Placeholder, might need to be calculated if actual portfolio returns are available
                'mae_daily': np.nan, # Placeholder
                'r2_daily': np.nan,  # Placeholder
                'mse_annual': np.nan,
                'mae_annual': np.nan,
                'r2_annual': np.nan
            }])
            metrics_df = pd.concat([metrics_df, portfolio_summary], ignore_index=True)

        # Save detailed metrics
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(self.output_dir, 'capm_detailed_metrics.csv'), index=False)
        print(f"✓ Saved detailed CAPM metrics to '{os.path.join(self.output_dir, 'capm_detailed_metrics.csv')}'")

        # Save expected returns to CSV
        returns_df = pd.DataFrame({
            'Ticker': list(self.expected_returns.keys()),
            'Expected_Return': list(self.expected_returns.values())
        })
        returns_df.to_csv(os.path.join(self.output_dir, 'expected_returns_capm.csv'), index=False)
        print(f"✓ Saved expected returns to '{os.path.join(self.output_dir, 'expected_returns_capm.csv')}'")


        return expected_returns_array, cov_matrix_array

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get detailed CAPM metrics as DataFrame.

        Returns:
            DataFrame with CAPM metrics for all tickers
        """
        return pd.DataFrame(self.capm_metrics)


class MPTOptimizer:
    """
    Modern Portfolio Theory (MPT) Optimizer for constructing optimal portfolios.
    Implements Mean-Variance Optimization (MVO).
    """

    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray, tickers: List[str], risk_score: int = 5):
        """
        Initialize MPT Optimizer with dynamic regularization based on risk score.

        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            tickers: List of ticker symbols
            risk_score: Client's risk score (1-9) to determine lambda/gamma
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.risk_score = risk_score
        
        # Dynamic hyperparameter mapping based on risk score
        self.lambda_ridge, self.gamma_lasso = self._get_regularization_params(risk_score)
    
    def _get_regularization_params(self, risk_score: int) -> tuple:
        """
        Get lambda_ridge and gamma_lasso based on risk score.
        
        Risk Averse (1-3): Pure Ridge (gamma=0, increasing lambda)
        Risk Neutral (4-6): Elastic Net (both non-zero)
        Risk Seeking (7-9): Pure Lasso (lambda=0, increasing gamma)
        
        Args:
            risk_score: Client's risk score (1-9)
            
        Returns:
            (lambda_ridge, gamma_lasso)
        """
        if risk_score <= 3:  # Risk Averse - Pure Ridge
            # Higher lambda for more conservative (score 1), lower for moderate (score 3)
            lambda_ridge = 0.5 - (risk_score - 1) * 0.15  # 0.5, 0.35, 0.2
            gamma_lasso = 0.0
        elif risk_score <= 6:  # Risk Neutral - Elastic Net
            # Balanced penalties, moderate magnitude
            lambda_ridge = 0.15 - (risk_score - 4) * 0.025  # 0.15, 0.125, 0.1
            gamma_lasso = 0.05 + (risk_score - 4) * 0.025   # 0.05, 0.075, 0.1
        else:  # Risk Seeking - Pure Lasso
            # Higher gamma for more aggressive (score 9), lower for moderate (score 7)
            lambda_ridge = 0.0
            gamma_lasso = 0.2 + (risk_score - 7) * 0.15  # 0.2, 0.35, 0.5
        
        return lambda_ridge, gamma_lasso

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and volatility.

        Args:
            weights: Array of portfolio weights

        Returns:
            Tuple of (annual_return, annual_volatility)
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        return portfolio_return, portfolio_volatility

    def portfolio_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """
        Calculate Sharpe ratio for a portfolio.

        Args:
            weights: Array of portfolio weights
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        ret, vol = self.portfolio_performance(weights)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0
        return sharpe

    def calculate_portfolio_alpha(self, weights: np.ndarray, benchmark_return: float, 
                                   benchmark_beta: float = 1.0, risk_free_rate: float = 0.03) -> float:
        """
        Calculate portfolio alpha (excess return above benchmark adjusted for risk).

        Args:
            weights: Array of portfolio weights
            benchmark_return: Benchmark return (e.g., S&P 500)
            benchmark_beta: Portfolio beta relative to benchmark
            risk_free_rate: Risk-free rate

        Returns:
            Portfolio alpha
        """
        portfolio_return, _ = self.portfolio_performance(weights)

        # Expected return based on CAPM given portfolio's beta
        expected_return = risk_free_rate + benchmark_beta * (benchmark_return - risk_free_rate)

        # Alpha is the excess return above CAPM prediction
        alpha = portfolio_return - expected_return
        return alpha

    def calculate_mea(self, weights: np.ndarray, benchmark_return: float) -> float:
        """
        Calculate Maximum Excess Above (MEA) benchmark.

        MEA = Portfolio Return - Benchmark Return

        Args:
            weights: Array of portfolio weights
            benchmark_return: Benchmark return

        Returns:
            Excess return above benchmark
        """
        portfolio_return, _ = self.portfolio_performance(weights)
        mea = portfolio_return - benchmark_return
        return mea

    def minimize_volatility(self, target_return: float = None) -> Dict:
        """
        Find optimal portfolio using Elastic Net regularization.

        Objective: Maximize U = w^T μ - (1/2)λ(w^T Σ w) - γ Σ|w_i|

        Args:
            target_return: Target annual return (optional)

        Returns:
            Dictionary with optimal weights and performance metrics
        """
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })

        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Elastic Net Objective: Maximize utility = returns - ridge_penalty - lasso_penalty
        # We minimize the negative utility
        def objective(weights):
            # Expected return term
            portfolio_return = np.dot(weights, self.expected_returns)

            # Ridge penalty: (1/2) λ (w^T Σ w) - penalizes overall variance
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            ridge_penalty = 0.5 * self.lambda_ridge * portfolio_variance

            # Lasso penalty: γ Σ|w_i| - penalizes large individual weights
            lasso_penalty = self.gamma_lasso * np.sum(np.abs(weights))

            # Utility to maximize (so we minimize the negative)
            utility = portfolio_return - ridge_penalty - lasso_penalty

            return -utility  # Minimize negative utility = maximize utility

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            print("Warning: Elastic Net optimization did not converge for target return {}".format(target_return))

        optimal_weights = result.x
        ret, vol = self.portfolio_performance(optimal_weights)

        return {
            'weights': dict(zip(self.tickers, optimal_weights)),
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': self.portfolio_sharpe_ratio(optimal_weights),
            'alpha': None,  # Will be calculated when benchmark is available
            'mea': None,  # Will be calculated when benchmark is available
            'lambda_ridge': self.lambda_ridge,
            'gamma_lasso': self.gamma_lasso
        }


    def maximize_sharpe_ratio(self, risk_free_rate: float = 0.03) -> Dict:
        """
        Find the portfolio with maximum Sharpe ratio.

        Args:
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with optimal weights and performance metrics
        """
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            return -self.portfolio_sharpe_ratio(weights, risk_free_rate)

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            print("Warning: Max Sharpe optimization did not converge")

        optimal_weights = result.x
        ret, vol = self.portfolio_performance(optimal_weights)
        sharpe = self.portfolio_sharpe_ratio(optimal_weights, risk_free_rate)

        return {
            'weights': dict(zip(self.tickers, optimal_weights)),
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'alpha': None,  # Will be calculated when benchmark is available
            'mea': None  # Will be calculated when benchmark is available
        }

    def generate_efficient_frontier(self, num_points: int = 99) -> pd.DataFrame:
        """
        Generate the efficient frontier with exactly num_points portfolios.
        
        Portfolio 1: Minimum Variance Portfolio (MVP) - absolute lowest volatility
        Portfolio num_points: Maximum Return Portfolio - highest expected return
        Portfolios 2 to num_points-1: Maximize return (w^T μ) for evenly-spaced volatility targets (w^T Σ w = target_vol²)
        
        This simplified approach focuses on RISK (volatility) without regularization complexity.

        Args:
            num_points: Number of portfolios to generate (default 99 for 9 risk bands × 11 portfolios, use 99 for risk bands)

        Returns:
            DataFrame with frontier portfolios
        """
        # Portfolio 1: ABSOLUTE minimum variance (no return constraint)
        constraints_mvp = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        def min_variance_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            return portfolio_variance
        
        result_mvp = minimize(
            min_variance_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_mvp
        )
        
        mvp_weights = result_mvp.x
        mvp_return, mvp_vol = self.portfolio_performance(mvp_weights)
        
        # Portfolio num_points: Maximum return (100% in highest expected return asset)
        max_return_idx = np.argmax(self.expected_returns)
        max_return = self.expected_returns[max_return_idx]
        max_return_weights = np.zeros(self.n_assets)
        max_return_weights[max_return_idx] = 1.0
        max_return_vol = np.sqrt(self.covariance_matrix[max_return_idx, max_return_idx])

        # Generate target VOLATILITIES linearly spaced from min_var_vol to max_ret_vol
        target_vols = np.linspace(mvp_vol, max_return_vol, num_points)

        frontier_portfolios = []

        for i, target_vol in enumerate(target_vols):
            if i == 0:
                # Portfolio 1: MVP
                portfolio = {
                    'return': mvp_return,
                    'volatility': mvp_vol,
                    'sharpe_ratio': self.portfolio_sharpe_ratio(mvp_weights),
                    'weights': dict(zip(self.tickers, mvp_weights))
                }
            elif i == num_points - 1:
                # Portfolio num_points: Max Return
                portfolio = {
                    'return': max_return,
                    'volatility': max_return_vol,
                    'sharpe_ratio': (max_return - 0.03) / max_return_vol if max_return_vol > 0 else 0,
                    'weights': dict(zip(self.tickers, max_return_weights))
                }
            else:
                # Portfolios 2 to num_points-1: MAXIMIZE return for target volatility
                # Objective: max w^T μ
                # Constraint: w^T Σ w ≤ target_vol² (inequality to allow solver flexibility)
                
                # Use inequality constraint instead of strict equality for better convergence
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x, tv=target_vol: tv**2 - np.dot(x.T, np.dot(self.covariance_matrix, x))}
                ]
                
                def objective(weights):
                    # Maximize return = minimize negative return
                    portfolio_return = np.dot(weights, self.expected_returns)
                    # Add small penalty to push towards target volatility
                    portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
                    vol_penalty = 0.001 * (np.sqrt(portfolio_variance) - target_vol)**2
                    return -portfolio_return + vol_penalty
                
                # Try with better initial guess: interpolate between MVP and max return
                alpha = (target_vol - mvp_vol) / (max_return_vol - mvp_vol) if (max_return_vol - mvp_vol) > 0 else 0
                alpha = np.clip(alpha, 0, 1)
                initial_guess = (1 - alpha) * mvp_weights + alpha * max_return_weights
                initial_guess /= initial_guess.sum()
                
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-10}
                )
                
                # Check if result is valid and monotonically increasing
                if result.success:
                    opt_weights = result.x
                    ret, vol = self.portfolio_performance(opt_weights)
                    
                    # Verify monotonicity: return should increase with volatility
                    if i > 1 and ret < frontier_portfolios[i-1]['return']:
                        # If return decreased, use interpolation instead
                        use_interpolation = True
                    else:
                        use_interpolation = False
                        portfolio = {
                            'return': ret,
                            'volatility': vol,
                            'sharpe_ratio': self.portfolio_sharpe_ratio(opt_weights),
                            'weights': dict(zip(self.tickers, opt_weights))
                        }
                else:
                    use_interpolation = True
                
                if use_interpolation:
                    # Fallback: linear interpolation between MVP and max return
                    alpha = (target_vol - mvp_vol) / (max_return_vol - mvp_vol) if (max_return_vol - mvp_vol) > 0 else 0
                    alpha = np.clip(alpha, 0, 1)
                    interp_weights = (1 - alpha) * mvp_weights + alpha * max_return_weights
                    interp_weights /= interp_weights.sum()
                    ret, vol = self.portfolio_performance(interp_weights)
                    portfolio = {
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': self.portfolio_sharpe_ratio(interp_weights),
                        'weights': dict(zip(self.tickers, interp_weights))
                    }
            
            frontier_portfolios.append({
                'return': portfolio['return'],
                'volatility': portfolio['volatility'],
                'sharpe_ratio': portfolio['sharpe_ratio'],
                'weights': portfolio['weights'],
                'alpha': None,
                'mea': None,
                'portfolio_number': i + 1
            })

        frontier_df = pd.DataFrame(frontier_portfolios)

        return frontier_df


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    mvp: Dict,
    max_sharpe: Dict,
    benchmark_return: float,
    benchmark_volatility: float,
    max_return: Dict,
    lower_bound: Dict = None,
    selected: Dict = None,
    upper_bound: Dict = None,
    risk_score: int = None
):
    """
    Generate efficient frontier visualization with confidence band markers.

    Args:
        frontier_df: DataFrame with efficient frontier portfolios
        mvp: Minimum Variance Portfolio dict with 'return' and 'volatility'
        max_sharpe: Maximum Sharpe Portfolio dict with 'return' and 'volatility'
        benchmark_return: Benchmark expected return
        benchmark_volatility: Benchmark volatility
        max_return: Maximum Return Portfolio dict with 'return' and 'volatility'
        lower_bound: Lower bound portfolio (1st in risk band)
        selected: Selected portfolio (6th in risk band)
        upper_bound: Upper bound portfolio (11th in risk band)
        risk_score: Client's risk score (1-9) for legend display
    """
    plt.figure(figsize=(14, 9))

    # Plot efficient frontier
    plt.plot(frontier_df['volatility'], frontier_df['return'], 'b-', linewidth=2, label='Efficient Frontier')

    # Mark minimum variance portfolio
    plt.scatter(mvp['volatility'], mvp['return'], color='green', s=250, marker='o', 
                label=f"Min Variance\nR: {mvp['return']:.2%}, Vol: {mvp['volatility']:.2%}", 
                zorder=3, edgecolors='black', linewidths=1.5)

    # Mark max sharpe portfolio
    plt.scatter(max_sharpe['volatility'], max_sharpe['return'], color='blue', s=250, marker='D', 
                label=f"Max Sharpe\nR: {max_sharpe['return']:.2%}, Vol: {max_sharpe['volatility']:.2%}", 
                zorder=3, edgecolors='black', linewidths=1.5)

    # Mark maximum return portfolio if provided
    if max_return:
        plt.scatter(max_return['volatility'], max_return['return'], color='purple', s=250, marker='^', 
                    label=f"Max Return\nR: {max_return['return']:.2%}, Vol: {max_return['volatility']:.2%}", 
                    zorder=3, edgecolors='black', linewidths=1.5)

    # Mark benchmark (SPY)
    plt.scatter(benchmark_volatility, benchmark_return, color='orange', s=250, marker='D', 
                label=f"Benchmark (SPY)\nR: {benchmark_return:.2%}, Vol: {benchmark_volatility:.2%}", 
                zorder=3, edgecolors='black', linewidths=1.5)

    # Mark confidence band portfolios if provided
    if lower_bound:
        plt.scatter(lower_bound['volatility'], lower_bound['return'], color='cyan', s=250, marker='s', 
                    label=f"Lower Bound (1st)\nR: {lower_bound['return']:.2%}, Vol: {lower_bound['volatility']:.2%}", 
                    zorder=4, edgecolors='black', linewidths=1.5)

    if upper_bound:
        plt.scatter(upper_bound['volatility'], upper_bound['return'], color='magenta', s=250, marker='s', 
                    label=f"Upper Bound (11th)\nR: {upper_bound['return']:.2%}, Vol: {upper_bound['volatility']:.2%}", 
                    zorder=4, edgecolors='black', linewidths=1.5)

    if selected:
        plt.scatter(selected['volatility'], selected['return'], color='red', s=400, marker='*', 
                    label=f"★ SELECTED (6th - Winner)\nR: {selected['return']:.2%}, Vol: {selected['volatility']:.2%}", 
                    zorder=5, edgecolors='black', linewidths=2)

    # Add labels and title
    plt.xlabel('Monthly Volatility (Standard Deviation)', fontsize=13, fontweight='bold')
    plt.ylabel('Expected Monthly Return', fontsize=13, fontweight='bold')
    plt.title('Efficient Frontier - CAPM Engine with MPT (Monthly Projections)\nKey Portfolios Highlighted', fontsize=15, fontweight='bold')

    # Add legend elements matching actual plot markers and colors
    selected_label = f'Selected (6th) - Risk Score {risk_score}' if risk_score else 'Selected (6th)'

    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Efficient Frontier'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Min Variance', linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Max Sharpe', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Max Return', linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Benchmark (SPY)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Lower Bound (1st)', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, 
               markeredgecolor='black', markeredgewidth=2, label=selected_label, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta', markersize=10, 
               markeredgecolor='black', markeredgewidth=1.5, label='Upper Bound (11th)', linestyle='None')
    ]

    plt.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Format axes as percentages
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Save the plot
    plt.tight_layout()
    os.makedirs(self.output_dir, exist_ok=True)
    plot_path_png = os.path.join(self.output_dir, 'efficient_frontier.png')
    plot_path_pdf = os.path.join(self.output_dir, 'efficient_frontier.pdf')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\nEfficient frontier plot saved as '{plot_path_png}' and '{plot_path_pdf}'")
    plt.close()


def demo_capm_engine():
    """
    Demonstration of CAPM Engine and MPT Optimizer.
    """
    print("=" * 80)
    print("WEEK 1: CAPM Engine with Modern Portfolio Theory (MPT)")
    print("=" * 80)

    # Select a subset of tickers for demonstration
    demo_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'PG', 'SPY']

    # Check which tickers have data
    available_tickers = [t for t in demo_tickers if t in cleaned_financial_data and not cleaned_financial_data[t].empty]

    if len(available_tickers) < 3:
        print("Error: Insufficient data available. Please run Library_DataPipelines.py first.")
        return

    print(f"\nUsing {len(available_tickers)} tickers: {available_tickers}\n")

    # Initialize CAPM Engine
    capm = CAPMEngine(risk_free_rate=0.03)

    # Prepare portfolio inputs
    expected_returns, cov_matrix = capm.prepare_portfolio_inputs(available_tickers)

    # Initialize MPT Optimizer
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    # Dynamically adjust lambda and gamma based on risk score
    # This is a simplified example; a real implementation would have a more nuanced mapping
    hypothetical_risk_score = 5 # Example risk score
    
    if hypothetical_risk_score <= 3: # Risk-Averse
        lambda_ridge = 0.5
        gamma_lasso = 0.05
    elif hypothetical_risk_score >= 7: # Risk-Seeking
        lambda_ridge = 0.05
        gamma_lasso = 0.5
    else: # Risk-Neutral
        lambda_ridge = 0.1
        gamma_lasso = 0.1

    optimizer = MPTOptimizer(expected_returns, cov_matrix, available_tickers, lambda_ridge=lambda_ridge, gamma_lasso=gamma_lasso)

    # Get benchmark return (SPY)
    benchmark_return = capm.estimate_market_return()

    # 1. Minimum Variance Portfolio (using Elastic Net objective)
    print("\n1. Minimum Variance Portfolio (MVP) with Elastic Net:")
    print("-" * 80)
    mvp = optimizer.minimize_volatility()
    mvp_weights_array = np.array([mvp['weights'].get(t, 0) for t in available_tickers])
    mvp['alpha'] = optimizer.calculate_portfolio_alpha(mvp_weights_array, benchmark_return)
    mvp['mea'] = optimizer.calculate_mea(mvp_weights_array, benchmark_return)

    print(f"Expected Return: {mvp['return']:.2%}")
    print(f"Volatility: {mvp['volatility']:.2%}")
    print(f"Sharpe Ratio: {mvp['sharpe_ratio']:.3f}")
    print(f"Alpha: {mvp['alpha']:.2%}")
    print(f"MEA (vs Benchmark): {mvp['mea']:.2%}")
    print(f"Lambda (Ridge): {mvp['lambda_ridge']:.3f}, Gamma (Lasso): {mvp['gamma_lasso']:.3f}")
    print("\nWeights:")
    for ticker, weight in mvp['weights'].items():
        if weight > 0.01:  # Only show significant weights
            print(f"  {ticker}: {weight:.2%}")

    # 2. Maximum Sharpe Ratio Portfolio (standard MVO for comparison, not using Elastic Net here)
    print("\n2. Maximum Sharpe Ratio Portfolio (Standard MVO):")
    print("-" * 80)
    max_sharpe = optimizer.maximize_sharpe_ratio()
    max_sharpe_weights_array = np.array([max_sharpe['weights'].get(t, 0) for t in available_tickers])
    max_sharpe['alpha'] = optimizer.calculate_portfolio_alpha(max_sharpe_weights_array, benchmark_return)
    max_sharpe['mea'] = optimizer.calculate_mea(max_sharpe_weights_array, benchmark_return)

    print(f"Expected Return: {max_sharpe['return']:.2%}")
    print(f"Volatility: {max_sharpe['volatility']:.2%}")
    print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
    print(f"Alpha: {max_sharpe['alpha']:.2%}")
    print(f"MEA (vs Benchmark): {max_sharpe['mea']:.2%}")
    print("\nWeights:")
    for ticker, weight in max_sharpe['weights'].items():
        if weight > 0.01:
            print(f"  {ticker}: {weight:.2%}")

    # 3. Generate Efficient Frontier (using Elastic Net objective)
    print("\n3. Efficient Frontier (with Elastic Net):")
    print("-" * 80)
    frontier = optimizer.generate_efficient_frontier(num_points=20)

    # Calculate alpha and MEA for frontier portfolios
    for i, portfolio in enumerate(frontier.to_dict('records')):
        weights_array = np.array([portfolio['weights'].get(t, 0) for t in available_tickers])
        frontier.at[i, 'alpha'] = optimizer.calculate_portfolio_alpha(weights_array, benchmark_return)
        frontier.at[i, 'mea'] = optimizer.calculate_mea(weights_array, benchmark_return)

    print(f"Generated {len(frontier)} optimal portfolios")
    print("\nSample portfolios from the frontier:")
    print(frontier[['return', 'volatility', 'sharpe_ratio', 'alpha', 'mea', 'lambda_ridge', 'gamma_lasso']].head(10).to_string(index=False))

    # 4. Plot Efficient Frontier
    print("\n4. Efficient Frontier Visualization:")
    print("-" * 80)

    # Calculate benchmark volatility for plotting
    benchmark_returns = capm.calculate_returns('SPY')
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    # Get the maximum return portfolio from the frontier (assuming it's the last point)
    # Note: The "Max Return" portfolio on the frontier is the one with the highest expected return,
    # not necessarily the one that maximizes return without considering risk.
    # We might want to explicitly calculate a "Maximum Return" portfolio if needed.
    max_return_portfolio_on_frontier = frontier.iloc[-1].to_dict()

    # The 'selected' portfolio is often determined by the client's risk profile.
    # For this example, we'll use the MVP as the 'selected' portfolio if the risk score is low,
    # or the max Sharpe if the risk score is high, as a placeholder.
    # In a real scenario, you'd directly select the portfolio that aligns with the risk score.
    selected_portfolio_for_plot = None
    if hypothetical_risk_score <= 3: # Risk-Averse
        selected_portfolio_for_plot = mvp
    elif hypothetical_risk_score >= 7: # Risk-Seeking
        # For risk-seeking, the Max Sharpe might be a good proxy, or a portfolio further out on the frontier.
        # Here, we'll just use max_sharpe for demonstration.
        selected_portfolio_for_plot = max_sharpe
    else: # Risk-Neutral
        # For risk-neutral, a point in the middle of the frontier might be chosen.
        # For simplicity, let's take the max Sharpe again or a specific point.
        selected_portfolio_for_plot = max_sharpe # Placeholder

    plot_efficient_frontier(
        frontier, 
        mvp, 
        max_sharpe, 
        benchmark_return, 
        benchmark_volatility, 
        max_return_portfolio_on_frontier, # Using the max return from the frontier
        lower_bound=None, # Placeholder
        selected=selected_portfolio_for_plot, # Use the determined selected portfolio
        upper_bound=None, # Placeholder
        risk_score=hypothetical_risk_score # Pass the risk score
    )

    print("\n" + "=" * 80)
    print("WEEK 1 IMPLEMENTATION COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    demo_capm_engine()