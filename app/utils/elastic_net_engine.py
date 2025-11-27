"""
Elastic Net Engine for Portfolio Return Prediction
Purpose: Generate expected return vector using regularized regression
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import sys
import os
sys.path.append('.')
from app.utils.data_loader import get_features_for_ticker

class ElasticNetEngine:
    """
    Elastic Net regression engine for predicting monthly stock returns.
    Uses L1 + L2 regularization for feature selection and stability.
    """

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        """
        Initialize Elastic Net Engine.

        Args:
            alpha: Regularization strength (default 0.1)
            l1_ratio: Mix of L1 (lasso) vs L2 (ridge). 0.5 = balanced
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.models = {}  # Store trained model per ticker
        self.scalers = {}  # Store scaler per ticker
        self.feature_names = None
        self.trading_days_per_month = 21  # Scaling factor
        self.output_dir = 'Results/ElasticNet'
        # Store metrics to add portfolio summary later
        self.model_metrics = {}

    def calculate_r2_asym(self, y_true: np.ndarray, y_pred: np.ndarray, bias: float = 0.0) -> float:
        """
        Calculate Asymmetrical R² using weighted residuals.

        R²_asym = 1 - (RSS_asym / TSS)

        where RSS_asym uses asymmetric penalties aligned with MAAE logic.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bias: Bias term added to predictions (default 0.0, disabled)

        Returns:
            R²_asym value
        """
        # No bias correction applied (bias = 0.0)
        y_pred_corrected = y_pred + bias

        # Calculate errors
        errors = y_true - y_pred_corrected
        upside_errors = errors[errors > 0]

        # Calculate thresholds for tiered penalties
        if len(upside_errors) > 0:
            mean_upside = np.mean(upside_errors)
            std_upside = np.std(upside_errors)
            threshold_1std = mean_upside + std_upside
            threshold_2std = mean_upside + 2 * std_upside
        else:
            mean_upside = 0
            std_upside = 0
            threshold_1std = 0
            threshold_2std = 0

        # Calculate RSS_asym with asymmetric weights
        rss_asym = 0.0
        for error in errors:
            if error > 0:  # Upside error (underprediction)
                if std_upside > 0 and error > threshold_2std:
                    weight = 1.0  # Extreme outlier
                elif std_upside > 0 and error > threshold_1std:
                    weight = 0.8  # Moderate surprise
                else:
                    weight = 0.2  # Acceptable conservatism
            else:  # Downside error (overprediction)
                weight = 1.0  # Full penalty

            rss_asym += weight * (error ** 2)

        # Calculate TSS (Total Sum of Squares)
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)

        # Calculate R²_asym
        r2_asym = 1.0 - (rss_asym / tss) if tss > 0 else 0.0

        return r2_asym

    def calculate_maae(self, y_true: np.ndarray, y_pred: np.ndarray, bias: float = 0.0) -> float:
        """
        Calculate Mean Absolute Asymmetrical Error (MAAE) without bias correction.

        MAAE penalizes upside and downside errors differently:
        - Upside error within 1 std: 0.2 penalty (conservative is acceptable)
        - Upside error within 2 std: 0.8 penalty (moderate surprise)
        - Upside error beyond 2 std: 1.0 penalty (extreme outlier)
        - Downside error (overprediction): 1.0 penalty (direct prediction failure)

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bias: Bias term to add to predictions (default 0.0, disabled)

        Returns:
            MAAE value
        """
        # No bias correction applied (bias = 0.0)
        y_pred_corrected = y_pred + bias

        errors = y_true - y_pred_corrected
        upside_errors = errors[errors > 0]  # Underpredictions (actual > predicted)

        # Calculate mean and std of upside errors for tier detection
        if len(upside_errors) > 0:
            mean_upside = np.mean(upside_errors)
            std_upside = np.std(upside_errors)
            threshold_1std = mean_upside + std_upside
            threshold_2std = mean_upside + 2 * std_upside
        else:
            mean_upside = 0
            std_upside = 0
            threshold_1std = 0
            threshold_2std = 0

        # Apply penalties
        total_weighted_error = 0.0
        total_count = 0

        for error in errors:
            if error > 0:  # Upside error (underprediction)
                if std_upside > 0 and error > threshold_2std:
                    # Beyond 2 std: 1.0 penalty (extreme outlier)
                    total_weighted_error += 1.0 * abs(error)
                elif std_upside > 0 and error > threshold_1std:
                    # Within 2 std: 0.8 penalty (moderate surprise)
                    total_weighted_error += 0.8 * abs(error)
                else:
                    # Within 1 std: 0.2 penalty (acceptable conservatism)
                    total_weighted_error += 0.2 * abs(error)
            else:  # Downside error (overprediction)
                # Full penalty: 1.0 (direct prediction failure)
                total_weighted_error += 1.0 * abs(error)
            total_count += 1

        maae = total_weighted_error / total_count if total_count > 0 else 0.0
        return maae

    def calculate_error_breakdown(self, y_true: np.ndarray, y_pred: np.ndarray, bias: float = 0.0) -> Dict:
        """
        Calculate detailed breakdown of upside vs downside errors.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bias: Bias term (default 0.0, disabled)

        Returns:
            Dictionary with error breakdown metrics
        """
        # No bias correction applied (bias = 0.0)
        y_pred_corrected = y_pred + bias
        errors = y_true - y_pred_corrected

        # Separate upside and downside errors
        upside_mask = errors > 0
        downside_mask = errors <= 0

        upside_errors = errors[upside_mask]
        downside_errors = errors[downside_mask]

        # Calculate simple RSS components with simplified penalties
        # Upside errors now only have two tiers: 0.5x and 1.0x penalty
        # Downside errors have a 1.0x penalty
        rss_upside_0_5x = 0.0
        rss_upside_1_0x = 0.0 # This will hold errors beyond the 0.5x penalty range

        for error in upside_errors:
            # Simplified: Only 0.5x penalty for upside errors (conservative prediction is acceptable)
            # In this simplified model, all upside errors are treated as acceptable conservatism
            rss_upside_0_5x += 0.5 * (error ** 2)
            # All upside errors are also considered in the total upside RSS (which will be used for ratios)
            rss_upside_1_0x += error ** 2 # This is effectively the full RSS of upside errors

        rss_downside = np.sum(downside_errors ** 2) if len(downside_errors) > 0 else 0.0
        rss_total = rss_upside_1_0x + rss_downside # Total RSS uses the full upside error sum

        # Calculate TSS
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)

        # Calculate MAE components
        mae_upside = np.mean(np.abs(upside_errors)) if len(upside_errors) > 0 else 0.0
        mae_downside = np.mean(np.abs(downside_errors)) if len(downside_errors) > 0 else 0.0

        # Calculate MAAE components (simplified)
        # For upside, use the 0.5x penalty directly
        maae_upside = 0.5 * mae_upside if mae_upside > 0 else 0.0

        # For downside, use the full 1.0x penalty
        maae_downside = mae_downside # Already represents 1.0x penalty

        # Calculate error reduction
        error_reduction = mae_upside - maae_upside
        reduction_pct = (error_reduction / mae_upside * 100) if mae_upside > 0 else 0

        return {
            'rss_total': rss_total,
            'rss_upside_0.5x': rss_upside_0_5x, # Simplified upside RSS
            'rss_downside_1.0x': rss_downside, # Downside RSS (1.0x penalty)
            'tss': tss,
            'upside_count': len(upside_errors),
            'downside_count': len(downside_errors),
            'upside_mae': mae_upside,
            'downside_mae': mae_downside,
            'upside_maae': maae_upside, # Simplified MAAE for upside
            'downside_maae': maae_downside,
            'upside_error_reduction': error_reduction,
            'upside_reduction_pct': reduction_pct
        }


    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical features
        """
        df = data.copy()

        # Flatten multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove duplicate columns if they exist (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Ensure we're working with Series for each column
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                # Force to Series if it's a DataFrame
                if isinstance(df[col], pd.DataFrame):
                    df[col] = df[col].iloc[:, 0]

        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['SMA_Ratio'] = df['SMA_10'] / df['SMA_30']

        # Volatility (20-day rolling std)
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Momentum indicators
        df['Return_Lag1'] = df['Daily_Return'].shift(1)
        df['Return_Lag5'] = df['Daily_Return'].shift(5)
        df['Price_Change_5d'] = df['Close'].pct_change(5)

        # Volume features
        volume_series = df['Volume']
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]

        df['Volume_SMA_20'] = volume_series.rolling(window=20).mean()

        volume_sma_series = df['Volume_SMA_20']
        if isinstance(volume_sma_series, pd.DataFrame):
            volume_sma_series = volume_sma_series.iloc[:, 0]

        # Calculate ratio ensuring both are Series
        df['Volume_Ratio'] = volume_series / volume_sma_series

        # Drop NaN rows from feature calculations
        df.dropna(inplace=True)

        return df

    def prepare_features_and_target(
        self,
        data: pd.DataFrame,
        forward_days: int = 21
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target (future monthly return).

        Args:
            data: DataFrame with technical features
            forward_days: Days to look forward for target (21 = 1 month)

        Returns:
            (X: features, y: target monthly returns)
        """
        # Work with a copy to avoid modifying original
        df = data.copy()

        # Flatten multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure Close is a Series
        if 'Close' not in df.columns:
            raise ValueError(f"Close column not found in data. Available columns: {df.columns.tolist()}")

        close_series = df['Close']

        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        # Calculate future monthly return (forward-looking)
        target_returns = close_series.pct_change(forward_days).shift(-forward_days)

        # Force to Series if needed
        if isinstance(target_returns, pd.DataFrame):
            target_returns = target_returns.iloc[:, 0]

        # Assign to DataFrame
        df['Target_Monthly_Return'] = target_returns

        # Define feature columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_30', 'SMA_Ratio',
            'Volatility_20', 'RSI',
            'Return_Lag1', 'Return_Lag5', 'Price_Change_5d',
            'Volume_Ratio'
        ]

        # Filter to available features
        available_features = [f for f in feature_cols if f in df.columns]
        self.feature_names = available_features

        # Validate that Target_Monthly_Return was created
        if 'Target_Monthly_Return' not in df.columns:
            raise KeyError(f"Target_Monthly_Return column not created. Available columns: {df.columns.tolist()}")

        # Remove rows with NaN target
        df_clean = df.dropna(subset=['Target_Monthly_Return'])

        X = df_clean[available_features]
        y = df_clean['Target_Monthly_Return']

        return X, y

    def train_model(self, ticker: str, lookback_days: int = 1260, n_folds: int = 5, forward_days: int = 21,
                    train_start_date: str = None, train_end_date: str = None,
                    val_start_date: str = None, val_end_date: str = None) -> Dict:
        """
        Train Elastic Net model for a single ticker with K-Fold cross-validation.
        SELECTS THE BEST MODEL BASED ON LOWEST K-FOLD MSE.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Training period (default ~5 years), used if date range not specified
            n_folds: Number of K-Fold splits (default 5)
            forward_days: Days ahead to predict (21=monthly, 252=annual)
            train_start_date: Training start date (YYYY-MM-DD)
            train_end_date: Training end date (YYYY-MM-DD)
            val_start_date: Validation start date (YYYY-MM-DD)
            val_end_date: Validation end date (YYYY-MM-DD)

        Returns:
            Training metrics dict
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from sklearn.metrics import r2_score # Import r2_score

        # Get data
        data = get_features_for_ticker(ticker)
        if data.empty:
            print(f"No data for {ticker}")
            return {}

        # Remove duplicate columns immediately after getting data
        data = data.loc[:, ~data.columns.duplicated(keep='first')]

        # Filter by date range if provided, otherwise use lookback_days
        if train_start_date and train_end_date:
            # Combine training and validation data for full dataset
            if val_start_date and val_end_date:
                data = data.loc[train_start_date:val_end_date]
            else:
                data = data.loc[train_start_date:train_end_date]
        else:
            # Use data from 2015-01-01 onwards (approximately 10 years)
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data[data.index >= '2015-01-01']

        # Calculate technical features
        data_with_features = self.calculate_technical_features(data)

        # Prepare X, y
        X, y = self.prepare_features_and_target(data_with_features, forward_days=forward_days)

        if len(X) < 100:
            print(f"Insufficient data for {ticker} ({len(X)} samples)")
            return {}

        # Scale features on full data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Fold Cross-Validation to detect overfitting - collect detailed metrics
        kfold = KFold(n_splits=n_folds, shuffle=False)
        model_cv = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)

        cv_r2_scores = cross_val_score(model_cv, X_scaled, y, cv=kfold, scoring='r2')
        cv_mse_scores = -cross_val_score(model_cv, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error')
        cv_mae_scores = -cross_val_score(model_cv, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')

        # Calculate MAAE for K-Fold cross-validation
        cv_maae_scores = []
        for train_idx, val_idx in kfold.split(X_scaled):
            X_fold_train = X_scaled[train_idx]
            X_fold_val = X_scaled[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            model_fold = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)
            model_fold.fit(X_fold_train, y_fold_train)
            y_pred_fold = model_fold.predict(X_fold_val)
            maae_fold = self.calculate_maae(y_fold_val.values, y_pred_fold)
            cv_maae_scores.append(maae_fold)

        # Calculate CV statistics
        cv_mean_r2 = cv_r2_scores.mean()
        cv_std_r2 = cv_r2_scores.std()

        # === K-FOLD ONLY (Consistent with LSTM Engine) ===
        # Select best performing K-Fold fold
        best_fold_mse = float('inf')
        best_fold_model = None
        best_fold_train_idx = None
        best_fold_val_idx = None
        best_fold_test_mae = float('inf') # Initialize for best fold test MAE
        best_fold_test_maae = float('inf') # Initialize for best fold test MAAE
        best_fold_test_r2_asym = float('inf') # Initialize for best fold test R2_asym

        # Find best fold based on MSE (MAAE already calculated above)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_fold_train = X_scaled[train_idx]
            X_fold_val = X_scaled[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            model_fold = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)
            model_fold.fit(X_fold_train, y_fold_train)
            y_pred_fold = model_fold.predict(X_fold_val)
            mse_fold = mean_squared_error(y_fold_val, y_pred_fold)
            mae_fold_val = mean_absolute_error(y_fold_val, y_pred_fold)
            maae_fold_val = self.calculate_maae(y_fold_val.values, y_pred_fold)
            r2_asym_fold_val = self.calculate_r2_asym(y_fold_val.values, y_pred_fold) # R2_asym is calculated using the method

            if mse_fold < best_fold_mse:
                best_fold_mse = mse_fold
                best_fold_model = model_fold
                best_fold_train_idx = train_idx
                best_fold_val_idx = val_idx
                best_fold_test_mae = mae_fold_val # Store MAE of the best fold
                best_fold_test_maae = maae_fold_val # Store MAAE of the best fold
                best_fold_test_r2_asym = r2_asym_fold_val # Store R2_asym of the best fold


        # Use the best K-Fold model
        best_candidate = {
            'model': best_fold_model,
            'scaler': scaler,
            'mse': best_fold_mse,
            'X_train': X_scaled[best_fold_train_idx],
            'X_test': X_scaled[best_fold_val_idx],
            'y_train': y.iloc[best_fold_train_idx],
            'y_test': y.iloc[best_fold_val_idx],
            'method': 'kfold_best'
        }

        # Store BEST model
        self.models[ticker] = best_candidate['model']
        self.scalers[ticker] = best_candidate['scaler']

        X_train_scaled = best_candidate['X_train']
        X_test_scaled = best_candidate['X_test']
        y_train = best_candidate['y_train']
        y_test = best_candidate['y_test']

        print(f"  ✓ {ticker}: Selected kfold_best (MSE={best_candidate['mse']:.6f})")

        # Calculate metrics for TRAINING set only (test metrics stored from fold selection)
        model = best_candidate['model']
        y_train_pred = model.predict(X_train_scaled)

        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = best_candidate['mse']  # Use stored MSE from fold selection
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = best_fold_test_mae  # Use stored value from fold selection

        # Calculate MAAE (asymmetrical error) for training only
        train_maae = self.calculate_maae(y_train.values, y_train_pred)
        test_maae = best_fold_test_maae  # Use stored value from fold selection

        # Calculate R²_asym (proper RSS_asym/TSS calculation) for training and testing
        train_r2_asym = self.calculate_r2_asym(y_train.values, y_train_pred)
        test_r2_asym = self.calculate_r2_asym(y_test.values, best_fold_model.predict(X_test_scaled))

        # Annualize metrics based on forward_days
        # If forward_days=21 (monthly), scale to annual (12 months)
        # If forward_days=252 (annual), no scaling needed
        if forward_days == 252:
            scale_factor = 1
        elif forward_days == 21:
            scale_factor = 12
        else:
            scale_factor = 252 / forward_days

        # MSE scales quadratically
        train_mse_annual = train_mse * (scale_factor ** 2)
        test_mse_annual = test_mse * (scale_factor ** 2)
        cv_mse_annual = cv_mse_scores.mean() * (scale_factor ** 2)
        cv_std_mse_annual = cv_mse_scores.std() * (scale_factor ** 2)

        # MAE scales linearly
        train_mae_annual = train_mae * scale_factor
        test_mae_annual = test_mae * scale_factor
        cv_mae_annual = cv_mae_scores.mean() * scale_factor
        cv_std_mae_annual = cv_mae_scores.std() * scale_factor

        # R² is scale-invariant (same regardless of time period)
        train_r2_annual = train_r2
        test_r2_annual = test_r2
        cv_r2_annual = cv_mean_r2

        # Calculate portfolio metrics for the current ticker's training data
        # This is a simplification; a true portfolio metric would consider all tickers
        # Here, we're just adding the ticker's own performance to the metrics
        # This part might need adjustment if you want to aggregate across tickers within this function
        metrics = {
            'ticker': ticker,
            'winning_method': 'kfold_best',  # Always kfold_best now
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_maae': train_maae,
            'test_maae': test_maae,
            'cv_mean_r2': cv_mean_r2,
            'cv_std_r2': cv_std_r2,
            'cv_mean_mse': cv_mse_scores.mean(),
            'cv_std_mse': cv_mse_scores.std(),
            'cv_mean_mae': cv_mae_scores.mean(),
            'cv_std_mae': cv_mae_scores.std(),
            'cv_mean_maae': np.mean(cv_maae_scores),
            'cv_std_maae': np.std(cv_maae_scores),
            'kfold_best_test_mae': best_fold_test_mae,
            'kfold_best_test_maae': best_fold_test_maae,
            'kfold_best_test_r2': best_fold_model.score(X_scaled[best_fold_val_idx], y.iloc[best_fold_val_idx]),
            'cv_coefficient_of_variation_r2': cv_std_r2 / cv_mean_r2 if cv_mean_r2 != 0 else 0,
            'n_features': len(self.feature_names),
            'n_samples_train': len(X_train_scaled),
            'n_samples_test': len(X_test_scaled),
            'n_folds': n_folds,
            'alpha_used': self.alpha,
            'l1_ratio': self.l1_ratio,
            'train_r2_asym': train_r2_asym, # Added R2_asym
            'test_r2_asym': test_r2_asym,   # Added R2_asym
        }
        self.model_metrics[ticker] = metrics # Store metrics for later aggregation
        return metrics

    def predict_annual_return(self, ticker: str, forward_days: int = 21) -> float:
        """
        Predict expected annual return for a ticker.

        Args:
            ticker: Stock ticker symbol
            forward_days: Days ahead that was used for training (21=monthly, 252=annual)

        Returns:
            Expected annual return
        """
        if ticker not in self.models:
            print(f"Model not trained for {ticker}")
            return 0.0

        # Get latest data
        data = get_features_for_ticker(ticker)
        if data.empty:
            return 0.0

        # Calculate features
        data_with_features = self.calculate_technical_features(data)

        # Get latest features
        latest_features = data_with_features[self.feature_names].tail(1)

        # Predict return
        X_scaled = self.scalers[ticker].transform(latest_features)
        predicted_return = self.models[ticker].predict(X_scaled)[0]

        # No bias correction - use raw model predictions

        # Annualize based on forward_days
        if forward_days == 252:
            # Already annual, no scaling needed
            annual_return = predicted_return
        elif forward_days == 21:
            # Monthly to annual (×12)
            annual_return = predicted_return * 12
        else:
            # General case: scale to annual
            annual_return = predicted_return * (252 / forward_days)

        return annual_return

    def generate_expected_returns(
        self,
        tickers: List[str],
        train_start_date: str = None,
        train_end_date: str = None,
        val_start_date: str = None,
        val_end_date: str = None
    ) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
        """
        Generate expected monthly return vector for portfolio optimization.

        Args:
            tickers: List of stock tickers
            train_start_date: Training start date (YYYY-MM-DD), defaults to using lookback_days
            train_end_date: Training end date (YYYY-MM-DD)
            val_start_date: Validation start date (YYYY-MM-DD)
            val_end_date: Validation end date (YYYY-MM-DD)

        Returns:
            (expected_returns_array, training_metrics_df, approved_tickers)
        """
        print("\n" + "="*60)
        print("ELASTIC NET ENGINE: TRAINING & PREDICTION")
        print("="*60)

        MAE_THRESHOLD = 0.05  # 5% monthly maximum acceptable prediction error

        expected_returns = {}
        metrics_list = []
        approved_tickers = []  # Track approval status

        for ticker in tickers:
            # Train model with date filtering
            metrics = self.train_model(
                ticker,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date
            )

            if metrics:
                # Predict annual return
                annual_return = self.predict_annual_return(ticker)
                expected_returns[ticker] = annual_return

                metrics['expected_annual_return'] = annual_return
                metrics_list.append(metrics)

                # Determine approval based on kfold_best_test_maae (K-Fold MAAE)
                # Use K-Fold MAAE as the primary quality metric for approval
                kfold_best_test_maae = metrics.get('kfold_best_test_maae', float('inf'))
                is_approved = 1 if kfold_best_test_maae <= MAE_THRESHOLD else 0
                approved_tickers.append(is_approved)
            else:
                expected_returns[ticker] = 0.0
                approved_tickers.append(0)  # Not approved if training failed

        # Convert expected_returns dictionary to a list in the same order as tickers
        expected_returns_list = [expected_returns.get(ticker, 0.0) for ticker in tickers]

        # Add winning method columns with proper naming BEFORE creating DataFrame
        for m in metrics_list:
            method = m.get('winning_method', 'unknown')
            # Add winning method prefix to columns
            if 'train_r2' in m:
                m[f'{method}_train_r2'] = m['train_r2']
            if 'test_r2' in m:
                m[f'{method}_test_r2'] = m['test_r2']
            if 'train_mse' in m:
                m[f'{method}_train_mse'] = m['train_mse']
            if 'test_mse' in m:
                m[f'{method}_test_mse'] = m['test_mse']
            if 'train_mae' in m:
                m[f'{method}_train_mae'] = m['train_mae']
            if 'test_mae' in m:
                m[f'{method}_test_mae'] = m['test_mae']
            # Add MAAE winning method columns
            if 'train_maae' in m:
                m[f'{method}_train_maae'] = m['train_maae']
            if 'test_maae' in m:
                m[f'{method}_test_maae'] = m['test_maae']
            # Add R2_asym winning method columns
            if 'train_r2_asym' in m:
                m[f'{method}_train_r2_asym'] = m['train_r2_asym']
            if 'test_r2_asym' in m:
                m[f'{method}_test_r2_asym'] = m['test_r2_asym']


        # Create DataFrame with all columns including winning method columns
        metrics_df = pd.DataFrame(metrics_list)

        # Add portfolio summary row with actual metrics (will be moved to bottom later)
        if self.model_metrics:
            # Calculate averages for winning method columns
            portfolio_summary_dict = {
                'ticker': 'PORTFOLIO',
                'winning_method': f"{len([m for m in self.model_metrics.values() if m.get('winning_method')])} models trained",
                'expected_annual_return': np.mean([m['expected_annual_return'] for m in self.model_metrics.values()]),
                'train_r2': np.mean([m['train_r2'] for m in self.model_metrics.values()]),
                'test_r2': np.mean([m['test_r2'] for m in self.model_metrics.values()]),
                'train_mse': np.mean([m['train_mse'] for m in self.model_metrics.values()]),
                'test_mse': np.mean([m['test_mse'] for m in self.model_metrics.values()]),
                'train_mae': np.mean([m['train_mae'] for m in self.model_metrics.values()]),
                'test_mae': np.mean([m['test_mae'] for m in self.model_metrics.values()]),
                'train_maae': np.mean([m['train_maae'] for m in self.model_metrics.values()]),
                'test_maae': np.mean([m['test_maae'] for m in self.model_metrics.values()]),
                'cv_mean_r2': np.mean([m.get('cv_mean_r2', 0) for m in self.model_metrics.values()]),
                'cv_mean_mse': np.mean([m.get('cv_mean_mse', 0) for m in self.model_metrics.values()]),
                'cv_mean_mae': np.mean([m.get('cv_mean_mae', 0) for m in self.model_metrics.values()]),
                'cv_mean_maae': np.mean([m.get('cv_mean_maae', 0) for m in self.model_metrics.values()])
            }

            # Add winning method columns with aggregated values
            for col in metrics_df.columns:
                if 'kfold_best_' in col or 'split_80_20_' in col:
                    # Extract base metric name (e.g., 'train_r2_annual' from 'kfold_best_train_r2_annual')
                    if 'kfold_best_' in col:
                        base_metric = col.replace('kfold_best_', '')
                    else:
                        base_metric = col.replace('split_80_20_', '')

                    # Calculate average from metrics_list for this winning method column
                    values = [m.get(col, np.nan) for m in metrics_list if col in m]
                    if values and not all(pd.isna(values)):
                        portfolio_summary_dict[col] = np.nanmean(values)
                    else:
                        portfolio_summary_dict[col] = np.nan
                # Also handle R2_asym for portfolio summary
                elif 'train_r2_asym' in col:
                    portfolio_summary_dict['train_r2_asym'] = np.nanmean([m.get('train_r2_asym', np.nan) for m in metrics_list])
                elif 'test_r2_asym' in col:
                    portfolio_summary_dict['test_r2_asym'] = np.nanmean([m.get('test_r2_asym', np.nan) for m in metrics_list])


            # Create portfolio summary row with actual aggregated metrics
            portfolio_summary = pd.DataFrame([portfolio_summary_dict])

            # Concatenate portfolio summary to END (bottom row)
            metrics_df = pd.concat([metrics_df, portfolio_summary], ignore_index=True)

        # Reorder columns: winning method columns first (MAE, R², MSE order), then basic columns
        base_cols = ['ticker', 'winning_method']

        # Find winning method columns in specific order: MAE first, then MAAE, then R², then R2_asym, then MSE
        winning_mae_cols = [col for col in metrics_df.columns if any(method in col for method in ['kfold_best', 'split_80_20'])
                           and 'mae' in col.lower() and col not in base_cols]
        winning_maae_cols = [col for col in metrics_df.columns if any(method in col for method in ['kfold_best', 'split_80_20'])
                           and 'maae' in col.lower() and col not in base_cols]
        winning_r2_cols = [col for col in metrics_df.columns if any(method in col for method in ['kfold_best', 'split_80_20'])
                          and 'r2' in col.lower() and col not in base_cols and 'r2_asym' not in col]
        winning_r2_asym_cols = [col for col in metrics_df.columns if any(method in col for method in ['kfold_best', 'split_80_20'])
                           and 'r2_asym' in col.lower() and col not in base_cols]
        winning_mse_cols = [col for col in metrics_df.columns if any(method in col for method in ['kfold_best', 'split_80_20'])
                           and 'mse' in col.lower() and col not in base_cols]

        # Combine winning method columns in MAE, MAAE, R², R2_asym, MSE order
        winning_cols = winning_mae_cols + winning_maae_cols + winning_r2_cols + winning_r2_asym_cols + winning_mse_cols

        # Basic metric columns in MAE, MAAE, R², R2_asym, MSE order
        basic_cols = ['train_mae', 'test_mae', 'train_maae', 'test_maae', 'train_r2', 'test_r2', 'train_r2_asym', 'test_r2_asym', 'train_mse', 'test_mse']
        basic_cols = [col for col in basic_cols if col in metrics_df.columns and col not in winning_cols]

        # CV and other columns
        other_cols = [col for col in metrics_df.columns if col not in base_cols + winning_cols + basic_cols]

        # Reorder: base -> winning method (MAE, MAAE, R², R2_asym, MSE) -> basic (MAE, MAAE, R², R2_asym, MSE) -> others
        ordered_cols = base_cols + winning_cols + basic_cols + other_cols
        metrics_df = metrics_df[[col for col in ordered_cols if col in metrics_df.columns]]

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(self.output_dir, 'elasticnet_detailed_metrics.csv'), index=False)
        print(f"✓ Saved detailed ElasticNet metrics to '{os.path.join(self.output_dir, 'elasticnet_detailed_metrics.csv')}'")

        # Create R²_asym vs R² comparison table
        r2_comparison = []
        for m in metrics_list:
            r2_comparison.append({
                'Ticker': m.get('ticker', 'N/A'),
                'Train_R2': m.get('train_r2', np.nan),
                'Train_R2_asym': m.get('train_r2_asym', np.nan),
                'Test_R2': m.get('test_r2', np.nan),
                'Test_R2_asym': m.get('test_r2_asym', np.nan),
                'Train_R2_Improvement': m.get('test_r2_asym', 0) - m.get('test_r2', 0) if not pd.isna(m.get('test_r2_asym')) else np.nan,
                'Test_R2_Improvement': m.get('test_r2_asym', 0) - m.get('test_r2', 0) if not pd.isna(m.get('test_r2_asym')) else np.nan
            })

        if r2_comparison:
            r2_comparison_df = pd.DataFrame(r2_comparison)
            # Add portfolio summary row
            portfolio_r2_summary = {
                'Ticker': 'PORTFOLIO',
                'Train_R2': r2_comparison_df['Train_R2'].mean(),
                'Train_R2_asym': r2_comparison_df['Train_R2_asym'].mean(),
                'Test_R2': r2_comparison_df['Test_R2'].mean(),
                'Test_R2_asym': r2_comparison_df['Test_R2_asym'].mean(),
                'Train_R2_Improvement': r2_comparison_df['Train_R2_Improvement'].mean(),
                'Test_R2_Improvement': r2_comparison_df['Test_R2_Improvement'].mean()
            }
            r2_comparison_df = pd.concat([r2_comparison_df, pd.DataFrame([portfolio_r2_summary])], ignore_index=True)
            r2_comparison_df.to_csv(os.path.join(self.output_dir, 'r2_asym_vs_r2_comparison.csv'), index=False)
            print(f"✓ Saved R²_asym vs R² comparison to '{os.path.join(self.output_dir, 'r2_asym_vs_r2_comparison.csv')}'")


        # Create simplified winning metrics table
        winning_metrics_cols = ['ticker', 'winning_method']

        # Add winning method columns (MAE, MAAE, R², R2_asym, MSE order)
        for col in metrics_df.columns:
            if any(method in col for method in ['kfold_best', 'split_80_20']):
                if any(metric in col for metric in ['mae', 'maae', 'r2', 'r2_asym', 'mse']):
                    winning_metrics_cols.append(col)

        # Add basic metrics (MAE, MAAE, R², R2_asym, MSE order)
        basic_metrics = ['train_mae', 'test_mae', 'train_maae', 'test_maae', 'train_r2', 'test_r2', 'train_r2_asym', 'test_r2_asym', 'train_mse', 'test_mse']
        for col in basic_metrics:
            if col in metrics_df.columns and col not in winning_metrics_cols:
                winning_metrics_cols.append(col)

        # Create simplified dataframe
        winning_metrics_df = metrics_df[[col for col in winning_metrics_cols if col in metrics_df.columns]].copy()
        winning_metrics_df.to_csv(os.path.join(self.output_dir, 'elasticnet_winning_metrics.csv'), index=False)
        print(f"✓ Saved simplified winning metrics to '{os.path.join(self.output_dir, 'elasticnet_winning_metrics.csv')}'")

        # Save expected returns to CSV
        returns_df = pd.DataFrame({
            'Ticker': list(expected_returns.keys()),
            'Expected_Return': list(expected_returns.values()),
            'Approved': approved_tickers
        })
        returns_df.to_csv(os.path.join(self.output_dir, 'expected_returns_elastic.csv'), index=False)
        print(f"✓ Saved expected returns to '{os.path.join(self.output_dir, 'expected_returns_elastic.csv')}'")

        # Print detailed metrics table matching LSTM format with winning method columns
        if metrics_list:
            print("\n" + "="*200)
            print("ELASTIC NET DETAILED METRICS TABLE (Monthly Prediction Horizon)")
            print("="*200)
            first_method = metrics_list[0].get("winning_method", "Method")
            # Updated header to include MAAE metrics and best fold MAE/MAAE and R2_asym
            print(f"{'Ticker':<8} {'Method':<15} {first_method}_TrainMAE{'':<10} {first_method}_TestMAE{'':<11} {'kfold_best_TestMAE':<15} {first_method}_TrainMAAE{'':<10} {first_method}_TestMAAE{'':<11} {'kfold_best_TestMAAE':<16} {first_method}_TrainR²{'':<8} {first_method}_TestR²{'':<9} {first_method}_TrainR²asym{'':<5} {first_method}_TestR²asym{'':<6} {first_method}_TrainMSE{'':<10} {first_method}_TestMSE{'':<11}")
            print("-"*200)

            for m in metrics_list:
                ticker = m.get('ticker', 'N/A')
                method = m.get('winning_method', 'N/A')
                # Use the winning method columns - ORDER: MAE, MAAE, R², R2_asym, MSE
                train_mae = m.get(f'{method}_train_mae', m.get('train_mae', 0.0))
                test_mae = m.get(f'{method}_test_mae', m.get('test_mae', 0.0))
                # Get best fold test MAE and MAAE
                best_fold_test_mae = m.get('kfold_best_test_mae', 0.0)
                best_fold_test_maae = m.get('kfold_best_test_maae', 0.0)
                train_maae = m.get(f'{method}_train_maae', m.get('train_maae', 0.0))
                test_maae = m.get(f'{method}_test_maae', m.get('test_maae', 0.0))
                train_r2 = m.get(f'{method}_train_r2', m.get('train_r2', 0.0))
                test_r2 = m.get(f'{method}_test_r2', m.get('test_r2', 0.0))
                train_r2_asym = m.get(f'{method}_train_r2_asym', m.get('train_r2_asym', 0.0))
                test_r2_asym = m.get(f'{method}_test_r2_asym', m.get('test_r2_asym', 0.0))
                train_mse = m.get(f'{method}_train_mse', m.get('train_mse', 0.0))
                test_mse = m.get(f'{method}_test_mse', m.get('test_mse', 0.0))

                print(f"{ticker:<8} {method:<15} {train_mae:>21.4f} {test_mae:>21.4f} {best_fold_test_mae:>15.4f} {train_maae:>21.4f} {test_maae:>21.4f} {best_fold_test_maae:>16.4f} {train_r2:>19.3f} {test_r2:>19.3f} {train_r2_asym:>20.3f} {test_r2_asym:>21.3f} {train_mse:>21.4f} {test_mse:>21.4f}")

            print("="*200)

            # Print portfolio summary
            print(f"\n{'PORTFOLIO SUMMARY':<8}")
            print("-"*200)
            # Get portfolio metrics using winning method columns - ORDER: MAE, MAAE, R², R2_asym, MSE
            non_portfolio = [m for m in metrics_list if m.get('ticker') != 'PORTFOLIO']
            avg_train_mae = np.nanmean([m.get(f'{m.get("winning_method")}_train_mae', m.get('train_mae', np.nan)) for m in non_portfolio])
            avg_test_mae = np.nanmean([m.get(f'{m.get("winning_method")}_test_mae', m.get('test_mae', np.nan)) for m in non_portfolio])
            avg_best_fold_mae = np.nanmean([m.get('kfold_best_test_mae', np.nan) for m in non_portfolio])
            avg_train_maae = np.nanmean([m.get(f'{m.get("winning_method")}_train_maae', m.get('train_maae', np.nan)) for m in non_portfolio])
            avg_test_maae = np.nanmean([m.get(f'{m.get("winning_method")}_test_maae', m.get('test_maae', np.nan)) for m in non_portfolio])
            avg_best_fold_maae = np.nanmean([m.get('kfold_best_test_maae', np.nan) for m in non_portfolio])
            avg_train_r2 = np.nanmean([m.get(f'{m.get("winning_method")}_train_r2', m.get('train_r2', np.nan)) for m in non_portfolio])
            avg_test_r2 = np.nanmean([m.get(f'{m.get("winning_method")}_test_r2', m.get('test_r2', np.nan)) for m in non_portfolio])
            avg_train_r2_asym = np.nanmean([m.get(f'{m.get("winning_method")}_train_r2_asym', m.get('train_r2_asym', np.nan)) for m in non_portfolio])
            avg_test_r2_asym = np.nanmean([m.get(f'{m.get("winning_method")}_test_r2_asym', m.get('test_r2_asym', np.nan)) for m in non_portfolio])
            avg_train_mse = np.nanmean([m.get(f'{m.get("winning_method")}_train_mse', m.get('train_mse', np.nan)) for m in non_portfolio])
            avg_test_mse = np.nanmean([m.get(f'{m.get("winning_method")}_test_mse', m.get('test_mse', np.nan)) for m in non_portfolio])

            num_models = len(non_portfolio)
            models_label = f'{num_models} models'
            print(f"{'Average':<8} {models_label:<15} {avg_train_mae:>21.4f} {avg_test_mae:>21.4f} {avg_best_fold_mae:>15.4f} {avg_train_maae:>21.4f} {avg_test_maae:>21.4f} {avg_best_fold_maae:>16.4f} {avg_train_r2:>19.3f} {avg_test_r2:>19.3f} {avg_train_r2_asym:>20.3f} {avg_test_r2_asym:>21.3f} {avg_train_mse:>21.4f} {avg_test_mse:>21.4f}")
            print("="*200)

        # Save error breakdown table with simplified upside/downside components
        error_breakdown_rows = []
        for ticker in tickers:
            if ticker not in self.models:
                continue

            # Get test data for this ticker
            data = get_features_for_ticker(ticker)
            if data.empty:
                continue

            data_with_features = self.calculate_technical_features(data)
            X, y = self.prepare_features_and_target(data_with_features, forward_days=21)

            if len(X) < 100:
                continue

            # Scale and split
            scaler = self.scalers[ticker]
            X_scaled = scaler.transform(X)

            # Use same split as training (80/20)
            split_idx = int(len(X_scaled) * 0.8)
            X_test = X_scaled[split_idx:]
            y_test = y.iloc[split_idx:]

            # Predict
            y_pred = self.models[ticker].predict(X_test)

            # Calculate error breakdown
            breakdown = self.calculate_error_breakdown(y_test.values, y_pred)

            error_breakdown_rows.append({
                'Ticker': ticker,
                'RSS_Total': breakdown['rss_total'],
                'RSS_Upside_0.5x': breakdown['rss_upside_0.5x'],
                'RSS_Downside_1.0x': breakdown['rss_downside_1.0x'],
                'TSS': breakdown['tss'],
                'Upside_Count': breakdown['upside_count'],
                'Downside_Count': breakdown['downside_count'],
                'Upside_MAE': breakdown['upside_mae'],
                'Downside_MAE': breakdown['downside_mae']
            })

        if error_breakdown_rows:
            error_breakdown_df = pd.DataFrame(error_breakdown_rows)

            # Calculate ratio columns
            error_breakdown_df['Total_Error_Count'] = error_breakdown_df['Upside_Count'] + error_breakdown_df['Downside_Count']
            error_breakdown_df['Upside_Count_Ratio'] = error_breakdown_df['Upside_Count'] / error_breakdown_df['Total_Error_Count']
            error_breakdown_df['RSS_Upside_Ratio'] = error_breakdown_df['RSS_Upside_0.5x'] / error_breakdown_df['RSS_Total']
            error_breakdown_df['RSS_Downside_Ratio'] = error_breakdown_df['RSS_Downside_1.0x'] / error_breakdown_df['RSS_Total']

            # Add portfolio summary
            portfolio_summary = {
                'Ticker': 'PORTFOLIO_AVG',
                'RSS_Total': error_breakdown_df['RSS_Total'].mean(),
                'RSS_Upside_0.5x': error_breakdown_df['RSS_Upside_0.5x'].mean(),
                'RSS_Downside_1.0x': error_breakdown_df['RSS_Downside_1.0x'].mean(),
                'TSS': error_breakdown_df['TSS'].mean(),
                'Upside_Count': error_breakdown_df['Upside_Count'].sum(),
                'Downside_Count': error_breakdown_df['Downside_Count'].sum(),
                'Upside_MAE': error_breakdown_df['Upside_MAE'].mean(),
                'Downside_MAE': error_breakdown_df['Downside_MAE'].mean(),
                'Total_Error_Count': error_breakdown_df['Total_Error_Count'].sum(),
                'Upside_Count_Ratio': error_breakdown_df['Upside_Count_Ratio'].mean(),
                'RSS_Upside_Ratio': error_breakdown_df['RSS_Upside_Ratio'].mean(),
                'RSS_Downside_Ratio': error_breakdown_df['RSS_Downside_Ratio'].mean()
            }
            error_breakdown_df = pd.concat([error_breakdown_df, pd.DataFrame([portfolio_summary])], ignore_index=True)

            error_breakdown_df.to_csv(os.path.join(self.output_dir, 'error_breakdown_analysis.csv'), index=False)
            print(f"✓ Saved error breakdown analysis to '{os.path.join(self.output_dir, 'error_breakdown_analysis.csv')}')")

        # Save rejected securities CSV
        rejected_securities = []
        for m in metrics_list:
            if m.get('ticker') != 'PORTFOLIO':
                # Use MAAE for rejection criteria as per the prompt's intention
                kfold_best_test_maae = m.get('kfold_best_test_maae', float('inf'))
                if kfold_best_test_maae > MAE_THRESHOLD:
                    rejected_securities.append({
                        'Ticker': m['ticker'],
                        'MAAE': kfold_best_test_maae,
                        'Threshold': MAE_THRESHOLD,
                        'Miss_Amount': kfold_best_test_maae - MAE_THRESHOLD,
                        'Best_MAAE': kfold_best_test_maae
                    })

        if rejected_securities:
            rejected_df = pd.DataFrame(rejected_securities)
            rejected_df.to_csv(os.path.join(self.output_dir, 'elasticnet_rejected_securities.csv'), index=False)
            print(f"✓ Saved rejected securities details to '{os.path.join(self.output_dir, 'elasticnet_rejected_securities.csv')}'")

        # Store approved_tickers as instance variable for use in optimize_portfolio_mvo
        self.approved_tickers = approved_tickers

        # Print approval summary
        num_approved = sum(approved_tickers)
        num_total = len(approved_tickers)
        print(f"\n{'='*60}")
        print(f"APPROVAL SUMMARY (MAAE Threshold: {MAE_THRESHOLD:.1%})")
        print(f"{'='*60}")
        print(f"  Total Securities: {num_total}")
        print(f"  Approved (MAAE ≤ {MAE_THRESHOLD:.1%}): {num_approved} ({num_approved/num_total*100:.1f}%)")
        print(f"  Rejected (MAAE > {MAE_THRESHOLD:.1%}): {num_total - num_approved} ({(num_total-num_approved)/num_total*100:.1f}%)")
        print(f"{'='*60}\n")

        return np.array(list(expected_returns.values())), metrics_df, approved_tickers

    def compute_covariance_matrix(
        self,
        tickers: List[str],
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Compute monthly covariance matrix from daily returns.

        Args:
            tickers: List of stock tickers
            lookback_days: Historical period for covariance

        Returns:
            Monthly covariance matrix
        """
        returns_data = {}

        for ticker in tickers:
            data = get_features_for_ticker(ticker)
            if not data.empty and 'Daily_Return' in data.columns:
                returns_data[ticker] = data['Daily_Return'].tail(lookback_days)

        # Create DataFrame and align
        returns_df = pd.DataFrame(returns_data).dropna()

        # Scale daily covariance to annual (multiply by 252 trading days)
        annual_cov_matrix = returns_df.cov() * 252

        # Save covariance matrix
        os.makedirs('Results/ElasticNet', exist_ok=True)
        annual_cov_matrix.to_csv('Results/ElasticNet/elasticnet_covariance_matrix.csv')
        print("✓ Saved covariance matrix to 'Results/ElasticNet/elasticnet_covariance_matrix.csv'")

        return annual_cov_matrix

    def optimize_portfolio_mvo(self, expected_returns: np.ndarray, cov_matrix: pd.DataFrame,
                                tickers: List[str], risk_score: int = 5):
        """
        Perform Mean-Variance Optimization using ElasticNet expected returns.

        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            tickers: List of tickers
            risk_score: Client risk score (1-9)

        Returns:
            Dictionary with optimal portfolio and metrics
        """
        from Libraries.CAPM_Engine import MPTOptimizer

        # Generate non-filtered frontier (all tickers)
        optimizer = MPTOptimizer(expected_returns, cov_matrix, tickers, risk_score=risk_score)

        try:
            frontier_all = optimizer.generate_efficient_frontier(num_points=99)
        except Exception as e:
            print(f"Could not generate frontier for all tickers: {e}")
            frontier_all = pd.DataFrame()

        # Generate filtered frontier (only approved tickers with MAAE <= 5%)
        frontier_filtered = pd.DataFrame()
        optimizer_filtered = None

        if hasattr(self, 'approved_tickers'):
            approved_indices = [i for i, app in enumerate(self.approved_tickers) if app == 1]
            approved_tickers_list = [tickers[i] for i in approved_indices]

            if approved_tickers_list and len(approved_tickers_list) < len(tickers):
                try:
                    # Re-filter expected returns and covariance matrix
                    approved_expected_returns = expected_returns[approved_indices]
                    approved_cov_matrix = cov_matrix.loc[approved_tickers_list, approved_tickers_list]

                    # Create a new optimizer instance for the filtered set
                    optimizer_filtered = MPTOptimizer(approved_expected_returns, approved_cov_matrix, approved_tickers_list, risk_score=risk_score)
                    frontier_filtered = optimizer_filtered.generate_efficient_frontier(num_points=99)

                    print(f"\n✓ Generated filtered frontier with {len(approved_tickers_list)} approved tickers (MAAE ≤ 5%)")
                    print(f"  Excluded {len(tickers) - len(approved_tickers_list)} tickers with MAAE > 5%")
                except Exception as e:
                    print(f"Could not generate filtered frontier: {e}")
                    frontier_filtered = pd.DataFrame()
            elif len(approved_tickers_list) == len(tickers):
                print(f"\n✓ All {len(tickers)} tickers approved (MAAE ≤ 5%), no filtering needed")
                frontier_filtered = frontier_all.copy()
                optimizer_filtered = optimizer

        # Get key portfolios from the ALL tickers set
        mvp = optimizer.minimize_volatility()
        max_sharpe = optimizer.maximize_sharpe_ratio()

        return {
            'frontier_all': frontier_all,
            'frontier_filtered': frontier_filtered,
            'mvp': mvp,
            'max_sharpe': max_sharpe,
            'optimizer': optimizer,
            'optimizer_filtered': optimizer_filtered
        }