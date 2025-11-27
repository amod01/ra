"""
LSTM Engine for Portfolio Return Prediction
Purpose: Time series forecasting using recurrent neural networks
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import sys
import os
import json
from datetime import datetime, date
sys.path.append('.')
from app.utils.data_loader import cleaned_financial_data, get_features_for_ticker
from sklearn.model_selection import TimeSeriesSplit, KFold


# TensorFlow imports (with graceful fallback)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None  # Define as None for type hints
    print("Warning: TensorFlow not available. LSTM engine will use fallback predictions.")


class LSTMEngine:
    """
    LSTM recurrent neural network for predicting monthly stock returns.
    Uses sequential time series data for pattern recognition.
    """

    def __init__(self, sequence_length: int = 30, lstm_units: int = 40, cache_file: str = 'Memory/lstm_cache.json'):
        """
        Initialize LSTM Engine.

        Args:
            sequence_length: Number of days to look back (default 30 - reduced from 60)
            lstm_units: LSTM layer neurons (default 40 - reduced from 50)
            cache_file: Path to cache file for storing daily expected returns
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.models = {}  # Store trained model per ticker
        self.feature_names = None
        self.trading_days_per_month = 21  # Scaling factor
        self.tf_available = TENSORFLOW_AVAILABLE
        self.cache_file = cache_file
        self.expected_returns_cache = {}
        self.last_calculated_date = {}
        self._load_cache()
        self.output_dir = 'Results/LSTM'
        # Placeholder for metrics that will be used in portfolio summary
        self.model_metrics = {}

    def _load_cache(self):
        """Load cached expected returns from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.expected_returns_cache = cache_data.get('expected_returns', {})
                    # Convert date strings back to date objects
                    self.last_calculated_date = {
                        ticker: datetime.strptime(date_str, '%Y-%m-%d').date()
                        for ticker, date_str in cache_data.get('last_calculated_date', {}).items()
                    }
            except Exception as e:
                print(f"Warning: Could not load cache from {self.cache_file}: {e}")
                self.expected_returns_cache = {}
                self.last_calculated_date = {}

    def _save_cache(self):
        """Save expected returns cache to file."""
        try:
            cache_data = {
                'expected_returns': self.expected_returns_cache,
                'last_calculated_date': {
                    ticker: date_obj.strftime('%Y-%m-%d')
                    for ticker, date_obj in self.last_calculated_date.items()
                }
            }
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")

    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached value for ticker is still valid (calculated today)."""
        if ticker not in self.last_calculated_date:
            return False
        current_date = date.today()
        return self.last_calculated_date[ticker] == current_date

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

    def calculate_error_breakdown(self, y_true: np.ndarray, y_pred: np.ndarray, bias: float = 0.0) -> Dict:
        """
        Calculate detailed breakdown of upside vs downside errors.
        Upside errors have 0.5 penalty, downside errors have 1.0 penalty.

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

        # Calculate weighted RSS components (0.5 for upside, 1.0 for downside)
        rss_upside_weighted = 0.5 * np.sum(upside_errors ** 2) if len(upside_errors) > 0 else 0.0
        rss_downside_weighted = 1.0 * np.sum(downside_errors ** 2) if len(downside_errors) > 0 else 0.0
        rss_total = rss_upside_weighted + rss_downside_weighted

        # Calculate TSS
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)

        # Calculate MAE components
        mae_upside = np.mean(np.abs(upside_errors)) if len(upside_errors) > 0 else 0.0
        mae_downside = np.mean(np.abs(downside_errors)) if len(downside_errors) > 0 else 0.0

        return {
            'rss_total': rss_total,
            'rss_upside_0.5x': rss_upside_weighted,
            'rss_downside_1.0x': rss_downside_weighted,
            'tss': tss,
            'upside_count': len(upside_errors),
            'downside_count': len(downside_errors),
            'upside_mae': mae_upside,
            'downside_mae': mae_downside
        }


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

    def calculate_lstm_features(self, data: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
        """
        Calculate features suitable for LSTM input.

        Args:
            data: DataFrame with OHLCV data
            forward_days: Days ahead to predict (21 = 1 month)

        Returns:
            DataFrame with LSTM features
        """
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

        # Price-based features (normalized)
        df['Price_Normalized'] = close_series / close_series.rolling(window=20).mean()

        # Returns (momentum)
        df['Return_1d'] = close_series.pct_change(1)
        df['Return_5d'] = close_series.pct_change(5)
        df['Return_20d'] = close_series.pct_change(20)

        # Volatility
        if 'Daily_Return' in df.columns:
            df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
            df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

        # Volume indicators
        if 'Volume' in df.columns:
            volume_series = df['Volume']
            if isinstance(volume_series, pd.DataFrame):
                volume_series = volume_series.iloc[:, 0]
            df['Volume_Normalized'] = volume_series / volume_series.rolling(window=20).mean()

        # Moving averages
        df['SMA_10'] = close_series.rolling(window=10).mean()
        df['SMA_30'] = close_series.rolling(window=30).mean()
        df['SMA_Cross'] = df['SMA_10'] / df['SMA_30']

        # High-low range
        if 'High' in df.columns and 'Low' in df.columns:
            high_series = df['High']
            low_series = df['Low']
            if isinstance(high_series, pd.DataFrame):
                high_series = high_series.iloc[:, 0]
            if isinstance(low_series, pd.DataFrame):
                low_series = low_series.iloc[:, 0]
            df['HL_Range'] = (high_series - low_series) / close_series

        # Calculate target (monthly return) - forward-looking
        target_returns = close_series.pct_change(forward_days).shift(-forward_days)

        # Force to Series if needed
        if isinstance(target_returns, pd.DataFrame):
            target_returns = target_returns.iloc[:, 0]

        df['Target_Monthly_Return'] = target_returns

        # Drop NaN
        df.dropna(inplace=True)

        return df

    def create_sequences(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        forward_days: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM input sequences and targets.

        Args:
            data: DataFrame with features (must include 'Target_Monthly_Return')
            feature_cols: List of feature column names
            forward_days: Days ahead to predict (21 = 1 month)

        Returns:
            (X: 3D array [samples, sequence_length, features], y: target returns)
        """
        # Ensure target column exists
        if 'Target_Monthly_Return' not in data.columns:
            raise ValueError(f"Target_Monthly_Return column not found. Available columns: {data.columns.tolist()}")

        # Target should already be calculated in calculate_lstm_features
        data_clean = data.dropna(subset=['Target_Monthly_Return'])

        features = data_clean[feature_cols].values
        targets = data_clean['Target_Monthly_Return'].values

        X, y = [], []

        for i in range(self.sequence_length, len(features) - forward_days):
            X.append(features[i - self.sequence_length:i])
            y.append(targets[i])

        return np.array(X), np.array(y)

    def maae_loss(self, y_true, y_pred):
        """
        Custom MAAE loss function for Keras (TensorFlow backend).

        This enables the model to learn asymmetric error penalties during training.
        Gradients flow through this function to update weights.
        """
        import tensorflow as tf

        # Calculate errors
        errors = y_true - y_pred

        # Separate upside (underprediction) and downside (overprediction) errors
        upside_mask = tf.cast(errors > 0, tf.float32)
        downside_mask = tf.cast(errors <= 0, tf.float32)

        upside_errors = errors * upside_mask

        # Calculate mean and std of upside errors for dynamic thresholds
        mean_upside = tf.reduce_mean(upside_errors)
        std_upside = tf.math.reduce_std(upside_errors)

        # Define thresholds (1σ and 2σ)
        threshold_1std = mean_upside + std_upside
        threshold_2std = mean_upside + 2 * std_upside

        # Apply tiered penalties for upside errors
        # Tier 1: Within 1σ (penalty = 0.2)
        tier1_mask = upside_mask * tf.cast(errors <= threshold_1std, tf.float32)
        tier1_penalty = 0.2 * tf.abs(errors) * tier1_mask

        # Tier 2: Between 1σ and 2σ (penalty = 0.8)
        tier2_mask = upside_mask * tf.cast((errors > threshold_1std) & (errors <= threshold_2std), tf.float32)
        tier2_penalty = 0.8 * tf.abs(errors) * tier2_mask

        # Tier 3: Beyond 2σ (penalty = 1.0)
        tier3_mask = upside_mask * tf.cast(errors > threshold_2std, tf.float32)
        tier3_penalty = 1.0 * tf.abs(errors) * tier3_mask

        # Downside errors: full penalty (1.0)
        downside_penalty = 1.0 * tf.abs(errors) * downside_mask

        # Combine all penalties
        total_penalty = tier1_penalty + tier2_penalty + tier3_penalty + downside_penalty

        # Return mean MAAE
        return tf.reduce_mean(total_penalty)

    def build_lstm_model(self, n_features: int) -> 'keras.Model':
        """
        Build LSTM neural network architecture.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)  # Regression output
        ])

        # Use custom MAAE loss function instead of MSE
        model.compile(optimizer='adam', loss=self.maae_loss, metrics=['mae'])
        return model

    def train_model(self, ticker: str, lookback_days: int = 1260, n_splits_ts: int = 2, n_splits_kfold: int = 2) -> Dict:
        """
        Train LSTM model for a single ticker with dual cross-validation (TimeSeriesSplit + KFold).

        Args:
            ticker: Stock ticker symbol
            lookback_days: Training period (default ~5 years)
            n_splits_ts: Number of time series splits (default 2)
            n_splits_kfold: Number of K-fold splits (default 2)

        Returns:
            Training metrics dict
        """
        if not self.tf_available:
            return {}

        # Get data
        data = get_features_for_ticker(ticker)
        if data.empty:
            return {}

        # Use full historical data (25 years) for better pattern recognition
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Filter to use last 'lookback_days' if specified, otherwise use all data
        if lookback_days is not None and lookback_days > 0:
            data = data.tail(lookback_days)


        # Calculate LSTM features with forward_days=21 (monthly prediction)
        data_with_features = self.calculate_lstm_features(data, forward_days=21)

        # Define features
        feature_cols = [
            'Price_Normalized', 'Return_1d', 'Return_5d', 'Return_20d',
            'Volatility_10', 'Volatility_20', 'Volume_Normalized',
            'SMA_Cross', 'HL_Range'
        ]
        available_features = [f for f in feature_cols if f in data_with_features.columns]
        self.feature_names = available_features

        # Create sequences
        X, y = self.create_sequences(data_with_features, available_features)

        if len(X) < 100:
            return {}

        import warnings
        warnings.filterwarnings('ignore')

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # 1. TimeSeriesSplit CV (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=n_splits_ts)
        cv_losses_ts = []
        cv_mae_ts = []
        cv_mse_ts = []
        cv_r2_ts = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]

            model_cv = self.build_lstm_model(len(available_features))
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history_cv = model_cv.fit(
                X_cv_train, y_cv_train,
                validation_data=(X_cv_val, y_cv_val),
                epochs=15,  # Reduced from 20 for faster CV
                batch_size=64,  # Increased batch size
                callbacks=[early_stop],
                verbose=0
            )

            # Calculate predictions and metrics
            y_pred_val = model_cv.predict(X_cv_val, verbose=0).flatten()
            cv_losses_ts.append(history_cv.history['val_loss'][-1])
            cv_mse_ts.append(mean_squared_error(y_cv_val, y_pred_val))
            cv_mae_ts.append(mean_absolute_error(y_cv_val, y_pred_val))
            cv_r2_ts.append(r2_score(y_cv_val, y_pred_val))

        # 2. K-Fold CV (tests generalization across shuffled splits)
        kfold = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=42)
        cv_losses_kfold = []
        cv_mae_kfold = []
        cv_maae_kfold = []
        cv_mse_kfold = []
        cv_r2_kfold = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]

            model_cv = self.build_lstm_model(len(available_features))
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history_cv = model_cv.fit(
                X_cv_train, y_cv_train,
                validation_data=(X_cv_val, y_cv_val),
                epochs=15,  # Reduced from 20 for faster CV
                batch_size=64,  # Increased batch size
                callbacks=[early_stop],
                verbose=0
            )

            # Calculate predictions and metrics
            y_pred_val = model_cv.predict(X_cv_val, verbose=0).flatten()
            cv_losses_kfold.append(history_cv.history['val_loss'][-1])
            cv_mse_kfold.append(mean_squared_error(y_cv_val, y_pred_val))
            cv_mae_kfold.append(mean_absolute_error(y_cv_val, y_pred_val))
            cv_maae_kfold.append(self.calculate_maae(y_cv_val, y_pred_val))
            cv_r2_kfold.append(r2_score(y_cv_val, y_pred_val))

        # Combine both CV metrics
        cv_mean_loss_ts = np.mean(cv_losses_ts)
        cv_std_loss_ts = np.std(cv_losses_ts)
        cv_mean_loss_kfold = np.mean(cv_losses_kfold)
        cv_std_loss_kfold = np.std(cv_losses_kfold)

        # Use worst-case variance for overfitting detection
        cv_mean_loss = max(cv_mean_loss_ts, cv_mean_loss_kfold)
        cv_std_loss = max(cv_std_loss_ts, cv_std_loss_kfold)

        cv_coefficient_of_variation = cv_std_loss / cv_mean_loss if cv_mean_loss > 0 else 0

        # Detect overfitting: stricter thresholds
        dropout_rate = 0.2
        lstm_units_adjusted = self.lstm_units

        if cv_coefficient_of_variation > 0.4:  # High variance - aggressive reduction
            dropout_rate = 0.5
            lstm_units_adjusted = self.lstm_units // 2
            print(f"  ⚠ {ticker}: High CV variance (CoV={cv_coefficient_of_variation:.2f}) - reducing capacity")
        elif cv_coefficient_of_variation > 0.25:  # Moderate variance
            dropout_rate = 0.35
            lstm_units_adjusted = int(self.lstm_units * 0.75)

        # === K-FOLD ONLY (Simplified - no time-based split) ===
        best_kfold_mse = float('inf')
        best_kfold_model = None
        best_kfold_test_mae = np.nan # Initialize with NaN
        best_kfold_test_maae = np.nan # Initialize with NaN
        best_kfold_train_idx = None
        best_kfold_val_idx = None

        # Scale features for the model training
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Store scalers as instance variables for prediction
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train, y_fold_val = y_scaled[train_idx], y_scaled[val_idx]

            model_fold = Sequential([
                LSTM(lstm_units_adjusted, return_sequences=True, input_shape=(self.sequence_length, len(available_features))),
                Dropout(dropout_rate),
                LSTM(lstm_units_adjusted // 2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(25, activation='relu'),
                Dropout(dropout_rate / 2),
                Dense(1)
            ])
            model_fold.compile(optimizer='adam', loss='mse', metrics=['mae'])
            early_stop_fold = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_fold.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_fold_val),
                          epochs=20, batch_size=64, callbacks=[early_stop_fold], verbose=0)

            y_pred_scaled_fold = model_fold.predict(X_fold_val, verbose=0).flatten()
            mse_fold = mean_squared_error(y_fold_val, y_pred_scaled_fold)

            if mse_fold < best_kfold_mse:
                best_kfold_mse = mse_fold
                best_kfold_model = model_fold
                # Inverse transform predictions to original scale for metric calculation
                y_pred_best_fold_unscaled = scaler_y.inverse_transform(y_pred_scaled_fold.reshape(-1, 1)).flatten()
                y_val_unscaled = scaler_y.inverse_transform(y_fold_val.reshape(-1, 1)).flatten()

                best_kfold_test_mae = mean_absolute_error(y_val_unscaled, y_pred_best_fold_unscaled)
                best_kfold_test_maae = self.calculate_maae(y_val_unscaled, y_pred_best_fold_unscaled)
                best_kfold_train_idx = train_idx
                best_kfold_val_idx = val_idx

        # Store the best K-Fold model
        self.models[ticker] = [best_kfold_model] # Store as a list for consistency

        X_train = X_scaled[best_kfold_train_idx]
        X_test = X_scaled[best_kfold_val_idx]
        y_train = y_scaled[best_kfold_train_idx]
        y_test = y_scaled[best_kfold_val_idx]

        print(f"  ✓ {ticker}: Selected K-Fold best fold (MSE={best_kfold_mse:.6f})")

        # Calculate predictions on the best model's training and testing sets (unscaled)
        y_pred_train_scaled = best_kfold_model.predict(X_train, verbose=0).flatten()
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
        y_train_unscaled = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

        y_pred_test_scaled = best_kfold_model.predict(X_test, verbose=0).flatten()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        ensemble_train_losses = [best_kfold_mse] # MSE from the best fold's validation set
        ensemble_val_losses = [best_kfold_mse]
        ensemble_train_mse = [mean_squared_error(y_train_unscaled, y_pred_train)]
        ensemble_val_mse = [mean_squared_error(y_test_unscaled, y_pred_test)] # MSE of the best fold's test set
        ensemble_train_mae = [mean_absolute_error(y_train_unscaled, y_pred_train)]
        ensemble_val_mae = [best_kfold_test_mae]  # Use stored value from fold selection
        ensemble_train_r2 = [r2_score(y_train_unscaled, y_pred_train)]
        ensemble_val_r2 = [r2_score(y_test_unscaled, y_pred_test)]

        # Calculate MAAE (asymmetrical error) for training only
        train_maae = self.calculate_maae(y_train_unscaled, y_pred_train)
        test_maae = best_kfold_test_maae  # Use stored value from fold selection

        # Calculate R²_asym (proper RSS_asym/TSS calculation) for training and testing
        train_r2_asym = self.calculate_r2_asym(y_train_unscaled, y_pred_train)
        test_r2_asym = self.calculate_r2_asym(y_test_unscaled, y_pred_test)

        # Average ensemble metrics for reporting
        train_loss = np.mean(ensemble_train_losses)
        val_loss = np.mean(ensemble_val_losses)

        # IMPORTANT: We predict MONTHLY returns and rebalance MONTHLY
        # Therefore, the "annual" metrics represent the sum of 12 monthly predictions
        # NOT a single annual prediction

        # For monthly rebalancing strategy:
        # - Monthly MAE measures average error per month
        # - Annual MAE = sum of 12 monthly errors = monthly_MAE * 12
        # This is correct because we make 12 independent monthly predictions

        # Calculate monthly MAE (primary metric)
        train_mae_monthly = np.mean(ensemble_train_mae)
        test_mae_monthly = np.mean(ensemble_val_mae)
        cv_mae_ts_monthly = np.mean(cv_mae_ts)
        cv_mae_kfold_monthly = np.mean(cv_mae_kfold)

        # Calculate annual MAE (sum of 12 monthly prediction errors)
        train_mae_annual = train_mae_monthly * 12  # Monthly MAE * 12 months
        test_mae_annual = test_mae_monthly * 12
        cv_mae_ts_annual = cv_mae_ts_monthly * 12
        cv_mae_kfold_annual = cv_mae_kfold_monthly * 12

        # Calculate monthly MSE (used for model comparison)
        train_mse_monthly = np.mean(ensemble_train_mse)
        test_mse_monthly = np.mean(ensemble_val_mse)
        cv_mse_ts_monthly = np.mean(cv_mse_ts)
        cv_mse_kfold_monthly = np.mean(cv_mse_kfold)

        # MSE for annual (scales quadratically: MSE_annual = MSE_monthly * 12^2)
        train_mse_annual = train_mse_monthly * (12 ** 2)
        test_mse_annual = test_mse_monthly * (12 ** 2)
        cv_mse_ts_annual = cv_mse_ts_monthly * (12 ** 2)
        cv_mse_kfold_annual = cv_mse_kfold_monthly * (12 ** 2)

        # Ensure output directory exists before saving models if needed
        os.makedirs(self.output_dir, exist_ok=True)

        # R² is scale-invariant, but include annual versions for consistency
        train_r2_annual = np.mean(ensemble_train_r2)
        test_r2_annual = np.mean(ensemble_val_r2)
        cv_r2_ts_annual = np.mean(cv_r2_ts)
        cv_std_r2_ts_annual = np.std(cv_r2_ts)
        cv_r2_kfold_annual = np.mean(cv_r2_kfold)
        cv_std_r2_kfold_annual = np.std(cv_r2_kfold)

        # Store metrics for portfolio summary calculation
        self.model_metrics[ticker] = {
            'winning_method': 'kfold_best', # Always kfold_best now
            'train_mae': np.mean(ensemble_train_mae),
            'test_mae': np.mean(ensemble_val_mae),
            'train_r2': np.mean(ensemble_train_r2),
            'test_r2': np.mean(ensemble_val_r2),
            'daily_mse': np.mean(ensemble_val_mse), # Using test_mse as proxy for daily
            'annual_mse': test_mse_annual,
            'daily_mae': np.mean(ensemble_val_mae), # Using test_mae as proxy for daily
            'annual_mae': test_mae_annual,
            'expected_return': np.nan # Placeholder, actual expected return calculated later
        }

        return {
            'ticker': ticker,
            'winning_method': 'kfold_best', # Always kfold_best now
            # Monthly metrics (match our prediction horizon)
            'train_loss_mse': train_loss,
            'test_loss_mse': val_loss,
            'train_mae': np.mean(ensemble_train_mae),
            'test_mae': best_kfold_test_mae,  # Different from kfold (training set)
            'train_maae': train_maae,
            'test_maae': best_kfold_test_maae,  # Different from kfold (validation set)
            'train_mse': train_mse_monthly,
            'test_mse': test_mse_monthly,
            'train_r2': np.mean(ensemble_train_r2),
            'test_r2': np.mean(ensemble_val_r2),
            # Cross-validation metrics
            'cv_mean_r2': np.mean(cv_r2_kfold), # Use KFold for mean R2
            'cv_mean_mse': cv_mse_kfold_monthly, # Use KFold for mean MSE
            'cv_std_mse': np.std(cv_mse_kfold), # Use KFold for std MSE
            'cv_mean_mae': np.mean(cv_mae_kfold), # Use KFold for mean MAE
            'cv_std_mae': np.std(cv_mae_kfold), # Use KFold for std MAE
            'cv_mean_maae': np.mean(cv_maae_kfold), # KFold MAAE
            'cv_std_maae': np.std(cv_maae_kfold), # KFold MAAE std
            'cv_mean_loss_ts': cv_mean_loss_ts, # Keep TimeSeriesSplit loss for reference
            'cv_std_loss_ts': cv_std_loss_ts,
            'cv_mean_loss_kfold': cv_mean_loss_kfold, # Use KFold loss
            'cv_std_loss_kfold': cv_std_loss_kfold,
            'cv_mean_mae_kfold': np.mean(cv_mae_kfold),
            'cv_std_mse_kfold': np.mean(cv_mse_kfold),
            'cv_std_mse_kfold': np.std(cv_mse_kfold),
            'cv_coefficient_of_variation': cv_coefficient_of_variation,
            'n_sequences': len(X),
            'n_features': len(available_features),
            'dropout_used': dropout_rate,
            'ensemble_size': 1, # Only one model selected now
            'kfold_best_test_mae': best_kfold_test_mae, # Add best K-Fold MAE
            'kfold_best_test_maae': best_kfold_test_maae, # Add best K-Fold MAAE
            'kfold_best_test_r2': r2_score(y_test_unscaled, y_pred_test), # Add best K-Fold R²
            'train_r2_asym': train_r2_asym, # Novel asymmetric R² for training
            'test_r2_asym': test_r2_asym, # Novel asymmetric R² for testing
        }

    def predict_annual_return(self, ticker: str) -> float:
        """
        Predict expected annual return for a ticker using ensemble averaging.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Expected annual return (ensemble-averaged and capped)
        """
        if not self.tf_available:
            data = get_features_for_ticker(ticker)
            if not data.empty and 'Daily_Return' in data.columns:
                daily_mean = data['Daily_Return'].tail(252).mean()
                return daily_mean * 252
            return 0.0

        if ticker not in self.models:
            return 0.0

        # Get latest data
        data = get_features_for_ticker(ticker)
        if data.empty:
            return 0.0

        # Calculate features
        data_with_features = self.calculate_lstm_features(data, forward_days=21)
        latest_features = data_with_features[self.feature_names].tail(self.sequence_length).values
        X_latest = np.array([latest_features])

        # Scale the latest features using the scaler fitted during training
        if hasattr(self, 'scaler_X') and hasattr(self, 'scaler_y'): # Check if scalers exist
            X_latest_scaled = self.scaler_X.transform(X_latest.reshape(-1, X_latest.shape[-1])).reshape(X_latest.shape)
        else:
            # Fallback if scalers were not fitted or not available (e.g., during initial run without scaling)
            # This part might need adjustment if scaling is always applied.
            # For now, assume X_latest is already scaled if models exist.
            X_latest_scaled = X_latest


        # Ensemble prediction: average across all models
        ensemble_predictions = []
        for model in self.models[ticker]:
            monthly_pred_scaled = model.predict(X_latest_scaled, verbose=0)[0][0]
            # Inverse transform the prediction
            monthly_pred = self.scaler_y.inverse_transform(np.array([[monthly_pred_scaled]]))[0][0]
            ensemble_predictions.append(monthly_pred)

        # Average ensemble predictions
        monthly_return = np.mean(ensemble_predictions)

        # No bias correction - use raw model predictions
        annual_return = float(monthly_return) * 12

        # Sanity check: Flag unrealistic predictions (but don't cap for Arbiter fairness)
        if abs(annual_return) > 2.0:  # >200% annual return is unrealistic
            print(f"  ⚠️  WARNING: {ticker} LSTM prediction is {annual_return:.2%} (monthly: {monthly_return:.2%})")
            print(f"     This suggests model instability or data issues.")
            print(f"     Raw ensemble predictions: {ensemble_predictions}")
            # Return a fallback value based on historical average
            data = get_features_for_ticker(ticker)
            if not data.empty and 'Daily_Return' in data.columns:
                daily_mean = data['Daily_Return'].tail(252).mean()
                annual_return = daily_mean * 252
                print(f"     Using historical average instead: {annual_return:.2%}")

        # Update the placeholder in model_metrics
        if ticker in self.model_metrics:
            self.model_metrics[ticker]['expected_return'] = annual_return

        return annual_return

    def generate_expected_returns(
        self,
        tickers: List[str],
        force_recalculate: bool = False # Default set to False as requested
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate expected monthly return vector for portfolio optimization.

        Args:
            tickers: List of stock tickers
            force_recalculate: If True, recalculate even if cached values exist for today

        Returns:
            (expected_returns_array, training_metrics_df, approved_tickers)
        """
        print("\n" + "="*60)
        print("LSTM ENGINE: TRAINING & PREDICTION")
        print("="*60)

        # Train on all provided tickers
        print(f"  Training LSTM models for {len(tickers)} securities...")


        # Check if user wants to use cached values
        if not force_recalculate:
            cached_count = sum(1 for ticker in tickers if self._is_cache_valid(ticker))

            # Debug: Show cache status
            print(f"\n📊 Cache Status:")
            print(f"   Cache file: {self.cache_file}")
            print(f"   Cache exists: {os.path.exists(self.cache_file)}")
            print(f"   Cached tickers: {list(self.expected_returns_cache.keys())}")
            print(f"   Valid for today: {cached_count}/{len(tickers)}")

            if cached_count > 0:
                print(f"\n{'='*60}")
                print(f"CACHE DETECTED: {cached_count}/{len(tickers)} tickers have cached returns from today.")
                print(f"{'='*60}")

                # Default to using cache automatically
                use_cache = True
                print("\n✓ Automatically using cached values (faster)")
                print("  (To recalculate, restart and set force_recalculate=True)")
            else:
                use_cache = False
                print("\n⚠ No valid cache found. Will calculate expected returns (this will take ~10-15 minutes)...")
        else:
            use_cache = False
            print("Force recalculation requested. Ignoring cache...")

        expected_returns = {}
        metrics_list = []
        current_date = date.today()
        calculated = 0  # Track number of newly calculated returns
        approved_tickers = [] # List to store approval status (1 or 0)

        MAE_THRESHOLD = 0.05  # 5% monthly maximum acceptable prediction error

        for idx, ticker in enumerate(tickers, 1):
            # Check if we can use cached value
            if use_cache and self._is_cache_valid(ticker):
                cached_return = self.expected_returns_cache[ticker]
                expected_returns[ticker] = cached_return
                approved_tickers.append(1) # Assume approved if using cache
                print(f"  ✓ {ticker}: Using cached ER = {cached_return:.2%} (calculated today)")
                metrics_list.append({
                    'ticker': ticker,
                    'expected_annual_return': cached_return,
                    'method': 'cached',
                    'Approved': 1 # Explicitly set approved status for cached items
                })
                continue

            # Calculate new expected return
            print(f"\n[{idx}/{len(tickers)}] Training LSTM for {ticker}... (this may take 2-3 minutes)")
            metrics = self.train_model(ticker)

            if metrics or not self.tf_available:
                annual_return = self.predict_annual_return(ticker)
                expected_returns[ticker] = annual_return

                # Cache the result
                self.expected_returns_cache[ticker] = annual_return
                self.last_calculated_date[ticker] = current_date

                # Determine approval status based on K-Fold MAAE threshold
                # Use kfold_best_test_maae as the primary quality metric
                kfold_best_test_maae = metrics.get('kfold_best_test_maae', float('inf')) if metrics else float('inf')
                
                is_approved = 1 if kfold_best_test_maae <= MAE_THRESHOLD else 0
                approved_tickers.append(is_approved)

                # Increment calculated counter
                calculated += 1

                if metrics:
                    metrics['expected_annual_return'] = annual_return
                    metrics['Approved'] = is_approved # Add approval status to metrics
                    metrics_list.append(metrics)
                else:
                    metrics_list.append({
                        'ticker': ticker,
                        'expected_annual_return': annual_return,
                        'method': 'fallback',
                        'Approved': 0 # Mark as not approved if fallback
                    })
            else:
                expected_returns[ticker] = 0.0
                approved_tickers.append(0) # Not approved if calculation failed
                metrics_list.append({
                    'ticker': ticker,
                    'expected_annual_return': 0.0,
                    'method': 'failed',
                    'Approved': 0
                })


        # Save cache after all calculations
        self._save_cache()

        expected_returns_array = np.array(list(expected_returns.values()))

        # Save detailed metrics matching ElasticNet format
        os.makedirs('Results/LSTM', exist_ok=True)

        # Rename winning method columns to include method name for non-cached entries
        # Match ElasticNet naming: method_metric (e.g., kfold_best_train_mae)
        for m in metrics_list:
            if m.get('method') != 'cached' and m.get('method') != 'fallback' and m.get('method') != 'failed':
                method = m.get('winning_method', 'unknown')
                # Add winning method prefix to columns - ORDER: MAE, R², MSE
                if 'train_mae' in m:
                    m[f'{method}_train_mae'] = m['train_mae']
                if 'test_mae' in m:
                    m[f'{method}_test_mae'] = m['test_mae']
                if 'train_r2' in m:
                    m[f'{method}_train_r2'] = m['train_r2']
                if 'test_r2' in m:
                    m[f'{method}_test_r2'] = m['test_r2']
                if 'train_mse' in m:
                    m[f'{method}_train_mse'] = m['train_mse']
                if 'test_mse' in m:
                    m[f'{method}_test_mse'] = m['test_mse']

        # Create detailed DataFrame from metrics_list
        metrics_df = pd.DataFrame(metrics_list)

        # Reorder columns to match ElasticNet format EXACTLY - BEFORE portfolio summary
        base_cols = ['ticker', 'winning_method', 'method', 'expected_annual_return', 'Approved']

        # Find winning method columns in specific order: MAE first, then R², then MSE
        winning_method_prefix = 'kfold_best' # Assuming kfold_best is always the winning method
        winning_mae_cols = [col for col in metrics_df.columns if winning_method_prefix in col
                           and 'mae' in col.lower() and col not in base_cols]
        winning_r2_cols = [col for col in metrics_df.columns if winning_method_prefix in col
                          and 'r2' in col.lower() and col not in base_cols]
        winning_mse_cols = [col for col in metrics_df.columns if winning_method_prefix in col
                           and 'mse' in col.lower() and col not in base_cols]

        # Sort each metric type: train before test
        def sort_metric_cols(cols):
            train = sorted([c for c in cols if 'train' in c])
            test = sorted([c for c in cols if 'test' in c])
            return train + test

        winning_mae_cols = sort_metric_cols(winning_mae_cols)
        winning_r2_cols = sort_metric_cols(winning_r2_cols)
        winning_mse_cols = sort_metric_cols(winning_mse_cols)

        # Combine winning method columns in MAE, R², MSE order
        winning_cols = winning_mae_cols + winning_r2_cols + winning_mse_cols

        # CV and other columns
        other_cols = [col for col in metrics_df.columns if col not in base_cols + winning_cols]

        # Separate CV columns from other columns - CV columns go LAST
        cv_cols = [col for col in other_cols if 'cv_' in col]
        non_cv_other_cols = [col for col in other_cols if 'cv_' not in col]

        # Order: base -> winning method (MAE, R², MSE) -> non-CV others -> CV columns LAST
        ordered_cols = base_cols + winning_cols + non_cv_other_cols + cv_cols
        # Ensure all columns in ordered_cols actually exist in metrics_df before selecting
        final_ordered_cols = [col for col in ordered_cols if col in metrics_df.columns]
        metrics_df = metrics_df[final_ordered_cols]

        # Filter out portfolio summary row (none exists yet)
        df_securities = metrics_df[metrics_df['ticker'] != 'PORTFOLIO'].copy()

        # Count securities failing threshold
        # Check for MAE columns with winning method prefix
        mae_column_to_check = None
        for col in df_securities.columns:
            if f'{winning_method_prefix}_test_mae' in col:
                mae_column_to_check = col
                break

        if mae_column_to_check:
            failed_securities = df_securities[df_securities[mae_column_to_check] > MAE_THRESHOLD]
        else:
            failed_securities = pd.DataFrame()  # No suitable MAE column found

        total_securities = len(df_securities)
        num_failed = len(failed_securities)

        failure_rate = num_failed / total_securities if total_securities > 0 else 0

        print(f"\nLSTM Security-Level Threshold Check:")
        print(f"  Securities Analyzed: {total_securities}")
        print(f"  Securities Failed (MAE > {MAE_THRESHOLD:.1%}): {num_failed}")
        print(f"  Failure Rate: {failure_rate:.1%}")

        # Auto-elimination: >50% of securities fail threshold
        if failure_rate > 0.50:
            print(f"\n[AUTO-ELIMINATION WARNING] LSTM model shows {failure_rate:.1%} security failure rate.")
            print(f"  Rule: Models with >50% security failure rate are fundamentally unreliable.")
            if not failed_securities.empty:
                print(f"  Failed Securities: {', '.join(failed_securities['ticker'].tolist())}")
            print(f"  Interpretation: Model cannot accurately predict majority of portfolio components.")
            print(f"  Note: This model will be disqualified in Stage 3 Arbiter evaluation.")

        metrics_lstm = metrics_df # Rename for consistency

        # Add portfolio summary row
        portfolio_summary = {
            'ticker': 'PORTFOLIO',
            'winning_method': 'portfolio_avg',
            'method': 'portfolio_avg',
            'Approved': 1 # Portfolio is approved if majority of its components are
        }

        # Calculate averages for all numeric columns, excluding non-numeric ones
        numeric_cols = metrics_lstm.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Approved']: # Exclude 'Approved' from averaging directly
                 portfolio_summary[col] = metrics_lstm[col].mean()

        # Calculate average 'Approved' status for the portfolio summary
        # Count how many individual securities were approved (Approved == 1)
        approved_count = metrics_lstm['Approved'].sum()
        # Calculate the approval percentage based on the total number of securities processed (excluding portfolio row itself)
        total_securities_for_approval = len(metrics_lstm) - 1 # Subtract the portfolio row
        portfolio_summary['Approved_Percent'] = (approved_count / total_securities_for_approval * 100) if total_securities_for_approval > 0 else 0


        # Add expected annual return average (only for non-cached/non-failed entries)
        valid_returns = [m.get('expected_annual_return', 0) for m in metrics_list if m.get('method') not in ['cached', 'fallback', 'failed']]
        portfolio_summary['expected_annual_return'] = np.mean(valid_returns) if valid_returns else 0


        # Append portfolio row
        portfolio_row_df = pd.DataFrame([portfolio_summary])
        metrics_lstm = pd.concat([metrics_lstm, portfolio_row_df], ignore_index=True)

        metrics_lstm.to_csv('Results/LSTM/lstm_detailed_metrics.csv', index=False)
        print("✓ Saved detailed LSTM metrics to 'Results/LSTM/lstm_detailed_metrics.csv'")

        # Save all metrics (including cached) separately
        metrics_df_all = pd.DataFrame(metrics_list)
        metrics_df_all.to_csv('Results/LSTM/lstm_all_expected_returns.csv', index=False)
        print("✓ Saved all expected returns to 'Results/LSTM/lstm_all_expected_returns.csv'")

        # Create R²_asym vs R² comparison table
        r2_comparison = []
        for m in metrics_list:
            if m.get('method') not in ['cached', 'fallback', 'failed']:
                ticker = m.get('ticker', 'N/A')
                method = m.get('winning_method', 'unknown')
                train_r2 = m.get('train_r2', np.nan)
                train_r2_asym = m.get('train_r2_asym', np.nan)
                test_r2 = m.get('test_r2', np.nan)
                test_r2_asym = m.get('test_r2_asym', np.nan)

                r2_comparison.append({
                    'Ticker': ticker,
                    'Train_R2': train_r2,
                    'Train_R2_asym': train_r2_asym,
                    'Test_R2': test_r2,
                    'Test_R2_asym': test_r2_asym,
                    'Train_R2_Improvement': train_r2_asym - train_r2 if not pd.isna(train_r2_asym) and not pd.isna(train_r2) else np.nan,
                    'Test_R2_Improvement': test_r2_asym - test_r2 if not pd.isna(test_r2_asym) and not pd.isna(test_r2) else np.nan
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
            r2_comparison_df.to_csv('Results/LSTM/r2_asym_vs_r2_comparison.csv', index=False)
            print("✓ Saved R²_asym vs R² comparison to 'Results/LSTM/r2_asym_vs_r2_comparison.csv'")

        # Create simplified winning metrics table (MAE, R², MSE only)
        winning_metrics_cols = ['ticker', 'winning_method', 'method']

        # Add winning method columns (MAE, R², MSE order)
        for col in metrics_lstm.columns:
            if winning_method_prefix in col:
                if any(metric in col for metric in ['mae', 'r2', 'mse']):
                    winning_metrics_cols.append(col)

        # Add basic metrics (MAE, R², MSE order) - only if not already included via winning method
        basic_metrics = ['train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_mse', 'test_mse']
        for col in basic_metrics:
            if col in metrics_lstm.columns and col not in winning_metrics_cols:
                winning_metrics_cols.append(col)

        # Ensure 'Approved' and 'expected_annual_return' are also included if they exist
        if 'Approved' in metrics_lstm.columns:
            winning_metrics_cols.append('Approved')
        if 'expected_annual_return' in metrics_lstm.columns:
            winning_metrics_cols.append('expected_annual_return')

        # Create simplified dataframe
        final_winning_cols = [col for col in winning_metrics_cols if col in metrics_lstm.columns]
        winning_metrics_df = metrics_lstm[final_winning_cols].copy()
        winning_metrics_df.to_csv('Results/LSTM/lstm_winning_metrics.csv', index=False)
        print("✓ Saved simplified winning metrics to 'Results/LSTM/lstm_winning_metrics.csv'")

        # Save expected returns to CSV
        returns_df = pd.DataFrame({
            'Ticker': [m.get('ticker', 'N/A') for m in metrics_list],
            'Expected_Return': [m.get('expected_annual_return', 0.0) for m in metrics_list],
            'Approved': [m.get('Approved', 0) for m in metrics_list] # Add approved status here
        })
        returns_df.to_csv('Results/LSTM/expected_returns_lstm.csv', index=False)

        # Save error breakdown table with simplified upside/downside components
        error_breakdown_rows = []
        for ticker in tickers:
            if ticker not in self.models:
                continue

            # Get test data for this ticker
            data = get_features_for_ticker(ticker)
            if data.empty:
                continue

            data_with_features = self.calculate_lstm_features(data, forward_days=21)
            available_features = [f for f in self.feature_names if f in data_with_features.columns]
            X, y = self.create_sequences(data_with_features, available_features)

            if len(X) < 100:
                continue

            # Scale and split
            X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()

            # Use same split as training (best K-fold)
            kfold = KFold(n_splits=2, shuffle=True, random_state=42)
            for train_idx, val_idx in kfold.split(X_scaled):
                X_test = X_scaled[val_idx]
                y_test = y_scaled[val_idx]
                break  # Use first fold

            # Predict and inverse transform
            y_pred_scaled = self.models[ticker][0].predict(X_test, verbose=0).flatten()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_unscaled = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Calculate error breakdown
            breakdown = self.calculate_error_breakdown(y_test_unscaled, y_pred)

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

            error_breakdown_df.to_csv('Results/LSTM/error_breakdown_analysis.csv', index=False)
            print("✓ Saved error breakdown analysis to 'Results/LSTM/error_breakdown_analysis.csv'")

        # Save rejected securities CSV
        rejected_securities = []
        for m in metrics_list:
            if m.get('method') not in ['cached', 'fallback', 'failed']:
                method = m.get('winning_method', 'unknown')
                test_maae_col = f'{method}_test_maae'
                if test_maae_col in m and m[test_maae_col] > MAE_THRESHOLD:
                    rejected_securities.append({
                        'Ticker': m['ticker'],
                        'MAAE': m[test_maae_col],
                        'Threshold': MAE_THRESHOLD,
                        'Miss_Amount': m[test_maae_col] - MAE_THRESHOLD,
                        'Best_MAAE': m.get('test_maae', np.nan) # Fallback if test_maae is missing
                    })

        if rejected_securities:
            rejected_df = pd.DataFrame(rejected_securities)
            rejected_df.to_csv('Results/LSTM/lstm_rejected_securities.csv', index=False)
            print("✓ Saved rejected securities details to 'Results/LSTM/lstm_rejected_securities.csv'")
        else:
            print("No securities failed the MAE threshold for rejection analysis.")


        # Summary
        cached_used = sum(1 for m in metrics_list if m.get('method') == 'cached')
        print(f"\n{'='*60}")
        print(f"LSTM ENGINE SUMMARY:")
        print(f"  Used cache: {cached_used} tickers")
        print(f"  Calculated: {calculated} tickers")
        print(f"  Total time saved: ~{cached_used * 2.5:.1f} minutes")
        print(f"{'='*60}")

        # Print detailed metrics table matching ElasticNet format EXACTLY
        if metrics_list:
            print("\n" + "="*200)
            print("LSTM DETAILED METRICS TABLE (Monthly Prediction Horizon)")
            print("="*200)
            # Get first method name to use in header (like ElasticNet does)
            first_method = next((m.get('winning_method') for m in metrics_list if m.get('method') not in ['cached', 'fallback', 'failed']), 'Method')

            # Header formatting: Adjust spacing based on expected content length
            header_format = "{:<8} {:<15} {:<15} {:<18} {TrainMAE:<19} {TestMAE:<19} {TrainR2:<19} {TestR2:<19} {TrainMSE:<19} {TestMSE:<19}"
            print(header_format.format('Ticker', 'Method', 'Total Security', 'Approve Percent', TrainMAE=f'{first_method}_TrainMAE', TestMAE=f'{first_method}_TestMAE', TrainR2=f'{first_method}_TrainR²', TestR2=f'{first_method}_TestR²', TrainMSE=f'{first_method}_TrainMSE', TestMSE=f'{first_method}_TestMSE'))
            print("-"*200)

            for m in metrics_list:
                ticker = m.get('ticker', 'N/A')
                method = m.get('winning_method', m.get('method', 'N/A'))
                approved = m.get('Approved', 0)
                total_security_count = len(tickers)
                approve_percent = (approved / total_security_count * 100) if total_security_count > 0 else 0

                # Get metrics using winning method prefix, or fallback to basic metrics if not present
                train_mae = m.get(f'{method}_train_mae', m.get('train_mae', 0.0))
                test_mae = m.get(f'{method}_test_mae', m.get('test_mae', 0.0))
                train_r2 = m.get(f'{method}_train_r2', m.get('train_r2', 0.0))
                test_r2 = m.get(f'{method}_test_r2', m.get('test_r2', 0.0))
                train_mse = m.get(f'{method}_train_mse', m.get('train_mse', 0.0))
                test_mse = m.get(f'{method}_test_mse', m.get('test_mse', 0.0))

                # Apply formatting for each metric
                print(f"{ticker:<8} {method:<15} {total_security_count:<15} {approve_percent:>17.2f}% {train_mae:>19.4f} {test_mae:>19.4f} {train_r2:>19.3f} {test_r2:>19.3f} {train_mse:>19.4f} {test_mse:>19.4f}")

            print("="*200)

            # Print portfolio summary
            print(f"\n{'PORTFOLIO SUMMARY':<8}")
            print("-"*200)
            # Get portfolio metrics using winning method columns - ORDER: MAE, R², MSE
            non_failed_or_cached = [m for m in metrics_list if m.get('method') not in ['cached', 'fallback', 'failed']]

            avg_train_mae = np.nanmean([m.get(f'{m.get("winning_method")}_train_mae', m.get('train_mae', np.nan)) for m in non_failed_or_cached])
            avg_test_mae = np.nanmean([m.get(f'{m.get("winning_method")}_test_mae', m.get('test_mae', np.nan)) for m in non_failed_or_cached])
            avg_train_r2 = np.nanmean([m.get(f'{m.get("winning_method")}_train_r2', m.get('train_r2', np.nan)) for m in non_failed_or_cached])
            avg_test_r2 = np.nanmean([m.get(f'{m.get("winning_method")}_test_r2', m.get('test_r2', np.nan)) for m in non_failed_or_cached])
            avg_train_mse = np.nanmean([m.get(f'{m.get("winning_method")}_train_mse', m.get('train_mse', np.nan)) for m in non_failed_or_cached])
            avg_test_mse = np.nanmean([m.get(f'{m.get("winning_method")}_test_mse', m.get('test_mse', np.nan)) for m in non_failed_or_cached])

            num_models = len(non_failed_or_cached)
            models_label = f'{num_models} models'
            
            # Calculate approval percent for summary row
            approved_count = sum(m.get('Approved', 0) for m in metrics_list if m.get('ticker') != 'PORTFOLIO')
            total_securities_for_approval = len([m for m in metrics_list if m.get('ticker') != 'PORTFOLIO'])
            approval_percent = (approved_count / total_securities_for_approval * 100) if total_securities_for_approval > 0 else 0

            print(f"{'Average':<8} {models_label:<15} {num_models:<15} {approval_percent:>17.2f}% {avg_train_mae:>19.4f} {avg_test_mae:>19.4f} {avg_train_r2:>19.3f} {avg_test_r2:>19.3f} {avg_train_mse:>19.4f} {avg_test_mse:>19.4f}")
            print("="*200)

        return expected_returns_array, metrics_df_all, approved_tickers # Return array, metrics, and approval status

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

        returns_df = pd.DataFrame(returns_data).dropna()

        # Scale to annual (252 trading days)
        annual_cov_matrix = returns_df.cov() * 252

        # Save covariance matrix
        os.makedirs('Results/LSTM', exist_ok=True)
        annual_cov_matrix.to_csv('Results/LSTM/lstm_covariance_matrix.csv')
        print("✓ Saved covariance matrix to 'Results/LSTM/lstm_covariance_matrix.csv'")

        return annual_cov_matrix

    def optimize_portfolio_mvo(self, expected_returns: np.ndarray, cov_matrix: pd.DataFrame,
                                tickers: List[str], risk_score: int = 5):
        """
        Perform Mean-Variance Optimization using LSTM expected returns.

        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            tickers: List of tickers
            risk_score: Client risk score (1-9)

        Returns:
            Dictionary with optimal portfolio and metrics
        """
        from Libraries.CAPM_Engine import MPTOptimizer

        # Ensure cov_matrix is a numpy array if it's a DataFrame
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix_np = cov_matrix.values
        else:
            cov_matrix_np = cov_matrix

        # Ensure expected_returns is a numpy array
        if isinstance(expected_returns, pd.Series):
            expected_returns_np = expected_returns.values
        else:
            expected_returns_np = expected_returns

        # Ensure tickers match the order of expected_returns and cov_matrix
        # Assuming expected_returns and cov_matrix are already aligned with tickers

        optimizer = MPTOptimizer(expected_returns_np, cov_matrix_np, tickers, risk_score=risk_score)

        # Generate efficient frontier
        frontier = optimizer.generate_efficient_frontier(num_points=99)

        # Get key portfolios
        mvp = optimizer.minimize_volatility()
        max_sharpe = optimizer.maximize_sharpe_ratio()

        return {
            'frontier': frontier,
            'mvp': mvp,
            'max_sharpe': max_sharpe,
            'optimizer': optimizer
        }