import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    """
    Data loading utility for financial data
    """
    
    @staticmethod
    def get_features_for_ticker(ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        Get features for a ticker (matching your existing function)
        """
        try:
            # Download data from yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate basic features
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Price_SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Price_SMA_50'] = data['Close'].rolling(window=50).mean()
            data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
            
            # Drop NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_multiple_tickers(tickers: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers
        """
        data_dict = {}
        for ticker in tickers:
            data = DataLoader.get_features_for_ticker(ticker, period)
            if not data.empty:
                data_dict[ticker] = data
        
        return data_dict

# Create alias for compatibility with your existing code
cleaned_financial_data = DataLoader.get_features_for_ticker
get_features_for_ticker = DataLoader.get_features_for_ticker