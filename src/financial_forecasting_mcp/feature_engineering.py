"""Feature engineering for financial forecasting."""

import pandas as pd
import numpy as np
import ta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FinancialFeatureEngineering:
    """Creates features from OHLC data for machine learning models."""
    
    def __init__(self):
        """Initialize the feature engineering class."""
        pass
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLC data.
        
        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        try:
            data = df.copy()
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Price-based indicators
            data['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
            data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
            data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
            data['bb_upper'] = bollinger.bollinger_hband()
            data['bb_middle'] = bollinger.bollinger_mavg()
            data['bb_lower'] = bollinger.bollinger_lband()
            data['bb_width'] = data['bb_upper'] - data['bb_lower']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # RSI
            data['rsi'] = ta.momentum.rsi(data['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            data['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'], window=20)
            data['volume_weighted_price'] = ta.volume.volume_weighted_average_price(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # Average True Range (volatility)
            data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
            
            logger.info(f"Created technical indicators for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {str(e)}")
            raise
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with additional price features
        """
        try:
            data = df.copy()
            
            # Price ratios
            data['high_low_ratio'] = data['high'] / data['low']
            data['close_open_ratio'] = data['close'] / data['open']
            data['high_close_ratio'] = data['high'] / data['close']
            data['low_close_ratio'] = data['low'] / data['close']
            
            # Price changes
            data['price_change'] = data['close'].pct_change()
            data['price_change_2d'] = data['close'].pct_change(periods=2)
            data['price_change_5d'] = data['close'].pct_change(periods=5)
            
            # Log returns
            data['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # Range indicators
            data['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            
            # Intraday patterns
            data['body_size'] = abs(data['close'] - data['open'])
            data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
            data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
            
            logger.info(f"Created price features for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error creating price features: {str(e)}")
            raise
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lag features for time series modeling.
        
        Args:
            df: DataFrame with features
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        try:
            data = df.copy()
            
            # Create lag features for close price
            for lag in lags:
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
                data[f'price_change_lag_{lag}'] = data['price_change'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                data[f'close_rolling_mean_{window}'] = data['close'].rolling(window=window).mean()
                data[f'close_rolling_std_{window}'] = data['close'].rolling(window=window).std()
                data[f'volume_rolling_mean_{window}'] = data['volume'].rolling(window=window).mean()
            
            logger.info(f"Created lag features for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        include_technical: bool = True,
        include_price: bool = True,
        include_lags: bool = True,
        lag_periods: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Prepare all features for modeling.
        
        Args:
            df: DataFrame with OHLC data
            include_technical: Whether to include technical indicators
            include_price: Whether to include price features
            include_lags: Whether to include lag features
            lag_periods: List of lag periods to create
            
        Returns:
            DataFrame with all features
        """
        try:
            data = df.copy()
            
            if include_technical:
                data = self.create_technical_indicators(data)
            
            if include_price:
                data = self.create_price_features(data)
            
            if include_lags:
                data = self.create_lag_features(data, lags=lag_periods)
            
            # Drop rows with NaN values (due to rolling windows and lags)
            initial_rows = len(data)
            data = data.dropna()
            final_rows = len(data)
            
            logger.info(f"Feature preparation complete. Rows: {initial_rows} -> {final_rows}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def get_feature_names(self, exclude_ohlc: bool = True) -> List[str]:
        """Get list of feature column names (excluding OHLC and target).
        
        Args:
            exclude_ohlc: Whether to exclude basic OHLC columns
            
        Returns:
            List of feature column names
        """
        basic_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        
        feature_patterns = [
            'sma_', 'ema_', 'bb_', 'rsi', 'macd', 'stoch_', 'volume_',
            'atr', '_ratio', '_change', 'log_return', 'true_range',
            'body_size', '_shadow', '_lag_', 'rolling_'
        ]
        
        return feature_patterns 