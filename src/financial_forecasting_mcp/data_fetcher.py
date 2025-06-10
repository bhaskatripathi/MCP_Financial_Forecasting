"""Yahoo Finance data fetcher for financial forecasting."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class YahooFinanceDataFetcher:
    """Fetches and processes financial data from Yahoo Finance."""
    
    def __init__(self, timeout: int = 30):
        """Initialize the data fetcher.
        
        Args:
            timeout: Timeout for Yahoo Finance requests in seconds
        """
        self.timeout = timeout
    
    def fetch_ohlc_data(
        self, 
        symbol: str, 
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLC data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLC data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, timeout=self.timeout)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'current_price': info.get('currentPrice', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
            }
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a stock symbol exists.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", timeout=self.timeout)
            return not data.empty
        except:
            return False 