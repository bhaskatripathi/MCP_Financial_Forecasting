# Financial Forecasting MCP

A Model Context Protocol (MCP) server that provides SVM-based financial forecasting capabilities with OpenAI-powered interpretation and explanation.

## Features

- **Yahoo Finance Integration**: Fetch real-time OHLC data for any stock symbol
- **Advanced Feature Engineering**: Technical indicators, price patterns, and lag features
- **SVM Forecasting**: Support Vector Machine models for price prediction
- **OpenAI Interpretation**: AI-powered explanation and analysis of forecast results
- **Hyperparameter Optimization**: Automated parameter tuning for better performance
- **Risk Assessment**: Confidence metrics and risk factor identification

## Tools Available

### Data Operations
- `fetch_stock_data`: Get OHLC data from Yahoo Finance
- `get_stock_info`: Retrieve basic stock information
- `validate_stock_symbol`: Check if a symbol is valid and tradeable

### Modeling & Forecasting
- `train_svm_model`: Train SVM model on historical data
- `make_forecast`: Generate price predictions with AI interpretation
- `explain_model_performance`: Get detailed model performance analysis

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   Create a `.env` file based on `env.example`:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Configuration

The MCP server can be configured through environment variables:

- `OPENAI_API_KEY`: Required for forecast interpretation
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `YAHOO_FINANCE_TIMEOUT`: Timeout for Yahoo Finance requests (default: 30)
- `SVM_KERNEL`: Default SVM kernel (default: rbf)
- `SVM_C`: Default regularization parameter (default: 1.0)
- `SVM_GAMMA`: Default kernel coefficient (default: scale)
- `FORECAST_DAYS`: Default forecast horizon (default: 5)
- `TRAIN_WINDOW_DAYS`: Default training window (default: 252)

## Usage Examples

### Basic Stock Data Fetching
```json
{
  "name": "fetch_stock_data",
  "arguments": {
    "symbol": "AAPL",
    "period": "1y"
  }
}
```

### Training and Forecasting
```json
{
  "name": "train_svm_model",
  "arguments": {
    "symbol": "AAPL",
    "forecast_days": 5,
    "optimize_params": true
  }
}
```

```json
{
  "name": "make_forecast",
  "arguments": {
    "symbol": "AAPL",
    "include_interpretation": true,
    "market_context": "Recent earnings report shows strong growth"
  }
}
```

## Technical Details

### SVM Model
- **Kernels**: Linear, Polynomial, RBF, Sigmoid
- **Features**: 40+ technical indicators and engineered features
- **Validation**: Time series cross-validation
- **Metrics**: R², RMSE, MAE, MAPE

### Feature Engineering
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic
- **Price Features**: Ratios, changes, log returns, ranges
- **Lag Features**: Historical price and volume lags
- **Rolling Statistics**: Moving averages and standard deviations

### OpenAI Integration
- **Model Performance Explanation**: Detailed analysis of metrics and reliability
- **Forecast Interpretation**: Professional analysis with risk assessment
- **Market Context**: Integration of external market information
- **Risk Factors**: Identification of potential risks and limitations

## Model Performance

The SVM models are evaluated using standard regression metrics:

- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## Limitations and Disclaimers

⚠️ **Important**: This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions.

- **Historical Performance**: Past performance does not guarantee future results
- **Market Volatility**: Models may not capture extreme market events
- **External Factors**: News, earnings, and other events are not included in technical models
- **Short-term Focus**: SVM models work best for short to medium-term predictions
- **Risk Management**: Always use proper risk management and position sizing

## Architecture

```
Financial Forecasting MCP
├── Data Layer (Yahoo Finance)
├── Feature Engineering (Technical Indicators)
├── ML Layer (SVM Models)
├── Interpretation Layer (OpenAI)
└── MCP Server (Tool Interface)
```

## Development

### Project Structure
```
src/financial_forecasting_mcp/
├── __init__.py
├── server.py              # Main MCP server
├── data_fetcher.py        # Yahoo Finance integration
├── feature_engineering.py # Technical indicators
├── svm_forecaster.py      # SVM modeling
└── openai_interpreter.py  # AI interpretation
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## License

This project is provided as-is for educational purposes. Please ensure you comply with Yahoo Finance's terms of service and OpenAI's usage policies.

## Contributing

Contributions are welcome! Please ensure all code follows the existing style and includes appropriate tests.

---

**Disclaimer**: This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions.