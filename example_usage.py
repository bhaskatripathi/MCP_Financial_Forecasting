#!/usr/bin/env python3
"""Example usage of the Financial Forecasting MCP."""

import asyncio
import json
import logging
from src.financial_forecasting_mcp.data_fetcher import YahooFinanceDataFetcher
from src.financial_forecasting_mcp.feature_engineering import FinancialFeatureEngineering
from src.financial_forecasting_mcp.svm_forecaster import SVMForecaster
from src.financial_forecasting_mcp.openai_interpreter import OpenAIInterpreter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_forecast_workflow():
    """Example workflow for stock forecasting with interpretation."""
    
    # Configuration
    SYMBOL = "AAPL"
    FORECAST_DAYS = 5
    
    logger.info(f"Starting forecast workflow for {SYMBOL}")
    
    try:
        # Step 1: Initialize components
        data_fetcher = YahooFinanceDataFetcher()
        feature_engineer = FinancialFeatureEngineering()
        forecaster = SVMForecaster(kernel='rbf')
        
        # Initialize OpenAI interpreter if API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        interpreter = None
        if openai_api_key:
            interpreter = OpenAIInterpreter(openai_api_key)
            logger.info("OpenAI interpreter initialized")
        else:
            logger.warning("OPENAI_API_KEY not found. Skipping interpretation.")
        
        # Step 2: Fetch stock data
        logger.info(f"Fetching data for {SYMBOL}...")
        stock_data = data_fetcher.fetch_ohlc_data(SYMBOL, period="2y")
        stock_info = data_fetcher.get_stock_info(SYMBOL)
        
        print(f"\n📊 Stock Information for {SYMBOL}:")
        print(f"Company: {stock_info.get('company_name', 'N/A')}")
        print(f"Sector: {stock_info.get('sector', 'N/A')}")
        print(f"Current Price: ${stock_info.get('current_price', 'N/A')}")
        print(f"Market Cap: {stock_info.get('market_cap', 'N/A')}")
        print(f"Data Range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        print(f"Records: {len(stock_data)}")
        
        # Step 3: Engineer features
        logger.info("Engineering features...")
        features = feature_engineer.prepare_features(stock_data)
        
        print(f"\n🔧 Feature Engineering:")
        print(f"Original columns: {len(stock_data.columns)}")
        print(f"Engineered features: {len(features.columns)}")
        print(f"Usable records: {len(features)}")
        
        # Step 4: Train SVM model
        logger.info("Training SVM model...")
        training_results = forecaster.train(features, forecast_days=FORECAST_DAYS)
        
        print(f"\n🤖 Model Training Results:")
        print(f"Training samples: {training_results['train_samples']}")
        print(f"Test samples: {training_results['test_samples']}")
        print(f"Features used: {training_results['feature_count']}")
        print(f"Test R² Score: {training_results['test_metrics']['r2']:.4f}")
        print(f"Test RMSE: {training_results['test_metrics']['rmse']:.2f}")
        print(f"Test MAE: {training_results['test_metrics']['mae']:.2f}")
        print(f"Test MAPE: {training_results['test_metrics']['mape']:.2f}%")
        
        # Step 5: Make forecast
        logger.info("Making forecast...")
        forecast_results = forecaster.predict(features)
        
        print(f"\n📈 Forecast Results:")
        print(f"Current Price: ${forecast_results['current_price']:.2f}")
        print(f"Predicted Price ({FORECAST_DAYS} days): ${forecast_results['predicted_price']:.2f}")
        print(f"Price Change: {forecast_results['percentage_change']:+.2f}%")
        print(f"Direction: {forecast_results['direction'].upper()}")
        print(f"Confidence: {forecast_results['confidence']:.2f}")
        
        # Step 6: Get OpenAI interpretation (if available)
        if interpreter:
            logger.info("Getting AI interpretation...")
            interpretation = interpreter.interpret_forecast(
                SYMBOL,
                forecast_results,
                training_results,
                stock_info,
                market_context="Standard market conditions"
            )
            
            print(f"\n🧠 AI Interpretation:")
            print("="*60)
            print(interpretation['interpretation'])
            print("="*60)
            
            print(f"\n📊 Confidence Assessment:")
            conf_assessment = interpretation['confidence_assessment']
            print(f"Overall Confidence: {conf_assessment['level'].upper()} ({conf_assessment['score']:.2f})")
            for factor in conf_assessment['factors']:
                print(f"  • {factor}")
            
            print(f"\n⚠️ Risk Factors:")
            for risk in interpretation['risk_factors']:
                print(f"  • {risk}")
            
            # Get model performance explanation
            logger.info("Getting model performance explanation...")
            performance_explanation = interpreter.explain_model_performance(
                SYMBOL,
                training_results,
                training_results['model_params']
            )
            
            print(f"\n📈 Model Performance Explanation:")
            print("="*60)
            print(performance_explanation['performance_explanation'])
            print("="*60)
            
            print(f"\nModel Reliability: {performance_explanation['model_reliability'].upper()}")
            print(f"\nRecommendations:")
            for rec in performance_explanation['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n✅ Forecast workflow completed successfully!")
        
        # Return results for further processing if needed
        return {
            'symbol': SYMBOL,
            'stock_info': stock_info,
            'training_results': training_results,
            'forecast_results': forecast_results,
            'interpretation': interpretation if interpreter else None
        }
        
    except Exception as e:
        logger.error(f"Error in forecast workflow: {str(e)}")
        raise


async def example_multi_stock_comparison():
    """Example of comparing forecasts for multiple stocks."""
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    results = {}
    
    logger.info(f"Comparing forecasts for: {', '.join(symbols)}")
    
    # Initialize components
    data_fetcher = YahooFinanceDataFetcher()
    feature_engineer = FinancialFeatureEngineering()
    
    print(f"\n📊 Multi-Stock Forecast Comparison")
    print("="*80)
    
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            
            # Fetch and process data
            stock_data = data_fetcher.fetch_ohlc_data(symbol, period="1y")
            features = feature_engineer.prepare_features(stock_data)
            
            # Train model
            forecaster = SVMForecaster(kernel='rbf')
            training_results = forecaster.train(features, forecast_days=5)
            
            # Make forecast
            forecast_results = forecaster.predict(features)
            
            results[symbol] = {
                'current_price': forecast_results['current_price'],
                'predicted_price': forecast_results['predicted_price'],
                'percentage_change': forecast_results['percentage_change'],
                'direction': forecast_results['direction'],
                'model_r2': training_results['test_metrics']['r2'],
                'confidence': forecast_results['confidence']
            }
            
            print(f"\n{symbol}:")
            print(f"  Current: ${forecast_results['current_price']:.2f}")
            print(f"  Predicted: ${forecast_results['predicted_price']:.2f}")
            print(f"  Change: {forecast_results['percentage_change']:+.2f}%")
            print(f"  Direction: {forecast_results['direction'].upper()}")
            print(f"  Model R²: {training_results['test_metrics']['r2']:.4f}")
            print(f"  Confidence: {forecast_results['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            results[symbol] = {'error': str(e)}
    
    print(f"\n✅ Multi-stock comparison completed!")
    return results


async def main():
    """Main function to run examples."""
    
    print("🚀 Financial Forecasting MCP - Example Usage")
    print("="*60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OpenAI API key not found. Interpretation features will be limited.")
        print("Set OPENAI_API_KEY in your .env file for full functionality.\n")
    
    try:
        # Example 1: Full forecast workflow
        print("\n📈 Example 1: Complete Forecast Workflow")
        print("-" * 40)
        await example_forecast_workflow()
        
        # Example 2: Multi-stock comparison
        print("\n📊 Example 2: Multi-Stock Comparison")
        print("-" * 40)
        await example_multi_stock_comparison()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 