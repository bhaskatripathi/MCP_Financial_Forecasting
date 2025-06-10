#!/usr/bin/env python3
"""Setup and test script for the Financial Forecasting MCP."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} is not compatible. Requires Python 3.8+")
        return False


def check_env_file():
    """Check if .env file exists, create from example if not."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    elif env_example.exists():
        print("🔄 Creating .env file from example...")
        try:
            env_example.read_text()
            with open(env_file, 'w') as f:
                f.write(env_example.read_text())
            print("✅ .env file created from example")
            print("⚠️ Please edit .env file to add your OPENAI_API_KEY")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {str(e)}")
            return False
    else:
        print("❌ No .env or env.example file found")
        return False


def test_imports():
    """Test if all required modules can be imported."""
    print("🔄 Testing imports...")
    
    required_modules = [
        "yfinance",
        "sklearn", 
        "pandas",
        "numpy",
        "ta",
        "openai",
        "dotenv",
        "mcp"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All imports successful")
        return True


def test_basic_functionality():
    """Test basic functionality of the MCP components."""
    print("🔄 Testing basic functionality...")
    
    try:
        # Test data fetcher
        from src.financial_forecasting_mcp.data_fetcher import YahooFinanceDataFetcher
        data_fetcher = YahooFinanceDataFetcher()
        
        # Test with a simple validation (doesn't require internet)
        print("  ✅ Data fetcher initialized")
        
        # Test feature engineering
        from src.financial_forecasting_mcp.feature_engineering import FinancialFeatureEngineering
        feature_engineer = FinancialFeatureEngineering()
        print("  ✅ Feature engineer initialized")
        
        # Test SVM forecaster
        from src.financial_forecasting_mcp.svm_forecaster import SVMForecaster
        forecaster = SVMForecaster()
        print("  ✅ SVM forecaster initialized")
        
        # Test OpenAI interpreter (without API key)
        from src.financial_forecasting_mcp.openai_interpreter import OpenAIInterpreter
        print("  ✅ OpenAI interpreter module loaded")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False


def main():
    """Main setup and test function."""
    print("🚀 Financial Forecasting MCP - Setup and Test")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check/create .env file
    if not check_env_file():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Setup and tests completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file to add your OPENAI_API_KEY")
        print("2. Run the example: python example_usage.py")
        print("3. Or start the MCP server: python -m src.financial_forecasting_mcp.server")
    else:
        print("❌ Setup or tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 