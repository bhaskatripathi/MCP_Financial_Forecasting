"""OpenAI-powered interpretation of financial forecasting results."""

import openai
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenAIInterpreter:
    """Interprets and explains financial forecasting results using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize the OpenAI interpreter.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for interpretation
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def interpret_forecast(
        self, 
        symbol: str,
        forecast_results: Dict[str, Any],
        training_metrics: Dict[str, Any],
        stock_info: Dict[str, Any],
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Interpret and explain forecasting results.
        
        Args:
            symbol: Stock symbol
            forecast_results: Results from SVM prediction
            training_metrics: Model training performance metrics
            stock_info: Basic stock information
            market_context: Optional market context or news
            
        Returns:
            Dictionary with interpretation and explanation
        """
        try:
            # Prepare context for OpenAI
            context = self._prepare_context(
                symbol, forecast_results, training_metrics, stock_info, market_context
            )
            
            # Create prompt
            prompt = self._create_interpretation_prompt(context)
            
            # Get interpretation from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            interpretation = response.choices[0].message.content
            
            results = {
                'symbol': symbol,
                'interpretation': interpretation,
                'forecast_summary': self._create_forecast_summary(forecast_results),
                'confidence_assessment': self._assess_confidence(forecast_results, training_metrics),
                'risk_factors': self._identify_risk_factors(forecast_results, training_metrics),
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model
            }
            
            logger.info(f"Generated interpretation for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error interpreting forecast: {str(e)}")
            raise
    
    def explain_model_performance(
        self, 
        symbol: str,
        training_metrics: Dict[str, Any],
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain model performance and parameter choices.
        
        Args:
            symbol: Stock symbol
            training_metrics: Model training metrics
            model_params: Model parameters used
            
        Returns:
            Dictionary with performance explanation
        """
        try:
            context = {
                'symbol': symbol,
                'training_metrics': training_metrics,
                'model_params': model_params
            }
            
            prompt = f"""
            Analyze the SVM model performance for stock {symbol}:
            
            Training Metrics:
            - R² Score: {training_metrics.get('test_metrics', {}).get('r2', 'N/A')}
            - RMSE: {training_metrics.get('test_metrics', {}).get('rmse', 'N/A')}
            - MAE: {training_metrics.get('test_metrics', {}).get('mae', 'N/A')}
            - MAPE: {training_metrics.get('test_metrics', {}).get('mape', 'N/A')}%
            
            Model Parameters:
            - Kernel: {model_params.get('kernel', 'N/A')}
            - C: {model_params.get('C', 'N/A')}
            - Gamma: {model_params.get('gamma', 'N/A')}
            
            Training Data:
            - Training Samples: {training_metrics.get('train_samples', 'N/A')}
            - Test Samples: {training_metrics.get('test_samples', 'N/A')}
            - Features: {training_metrics.get('feature_count', 'N/A')}
            
            Explain what these metrics mean, assess the model's reliability, 
            and provide recommendations for improvement.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data scientist explaining ML model performance to investors."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            explanation = response.choices[0].message.content
            
            results = {
                'symbol': symbol,
                'performance_explanation': explanation,
                'model_reliability': self._assess_model_reliability(training_metrics),
                'recommendations': self._generate_recommendations(training_metrics, model_params),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated performance explanation for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error explaining model performance: {str(e)}")
            raise
    
    def _prepare_context(
        self, 
        symbol: str,
        forecast_results: Dict[str, Any],
        training_metrics: Dict[str, Any],
        stock_info: Dict[str, Any],
        market_context: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare context information for interpretation."""
        return {
            'symbol': symbol,
            'company_name': stock_info.get('company_name', 'N/A'),
            'sector': stock_info.get('sector', 'N/A'),
            'industry': stock_info.get('industry', 'N/A'),
            'current_price': forecast_results.get('current_price'),
            'predicted_price': forecast_results.get('predicted_price'),
            'percentage_change': forecast_results.get('percentage_change'),
            'direction': forecast_results.get('direction'),
            'confidence': forecast_results.get('confidence'),
            'model_r2': training_metrics.get('test_metrics', {}).get('r2'),
            'model_rmse': training_metrics.get('test_metrics', {}).get('rmse'),
            'forecast_days': training_metrics.get('forecast_days'),
            'market_context': market_context
        }
    
    def _create_interpretation_prompt(self, context: Dict[str, Any]) -> str:
        """Create the interpretation prompt for OpenAI."""
        return f"""
        Analyze this stock forecast and provide a comprehensive interpretation:
        
        Stock Information:
        - Symbol: {context['symbol']}
        - Company: {context['company_name']}
        - Sector: {context['sector']}
        - Industry: {context['industry']}
        
        Forecast Results:
        - Current Price: ${context['current_price']:.2f}
        - Predicted Price: ${context['predicted_price']:.2f}
        - Price Change: {context['percentage_change']:+.2f}%
        - Direction: {context['direction']}
        - Model Confidence: {context['confidence']:.2f}
        
        Model Performance:
        - R² Score: {context['model_r2']:.4f}
        - RMSE: {context['model_rmse']:.2f}
        - Forecast Horizon: {context['forecast_days']} days
        
        {f"Market Context: {context['market_context']}" if context['market_context'] else ""}
        
        Please provide:
        1. A clear interpretation of the forecast
        2. Assessment of the prediction reliability
        3. Key factors that might influence this prediction
        4. Potential risks and limitations
        5. Actionable insights for investors
        
        Keep the explanation professional but accessible to both technical and non-technical audiences.
        """
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for OpenAI."""
        return """
        You are an expert financial analyst and data scientist specializing in quantitative trading and SVM-based forecasting models. 
        
        Your role is to interpret machine learning forecasts for stock prices and provide clear, actionable insights. 
        You understand the limitations of technical analysis and always emphasize proper risk management.
        
        Guidelines:
        - Be objective and balanced in your analysis
        - Clearly explain model limitations and uncertainties
        - Provide context about market conditions when relevant
        - Use professional but accessible language
        - Always include appropriate disclaimers about investment risks
        - Focus on educational value and risk awareness
        """
    
    def _create_forecast_summary(self, forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the forecast results."""
        return {
            'price_movement': forecast_results.get('direction', 'unknown'),
            'magnitude': abs(forecast_results.get('percentage_change', 0)),
            'strength': self._categorize_movement_strength(abs(forecast_results.get('percentage_change', 0))),
            'target_price': forecast_results.get('predicted_price'),
            'current_price': forecast_results.get('current_price')
        }
    
    def _assess_confidence(
        self, 
        forecast_results: Dict[str, Any], 
        training_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the confidence of the forecast."""
        model_confidence = forecast_results.get('confidence', 0)
        r2_score = training_metrics.get('test_metrics', {}).get('r2', 0)
        
        # Combine model confidence with R² score
        overall_confidence = (model_confidence + max(0, r2_score)) / 2
        
        confidence_level = 'low'
        if overall_confidence > 0.7:
            confidence_level = 'high'
        elif overall_confidence > 0.5:
            confidence_level = 'medium'
        
        return {
            'level': confidence_level,
            'score': overall_confidence,
            'factors': [
                f"Model R² score: {r2_score:.3f}" if r2_score else "R² score not available",
                f"Prediction confidence: {model_confidence:.3f}" if model_confidence else "Prediction confidence not available"
            ]
        }
    
    def _identify_risk_factors(
        self, 
        forecast_results: Dict[str, Any], 
        training_metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # Model performance risks
        r2_score = training_metrics.get('test_metrics', {}).get('r2', 0)
        if r2_score < 0.5:
            risks.append("Low model R² score indicates limited predictive power")
        
        # Forecast magnitude risks
        pct_change = abs(forecast_results.get('percentage_change', 0))
        if pct_change > 10:
            risks.append("Large predicted price movement increases uncertainty")
        
        # Sample size risks
        test_samples = training_metrics.get('test_samples', 0)
        if test_samples < 50:
            risks.append("Limited test data may affect model reliability")
        
        # General market risks
        risks.extend([
            "Market volatility can significantly impact short-term predictions",
            "External factors (news, events) not captured in technical indicators",
            "Model based on historical patterns may not reflect future market conditions"
        ])
        
        return risks
    
    def _categorize_movement_strength(self, percentage: float) -> str:
        """Categorize the strength of price movement."""
        if percentage < 2:
            return 'weak'
        elif percentage < 5:
            return 'moderate'
        elif percentage < 10:
            return 'strong'
        else:
            return 'very strong'
    
    def _assess_model_reliability(self, training_metrics: Dict[str, Any]) -> str:
        """Assess overall model reliability."""
        r2_score = training_metrics.get('test_metrics', {}).get('r2', 0)
        
        if r2_score > 0.7:
            return 'high'
        elif r2_score > 0.5:
            return 'moderate'
        elif r2_score > 0.3:
            return 'low'
        else:
            return 'very low'
    
    def _generate_recommendations(
        self, 
        training_metrics: Dict[str, Any], 
        model_params: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for model improvement."""
        recommendations = []
        
        r2_score = training_metrics.get('test_metrics', {}).get('r2', 0)
        
        if r2_score < 0.5:
            recommendations.append("Consider feature engineering or additional data sources")
            recommendations.append("Experiment with different kernel types or hyperparameters")
        
        train_samples = training_metrics.get('train_samples', 0)
        if train_samples < 500:
            recommendations.append("Increase training data size for better model stability")
        
        recommendations.extend([
            "Implement ensemble methods for improved predictions",
            "Consider regime-based modeling for different market conditions",
            "Regular model retraining with latest market data"
        ])
        
        return recommendations 