"""SVM-based financial forecasting model."""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple, Optional
import logging
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SVMForecaster:
    """SVM-based forecasting model for financial time series."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        epsilon: float = 0.1,
        scaler_type: str = 'standard'
    ):
        """Initialize the SVM forecaster.
        
        Args:
            kernel: SVM kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            epsilon: Epsilon parameter for SVR
            scaler_type: Type of scaler ('standard', 'robust')
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.scaler_type = scaler_type
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        self.is_trained = False
        
        # Initialize scaler
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
    
    def prepare_target(self, df: pd.DataFrame, forecast_days: int = 5) -> pd.DataFrame:
        """Prepare target variable for forecasting.
        
        Args:
            df: DataFrame with features
            forecast_days: Number of days to forecast ahead
            
        Returns:
            DataFrame with target variable
        """
        try:
            data = df.copy()
            
            # Create target: future close price
            data['target'] = data['close'].shift(-forecast_days)
            
            # Remove rows where target is NaN
            data = data.dropna(subset=['target'])
            
            logger.info(f"Prepared target for {len(data)} records, forecasting {forecast_days} days ahead")
            return data
            
        except Exception as e:
            logger.error(f"Error preparing target: {str(e)}")
            raise
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for training.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of selected feature column names
        """
        # Exclude basic columns and target
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'target']
        
        # Select all other columns as features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} features for training")
        return feature_cols
    
    def train(
        self, 
        df: pd.DataFrame, 
        forecast_days: int = 5,
        test_size: float = 0.2,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """Train the SVM model.
        
        Args:
            df: DataFrame with features
            forecast_days: Number of days to forecast ahead
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            # Prepare target
            data = self.prepare_target(df, forecast_days)
            
            # Select features
            self.feature_columns = self.select_features(data)
            self.target_column = 'target'
            
            # Prepare training data
            X = data[self.feature_columns].values
            y = data[self.target_column].values
            
            # Split data (time series split)
            split_idx = int(len(data) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                epsilon=self.epsilon
            )
            
            # Train model
            logger.info(f"Training SVM model with {len(X_train)} samples")
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            self.is_trained = True
            
            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_count': len(self.feature_columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'forecast_days': forecast_days,
                'model_params': {
                    'kernel': self.kernel,
                    'C': self.C,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon
                }
            }
            
            logger.info(f"Model training complete. Test R²: {test_metrics['r2']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make forecasting predictions.
        
        Args:
            df: DataFrame with features (latest data)
            
        Returns:
            Dictionary with predictions and confidence metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare features
            X = df[self.feature_columns].values
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Get latest actual values for comparison
            latest_close = df['close'].iloc[-1]
            prediction = predictions[-1]
            
            # Calculate percentage change
            pct_change = ((prediction - latest_close) / latest_close) * 100
            
            # Simple confidence estimation based on training performance
            # This is a simplified approach - in practice, you might use prediction intervals
            confidence = min(max(0.5, abs(self.model.score(X_scaled, [latest_close] * len(X_scaled)))), 1.0)
            
            results = {
                'current_price': float(latest_close),
                'predicted_price': float(prediction),
                'percentage_change': float(pct_change),
                'direction': 'up' if pct_change > 0 else 'down',
                'confidence': float(confidence),
                'prediction_date': datetime.now().isoformat(),
                'data_date': df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
            }
            
            logger.info(f"Prediction: {latest_close:.2f} -> {prediction:.2f} ({pct_change:+.2f}%)")
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def optimize_hyperparameters(
        self, 
        df: pd.DataFrame, 
        forecast_days: int = 5,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """Optimize SVM hyperparameters using grid search.
        
        Args:
            df: DataFrame with features
            forecast_days: Number of days to forecast ahead
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and results
        """
        try:
            # Prepare data
            data = self.prepare_target(df, forecast_days)
            self.feature_columns = self.select_features(data)
            
            X = data[self.feature_columns].values
            y = data[self.target_column].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Grid search
            svr = SVR(kernel=self.kernel)
            grid_search = GridSearchCV(
                svr, 
                param_grid, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Starting hyperparameter optimization...")
            grid_search.fit(X_scaled, y)
            
            # Update model with best parameters
            self.C = grid_search.best_params_['C']
            self.epsilon = grid_search.best_params_['epsilon']
            self.gamma = grid_search.best_params_['gamma']
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,  # Convert back from negative MSE
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Optimization complete. Best parameters: {grid_search.best_params_}")
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'params': {
                    'kernel': self.kernel,
                    'C': self.C,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon,
                    'scaler_type': self.scaler_type
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            params = model_data['params']
            self.kernel = params['kernel']
            self.C = params['C']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            self.scaler_type = params['scaler_type']
            
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 