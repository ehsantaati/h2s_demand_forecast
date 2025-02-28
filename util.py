"""
Utility functions for H2S demand forecasting system.
This module contains all helper functions for data processing, model training,
evaluation, and visualization.
"""

import datetime
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from joblib import dump, load
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_cv_parameters(total_days: int, desired_forecast_days: int = 180) -> Dict[str, str]:
    """Calculate optimal cross-validation parameters based on total available data.
    
    The function uses the following rules:
    1. Initial window: 40% of total data, minimum 1 year for yearly seasonality
    2. Horizon: Maximum 50% of initial window, capped at desired forecast days
    3. Period: 25% of horizon, minimum 30 days
    
    Args:
        total_days: Total number of days in your dataset
        desired_forecast_days: How many days ahead you want to forecast
        
    Returns:
        Dictionary with recommended initial, horizon, and period values as strings
        formatted as "X days"
        
    Example:
        >>> params = calculate_cv_parameters(1095)  # 3 years of data
        >>> print(params)
        {
            'initial': '450 days',
            'horizon': '180 days',
            'period': '90 days'
        }
    """
    try:
        # Initial window calculation
        min_initial = 365  # Minimum 1 year for yearly seasonality
        recommended_initial = max(min_initial, int(total_days * 0.4))
        
        # Horizon calculation
        max_horizon = int(recommended_initial * 0.5)
        horizon = min(desired_forecast_days, max_horizon)
        
        # Period calculation
        period = max(30, int(horizon * 0.25))  # At least 30 days
        
        params = {
            "initial": f"{recommended_initial} days",
            "horizon": f"{horizon} days",
            "period": f"{period} days"
        }
        
        logger.info(f"Calculated CV parameters for {total_days} days of data: {params}")
        return params
        
    except Exception as e:
        logger.error(f"Error calculating CV parameters: {str(e)}")
        raise

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file has invalid JSON
        ValueError: If required fields are missing
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Required configuration fields
        required_fields = ['model_id', 'normalize', 'initial', 'period', 'horizon']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {', '.join(missing_fields)}")

        # Set default values
        defaults = {
            'test_size': 6,
            'date_column': 'Created',
            'normalize': False,
            'initial': '730 days',
            'period': '180 days',
            'horizon': '365 days'
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
                logger.info(f"Using default value for {key}: {default_value}")

        return config

    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_and_prep_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data from Excel files and prepare for Prophet.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Prepared DataFrame for Prophet
        
    Raises:
        Exception: If data processing fails
    """
    try:
        data_path = Path("data")
        paths = list(data_path.glob("*.xlsx"))
        
        if not paths:
            raise FileNotFoundError("No Excel files found in data directory")
            
        # Load and combine data files
        dfs = [pd.read_excel(p) for p in tqdm(paths, desc="Loading data files")]
        df = pd.concat(dfs, ignore_index=True)
        
        # Validate input data
        validate_data(df, config)
        
        # Prepare data for Prophet
        df["ds"] = pd.to_datetime(df[config["date_column"]], format='%Y-%m-%d')
        df.drop(config["date_column"], axis=1, inplace=True)
        
        # Group by month
        df_g = (df.groupby(df["ds"].dt.to_period('M'))
                .size()
                .reset_index())
        df_g.columns = ["ds", "y"]
        df_g["ds"] = df_g["ds"].dt.to_timestamp(how='end').dt.normalize()
        
        logger.info(f"Data loaded and prepared successfully: {len(df_g)} rows")
        return df_g
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Validate input data format and contents.
    
    Args:
        df: Input DataFrame to validate
        config: Configuration dictionary
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        # Check required columns
        required_cols = [config['date_column']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df[config['date_column']]):
            raise ValueError(f"Column {config['date_column']} must be datetime type")
            
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Dataset contains missing values")
            
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise

def log_transform(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Apply log transformation to data.
    
    Args:
        data: Input data to transform
        
    Returns:
        Log-transformed data
    """
    try:
        return np.log1p(data)
    except Exception as e:
        logger.error(f"Error in log transformation: {str(e)}")
        raise

def inverse_log_transform(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Apply inverse log transformation to data.
    
    Args:
        data: Input data to transform
        
    Returns:
        Inverse log-transformed data
    """
    try:
        return np.expm1(data)
    except Exception as e:
        logger.error(f"Error in inverse log transformation: {str(e)}")
        raise

def create_visualization(df: pd.DataFrame, title: str, path_to_save: str = None) -> None:
    """Create and save visualization of forecasts.
    
    Args:
        df: DataFrame containing actual and predicted values
        title: Plot title
        path_to_save: Optional path to save the plot
    """
    try:
        fig = px.line(
            df,
            x="ds",
            y=["y_train", "y_test"],
            color_discrete_sequence=["green", "orange"])

        if "yhat" in df.columns:
            df_forecast = df[df["yhat"] > 0]
            fig.add_scatter(
                x=df_forecast['ds'],
                y=df_forecast['yhat'],
                mode='lines',
                line=dict(color='blue'),
                name='Forecast')

            if all(col in df_forecast.columns for col in ['yhat_upper', 'yhat_lower']):
                fig.add_traces([
                    dict(
                        type='scatter',
                        x=df_forecast['ds'],
                        y=df_forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False),
                    dict(
                        type='scatter',
                        x=df_forecast['ds'],
                        y=df_forecast['yhat_lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0,100,250,0.2)',
                        line=dict(width=0),
                        showlegend=False)
                ])

        fig.update_layout(
            title=title,
            title_x=0.5,
            template="plotly_white",
            height=600,
            xaxis_title="Date",
            yaxis_title="Demand"
        )

        if path_to_save:
            fig.write_html(f"{path_to_save}.html")
            logger.info(f"Visualization saved to {path_to_save}.html")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

def save_model_and_metrics(
    model: Prophet,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    model_name: str = None
) -> Tuple[Path, Path]:
    """Save trained model and its performance metrics.
    
    Args:
        model: Trained Prophet model
        metrics: Dictionary of performance metrics
        config: Configuration dictionary
        model_name: Optional custom model name
        
    Returns:
        Tuple of paths where model and metrics were saved
    """
    try:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_id = config["model_id"]
        data_type = "norm" if config["normalize"] else "notnorm"
        
        if model_name is None:
            model_name = f"{model_id}_{now}_{data_type}"
            
        # Create directories
        model_dir = Path(f"models/{model_id}")
        output_dir = Path(f"output/{model_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = output_dir / f"{model_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return model_path, metrics_path
        
    except Exception as e:
        logger.error(f"Error saving model and metrics: {str(e)}")
        raise

def load_model(model_path: Union[str, Path]) -> Prophet:
    """Load a trained Prophet model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded Prophet model
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate performance metrics.
    
    Args:
        y_true: List of actual values
        y_pred: List of predicted values
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
        
        metrics = {
            "mape": round(mape * 100, 2),
            "rmse": round(rmse, 2)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise 