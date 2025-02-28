import numpy as np
import json
import sys
import logging
from typing import Dict, Union, Any
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

        # Set default values with type checking
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

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in config file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
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
        # Check for required columns
        required_cols = [config['date_column']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Check for data types
        if not pd.api.types.is_datetime64_any_dtype(df[config['date_column']]):
            raise ValueError(f"Column {config['date_column']} must be datetime type")
            
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Dataset contains missing values")
            
        # Check for negative values in target variable
        if (df['y'] < 0).any():
            logger.warning("Dataset contains negative values in target variable")
            
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
