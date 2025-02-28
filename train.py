import datetime
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sktime.split import temporal_train_test_split
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from joblib import dump, Parallel, delayed
import logging
from tqdm import tqdm
from util import *
import os 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_prep_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data from data directory folder and prepare it for Prophet.
    
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
            
        # Use list comprehension for better performance
        dfs = [pd.read_excel(p) for p in tqdm(paths, desc="Loading data files")]
        df = pd.concat(dfs, ignore_index=True)
        
        # Validate input data
        validate_data(df, config)
        
        # Prepare data for Prophet
        df["ds"] = pd.to_datetime(df[config["date_column"]], format='%Y-%m-%d')
        df.drop(config["date_column"], axis=1, inplace=True)
        
        # Group by month more efficiently
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

def visualise(df: pd.DataFrame, title: str, path_to_save: str) -> None:
    """Create and save visualization of forecasts.
    
    Args:
        df: DataFrame containing actual and predicted values
        title: Plot title
        path_to_save: Path to save the plot
    """
    try:
        fig = px.line(
            df,
            x="ds",
            y=["y_train", "y_test"],
            color_discrete_sequence=["green", "orange"])

        df = df[df["yhat"] > 0]
        fig.add_scatter(
            x=df['ds'],
            y=df['yhat'],
            mode='lines',
            line=dict(color='blue'),
            name='y_forecast')

        fig.add_traces(
            [
                dict(
                    type='scatter',
                    x=df['ds'],
                    y=df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False),
                dict(
                    type='scatter',
                    x=df['ds'],
                    y=df['yhat_lower'],
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
            height=600
        )

        # Save plot
        fig.write_html(f"{path_to_save}.html")
        logger.info(f"Visualization saved to {path_to_save}.html")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

def evaluate_prophet_params(params: Dict[str, Any], df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """Evaluate one set of Prophet parameters using cross-validation.
    
    Args:
        params: Dictionary of Prophet parameters
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        MAPE score for the parameter set
    """
    try:
        m = Prophet(**params).fit(df)
        df_cv = cross_validation(
            m,
            initial=config["initial"],
            period=config["period"],
            horizon=config["horizon"],
            parallel="processes"
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        return df_p['mape'].values[0]
    except Exception as e:
        logger.error(f"Error in parameter evaluation: {str(e)}")
        return float('inf')

def cv_train(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Train Prophet model with cross-validation and hyperparameter tuning.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
    """
    try:
        model_id = config["model_id"]
        data_type = "notnorm"

        # Split data
        train_df, test_df = temporal_train_test_split(df, test_size=config["test_size"])
        
        if config["normalize"]:
            data_type = "norm"
            train_df["y"] = log_transform(train_df["y"])
            test_df["y"] = log_transform(test_df["y"])

        # Parameter grid
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.5],
            'seasonality_prior_scale': [0.01, 1, 5, 10, 12],
            'seasonality_mode': ['additive', 'multiplicative']
        }

        # Generate parameter combinations
        all_params = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        # Parallel parameter evaluation
        logger.info("Starting hyperparameter tuning...")
        with Parallel(n_jobs=-1) as parallel:
            mapes = parallel(
                delayed(evaluate_prophet_params)(params, train_df, config)
                for params in tqdm(all_params, desc="Evaluating parameters")
            )

        # Find best parameters
        best_params = all_params[np.argmin(mapes)]
        logger.info(f"Best parameters found: {best_params}")

        # Train final model
        auto_model = Prophet(**best_params)
        auto_model.fit(train_df)

        # Generate forecast
        future = auto_model.make_future_dataframe(periods=12, freq='M')
        forecast = auto_model.predict(future)
        
        if config["normalize"]:
            forecast['yhat'] = inverse_log_transform(forecast['yhat'])
            forecast['yhat_lower'] = inverse_log_transform(forecast['yhat_lower'])
            forecast['yhat_upper'] = inverse_log_transform(forecast['yhat_upper'])
            
        # Set negative values to zero
        forecast_cols = [col for col in forecast.columns if col != "ds"]
        forecast[forecast_cols] = forecast[forecast_cols].clip(lower=0)

        # Calculate metrics
        y_true = inverse_log_transform(test_df["y"]).tolist() if config["normalize"] else test_df["y"].tolist()
        y_hat = forecast[
            (forecast["ds"] >= test_df["ds"].iloc[0]) & 
            (forecast["ds"] <= test_df["ds"].iloc[-1])
        ]["yhat"].tolist()

        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_hat)
        rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_hat))
        metrics = {"mape": round(mape, 2), "rmse": round(rmse, 2)}
        
        logger.info(f"Final model metrics: {metrics}")

        # Save model and results
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_dir = Path(f"models/{model_id}")
        model_name = f"{model_id}_{now}_{data_type}"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        dump(auto_model, model_dir / f"{model_name}.joblib")
        
        output_dir = Path(f"output/{model_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"{model_name}.json", "w") as f:
            json.dump(metrics, f)

        # Create visualization
        train_df.columns = ["ds", "y_train"]
        test_df.columns = ["ds", "y_test"]
        preds = forecast[forecast["ds"] >= test_df["ds"].iloc[0]]
        vis_df = pd.concat([train_df, test_df, preds], ignore_index=True)
        visualise(vis_df, title=model_id, path_to_save=str(model_dir / model_name))
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting demand forecasting pipeline")
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Load and prepare data
        df = load_and_prep_data(config)
        
        # Train model
        cv_train(df, config)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
