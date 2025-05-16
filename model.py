"""
Combined training and forecasting model for H2S demand forecasting.
This script handles both training and forecasting in a single workflow.
"""
import argparse
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
from joblib import dump, load, Parallel, delayed
import logging
from tqdm import tqdm
from util import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_prophet_params(params: Dict[str, Any], df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """Evaluate one set of Prophet parameters using cross-validation.
    
    Args:
        params: Dictionary of Prophet parameters
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        MAPE score for the parameter set
    """
    total_days = calculate_days(df) + config["test_size"] * 30
    
    cv_parameters = calculate_cv_parameters(total_days)
    try:
        m = Prophet(**params).fit(df)
        df_cv = cross_validation(
            m,
            initial=cv_parameters["initial"],
            period=cv_parameters["period"],
            horizon=cv_parameters["horizon"],
            parallel="processes"
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        return df_p['mape'].values[0]
    except Exception as e:
        logger.error(f"Error in parameter evaluation: {str(e)}")
        return float('inf')

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

def train_model(config: Dict[str, Any]) -> Tuple[Prophet, Path, str]:
    """Train Prophet model with cross-validation and hyperparameter tuning.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of trained model, model path, and data type (normalized or not)
    """
    logger.info("Starting model training...")
    
    try:
        # Load and prepare data
        df = load_and_prep_data(config)
        
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
        results_dir = Path(f"results/{model_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"{model_id}_{now}_{data_type}"
        
        model_path = results_dir / f"{model_name}.joblib"
        dump(auto_model, model_path)
        
        with open(results_dir / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f)

        # Create visualization
        train_df.columns = ["ds", "y_train"]
        test_df.columns = ["ds", "y_test"]
        preds = forecast[forecast["ds"] >= test_df["ds"].iloc[0]]
        
        visualization_df = train_df.merge(
            test_df, on="ds", how="outer"
        ).merge(
            preds[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="outer"
        )
        
        visualise(
            visualization_df,
            f"{model_id} Forecast",
            f"results/{model_id}/{model_name}_evaluation"
        )
        
        logger.info(f"Model trained and saved successfully at: {model_path}")
        return auto_model, model_path, data_type
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def forecast(model_path: Path = None, model: Prophet = None, data_type: str = None, horizon: int = 6) -> None:
    """Generate forecast using trained model.
    
    Args:
        model_path: Path to the trained model file (ignored if model is provided)
        model: Trained Prophet model (optional, loaded from model_path if not provided)
        data_type: Type of data used for training ("norm" or "notnorm")
        horizon: Prediction horizon in months
    """
    try:
        if model is None:
            if model_path is None:
                raise ValueError("Either model or model_path must be provided")
            
            model = load(model_path)
            model_id = model_path.stem
            data_type = "norm" if "norm" in str(model_path) and "notnorm" not in str(model_path) else "notnorm"
            logger.info(f"Model loaded from {model_path}")
        else:
            if data_type is None:
                raise ValueError("data_type must be provided when model is provided")
            model_id = model_path.stem if model_path else "custom_model"
        
        logger.info(f"Generating forecast with horizon {horizon} months...")
        
        # Generate future dataframe for prediction
        future = model.make_future_dataframe(periods=horizon, freq='ME', include_history=False)
        forecast_df = model.predict(future)

        # Apply inverse transformation if needed
        if data_type == "norm":
            forecast_df['yhat'] = inverse_log_transform(forecast_df['yhat'])
            forecast_df['yhat_lower'] = inverse_log_transform(forecast_df['yhat_lower'])
            forecast_df['yhat_upper'] = inverse_log_transform(forecast_df['yhat_upper'])
            
        # Set negative values to zero and round up to integers
        forecast_cols = [col for col in forecast_df.columns if col != "ds"]
        forecast_df[forecast_cols] = forecast_df[forecast_cols].clip(lower=0)
        forecast_df[forecast_cols] = np.ceil(forecast_df[forecast_cols])
        
        # Prepare output
        out = forecast_df[["ds", "yhat_lower", "yhat", "yhat_upper"]]
        
        # Create output directory and save results
        results_dir = Path(f"results/forecast_{model_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        out.columns = ["Date", "Worst Case Scenario", "Central Case", "Best Case Scenario"]
        out.to_excel(f"{results_dir}/forecast_{model_id}.xlsx", index=False)
        
        logger.info(f"Forecast generated and saved at: {results_dir}/forecast_{model_id}.xlsx")
        return out
        
    except Exception as e:
        logger.error(f"Error in forecasting: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Combined H2S demand training and forecasting")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to configuration file")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction horizon in months (default: 6)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
        
        # Train model
        logger.info("Starting model training followed by forecasting...")
        _, model_path, data_type = train_model(config)
        
        # Run forecasting
        forecast(model_path=model_path, data_type=data_type, horizon=args.horizon)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 