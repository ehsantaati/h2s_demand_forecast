import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error
from joblib import dump
from utils import *
import os 


def load_and_prep_data(config):
  """Load data from data directory folder and 
    prepare the data for Prophet"""
  try:
    data_path = Path("data")
    paths = list(data_path.glob("*.xlsx"))
    df = pd.DataFrame()
    for p in paths:
      df_ = pd.read_excel(p)
      df = pd.concat([df, df_])
    # Prepare data for Prophet
    df["ds"] = pd.to_datetime(df[config["date_column"]], format='%Y-%m-%d')
    df.drop(config["date_column"], axis=1, inplace=True)
    df_g = df.groupby(df["ds"].dt.to_period('M')).size().reset_index()
    df_g.columns = ["ds", "y"]

    df_g["ds"] = df_g["ds"].dt.to_timestamp(how='end').dt.normalize()
    print(f"Data is loaded and prepared.")
  except Exception as e:
    print(f"Error processing data: {str(e)}")
    sys.exit(1)

  return df_g


def visualise(df, title, path_to_save):
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
    title_x=0.5,  # Set figure height (decrease size)
  )

  fig.write_html(f"{path_to_save}.html")


def cv_train(df, config):
  model_id = config["model_id"]
  data_type = "notnorm"

  train_df, test_df = temporal_train_test_split(
    df, test_size=config["test_size"])
  if config["normalize"]:
    data_type = "norm"
    train_df["y"] = log_transform(train_df["y"])
    test_df["y"] = log_transform(test_df["y"])  
  # Set up parameter grid
  param_grid = {
    'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.5],
    'seasonality_prior_scale': [0.01, 1, 5, 10, 12],
    'seasonality_mode': ['additive', 'multiplicative']
  }

  # Generate all combinations of parameters
  all_params = [
    dict(zip(param_grid.keys(), v))
    for v in itertools.product(*param_grid.values())
  ]

  # Create a list to store MAPE values for each combination
  mapes = []

  # Use cross validation to evaluate all parameters
  for params in all_params:
    # Fit a model using one parameter combination
    m = Prophet(**params).fit(df)
    # Cross-validation
    df_cv = cross_validation(
      m,
      initial=config["initial"],
      period=config["period"],
      horizon=config["horizon"],
      parallel="processes")
    # Model performance
    df_p = performance_metrics(df_cv, rolling_window=1)
    # Save model performance metrics
    mapes.append(df_p['mape'].values[0])

  # Tuning results
  tuning_results = pd.DataFrame(all_params)
  tuning_results['mape'] = mapes

  # Find the best parameters
  best_params = all_params[np.argmin(mapes)]

  # Fit the model using the best parameters
  auto_model = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
  )

  # Fit the model on the training dataset
  auto_model.fit(train_df)

  # Cross validation
  auto_model_cv = cross_validation(
    auto_model,
    initial=config["initial"],
    period=config["period"],
    horizon=config["horizon"],
    parallel="processes")

  # Model performance metrics
  auto_model_p = performance_metrics(auto_model_cv, rolling_window=1)
  auto_model_p = performance_metrics(auto_model_cv, rolling_window=1)
  mape = auto_model_p['mape'].values[0]
  print(f"Best MAPE: {mape}")
  print(f"Best Paramters: {best_params}")

  #test model
  future = auto_model.make_future_dataframe(periods=12, freq='M')
  forecast = auto_model.predict(future)
  if config["normalize"]:
    forecast['yhat'] = inverse_log_transform(forecast['yhat'])
    forecast['yhat_lower'] = inverse_log_transform(forecast['yhat_lower'])
    forecast['yhat_upper'] = inverse_log_transform(forecast['yhat_upper'])
  # set negative values to zero
  forecast[[col for col in forecast.columns if col!="ds"]] = forecast[[col for col in forecast.columns if col!="ds"]].clip(lower=0)

  y_true = test_df["y"].tolist()
  if config["normalize"]:
    y_true = inverse_log_transform(test_df["y"]).tolist()

  y_hat = forecast[(forecast["ds"] >= test_df["ds"].iloc[0]) & (
    forecast["ds"] <= test_df["ds"].iloc[-1])]["yhat"].tolist()

  mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_hat)
  rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_hat))
  res = {"mape": mape.round(2), "rmse": rmse.round(2)}

  # Get current date and time
  now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
  # Save the final model
  model_dir = f"models/{model_id}"
  model_name = f"{model_id}_{now}_{data_type}"
  Path(model_dir).mkdir(parents=True, exist_ok=True)
  dump(auto_model, f"{model_dir}/{model_name}.joblib")
  print(f"Model saved to {model_dir}")
  # save test reults
  output_dir = f"output/{model_id}"
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  with open(f"{output_dir}/{model_name}.json", "w") as f:
    json.dump(res, f)

  # plot and save
  train_df.columns = ["ds", "y_train"]
  test_df.columns = ["ds", "y_test"]
  preds = forecast[forecast["ds"] >= test_df["ds"].iloc[0]]
  vis_df = pd.concat([train_df, test_df, preds], ignore_index=True)
  visualise(vis_df, title=model_id, path_to_save=f"{model_dir}/{model_name}")


if __name__ == "__main__":
  # Load configuration
  config = load_config()

  # Load and prepare the data
  df = load_and_prep_data(config)
  # train
  cv_train(df, config)
