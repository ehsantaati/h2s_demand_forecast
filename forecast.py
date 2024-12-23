import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from utils import *


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the saved model file")
  parser.add_argument(
    "--horizon", type=int, default=6, help="Prediction horizon (default: 6)")
  parser.add_argument(
    "--normalize",
    type=bool,
    default=False,
    help="Normalize(log) input data (default: False)")

  args = parser.parse_args()
  __model_path__ = Path(args.model_path)
  __model_id__ = __model_path__.stem
  __fh__ = args.horizon
  __normalize__ = args.normalize

  print(f"Model path provided: {__model_path__}")
  print(f"Forcasing horizon is {__fh__} months.")

  model = load(__model_path__)
  print("Model reloaded successfully")

  future = model.make_future_dataframe(
    periods=12, freq='M', include_history=False)
  forecast = model.predict(future)

  if __normalize__:
    forecast['yhat'] = inverse_log_transform(forecast['yhat'])
    forecast['yhat_lower'] = inverse_log_transform(forecast['yhat_lower'])
    forecast['yhat_upper'] = inverse_log_transform(forecast['yhat_upper'])
    # set negative values to zero
  forecast[[col for col in forecast.columns if col != "ds"
            ]] = forecast[[col for col in forecast.columns
                           if col != "ds"]].clip(lower=0)
  cut_off_date = pd.Timestamp(
    future.loc[0].values[0]) + pd.DateOffset(months=__fh__)
  forecast[[col for col in forecast.columns if col != "ds"]] = np.ceil(
    forecast[[col for col in forecast.columns if col != "ds"]])
    
  out = forecast[["ds", "yhat_lower", "yhat", "yhat_upper"]]
  out = out[out["ds"] >= cut_off_date]
  output_dir = f"output/{__model_id__}"
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  out.columns = [
    "Date", "Worst Case Scenario", "Central Case", "Best Case Scenario"
  ]
  out.to_excel(f"{output_dir}/forecast_{__model_id__}.xlsx", index=False)
  print(f"Model output saved successfully at: {output_dir}")


if __name__ == "__main__":
  main()
