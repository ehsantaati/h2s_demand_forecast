import pandas as pd
import numpy as np
from utils import *
from joblib import load
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,  help="Path to the saved model file")

    args = parser.parse_args()
    model_path = Path(args.model_path)

    print(f"Model path provided: {model_path}")

    config = load_config()
    model_id = config["model_id"]
    model= load(model_path)
    print("Model reloaded successfully")

    future = model.make_future_dataframe(periods=12, freq='M',include_history = False)
    forecast = model.predict(future)
    if config["normalize"]:
      forecast['yhat'] = inverse_log_transform(forecast['yhat'])
      forecast['yhat_lower'] = inverse_log_transform(forecast['yhat_lower'])
      forecast['yhat_upper'] = inverse_log_transform(forecast['yhat_upper'])
      # set negative values to zero
    forecast[[col for col in forecast.columns if col!="ds"]] = forecast[[col for col in forecast.columns if col!="ds"]].clip(lower=0)
    cut_off_date = pd.Timestamp(future.loc[0].values[0]) +pd.DateOffset(months=config["test_size"])
    forecast[[col for col in forecast.columns if col!="ds"]] = np.ceil(forecast[[col for col in forecast.columns if col!="ds"]])
    out = forecast[["ds","yhat_lower","yhat","yhat_upper"]]
    out = out[out["ds"]>=cut_off_date]
    output_dir = f"output/{model_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out.columns = ["Date","Worst Case Scenario","Central Case","Best Case Scenario"]
    out.to_excel(f"{output_dir}/forecast_{model_path.stem}.xlsx",index=False)
    print(f"Model output saved successfully at: {output_dir}")


if __name__ == "__main__":
  main()