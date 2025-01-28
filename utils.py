import numpy as np
import json
import sys


# Function to log-transform and inverse-transform data
def log_transform(data):
    return np.log1p(data)


def inverse_log_transform(data):
    return np.expm1(data)

def load_config(config_path="config/config.json"):
  """Load configuration from JSON file."""
  try:
    with open(config_path, 'r') as f:
      config = json.load(f)

    # # Validate required fields
    # required_fields = ['data_dir']
    # for field in required_fields:
    #   if field not in config:
    #     raise ValueError(f"Missing required field '{field}' in config file")

    # Set default values if not specified
    config.setdefault('test_size', 6)
    config.setdefault('date_column', 'Created')

    return config

  except FileNotFoundError:
    print(f"Config file not found: {config_path}")
    print("Please create a config.json file with the following structure:")
    print(
      """
                    {
                        "data_directory": "path/to/your/data",
                        "date_column": "name of the date column"
                    }
                            """)
    sys.exit(1)
  except json.JSONDecodeError:
    print(f"Invalid JSON format in config file: {config_path}")
    sys.exit(1)
  except Exception as e:
    print(f"Error reading config file: {str(e)}")
    sys.exit(1)
