# H2S Demand Forecasting System

A robust and adaptable demand forecasting system for Home To School Transport Service, leveraging machine learning to predict resource needs across eligibility, delivery, complaints, and appeals activities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Forecasting](#forecasting)
  - [Interactive Analysis](#interactive-analysis)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Model Architecture](#-model-architecture)

## 🎯 Overview

This project implements a demand forecasting system for Home To School Transport services. By analysing historical service data, critical path points, and seasonal patterns, the system predicts future resource requirements (permanent, fixed-term, and bank staff) across various service activities.

![Model Overview](assets/Model_Overview.jpg)

## ✨ Features

- **Automated Data Processing**: Converts daily records to monthly time series automatically
- **Advanced Forecasting**: Uses Facebook's Prophet model with automatic hyperparameter tuning
- **Cross-Validation**: Implements robust model validation techniques
- **Interactive Analysis**: Jupyter notebook for detailed analysis and visualization
- **Flexible Configuration**: Easy-to-modify JSON configuration system
- **Performance Metrics**: Comprehensive model evaluation with MAPE and RMSE
- **Visualization**: Interactive plots for time series and forecasts
- **Resource Allocation**: Under development.

## 📁 Project Structure

```
h2s_demand_forecast/
├── assets/              # Project assets and images
├── config/             # Configuration files
│   └── config.json    # Main configuration file
├── data/              # Data directory for input files
├── models/            # Saved trained models
├── output/            # Model outputs and metrics
├── forecast.py        # Forecasting script
├── train.py          # Model training script
├── utils.py          # Utility functions
├── modelling.ipynb  # Interactive analysis notebook
└── requirements.txt   # Python dependencies
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ehsantaati/h2s_demand_forecast.git
   cd h2s_demand_forecast
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Configuration

Create or modify `config/config.json` with your settings:

```json
{
    "model_id": "h2s_forecast",
    "normalize": false,
    "test_size": 6,
    "date_column": "Created"
}
```

Key parameters:
- `model_id`: Unique identifier for the model and its outputs
- `normalize`: Whether to apply log transformation to the data
- `test_size`: Number of months for testing (default: 6)
- `date_column`: Name of the date column in your data

### Training

1. **Prepare Your Data**
   - Place your Excel files in the `data` directory
   - Ensure files contain the specified date column
   - Data should be in daily format (will be aggregated to monthly)

2. **Train the Model**
   ```bash
   python -m train
   ```
   The training process includes:
   - Data preprocessing and validation
   - Automatic calculation of initial, horizon and period for Prophet based on teh input data
   - Hyperparameter tuning via cross-validation
   - Model training and evaluation
   - Saving model and metrics

3. **Check Results**
   - Trained model saved in `models/{model_id}/`
   - Performance metrics saved in `models/{model_id}/`

### Forecasting

1. **Generate Forecasts**
   ```bash
   python -m forecast
   ```
   This will:
   - Load trained models from the `models` directory
   - Generate forecasts for future periods
   - Save results in the `output` directory


## 🔍 Model Details

The system uses Facebook's Prophet model, which is particularly effective for time series with:
- Strong seasonal patterns
- Missing data
- Outliers
- Trend changes

Key features of the modeling approach:
- Automatic seasonality detection
- Robust to missing data
- Handles outliers effectively
- Configurable changepoint detection
- Uncertainty estimation

## 📊 Performance Metrics

The model's performance is evaluated using:
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)

Results are saved in JSON format with:
- Overall metrics
- Cross-validation results
- Forecast confidence intervals

## 🔄 Model Architecture

```mermaid
graph TD
    subgraph Data Layer["Data Layer"]
        style Data Layer fill:#e6f3ff,stroke:#4d94ff
        A[Excel Files] --> B[Data Processing]
        B --> C[Monthly Time Series]
        style A fill:#cce6ff,stroke:#4d94ff
        style B fill:#cce6ff,stroke:#4d94ff
        style C fill:#cce6ff,stroke:#4d94ff
    end

    subgraph Training Pipeline["Training Pipeline"]
        style Training Pipeline fill:#fff0f3,stroke:#ff4d6d
        C --> D[Data Validation]
        D --> E[Cross-Validation]
        E --> F[Hyperparameter Tuning]
        F --> G[Model Training]
        G --> H[Model Evaluation]
        H --> I[Save Model & Metrics]
        H -.-> E
        style D fill:#ffe6eb,stroke:#ff4d6d
        style E fill:#ffe6eb,stroke:#ff4d6d
        style F fill:#ffe6eb,stroke:#ff4d6d
        style G fill:#ffe6eb,stroke:#ff4d6d
        style H fill:#ffe6eb,stroke:#ff4d6d
        style I fill:#ffe6eb,stroke:#ff4d6d
    end

    subgraph Forecasting Pipeline["Forecasting Pipeline"]
        style Forecasting Pipeline fill:#f0fff4,stroke:#4dff88
        J[Load Model] --> K[Generate Forecast]
        K --> L[Post-Processing]
        L --> M[Output Results]
        style J fill:#e6ffe6,stroke:#4dff88
        style K fill:#e6ffe6,stroke:#4dff88
        style L fill:#e6ffe6,stroke:#4dff88
        style M fill:#e6ffe6,stroke:#4dff88
    end

    subgraph Utility Functions["Utility Functions"]
        style Utility Functions fill:#fff8e6,stroke:#ffb84d
        N[Data Validation]
        O[CV Parameters]
        P[Visualization]
        Q[Metrics Calculation]
        style N fill:#fff2cc,stroke:#ffb84d
        style O fill:#fff2cc,stroke:#ffb84d
        style P fill:#fff2cc,stroke:#ffb84d
        style Q fill:#fff2cc,stroke:#ffb84d
    end

    B -.-> N
    E -.-> O
    H -.-> P
    H -.-> Q
    K -.-> P
    I --> J
```

The diagram above illustrates the main components and data flow of the H2S Demand Forecasting System:

1. **Data Layer**: Handles data ingestion and preprocessing
2. **Training Pipeline**: Manages model training and evaluation
3. **Forecasting Pipeline**: Handles prediction generation
4. **Utility Functions**: Provides shared functionality across components

Each component is designed to be modular and maintainable, with clear separation of concerns and robust error handling.

