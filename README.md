<h1 style="text-align: center;">Demand forecasing and Resource Allocation: Home to School Transport Service</h1>


This project aims to develop a robust and adaptable resourcing model for Home To School Transport, focusing on the core activities of eligibility, delivery, complaints, and appeals. By analyzing historical service data, critical path points, and known seasonal fluctuations, the model will predict future resource needs (permanent, fixed-term, and bank staff) across these activities.


## Model Overview
![Model Overview](assets/Model_Overview.jpg)
## Getting Started

### Prerequisites

* Python 3.12
* pip (package installer for Python)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ehsantaati/h2s_demand_forecast.git
```
2. Navigate to the ```Python``` environment which already should be installed in your machine.

3. Install required packages:
```bash
pip install -r requirements.txt
```
### Train a forecatser
The forecasting model utilises [Prophet](https://facebook.github.io/prophet/), which undergoes automatic training via cross-validation to tune its hyperparameters. The entire pipeline is designed to be automated, encompassing data preparation, splitting, model training, and performance evaluation. To train the model, the following steps should be adhered to:
1. The pipeline consumes daily recorded data and transforms it into a monthly time series. To achieve this, configure the ```model_id``` and ```date_column``` parameters in ```config/config.json``` according to the data. The data should be provided in an Excel format and placed within the ```data```directory.
2. To start training process execute the following command in the command line:

    ```bash
    python -m train
    ```
The training process can be time-consuming, depending on the dataset size. Upon completion, the trained model will be saved in the designated ```model``` directory recognisable with the provided ```model_id```, and the corresponding evaluation results will be written to the ```output``` directory.
### Forecast with a trained model
To perform forecasts with a trained model, you need the path to the trained model file (models_dir). By default, this is set to the models directory. Ensure that the trained forecaster(s) are in this folder before executing the following command.<br>

1. Navigate to the Python environment that should already be installed on your machine.
2. To start the forecasting pipeline, execute the following command in the command line:<br>

    ```bash
    python -m forecast
    ```
<br>
All trained models with the ```.joblib``` extension within the models folder will be automatically utilized to forecast future demands.
The results will be saved in the ```output``` directory recognisable with ```model_id```.

### Parameters
```model_id```: Model name which all the model's output will be identified with.<br>
```test_size```: The number of months used to create the test set and evaluate model performance defaults to 6.<br>
```date_column```: Name of the column containing dates in the data.<br>
 outliers are present in the data. The default value is ```False```.<br>
The following options are for the [Prophet model for Cross-Validation](https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation). These values have been determined based on experiments conducted on the available data.<br>
```initial```: 365 days<br>
```period```: 30 days<br>
```horizon```: 90 days<br>
