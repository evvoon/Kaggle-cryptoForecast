# Crypto Forecasting
This project is based on the G-Research Crypto Forecasting competition hosted on Kaggle. The goal of this competition is to develop a model that can predict the next minute's crypto asset price.

## Data
The data used for this project is provided by G-Research and consists of two CSV files:

train.csv: The training data, containing historical crypto asset data
asset_details.csv: Information about the crypto assets

## Features
The following features are extracted from the raw data:

Count: The number of candles in the interval
Open: The price of the asset at the beginning of the interval
High: The highest price of the asset in the interval
Low: The lowest price of the asset in the interval
Close: The price of the asset at the end of the interval
Volume: The volume of trades in the interval
VWAP: The volume-weighted average price in the interval
Upper_Shadow: The difference between the highest price and the maximum of the close and open price
Lower_Shadow: The difference between the minimum of the close and open price and the lowest price
Model
An XGBoost regressor is trained for each crypto asset using the extracted features.

## Predictions
The predictions for each asset are made by using the trained model to predict the target (price) for each row in the test data.

## Usage
This project can be run on a local machine or a cloud-based system. It requires the following packages:

numpy
pandas
xgboost
gresearch_crypto
Once these packages are installed, the project can be run by executing the provided code.

## Results
The model achieved an RMSLE score of approximately 0.08. This is a good result, but there is still room for improvement. To further improve the model, more features or a different model could be used.
