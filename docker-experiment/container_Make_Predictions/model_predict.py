import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import joblib
import requests
import json

#load model
ensemble = joblib.load('ensemble_model.pkl')

#load data
base_url: str = "http://server_con:80/items/2"
response = requests.get(base_url)
dataset = json.loads(response.text)
to_predict = pd.read_json(dataset)[0:10]

#to_predict = pd.read_csv('processed_dataset_YUL-Flights-Weather.csv').iloc[0:10]
to_predict = to_predict.drop('Departure Delay (min)', axis=1)

# Predict
predictions = ensemble.predict(to_predict).tolist()

# Save predictions
preds = pd.DataFrame(predictions, columns=['Predicted Departure Delay (min)'])
json_data = preds.to_json(orient = "records")
base_url: str = "http://server_con:80/items/3"
response = requests.put(base_url, json = json_data)

# Check the response
if response.status_code == 200:
    print("Results DataFrame sent successfully to Key 3.")
else:
    print("Error:", response.text)

