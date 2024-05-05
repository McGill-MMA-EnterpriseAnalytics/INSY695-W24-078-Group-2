"""Microservice to train and make predictions"""
import time
import json
import requests
import joblib
import xgboost as xgb
import pandas as pd
from io import StringIO
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR

def listen_for_readiness(url, item_id):
    """Polls the server to check if the data is ready for modeling."""
    while True:
        try:
            readiness_response: requests.Response = requests.get(
                f"{url}/ready/{item_id}", timeout = 90)
            if readiness_response.status_code == 200 and readiness_response.json()['status'] == 'ready':
                print("Data is ready for modeling.")
                break
            else:
                print("Data not ready, waiting...")
        except requests.exceptions.RequestException as e:
            print("Error checking data readiness:", str(e))
        time.sleep(10)  

# Check for readiness of the pre-processing output prior to triggering the modeling workflow
listen_for_readiness(url="http://server_con:80", item_id=2)

# Load the dataset
base_url: str = "http://server_con:80/items/2"
response: requests.Response = requests.get(base_url)
dataset = json.loads(response.text)
df = pd.read_json(StringIO(dataset))

# Separate features and target variable
X = df.drop('Delay (in Minutes)', axis=1)
y = df['Delay (in Minutes)']

ensemble = VotingRegressor(
        estimators=[
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42)),
            ('xgb', xgb.XGBRegressor(random_state=42)),
            ('svm', SVR())
        ])

# Train models
ensemble.fit(X, y)

print("Model trained and saved successfully")

#load data
base_url: str = "http://server_con:80/items/2"
response: requests.Response = requests.get(base_url)
dataset = json.loads(response.text)
to_predict = pd.read_json(dataset)[0:10]

#to_predict = pd.read_csv('processed_dataset_YUL-Flights-Weather.csv').iloc[0:10]
to_predict = to_predict.drop('Delay (in Minutes)', axis=1)

# Predict
predictions = ensemble.predict(to_predict).tolist()

# Save predictions
preds = pd.DataFrame(predictions, columns=['Predicted Delay (in Minutes)'])
# Write predictions to csv file locally for data redundancy
preds.to_csv('/app/output/predictions-output.csv', index = False)
json_data = preds.to_json(orient = "records")
base_url: str = "http://server_con:80/items/3"
response: requests.Response = requests.put(base_url, json = json_data)

# Check the response after sending predictions
if response.status_code == 200:
    print("Results DataFrame sent successfully to Key 3.")
    # Mark the modeling output as ready
    ready_url: str = "http://server_con:80/set_ready/3"
    ready_response: requests.Response = requests.put(ready_url)
    if ready_response.status_code == 200:
        print("Modeling output marked as ready.")
    else:
        print("Failed to mark modeling output as ready:", ready_response.text)
else:
    print("Error sending modeling output:", response.text)


