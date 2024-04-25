import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import requests
import json

# Load the dataset
base_url: str = "http://server_con:80/items/2"
response = requests.get(base_url)
dataset = json.loads(response.text)
df = pd.read_json(dataset)

# Separate features and target variable
X = df.drop('Departure Delay (min)', axis=1)
y = df['Departure Delay (min)']

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

# Save the model
import joblib

joblib.dump(ensemble, '/app/output/ensemble_model.pkl')
print("Model trained and saved successfully")


