import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb


# Load the dataset
df = pd.read_csv('processed_dataset_YUL-Flights-Weather.csv')

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

joblib.dump(ensemble, 'ensemble_model.pkl')
print("Model trained and saved successfully")


