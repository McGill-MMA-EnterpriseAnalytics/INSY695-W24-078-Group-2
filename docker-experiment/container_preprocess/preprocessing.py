
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
import json

#load data 
base_url: str = "http://server_con:80/items/1"
response = requests.get(base_url)
dataset = json.loads(response.text)

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Continue with the preprocessing steps

def custom_preprocess_data(df):
    # Convert Scheduled Departure Time and Estimated Departure Time to datetime
    df['Scheduled Departure Time'] = pd.to_datetime(df['Scheduled Departure Time'])
    df['Estimated Departure Time'] = pd.to_datetime(df['Estimated Departure Time'])
    
    # Calculate the difference in minutes
    # df['Estimated Departure Delay (min)'] = (df['Estimated Departure Time'] - df['Scheduled Departure Time']).dt.total_seconds() / 60

    # Calculate the time of day
    df['Departure Time of Day'] = pd.cut(df['Scheduled Departure Time'].dt.hour, 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                                     right=False)

    # Weekday of departure
    df['Weekday of Departure'] = df['Scheduled Departure Time'].dt.day_name()

    # Calculate weather severety
    df['Weather Severity'] = np.where((df['Rain 1h'] > 0) | (df['Snow 1h'] > 0), 'Bad', 'Good')

    # Filter out detinations with a frequency less than 100
    destintaiton_counts = df['Arrival IATA Code'].value_counts()
    destinations_to_keep = destintaiton_counts[destintaiton_counts >= 100].index
    df = df[df['Arrival IATA Code'].isin(destinations_to_keep)]

    # Filter out infrequent airlines
    # airline_counts = df['Airline Name'].value_counts()
    # airlines_to_keep = airline_counts[airline_counts >= 50].index
    # df = df[df['Airline Name'].isin(airlines_to_keep)]

    # Feature engineering: Create a feature for delay status
    # df['Delay Status'] = pd.cut(df['Departure Delay (min)'], 
    #                             bins=[-np.inf, 0, 15, 60, np.inf], 
    #                             labels=['On Time', 'Slight Delay', 'Moderate Delay', 'Severe Delay'])

    # Feature engineering: Create a feature for season based on month
    df['Season'] = pd.cut(df['Scheduled Departure Time'].dt.month, 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                          right=False)

    # Feature engineering: Create a binary feature for weekend departure
    df['Weekend Departure'] = df['Weekday of Departure'].isin(['Saturday', 'Sunday']).astype(int)

    # Feature engineering: Create a feature for visibility based on weather conditions
    df['Visibility'] = np.where((df['Weather Main'].isin(['Fog', 'Mist', 'Haze', 'Snow', 'Rain'])), 'Low', 'High')

    # # Convert Scheduled Arrival Time and Scheduled Departure Time to datetime before calculating duration
    # df['Scheduled Arrival Time'] = pd.to_datetime(df['Scheduled Arrival Time'])
    # df['Scheduled Departure Time'] = pd.to_datetime(df['Scheduled Departure Time'])
    # df['Flight Duration (min)'] = (df['Scheduled Arrival Time'] - df['Scheduled Departure Time']).dt.total_seconds() / 60

    # Drop unwanted columns
    df = df.drop(columns=['Type', 'Departure IATA Code', 'Scheduled Departure Time', 'Estimated Departure Time', 
    'Actual Departure Time', 'Arrival Terminal', 'Scheduled Arrival Time', 'Estimated Arrival Time', 'Flight Number',
    'IATA Flight Number', 'Timestamp', 'Weather Description'])
    

    df = df[df['Status'] == 'active']

    df = df.drop(columns=['Status'])
    
    return df

df = custom_preprocess_data(df)


categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns


numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),  # Using 5 neighbors for imputation
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


X_processed = pipeline.fit_transform(df)

def get_feature_names_out(column_transformer):
    """Get output feature names for the given ColumnTransformer."""
    feature_names = []

    # Loop through each transformer within the ColumnTransformer
    for transformer_name, transformer, original_features in column_transformer.transformers_:
        if transformer_name == 'remainder':
            continue
        
        if hasattr(transformer, 'get_feature_names_out'):
            # If the transformer can generate feature names
            names = transformer.get_feature_names_out(original_features)
        else:
            # Otherwise, use the original feature names
            names = original_features
        
        feature_names.extend(names)
    
    return feature_names

def transform_output_to_df(X_processed, preprocessor, original_df):
    """Convert the output of the processing pipeline back to a pandas DataFrame."""
    feature_names = get_feature_names_out(preprocessor)
    processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed, 
                                columns=feature_names, 
                                index=original_df.index)
    return processed_df

processed_df = transform_output_to_df(X_processed, pipeline['preprocessor'], df)

# Save the processed data to a JSON file to the server
json_data = processed_df.to_json(orient='records')
base_url: str = "http://server_con:80/items/2"
response = requests.put(base_url, json=json_data)

# Check the response
if response.status_code == 200:
    print("Processed DataFrame sent successfully to Key 2.")
else:
    print("Error:", response.text)



