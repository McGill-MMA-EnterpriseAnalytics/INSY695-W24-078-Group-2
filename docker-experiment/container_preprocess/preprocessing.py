"""Microservice for the preprocessing steps taken."""
import json
import requests
import time
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def listen_for_readiness(url: str, item_id: int):
    """Setting up the polling system piece"""
    while True:
        try:
            response = requests.get(f"{url}/ready/{item_id}")
            if response.status_code == 200 and response.json()['status'] == 'ready':
                print(f"Data for item {item_id} is now ready for the next process.")
                break
            else:
                print(f"Waiting on data for item {item_id} to be ready for the next process...")
        except requests.exceptions.RequestException as ek:
            print("Error encountered while checking data readiness: ", str(ek))
        time.sleep(30)

# Use the function to wait for item 2
listen_for_readiness(url="http://server_con:80", item_id=1)

# After the data is confirmed ready, retrieve and process it
response = requests.get("http://server_con:80/items/1")
if response.status_code == 200:
    try:
        dataset = json.loads(response.text)
        df_in = pd.read_json(StringIO(dataset))
    except json.JSONDecodeError as json_err:
        print(f"Failed to parse JSON: {str(json_err)}")
    except pd.errors.ParserError as df_err:
        print("Failed to create dataframe: ", str(df_err))
else:
    print("Failed to retrieve any data for item 2", response.status_code)

# Attempt to create a DataFrame
# try:
#     df_in = pd.DataFrame.from_dict(dataset)
#     print("DataFrame created successfully.")
# except Exception as e:
#     print("Error creating DataFrame:", e)

# print(f"shape of df: {df_in.shape}")

# Continue with the preprocessing steps

def custom_preprocess_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Method to implement custom pre-processing steps."""
    # Convert Scheduled Departure and Actual Departure Time to datetime
    input_df['Scheduled Departure'] = pd.to_datetime(input_df['Scheduled Departure'])
    input_df['Actual Departure'] = pd.to_datetime(input_df['Actual Departure'])

    # Calculate the time of day
    input_df['Departure Time of Day'] = pd.cut(input_df['Scheduled Departure'].dt.hour, 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                                     right=False)

    # Weekday of departure
    input_df['Weekday of Departure'] = input_df['Scheduled Departure'].dt.day_name()

    # Calculate weather severety
    input_df['Weather Severity'] = np.where((input_df['Rain 1h'] > 0) | (input_df['Snow 1h'] > 0), 'Bad', 'Good')

    # Filter out destinations with a frequency less than 100
    destination_counts = input_df['Arrival IATA Code'].value_counts()
    destinations_to_keep = destination_counts[destination_counts >= 100].index
    input_df = input_df[input_df['Arrival IATA Code'].isin(destinations_to_keep)]

    # Feature engineering: Create a feature for season based on month
    input_df['Season'] = pd.cut(input_df['Scheduled Departure'].dt.month, 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                          right=False)

    # Feature engineering: Create a binary feature for weekend departure
    input_df['Weekend Departure'] = input_df['Weekday of Departure'].isin(['Saturday', 'Sunday']).astype(int)

    # Feature engineering: Create a feature for visibility based on weather conditions
    input_df['Visibility'] = np.where(
        (input_df['Weather Main'].isin(['Fog', 'Mist', 'Haze', 'Snow', 'Rain'])), 'Low', 'High')

    # Drop unwanted columns
    input_df = input_df.drop(columns=['Departure Airport', 'Scheduled Departure',
    'Actual Departure','Flight Number', 'Timestamp', 'Weather Description'])

    input_df = input_df.drop(columns=['Flight Status'])

    return input_df

cp_df = custom_preprocess_data(df_in)


categorical_cols = cp_df.select_dtypes(include=['object', 'category']).columns
numerical_cols = cp_df.select_dtypes(include=['int64', 'float64']).columns


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


X_processed = pipeline.fit_transform(cp_df)

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

def transform_output_to_df(X_processed, preprocessor, original_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the output of the processing pipeline back to a pandas DataFrame."""
    feature_names = get_feature_names_out(preprocessor)
    processing_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed, 
                                columns=feature_names, 
                                index=original_df.index)
    return processing_df

processed_df = transform_output_to_df(X_processed, pipeline.named_steps['preprocessor'], cp_df)

# Save the processed data to a JSON file to the server
json_data = processed_df.to_json(orient='records')
base_url: str = "http://server_con:80/items/2"
response = requests.put(base_url, json=json_data)

# Check the response
if response.status_code == 200:
    print("Processed DataFrame sent successfully to Key 2.")
    ready_url = f"http://server_con:80/set_ready/2"
    ready_response = requests.put(ready_url)
    if ready_response.status_code == 200:
        print("Preprocessed data marked as ready.")
    else:
        print("Failed to mark preprocessed data as ready:", ready_response.text)
else:
    print("Error:", response.text)
