import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from geopy.geocoders import Nominatim
from folium import plugins
from geopy.distance import geodesic
import streamlit as st
import joblib
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import optuna
import pipeline


# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set seaborn style
sns.set(style="whitegrid")

# Set streamlit page config
st.set_page_config(layout="wide")

# Set random seed for reproducibility
np.random.seed(42)




def load_data():
    """
    Load dataset and set project filters

    """

    df = pd.read_csv('dataset_YUL-Flights-Weather.csv')
    df = df.drop_duplicates()

    # Filter out destinations with a frequency less than 100
    destination_counts = df['Arrival IATA Code'].value_counts()
    destinations_to_keep = destination_counts[destination_counts >= 100].index
    df = df[df['Arrival IATA Code'].isin(destinations_to_keep)]

    # make arrival IATA Code uppercase
    df['Arrival IATA Code'] = df['Arrival IATA Code'].str.upper()
    df['Airline Name'] = df['Airline Name'].str.upper()

    # Filter out airlines with a frequency less than 100
    airline_counts = df['Airline Name'].value_counts()
    airlines_to_keep = airline_counts[airline_counts >= 100].index
    df = df[df['Airline Name'].isin(airlines_to_keep)]

    
    # Filter rows where 'Status' is not 'active'
    df = df[df['Status'] == 'active']
    
    # Drop the 'Status' column as it's no longer needed
    df = df.drop(columns=['Status','Departure Gate','IATA Flight Number'])

    return df


def preprocess_dates(df):
    """
    Engineering variables from date columns
    
    """

    # Convert Scheduled Departure Time and Estimated Departure Time to datetime
    df['Scheduled Departure Time'] = pd.to_datetime(df['Scheduled Departure Time'])
    df['Estimated Departure Time'] = pd.to_datetime(df['Estimated Departure Time'])
    
    # Calculate the difference in minutes
    df['Estimated Departure Delay (min)'] = (df['Estimated Departure Time'] - df['Scheduled Departure Time']).dt.total_seconds() / 60

    # Calculate the time of day
    df['Departure Time of Day'] = pd.cut(df['Scheduled Departure Time'].dt.hour, 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                                     right=False)

    # Weekday of departure
    df['Weekday of Departure'] = df['Scheduled Departure Time'].dt.day_name()

    # Feature engineering: Create a binary feature for weekend departure
    df['Weekend Departure'] = df['Weekday of Departure'].isin(['Saturday', 'Sunday']).astype(int)


    #### experimental
    df['Airline Delay Rate'] = df.groupby('Airline Name')['Departure Delay (min)'].transform('mean')
    # Calculate the overall delay rate of destination
    df['Destination Delay Rate'] = df.groupby('Arrival IATA Code')['Departure Delay (min)'].transform('mean')

    return df

def preprocess_weather(df):
    # Calculate weather severity
    df['Weather Severity'] = np.where((df['Rain 1h'] > 0) | (df['Snow 1h'] > 0), 'Bad', 'Good')

    # Feature engineering: Create a feature for season based on month
    df['Season'] = pd.cut(df['Scheduled Departure Time'].dt.month, 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                          right=False)

    # Feature engineering: Create a feature for visibility based on weather conditions
    df['Visibility'] = np.where((df['Weather Main'].isin(['Fog', 'Mist', 'Haze', 'Snow', 'Rain'])), 'Low', 'High')    
    return df

def unwanted_columns(df):
    # Drop unwanted columns #IATA flight number is kept
    df = df.drop(columns=['Type', 'Departure IATA Code', 'Estimated Departure Time', 
    'Actual Departure Time', 'Arrival Terminal', 'Scheduled Arrival Time', 'Estimated Arrival Time', 'Flight Number',
    'Timestamp', 'Weather Description', 'Scheduled Departure Time'])
    return df


#--------------------------------------------------------------------------------------
##### encoder decoder

def encodings_imputers(df):
    try:
        X = df.drop(columns=['Departure Delay (min)'])
        y = df['Departure Delay (min)'] 
    except:
        X = df
        y = None
  
    # Define categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing for numerical data with KNNImputer
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  # Using 5 neighbors for imputation
        ('scaler', StandardScaler())])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define the model preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply the pipeline to the dataset
    X_processed = pipeline.fit_transform(df)
    
    return X_processed, y, pipeline


### experimental
def encode_new_instance(instance_values, pipeline):
    """
    Encode a new instance of data using the preprocessing pipeline.

    Parameters:
    instance_values (dict or list): Values of predictors for the new instance.
                                    It can be a dictionary where keys are column names
                                    and values are corresponding values for each column,
                                    or a list where values correspond to columns in the
                                    same order as in the original DataFrame.
    pipeline (Pipeline): Preprocessing pipeline learned from the training data.

    Returns:
    array: Encoded representation of the new instance.
    """
    # Convert instance_values to DataFrame if it's a dictionary
    if isinstance(instance_values, dict):
        instance_df = pd.DataFrame([instance_values])
    else:
        instance_df = pd.DataFrame([instance_values], columns=pipeline.named_steps['preprocessor'].get_feature_names_out())

    # Apply the preprocessing pipeline to the new instance
    encoded_instance = pipeline.transform(instance_df)
    joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

    return encoded_instance


#--------------------------------------------------------------------------------------


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



def split_data(X_processed, y, test_size=0.2, random_state=42):
    """
    Split the DataFrame into features (X) and target variable (y),
    then split them into training and testing sets.
    
    Parameters:
    df (DataFrame): The input DataFrame containing features and target variable.
    target_column (str): The name of the target column.
    test_size (float, optional): The proportion of the dataset to include in the test split.
    random_state (int, optional): Controls the shuffling applied to the data before splitting.
    
    Returns:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.
    y_train (Series): Training target.
    y_test (Series): Testing target.
    """

    # Separate features and target variable
    X = X_processed
    y = y
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

## --------------------------------------------------------------------------------------
##### Front end Functions

def display_delay_prediction(unscaled_delay):
    
    if unscaled_delay > 60:
        message = "⛔️ Don't buy this flight by any chance. There is a strong likelihood of significant delay (> 1 hour)."
    elif 30 < unscaled_delay <= 60:
        message = "⚠️ Consider booking another flight. The delay for this flight is likely to be more than 30 minutes."
    elif 0 < unscaled_delay <= 30:
        message = "✈️ There will be some delay, but if the flight is significantly cheaper than other options, you can consider booking it. Expect a delay of approximately 0-30 minutes."
    else:
        message = "🛫 This flight is expected to be on time or may experience only a minimal delay. You can proceed with booking."

    st.write('The delay for this flight is', unscaled_delay, 'minutes.')
    st.write(message)


def get_coordinates(airport):
    geolocator = Nominatim(user_agent="flight_app")
    location = geolocator.geocode(airport)
    return location.latitude, location.longitude

def plot_flight_curve(origin, destination):
    origin_coords = montreal_coords = (45.5017, -73.5673)
    dest_coords = get_coordinates(destination)
    
    m = folium.Map(location=[origin_coords[0], origin_coords[1]], zoom_start=4)
    
    # Add markers for origin and destination
    folium.Marker(location=[origin_coords[0], origin_coords[1]], popup=origin).add_to(m)
    folium.Marker(location=[dest_coords[0], dest_coords[1]], popup=destination).add_to(m)
    
    # Plot flight curve
    points = [origin_coords, dest_coords]
    folium.PolyLine(locations=points, color='blue').add_to(m)
    
    return m

#--------------------------------------------------------------------------------------

def match_delay_rate(airline_name, fl):
    tempQ= fl[['Airline Name', 'Airline Delay Rate']]
    tempQ = tempQ.drop_duplicates()
    airline_delay_rate = tempQ[tempQ['Airline Name'] == airline_name]['Airline Delay Rate'].values[0]
    return airline_delay_rate

def match_dest_delay_rate(airline_name,fl):
    tempE= fl[['Arrival IATA Code', 'Destination Delay Rate']]
    tempE = tempE.drop_duplicates()
    destination_delay_rate = tempE[tempE['Arrival IATA Code'] == airline_name]['Destination Delay Rate'].values[0]
    return destination_delay_rate

def Deployment_Feature_Creation_From_User_Input(fl):
    # Convert 'Scheduled Departure Time' to date
    Temp = fl
    Temp['Date'] = pd.to_datetime(Temp['Scheduled Departure Time']).dt.date

    # Reorder and select columns
    Temp = Temp[['Date', 'Departure Time of Day', 'Temperature', 'Feels Like', 'Pressure', 'Humidity', 'Wind Speed',
                 'Wind Degree', 'Clouds', 'Weather Main', 'Rain 1h', 'Snow 1h',
                 'Weekday of Departure', 'Weekend Departure', 'Weather Severity', 'Season', 'Visibility']]

    # Drop duplicates based on the 'Date' column
    Temp.drop_duplicates(inplace=True)

    # Reset the index to make 'Date' a regular column again
    Temp.reset_index(drop=True, inplace=True)

    return Temp

def Match_DateTime_Fields(Temp, Date, Time_of_Day):
    # find row where date and time of day, both match
    TempW = Temp[(Temp['Date'] == Date) & (Temp['Departure Time of Day'] == Time_of_Day)]
    TempW.drop(columns=['Date'], inplace=True)
    # take first row
    TempW = TempW.head(1)
    return TempW


#--------------------------------------------------------------------------------------

# frequency distribution of actual delay for the airline and route
def plot_delay_frequency(fl, airline, destination):
    # Filter the data for the specific airline and destination
    data = fl[(fl['Airline Name'] == airline) & (fl['Arrival IATA Code'] == destination)]
    
    # Plot the distribution of actual delays
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Departure Delay (min)'], kde=True, ax=ax)
    ax.set_title(f'Distribution of Actual Delays for {airline} to {destination}')
    ax.set_xlabel('Departure Delay (min)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)