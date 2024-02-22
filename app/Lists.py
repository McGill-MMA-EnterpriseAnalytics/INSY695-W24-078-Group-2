from utils_v2 import *
import pandas as pd 

fl = load_data()
fl = preprocess_dates(fl)
fl = preprocess_weather(fl)
# fl = unwanted_columns(fl)

# Assuming 'fl' is the DataFrame containing the data
Date_Features = Deployment_Feature_Creation_From_User_Input(fl)
DateList = Date_Features['Date'].unique().tolist()
Time_of_Day_List = Date_Features['Departure Time of Day'].unique().tolist()

# unique list of arrival iata code
Arrival_IATA_Code_List = fl['Arrival IATA Code'].unique().tolist()

#unique list of airline name
Airline_Name_List = fl['Airline Name'].unique().tolist()




