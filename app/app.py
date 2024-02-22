from datetime import date
from turtle import width

from utils_v2 import *
from Lists import *
# from model_pickle_creator import *

# Add content to the left container
# Set the title
# st.title('MTL Outbound Flight Delay Prediction')
# left_column, right_column = st.columns([2, 2])
# with left_column:
#     # select date from datelist
#     selected_date = st.selectbox('Select a Date', DateList)
#     origin = st.selectbox('Origin', ['YUL'])
# with right_column:
#     selected_TOD = st.selectbox('Select a Time of Day', Time_of_Day_List)
#     selected_destination = st.selectbox('Destination', Arrival_IATA_Code_List, index = 16)
# selected_airline = st.selectbox('Airline', Airline_Name_List, index =3)


# Date_Related_Predictors = Match_DateTime_Fields(Date_Features, selected_date, selected_TOD)

# airline_delay_rate = match_delay_rate(selected_airline,fl)
# destination_delay_rate = match_dest_delay_rate(selected_destination,fl)

# # make a dataframe from the user input
# instance = pd.DataFrame({
#     'Arrival IATA Code': selected_destination,
#     'Airline Name': selected_airline,
#     'Temperature': Date_Related_Predictors['Temperature'],
#     'Feels Like': Date_Related_Predictors['Feels Like'],
#     'Pressure': Date_Related_Predictors['Pressure'],
#     'Humidity': Date_Related_Predictors['Humidity'],
#     'Wind Speed': Date_Related_Predictors['Wind Speed'],
#     'Wind Degree': Date_Related_Predictors['Wind Degree'],
#     'Clouds': Date_Related_Predictors['Clouds'],
#     'Weather Main': Date_Related_Predictors['Weather Main'],
#     'Rain 1h': Date_Related_Predictors['Rain 1h'],
#     'Snow 1h': Date_Related_Predictors['Snow 1h'],
#     'Estimated Departure Delay (min)': 10,
#     'Departure Time of Day': Date_Related_Predictors['Departure Time of Day'],
#     'Weekday of Departure': Date_Related_Predictors['Weekday of Departure'],
#     'Weekend Departure': Date_Related_Predictors['Weekend Departure'],
#     'Airline Delay Rate': airline_delay_rate,
#     'Destination Delay Rate': destination_delay_rate,
#     'Weather Severity': Date_Related_Predictors['Weather Severity'],
#     'Season': Date_Related_Predictors['Season'],
#     'Visibility': Date_Related_Predictors['Visibility'],
# }, index=[0])

# st.set_option('deprecation.showPyplotGlobalUse', False)

# # load model
# model = joblib.load('best_model.pkl')
# pipeline = joblib.load('pipeline.pkl')
# instance = pipeline.transform(instance)
# delay = model.predict(instance)


# # Add content to the right container as needed
# display_delay_prediction(delay)


# # Add content to the right container

# if origin and selected_destination:
#     flight_map = plot_flight_curve(origin, selected_destination)
#     st.write(flight_map, width = 150)
# else:
#     st.error("Please enter both origin and destination airports.")

# # Adjust the width of the plot
# st.write("#### Delay Frequency Plot")
# if origin and selected_destination:
#     plot_delay_frequency(fl, selected_airline, selected_destination)
#     # Adjust the width of the plot
#     # st.pyplot()  # Adjust the width as needed


left_column, right_column = st.columns([2, 3])
# wide mode


# Add content to the left container
with left_column:
    # Set the title
    st.title("""MTL Outbound Flight Delay Prediction""")
    l,r = st.columns([1,1])
    with l:
        # select date from datelist
        selected_date = st.selectbox('Select a Date', DateList)
        # selected_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d').date()

        origin = st.selectbox('Origin', ['YUL'])
    with r:
        selected_TOD = st.selectbox('Select a Time of Day', Time_of_Day_List)
        selected_destination = st.selectbox('Destination', Arrival_IATA_Code_List, index = 16)
    
    selected_airline = st.selectbox('Airline', Airline_Name_List, index =3)



Date_Related_Predictors = Match_DateTime_Fields(Date_Features, selected_date, selected_TOD)

airline_delay_rate = match_delay_rate(selected_airline,fl)
destination_delay_rate = match_dest_delay_rate(selected_destination,fl)

# make a dataframe from the user input
instance = pd.DataFrame({
    'Arrival IATA Code': selected_destination,
    'Airline Name': selected_airline,
    'Temperature': Date_Related_Predictors['Temperature'],
    'Feels Like': Date_Related_Predictors['Feels Like'],
    'Pressure': Date_Related_Predictors['Pressure'],
    'Humidity': Date_Related_Predictors['Humidity'],
    'Wind Speed': Date_Related_Predictors['Wind Speed'],
    'Wind Degree': Date_Related_Predictors['Wind Degree'],
    'Clouds': Date_Related_Predictors['Clouds'],
    'Weather Main': Date_Related_Predictors['Weather Main'],
    'Rain 1h': Date_Related_Predictors['Rain 1h'],
    'Snow 1h': Date_Related_Predictors['Snow 1h'],
    'Estimated Departure Delay (min)': 10,
    'Departure Time of Day': Date_Related_Predictors['Departure Time of Day'],
    'Weekday of Departure': Date_Related_Predictors['Weekday of Departure'],
    'Weekend Departure': Date_Related_Predictors['Weekend Departure'],
    'Airline Delay Rate': airline_delay_rate,
    'Destination Delay Rate': destination_delay_rate,
    'Weather Severity': Date_Related_Predictors['Weather Severity'],
    'Season': Date_Related_Predictors['Season'],
    'Visibility': Date_Related_Predictors['Visibility'],
}, index=[0])

st.set_option('deprecation.showPyplotGlobalUse', False)

# load model
model = joblib.load('best_model.pkl')
pipeline = joblib.load('pipeline.pkl')
instance = pipeline.transform(instance)
delay = model.predict(instance)

with left_column:
    # Add content to the right container as needed
    display_delay_prediction(delay)


# Add content to the right container
with right_column:
    if origin and selected_destination:
        flight_map = plot_flight_curve(origin, selected_destination)
        st.write(flight_map, width = 150)
    else:
        st.error("Please enter both origin and destination airports.")

    # Adjust the width of the plot
    st.write("#### Delay Frequency Plot")
    if origin and selected_destination:
        plot_delay_frequency(fl, selected_airline, selected_destination)
        # Adjust the width of the plot
        # st.pyplot()  # Adjust the width as needed





