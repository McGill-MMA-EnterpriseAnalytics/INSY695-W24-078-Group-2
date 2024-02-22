from datetime import date
from utils import *


# Set the title
st.title('Flight Delay Prediction')

## description of usage

# Define the minimum and maximum allowed dates
min_date = datetime.date(2024, 1, 15)
max_date = datetime.date(2024, 2, 1)
date = st.date_input('Date of the flight', min_value=min_date, max_value=max_date)


# input through streamlit Departure_Time_of_day = ['Night', 'Morning', 'Afternoon', 'Evening']
Departure_Time_of_day = st.selectbox('Departure Time of Day', ['Night', 'Morning', 'Afternoon', 'Evening'])
Departure_DoW = st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
Weekend_Departure = st.selectbox('Weekend Departure', ['0', '1'])



# # input the origin of the flight from drop down 
# origin = st.selectbox('Origin', ['YUL'])
# destination = st.selectbox('Destination', ['YYZ', 'YVR', 'YEG', 'YHZ', 'YOW', 'YQB', 'YQR', 'YXE', 'YWG'])


origin = st.selectbox('Origin', ['YUL'])
destination = st.selectbox('Destination', ['YYZ', 'YVR', 'YEG', 'YHZ', 'YOW', 'YQB', 'YQR', 'YXE', 'YWG'])

# load model
model = joblib.load('super_ensemble_model.pkl')


# predict the delay
delay = model.predict(instance)
display_delay_prediction( unscale_target(delay))


if origin and destination:
    flight_map = plot_flight_curve(origin, destination)
    st.write(flight_map)
else:
    st.error("Please enter both origin and destination airports.")

# function to create a timeline of number of delayed flights for this destination
#






