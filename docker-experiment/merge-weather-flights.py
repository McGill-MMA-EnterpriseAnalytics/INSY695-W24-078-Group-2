import pandas as pd 
import arrow
import requests 

print("this shit is running bro")

flights = pd.read_csv("./data/Full_Departures_Data.csv")
weather = pd.read_csv("./data/Enterprise-II-YUL-Full-Weather.csv")


# can I write a function to do this? Yes. 
# writing it out to test if a vectorized operation is better than writing a function

flights['Scheduled Departure'] = flights['Scheduled Departure'].apply(
    lambda x: arrow.get(x).format('YYYY-MM-DD HH:00:00') if isinstance(x, str) else x)

flights['Actual Departure'] = flights['Actual Departure'].apply(
    lambda x: arrow.get(x).format('YYYY-MM-DD HH:00:00') if isinstance(x, str) else x)


flights['Scheduled Departure'] = flights['Scheduled Departure'].apply(
    lambda x: arrow.get(x).format("YYYY-MM-DD HH:00:00") if pd.notna(x) else x)

flights['Actual Departure'] = flights['Actual Departure'].apply(
    lambda x: arrow.get(x).format("YYYY-MM-DD HH:00:00") if pd.notna(x) else x)

flights_and_weather = flights.merge(weather, how = "left", left_on = "Scheduled Departure", right_on = "Timestamp")

# flights_and_weather.to_csv("/app/output/Enterprise-II-YUL-Flights-Weather-Merged.csv", index = False)

json_data = flights_and_weather.to_json(orient = "records")
base_url: str = "http://server_cond:80/items/1"

response = requests.put(base_url, json = json_data)

# Check the response
if response.status_code == 200:
    print("DataFrame sent successfully.")
else:
    print("Error:", response.text)

print("yo this is done running")
