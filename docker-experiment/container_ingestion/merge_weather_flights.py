"""Module merges weather data
and flight data as pulled from
the respective APIs."""
from pathlib import Path
import json
import requests
import pandas as pd
import arrow

def read_data(input_filepath: Path) -> pd.DataFrame:

    """function to read any input data from csv"""

    return pd.read_csv(input_filepath)


if __name__ == "__main__":

    # setting up the input df file path
    infile_path: Path = Path().absolute()/"data"

    # read in flight and weather data
    flights: pd.DataFrame = read_data(infile_path/"Full_Departures_Data.csv")
    weather: pd.DataFrame = read_data(infile_path/"Enterprise-II-YUL-Full-Weather.csv")

    # vectorized operations to make timestamp changes
    # so as to enable merging with hourly weather timestamps

    flights['Scheduled Departure'] = flights['Scheduled Departure'].apply(
    lambda x: arrow.get(x).format('YYYY-MM-DD HH:00:00') if isinstance(x, str) else x)

    flights['Actual Departure'] = flights['Actual Departure'].apply(
        lambda x: arrow.get(x).format('YYYY-MM-DD HH:00:00') if isinstance(x, str) else x)

    flights['Scheduled Departure'] = flights['Scheduled Departure'].apply(
        lambda x: arrow.get(x).format("YYYY-MM-DD HH:00:00") if pd.notna(x) else x)

    flights['Actual Departure'] = flights['Actual Departure'].apply(
        lambda x: arrow.get(x).format("YYYY-MM-DD HH:00:00") if pd.notna(x) else x)
    
    # Add and adjusting delays to ensure data is consistent
    
    # Case 1: Scheduled Departure == Actual Departure but Delay (in Minutes) != 0
    mask = (flights['Scheduled Departure'] == flights['Actual Departure']) & (flights['Delay (in Minutes)'].notna())
    flights.loc[mask, 'Actual Departure'] = flights.loc[mask].apply(
        lambda row: arrow.get(row['Actual Departure']).shift(
            minutes=row['Delay (in Minutes)']).format('YYYY-MM-DD HH:00:00'), axis=1)

    # Case 2: Scheduled = Actual and Delay is NaN
    mask_nan = (flights['Scheduled Departure'] == flights['Actual Departure']) & (flights['Delay (in Minutes)'].isna())
    flights.loc[mask_nan, 'Delay (in Minutes)'] = 0

    flights_and_weather: pd.DataFrame = flights.merge(
        weather, how = "left", left_on = "Scheduled Departure", right_on = "Timestamp")

    flights_and_weather.to_csv("/app/output/Enterprise-II-YUL-Flights-Weather-Merged.csv", index = False)

    # Setting up code to communicate with the FastAPI Server

    json_data: str = flights_and_weather.to_json(orient = "records")
    base_url: str = "http://server_con:80/items/1"

    response: requests.Response = requests.put(base_url, json = json_data, timeout = 90)

    # Check the response
    if response.status_code == 200:
        print("DataFrame sent successfully.")
        ready_response: requests.Response = requests.put(
            "http://server_con:80/set_ready/1", timeout = 90)
        if ready_response.status_code == 200:
            print("Data marked as ready.")
        else:
            print("Failed to mark data readiness: ", ready_response.status_code)
    else:
        print("Error sending dataframe:", response.text)
