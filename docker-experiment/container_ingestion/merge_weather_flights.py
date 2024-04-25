"""Module merges weather data
and flight data as pulled from
the respective APIs."""
from pathlib import Path
import pandas as pd
import arrow
import requests

def read_data(input_filepath: Path) -> pd.DataFrame:

    """function to read any input data from csv"""

    return pd.read_csv(input_filepath)


if __name__ == "__main__":

    # setting up the input df file path
    infile_path: Path = Path().absolute()/"data"

    # read in flight and weather data
    flights = read_data(infile_path/"Full_Departures_Data.csv")
    weather = read_data(infile_path/"Enterprise-II-YUL-Full-Weather.csv")

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

    flights_and_weather: pd.DataFrame = flights.merge(
        weather, how = "left", left_on = "Scheduled Departure", right_on = "Timestamp")

    # flights_and_weather.to_csv("/app/output/Enterprise-II-YUL-Flights-Weather-Merged.csv", index = False)

    # Setting up code to communicate with the FastAPI Server

    json_data: str = flights_and_weather.to_json(orient = "records")
    base_url: str = "http://server_con:80/items/1"

    response = requests.put(base_url, json = json_data)

    # Check the response
    if response.status_code == 200:
        print("DataFrame sent successfully.")
    else:
        print("Error:", response.text)





