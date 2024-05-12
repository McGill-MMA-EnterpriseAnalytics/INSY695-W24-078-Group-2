"""
This module contains the code required to extract
weather data at the Montreal-Trudeau Airport (YUL)
using the OpenweatherMap API.
"""
import glob
import pandas as pd
import requests
import arrow

# Setting up global variables

# Setting up API base URL + API secret
# For more information on the API, please refer:
# https://openweathermap.org/api
base_url: str = "https://history.openweathermap.org/data/2.5/history/city"
api_key: str = "************************"

# Setting up date objects using arrow
# uppercased variables are constants
START_DATE = arrow.get("2024-01-24")
END_DATE = arrow.get("2024-04-22")

def pull_openweather_data():
    """This function grabs data from the
    OpenWeatherMap API upon being called."""

    current_date = START_DATE

    while current_date <= END_DATE:

        query_params:dict = {
            "id": 6077243,
            "type": "hour",
            "appid": api_key,
            "start": current_date.timestamp(),
            "units": "metric",
        }

        response = requests.get(base_url, params = query_params)

        if response.status_code == 200:

            json_response = response.json()

            weather_data = []

            for entry in json_response['list']:

                timestamp = arrow.get(entry['dt']).format("YYYY-MM-DD HH:mm:ss")

                weather_data.append({
                    'Timestamp': timestamp,
                    'Temperature': entry['main']['temp'],
                    'Feels Like': entry['main']['feels_like'],
                    'Pressure': entry['main']['pressure'],
                    'Humidity': entry['main']['humidity'],
                    'Wind Speed': entry['wind']['speed'],
                    'Wind Degree': entry['wind'].get('deg', ''),
                    'Clouds': entry['clouds']['all'],
                    'Weather Main': entry['weather'][0]['main'],
                    'Weather Description': entry['weather'][0]['description'],
                    'Rain 1h': entry.get('rain', {}).get('1h', 0),
                    'Snow 1h': entry.get('snow', {}).get('1h', 0)
                    })
                       
            weather_df = pd.DataFrame(weather_data)

            weather_df.to_csv(f"/data/YUL_Weather_{current_date.format('YYYY-MM-DD')}.csv", index = False)

        current_date = current_date.shift(days = 1)

# Concatenating all of the output CSV files from the API pull into one big DataFrame

def concatenate_weather_data():
    """Putting together all available weather data."""
    overall_weather_df = pd.DataFrame()

    for csv_file in glob.glob("/data/YUL_Weather_*.csv"):

        try:

            file: pd.DataFrame = pd.read_csv(csv_file)

            if not file.empty:

                overall_weather_df = pd.concat([overall_weather_df, file], ignore_index = True)

        except pd.errors.EmptyDataError:

            print(f"Skipping empty file: {csv_file}")

    overall_weather_df.to_csv('/data/Enterprise-II-YUL-Full-Weather.csv', index = False)


if __name__ == "__main__":

    pull_openweather_data()

    concatenate_weather_data()
