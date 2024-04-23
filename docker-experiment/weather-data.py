import glob
import pandas as pd
import requests
import arrow                # i don't want to work with datetime

# OpenWeather API Key
api_key: str = "8eff9bb75e22ae972cb3954d01acfccb"

base_url: str = "https://history.openweathermap.org/data/2.5/history/city"

start_date = arrow.get("2024-01-24")
end_date = arrow.get("2024-04-22")

current_date = start_date

while current_date <= end_date:

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
        
        df = pd.DataFrame(weather_data)

        df.to_csv(f"/data/YUL_Weather_{current_date.format('YYYY-MM-DD')}.csv", index = False)

    current_date = current_date.shift(days = 1)

# Concatenating all of the output CSV files from the API pull into one big DataFrame

weather_df = pd.DataFrame()

for csv_file in glob.glob("/data/YUL_Weather_*.csv"):

    try:

        file: pd.DataFrame = pd.read_csv(csv_file)

        if not file.empty:

            weather_df = pd.concat([weather_df, file], ignore_index = True)

    except pd.errors.EmptyDataError:

        print(f"Skipping empty file: {csv_file}")

weather_df.to_csv('/data/Enterprise-II-YUL-Full-Weather.csv', index = False)
