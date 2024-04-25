import glob
import requests
import pandas as pd
import arrow

# Setting up API Base URL + Credentials

base_url: str = "https://api.aviationstack.com/v1/flights"

api_key: str = "d2a6e0d7da1d79dd85f9c617082d2d56"

today = arrow.now()                             # grabbing today's date
start_date = today.shift(days = -90)            # date when to start the 90 day lookback window
offsets: list[int] = [0, 100, 200]              # working around the flightstack API response pagination

for i in range(91):

    dte = start_date.shift(days = i).format('YYYY-MM-DD')

    for offset in offsets:

        query_params: dict = {
            "access_key": api_key,
            "flight_date": dte,
            "limit": 100,
            "offset": offset,
            "dep_iata": "YUL",
        }

        response = requests.get(base_url, query_params)

        if response.status_code == 200:

            # print(f"Pull Successful - Writing to file # {i+1}...")

            response_json = response.json()

            required_data = response_json.get('data', [])

            structured_data: list = []

            for flight in required_data:

                if isinstance(flight, dict):

                    if flight['flight'].get('codeshared') is None:

                        structured_data.append({
                            'Flight Status': flight['flight_status'],
                            'Departure Airport': flight['departure'].get('iata', ''),
                            'Departure Gate': flight['departure'].get('gate', ''),
                            'Arrival IATA Code': flight['arrival'].get('iata', ''),
                            'Scheduled Departure': flight['departure'].get('scheduled', ''),
                            'Actual Departure': flight['departure'].get('actual', ''),
                            'Delay (in Minutes)': flight['departure'].get('delay', 0),
                            'Airline Name': flight['airline'].get('name', ''),
                            'Flight Number': flight['flight'].get('iata', ''),
                        })
                
            departures_df: pd.DataFrame = pd.DataFrame(structured_data)

            departures_df.to_csv(f"/data/YUL_Departures_{dte}_{offset}.csv", index = False)


full_departures_df = pd.DataFrame()             # initializing empty dataframe to concatenate all the daily dataframes

for csv_file in glob.glob('/data/YUL_Departures_*.csv'):

    try:

        file: pd.DataFrame = pd.read_csv(csv_file)

        if not file.empty:

            full_departures_df = pd.concat([full_departures_df, file], ignore_index = True)
    
    except pd.errors.EmptyDataError:

        print(f"Skipping empty file {csv_file}")


full_departures_df.head(100)

full_departures_df.to_csv('/data/Full_Departures_Data.csv', index = False)



