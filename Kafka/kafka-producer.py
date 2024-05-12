#!/usr/bin/env python3

from random import sample
import sys
import json
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
from confluent_kafka import Producer
import requests

if __name__ == '__main__':
    # Parse the command line.
    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    args = parser.parse_args()

    # Parse the configuration.
    # See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    config = dict(config_parser['default'])

    # Create Producer instance
    producer = Producer(config)

    # Optional per-message delivery callback (triggered by poll() or flush())
    # when a message has been successfully delivered or permanently
    # failed delivery (after retries).
    def delivery_callback(err, msg):
        if err:
            print('ERROR: Message failed delivery: {}'.format(err))
        else:
            print("Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))

    # 1. Setting up API Base URL + Credentials

    base_url: str = "https://api.aviationstack.com/v1/flights"

    api_key: str = "**********************"

    # 2. Testing API Response - Historical Flights Endpoint

    query_params: dict = {
            "access_key" : api_key,
            "flight_date": "2024-03-01",
            "offset": 100,
            "limit": 100,
            "dep_iata": "YUL"
    }

    # 3. Initializing kafka topic

    topic: str = "departures-data-topic"

    sample_response = requests.get(base_url, params = query_params, timeout = 70)

    print(f"API code: {sample_response.status_code}")

    if sample_response.status_code == 200:

        response_json = sample_response.json()

        required_data = response_json.get('data', [])

        for flight in required_data:

            if isinstance(flight, dict):

                if flight['flight'].get('codeshared') is None:

                    departure_record = {
                        'flight_status': flight['flight_status'],
                        'departure_airport': flight['departure'].get('iata', ''),
                        'departure_gate': flight['departure'].get('gate', ''),
                        'arrival_airport': flight['arrival'].get('iata', ''),
                        'scheduled_departure': flight['departure'].get('scheduled', ''),
                        'actual_departure': flight['departure'].get('actual', ''),
                        'delay_minutes': flight['departure'].get('delay', 0),
                        'airline_name': flight['airline'].get('name', ''),
                        'flight_number': flight['flight'].get('iata', '')
                    }

                    departure_record_json = json.dumps(departure_record)

                    producer.produce(topic, key = str(flight['flight'].get('iata', '')), value = departure_record_json, on_delivery = delivery_callback)


    # Block until the messages are sent.
    producer.poll(10000)
    producer.flush()
