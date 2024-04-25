"""
This module contains unit tests to ensure weather_data.py
works correctly.
"""
import unittest
from unittest.mock import patch
import pandas as pd
from weather_data import concatenate_weather_data

class TestConcatenateWeatherData(unittest.TestCase):
    """unit testing class"""
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_concatenate_weather_data(self, mock_to_csv, mock_read_csv, mock_glob):
        """Actual test code"""
        # Mock the glob.glob to return a list of filenames
        mock_glob.return_value = ['/data/YUL_Weather_1.csv', '/data/YUL_Weather_2.csv']

        # Create some mock DataFrames to be returned by pd.read_csv
        df1 = pd.DataFrame({'Temperature': [20, 21], 'Humidity': [30, 35]})
        df2 = pd.DataFrame({'Temperature': [22, 23], 'Humidity': [36, 37]})
        mock_read_csv.side_effect = [df1, df2]  # Each call to read_csv will return one of these DataFrames

        # Execute the function
        concatenate_weather_data()

        # Check if pd.concat was called correctly, check the actual DataFrame content if needed
        self.assertTrue(mock_read_csv.call_count, 2)

        # Check if the final DataFrame was written to CSV
        mock_to_csv.assert_called_once_with('/data/Enterprise-II-YUL-Full-Weather.csv', index=False)


if __name__ == '__main__':

    unittest.main()
