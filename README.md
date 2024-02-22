# Introduction

When was the last time your flight was delayed? The aviation industry faces an ongoing challenge with flight delays, impacting millions of passengers annually. In 2022, nearly 199,000 flights were delayed across Canada, impacting approximately 29.85 million passengers, and leading to substantial economic and emotional consequences. On average, each affected passenger incurs costs of $530 CAD, a significant amount that can severely disrupt travel budgets. Despite the inconvenience caused, only 37% of passengers receive basic care from airlines during such disruptions. Moreover, 83% of passengers report not being adequately informed about their rights in these situations.

Current market solutions for flight delays include real-time tracking apps like FlightAware and Flighty, which start predictions after the inbound flight has departed and alert users to delays, as well as services like AirHelp that focus on post-event compensation. However, these approaches primarily react to delays after they have occurred or are imminent. We aim to explore the possibility of predicting delays as early as during the ticket purchasing process.

Our demo product, **FlyOnTime**, is set to revolutionize the industry by introducing a predictive analytics-based web application designed to forecast potential flight delays. It achieves this by analyzing historical departure data, weather conditions, and employing a supervised learning model. The demo has been developed using a dataset that includes history records from October 1, 2023, to January 31, 2024, focusing on flights departing from Montreal-Trudeau International Airport (YUL).

## Our Approaches

### 🛫 Data Acquisition
- **Flight Data Acquisition**: Utilized FlightLabs API to gather comprehensive flight data, including departure and arrival times, flight status, and aircraft information.
- **Weather Data Acquisition**: Employed OpenWeather API to fetch detailed weather conditions for Montreal, focusing on the YUL airport area to correlate weather patterns with flight data.
- **Airport Distances Calculation**: Calculated distances between airports using geographical coordinates, essential for analyzing flight durations and potential delays.
- **Data Integration**

### 🧹 EDA & Data Pre-processing
- Conducted comprehensive EDA utilizing visualization tools (Matplotlib, Seaborn), statistical analysis techniques, and automated profiling tools (YDataProfiling, Sweetviz, D-Tale) to uncover patterns, correlations, and outliers in the flight and weather data, providing insights into factors affecting flight schedules and weather conditions.
- Implemented preprocessing pipeline using pandas, numpy, and sklearn (including train_test_split, OrdinalEncoder, StandardScaler, KNNImputer, SimpleImputer, and Pipeline) to clean, impute missing values, encode categorical variables, and standardize features, ensuring the dataset is optimized for accurate analysis and modeling.

### ⚙️ Modelling
**Model Selection**: We experimented with a variety of machine learning models, including:
- **Linear Regression**: For establishing a baseline performance with a simple linear approach.
- **Random Forest Regressor**: Leveraging an ensemble method for more robust predictions and handling non-linear relationships.
- **Gradient Boosting Regressor**: To improve prediction accuracy through sequential learning from previous model errors.
- **Support Vector Regression (SVR)**: Applying a different learning strategy to capture complex patterns in the data.
- **Voting Regressor**: Combining predictions from multiple models to enhance overall performance.

**Model Evaluation**: Utilized cross_val_score for cross-validation, along with mean_squared_error and r2_score for assessing model accuracy and fit.

## Key Results

