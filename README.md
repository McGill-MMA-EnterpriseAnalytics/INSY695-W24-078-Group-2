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

Our predictive models have undergone extensive training and testing, and the key results, including biases, are summarized below:

- **Linear Regression**: This baseline model achieved an RMSE of 0.7772, an R^2 Score of 0.0136, and a bias of 0.0427. The training-validation loss curve (as seen in LinearRegression_learning_curve.png) indicates that the model may benefit from more complex features or regularization techniques.
- **Random Forest Regressor**: It had an RMSE of 0.8592, an R^2 Score of -0.2054, and a bias of 0.0897. The RandomForestRegressor_learning_curve.png shows that the model might be overfitting, as indicated by the gap between training and validation loss.
- **Gradient Boosting Regressor**: This model yielded an RMSE of 0.7672, an R^2 Score of 0.0389, and a bias of 0.0418. The GradientBoostingRegressor_learning_curve.png suggests a good balance between bias and variance.
- **XGBRegressor**: With an RMSE of 0.8723, an R^2 Score of -0.2425, and a bias of 0.0555, the XGBRegressor_learning_curve.png indicates that the model struggles with this dataset, potentially due to hyperparameter settings.
- **SVR (Support Vector Regression)**: The SVR model had an RMSE of 0.7823, an R^2 Score close to zero, and a negative bias of -0.1184. The SVR_learning_curve.png reveals that the model is not capturing the complexity of the data well.
- **Ensemble**: Our ensemble approach, combining predictions from multiple models, achieved the best results with an RMSE of 0.7705, an R^2 Score of 0.0306, and a bias of 0.0223. The Ensemble_learning_curve.png shows a promising convergence of training and validation loss, suggesting a good generalization capability.

These results indicate the challenges in predicting flight delays accurately and point towards the need for further model tuning, feature engineering, and possibly the incorporation of additional data sources.
