# from utils import *
from Lists import *
from math import sqrt


fl = load_data()
fl = preprocess_dates(fl)
fl = preprocess_weather(fl)
fl = unwanted_columns(fl)
X_processed, y, pipeline = encodings_imputers(fl)
X_train, X_test, y_train, y_test = split_data(X_processed, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    # 'XGBRegressor': xgb.XGBRegressor(random_state=42),
    # 'SVR': SVR(),
    # 'Ensemble': VotingRegressor(
    #     estimators=[
    #         ('lr', LinearRegression()),
    #         ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    #         ('gb', GradientBoostingRegressor(random_state=42)),
    #         ('xgb', xgb.XGBRegressor(random_state=42)),
    #         ('svm', SVR())
    #     ]
    # )
}

# Test and validate different models
results = {}



for name, model in models.items():
    mse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'RMSE': sqrt(mse), 'R2': r2}
    print(f"{name} - RMSE: {sqrt(mse):.2f}, R2: {r2:.2f}")

# pickle the best model
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_model.pkl')


