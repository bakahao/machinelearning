import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# Read CSV file and preprocess the data
file_path = 'C:/Users/shenhao/GitHub/machinelearning/Datasets/Tesla Dataset.csv'
tesla_data = pd.read_csv(file_path)

# Check for missing values
print("Missing values in each column before removal:\n", tesla_data.isnull().sum())

#Remove rows with any missing values
tesla_data = tesla_data.dropna()

#Verify if missing values are removed
print("Missing values in each column after removal:\n", tesla_data.isnull().sum())

#Convert Date column to datetime format
tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])

# Extract useful features from 'Date'
tesla_data['Year'] = tesla_data['Date'].dt.year
tesla_data['Month'] = tesla_data['Date'].dt.month
tesla_data['Day'] = tesla_data['Date'].dt.day
tesla_data['DayOfWeek'] = tesla_data['Date'].dt.dayofweek

# Sort data by 'Date'
tesla_data = tesla_data.sort_values(by='Date')

# Drop the 'Date' column
tesla_data = tesla_data.drop(columns=['Date'])

# Split the data into X and y
# Features
X = tesla_data.drop(columns=['Close', 'Adj Close'])  
# Target
y = tesla_data['Adj Close'] 

# Scale the features using MinMaxScaler to normalize values between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed dataset to a new CSV file
preprocessed_file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 7/Machine Learning/Preprocessed_Tesla_Dataset.csv'
tesla_data.to_csv(preprocessed_file_path, index=False)

# Implement Linear Regression Model
# Fit Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and Evaluate Linear Regression
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
cv_scores_linear = cross_val_score(linear_model, X_scaled, y, cv=5, scoring='r2')

print("Linear Regression MSE:", mse_linear)
print("Linear Regression R² Score:", r2_linear)
print("Linear Regression MAE:", mae_linear)
print("Linear Regression RMSE:", rmse_linear)
print("")


print("Linear Regression Cross-Validation Scores:", cv_scores_linear)
print("Linear Regression Average CV Score:", cv_scores_linear.mean())
print("")

# Ridge Regression (Tuned Linear Regression)
# Define the parameter grid for Ridge regression
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Regularization strength

# Initialize Ridge regression
ridge_model = Ridge()

# Perform GridSearchCV
grid_search_ridge = GridSearchCV(estimator=ridge_model,
                                 param_grid=param_grid_ridge,
                                 cv=5,
                                 scoring='r2',
                                 n_jobs=-1)

# Fit the GridSearchCV
grid_search_ridge.fit(X_train, y_train)

# Best parameters and best Ridge estimator
best_params_ridge = grid_search_ridge.best_params_
best_ridge_model = grid_search_ridge.best_estimator_

# Predict using the best Ridge model
y_pred_ridge = best_ridge_model.predict(X_test)

# Evaluate Ridge regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)  # RMSE
cv_scores_ridge = cross_val_score(best_ridge_model, X_scaled, y, cv=5, scoring='r2')

print("Ridge Regression Best Parameters:", best_params_ridge)
print("Ridge Regression MSE:", mse_ridge)
print("Ridge Regression R² Score:", r2_ridge)
print("Ridge Regression MAE:", mae_ridge)
print("Ridge Regression RMSE:", rmse_ridge)
print("")

print("\nRidge Regression Cross-Validation Scores:", cv_scores_ridge)
print("Ridge Regression Average CV Score:", cv_scores_ridge.mean())
print("")

# Random Forest Model Implementation
# Fit Random Forest
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Predict and Evaluate Random Forest
y_pred_rf = random_forest_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)  
cv_scores_rf = cross_val_score(random_forest_model, X_scaled, y, cv=5, scoring='r2')

print("\nRandom Forest Regression MSE:", mse_rf)
print("Random Forest Regression R² Score:", r2_rf)
print("Random Forest Regression MAE:", mae_rf)
print("Random Forest Regression RMSE:", rmse_rf)
print("")


print("\nRandom Forest Cross-Validation Scores:", cv_scores_rf)
print("Random Forest Average CV Score:", cv_scores_rf.mean())
print("")

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 10, 20]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                              param_grid=param_grid_rf,
                              cv=3,
                              scoring='r2',
                              n_jobs=-1)

# Fit the tuned Random Forest model
grid_search_rf.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params_rf = grid_search_rf.best_params_
print("\nBest Parameters for Random Forest:", best_params_rf)

# Predict using the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)

# Evaluate Tuned Random Forest
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
rmse_best_rf = mean_squared_error(y_test, y_pred_best_rf, squared=False)

cv_scores_best_rf = cross_val_score(
    best_rf_model, 
    X_scaled, 
    y, 
    cv=5,  
    scoring='r2', 
    n_jobs=-1  
)

print("\nTuned Random Forest MSE:", mse_best_rf)
print("Tuned Random Forest R² Score:", r2_best_rf)
print("Tuned Random Forest MAE:", mae_best_rf)
print("Tuned Random Forest RMSE:", rmse_best_rf)
print("")



print("\nTuned Random Forest Cross-Validation Scores:", cv_scores_best_rf)
print("Tuned Random Forest Average CV Score:", cv_scores_best_rf.mean())


# Plot for Linear and Ridge Regression
plt.figure(figsize=(15, 7))

# Linear Regression
plt.subplot(1, 4, 1)
plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.5, label='Linear Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Linear Regression: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Ridge Regression
plt.subplot(1, 4, 2)
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.5, label='Ridge Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Ridge Regression: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Random Forest Regression
plt.subplot(1, 4, 3)
plt.scatter(y_test, y_pred_rf, color='purple', alpha=0.5, label='RF Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Random Forest: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Random Forest Regression
plt.subplot(1, 4, 4)
plt.scatter(y_test, y_pred_best_rf, color='black', alpha=0.5, label='RF Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Tuned Random Forest: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

plt.tight_layout()
plt.show()




