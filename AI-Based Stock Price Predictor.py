# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
data = pd.read_csv('stock_data.csv')  # Replace with your file path
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering: Create day-based numerical feature for date
data['Day'] = np.arange(len(data))  # Adds a numerical index for date

# Select features and target variable
X = data[['Day']]  # Feature (independent variable)
y = data['Close Price']  # Target (dependent variable)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Make predictions
linear_predictions = linear_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)

# Evaluate models
linear_mae = mean_absolute_error(y_test, linear_predictions)
tree_mae = mean_absolute_error(y_test, tree_predictions)

linear_mse = mean_squared_error(y_test, linear_predictions)
tree_mse = mean_squared_error(y_test, tree_predictions)

print(f"Linear Regression MAE: {linear_mae}, MSE: {linear_mse}")
print(f"Decision Tree MAE: {tree_mae}, MSE: {tree_mse}")

# Plotting actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close Price'], label='Actual Price', color='blue')
plt.plot(X_test.index, linear_predictions, label='Linear Regression Predictions', color='orange')
plt.plot(X_test.index, tree_predictions, label='Decision Tree Predictions', color='green')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
