import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV
df = pd.read_csv("global_temp_co2_1960_2024.csv")

# Prepare data
X_simple = df[['CO2_ppm']]
X_multi = df[['Year', 'CO2_ppm']]
y = df['Temp_Anomaly_C']

# Future years
future_years = np.arange(2025, 2051)
future_co2 = np.linspace(423, 500, len(future_years))


# Model 1: Simple Linear Regression
'''
model = LinearRegression()
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model.fit(X_train_s, y_train_s)
predictions = model.predict(X_test_s)

print("Model: Simple Linear Regression")
print("R² Score:", r2_score(y_test_s, predictions))
print("MSE:", mean_squared_error(y_test_s, predictions))

# Predict future
future_X = pd.DataFrame({'CO2_ppm': future_co2})
future_preds = model.predict(future_X)

print("\n--- Future Predictions (Simple Linear Regression) ---")
for year, temp in zip(future_years, future_preds):
    print(f"{year}: Predicted Temp = {temp:.2f} °C")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X_simple, y, color='blue', label='Actual Data')
plt.plot(future_co2, future_preds, color='red', label='Prediction')
plt.title('Simple Linear Regression: CO₂ vs Temperature Anomaly')
plt.xlabel('CO2 (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# Model 2: Multiple Linear Regression


model = LinearRegression()
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model.fit(X_train_m, y_train_m)
predictions = model.predict(X_test_m)

print("Model: Multiple Linear Regression")
print("R² Score:", r2_score(y_test_m, predictions))
print("MSE:", mean_squared_error(y_test_m, predictions))

# Predict future
future_X = pd.DataFrame({'Year': future_years, 'CO2_ppm': future_co2})
future_preds = model.predict(future_X)

print("\n--- Future Predictions (Multiple Linear Regression) ---")
for year, temp in zip(future_years, future_preds):
    print(f"{year}: Predicted Temp = {temp:.2f} °C")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(df['Year'], y, color='blue', label='Actual Data')
plt.plot(future_years, future_preds, color='green', label='Prediction')
plt.title('Multiple Linear Regression: Year + CO₂ vs Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Model 3: Polynomial Regression

'''
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_simple)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_p, y_train_p)
predictions = model.predict(X_test_p)

print("Model: Polynomial Regression (Degree 2)")
print("R² Score:", r2_score(y_test_p, predictions))
print("MSE:", mean_squared_error(y_test_p, predictions))

# Predict future
future_X = pd.DataFrame({'CO2_ppm': future_co2})
future_X_poly = poly.transform(future_X)
future_preds = model.predict(future_X_poly)

print("\n--- Future Predictions (Polynomial Regression) ---")
for year, temp in zip(future_years, future_preds):
    print(f"{year}: Predicted Temp = {temp:.2f} °C")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X_simple, y, color='blue', label='Actual Data')
plt.plot(future_co2, future_preds, color='purple', label='Prediction')
plt.title('Polynomial Regression: CO₂ vs Temperature Anomaly (Degree 2)')
plt.xlabel('CO2 (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

