
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the dataset
df = pd.read_csv("city_day.csv")

# Removing duplicate rows and records without AQI
df = df.drop_duplicates()
df = df.dropna(subset=['AQI'])

# Selecting input and output columns
X = df[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
        'NH3', 'CO', 'SO2', 'O3', 'Benzene',
        'Toluene', 'Xylene']].copy()

y = df['AQI']

# Separating numerical and categorical columns
num_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
            'NH3', 'CO', 'SO2', 'O3', 'Benzene',
            'Toluene', 'Xylene']

cat_cols = ['City']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Filling missing values and encoding city names
num_process = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_process = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_process, num_cols),
    ('cat', cat_process, cat_cols)
])

# Creating and training the Random Forest model
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regression', RandomForestRegressor(
        n_estimators=100, random_state=42
    ))
])

model.fit(X_train, y_train)

# Predicting AQI
y_pred = model.predict(X_test)

# Checking model performance
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2_score(y_test, y_pred))

# Comparing actual and predicted AQI values
result = pd.DataFrame({
    'Actual AQI': y_test.values,
    'Predicted AQI': y_pred
})

print(result.head(10))

# Plotting actual and predicted AQI
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest - Actual vs Predicted AQI")
plt.tight_layout()
plt.show()
