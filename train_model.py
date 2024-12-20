import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("ai4i2020.csv")

# Preprocess the dataset
data['Air temperature [K]'] -= 273.15  # Convert Kelvin to Celsius
data['Process temperature [K]'] -= 273.15  # Convert Kelvin to Celsius

# Features and Target
X = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = data['Machine failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'failure_prediction_model.pkl')
print("Model trained and saved as 'failure_prediction_model.pkl'")
