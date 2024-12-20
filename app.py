import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('failure_prediction_model.pkl')

# Streamlit App Title
st.title("Predictive Maintenance - Device Failure Warning 🚨")

# Sidebar navigation
sidebar_options = st.sidebar.radio("Navigation", ["Home", "Device Status Prediction", "About"])

# Custom CSS for buttons and styling
st.markdown("""
    <style>
    /* Custom button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 18px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Make subheadings smaller */
    h3 {
        font-size: 20px;
        font-weight: normal;
        color: #333;
    }
    h4 {
        font-size: 18px;
        font-weight: normal;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Home Page
if sidebar_options == "Home":
    st.write("""
    # Welcome to Predictive Maintenance
    This app predicts if a device is at risk of failure based on sensor data.
    Use the navigation bar on the left to check device status.
    """)

# Device Status Prediction Page
elif sidebar_options == "Device Status Prediction":
    st.write("""### Input the Device Sensor Data
    Enter the sensor values below to check if the device is at risk of failure.
    """)

    # Input fields for user to enter sensor data
    air_temp = st.number_input("Air Temperature (in °C)", min_value=0.0, max_value=100.0, value=25.0)
    process_temp = st.number_input("Process Temperature (in °C)", min_value=0.0, max_value=200.0, value=60.0)
    rotational_speed = st.number_input("Rotational Speed (in RPM)", min_value=1000, max_value=3000, value=1500)
    torque = st.number_input("Torque (in Nm)", min_value=0.0, max_value=100.0, value=30.0)
    tool_wear = st.number_input("Tool Wear (in mins)", min_value=0, max_value=500, value=100)

    # Predict button
    if st.button("Check Device Status"):
        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })

        # Make a prediction
        prediction = model.predict(input_data)

        # Display results
        if prediction[0] == 1:  # Assuming 1 means failure risk
            st.error("⚠️ Warning: The device is at risk of failure!")
        else:
            st.success("✅ The device is operating normally.")

# About Page
elif sidebar_options == "About":
    st.write("""
    # About Predictive Maintenance
    This model uses sensor data to predict if a device is at risk of failure. 
    By analyzing sensor values such as air temperature, process temperature, 
    rotational speed, torque, and tool wear, the model helps in preventing 
    unexpected failures, thus saving time and cost on repairs.
    """)
