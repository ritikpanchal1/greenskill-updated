import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model, scaler, and encoder
model = joblib.load('Greenskill_AI_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Set up the Streamlit page
st.set_page_config(page_title="GreenSkill AI", page_icon="🍃", layout="wide")

# Custom CSS for a more attractive look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .st-title {
        color: #1a73e8;
    }
    .st-header {
        color: #1a73e8;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🍃 GreenSkill AI: Wind Turbine Power Prediction")
st.markdown("Predict the power output of a wind turbine with our advanced AI model. This tool is designed to be user-friendly, even for those without a technical background.")

# Sidebar for user inputs
st.sidebar.header("Input Features")

def user_input_features():
    date_time = st.sidebar.text_input("Date/Time (dd mm yyyy hh:mm)", "01 01 2018 00:00")
    wind_direction = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 260.0)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 5.3)
    theoretical_power = st.sidebar.slider("Theoretical Power Curve (KWh)", 0.0, 3600.0, 416.0)

    # Create a dictionary of the inputs
    data = {
        'Date/Time': [date_time],
        'Wind Direction (°)': [wind_direction],
        'Wind Speed (m/s)': [wind_speed],
        'Theoretical_Power_Curve (KWh)': [theoretical_power]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# Display user inputs
st.header("Your Inputs")
st.write(input_df)

# Prediction
if st.button("Predict Power Output"):
    try:
        # Preprocess the input data
        input_df['Date/Time'] = encoder.transform(input_df['Date/Time'])
        input_scaled = scaler.transform(input_df[['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)','Wind Direction (°)']])
        
        # Create a DataFrame with the scaled data and correct column names
        input_processed = pd.DataFrame(input_scaled, columns=['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)','Wind Direction (°)'])
        input_processed['Date/Time'] = input_df['Date/Time']

        # Make prediction
        prediction = model.predict(input_processed[['Date/Time','Wind Direction (°)','Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)']])

        # Display the prediction in a visually appealing way
        st.subheader("Predicted Power Output (kW)")
        st.success(f"{prediction[0]:.2f} kW")

        # Add a gauge chart for a more visual representation
        st.subheader("Power Output Visualization")
        # A simple bar chart to act as a gauge
        power_percentage = (prediction[0] / 3600) * 100  # Assuming max power is 3600 kW
        st.progress(int(power_percentage))

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please make sure the Date/Time format is correct (dd mm yyyy hh:mm) and that it exists in the original dataset.")

# Add some information about the model
st.header("About the AI Model")
st.info("""
This application uses a RandomForestRegressor model to predict wind turbine power output.
- *Features Used*: Date/Time, Wind Direction, Wind Speed, and Theoretical Power Curve.
- *Target Variable*: LV ActivePower (kW), which is the actual power generated.
- *Model Performance*: The model has an R2 score of approximately 0.97, which indicates a high level of accuracy.

""")

