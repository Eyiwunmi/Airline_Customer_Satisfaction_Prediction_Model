import streamlit as st
import pickle
import numpy as np

# Load your trained model


with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)



st.title("📊 Customer Satisfaction Prediction App")
st.write("Enter customer details below to predict satisfaction:")

# Example input fields - replace with your model's real features
Departure_Delay = st.number_input("Departure Delay(mins)", min_value=0, max_value=1128, value=60)
Leg_room_service= st.number_input("Leg room service", min_value=0, max_value=5, value=1)
Cleanliness = st.slider("Cleanliness", 0, 5, 2)
Seat_comfort = st.number_input("Seat comfort", min_value=0, max_value=5, value=1)

if st.button("Predict"):
    # Arrange inputs as your model expects
    features = np.array([[Departure_Delay, Leg_room_service, Cleanliness, Seat_comfort]])
    prediction = model.predict(features)

    # Example: mapping result to label
    labels = {0: "Not Satisfied", 1: "Satisfied"}
    st.success(f"Prediction: {labels.get(prediction[0], prediction[0])}")

