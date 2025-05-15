import streamlit as st
from joblib import load


# Load the saved model
model = load('random_forest_model.joblib')


# Create the Streamlit interface
st.title('Climate Change Prediction for Tanzania')

# Add input components based on your model's features
#feature1 = st.number_input('Feature 1', value=0.0)
#feature2 = st.number_input('Feature 2', value=0.0)

user_input = st.text_input("What's your name?")
if user_input:
    st.write(f"Hello, {user_input}!")

# Make prediction when button is clicked
#if st.button('Predict'):
#    input_data = [[feature1, feature2]]  # format according to your model's requirements
#    prediction = model.predict(input_data)
#    st.success(f'Predicted Value: {prediction[0]:.2f}')