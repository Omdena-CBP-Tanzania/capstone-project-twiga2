import streamlit as st
from pages import Data_Exploration, Model_Training, Prediction_Page
from data_utils import load_data

# Set the page configuration

st.set_page_config(
    page_title= "Climate Trend Predictor",
    page_icon=" ",
    layout = "wide" 
)

# Give the title and desc
st.title("Climate Trend Analysis and Prediction")
st.markdown("Analyze the historical temperature and rainfall data for Tanzania and predict future trends")

# Load the data
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ['Data Exploration', 'Model Training', 'Prediction'])

# Display the selected page
if page == "Data Exploration":
    Data_Exploration.show(df)

elif page == "Model Training":
    Model_Training.show(df)

else:
    Prediction_Page.show(df)