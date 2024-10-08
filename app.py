import streamlit as st
import pandas as pd
import pickle as pk
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_elements import elements, mui, html

# Load your model (replace with your actual model path)
model = pk.load(open('model.pkl', 'rb'))

# Apply Bootstrap for enhanced UI styling
st.markdown(
    """
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
    }
    .title {
        color: #ffffff;
        background-color: #007bff;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display enhanced title
st.markdown('<h1 class="title">Car Price Prediction ML Model</h1>', unsafe_allow_html=True)

# Custom form with Streamlit UI components
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Use AgGrid for enhanced table display of car data
st.markdown("### Car Details Table")
gb = GridOptionsBuilder.from_dataframe(cars_data)
grid_options = gb.build()
AgGrid(cars_data, gridOptions=grid_options)

# Car input form
name = st.selectbox('Select Car Brand', cars_data['name'].unique(), key='brand')
year = st.slider('Car Manufactured Year', 1994, 2024, key='year')
km_driven = st.slider('No of kms Driven', 11, 200000, key='km_driven')
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique(), key='fuel')
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique(), key='seller')
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique(), key='transmission')
owner = st.selectbox('Owner type', cars_data['owner'].unique(), key='owner')
mileage = st.slider('Car Mileage (in KM/L)', 10, 40, key='mileage')
engine = st.slider('Engine Capacity (CC)', 700, 5000, key='engine')
max_power = st.slider('Max Power (BHP)', 0, 200, key='power')
seats = st.slider('No of Seats', 5, 10, key='seats')

# Prediction logic
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'name': [name], 'year': [year], 'km_driven': [km_driven],
        'fuel': [fuel], 'seller_type': [seller_type],
        'transmission': [transmission], 'owner': [owner],
        'mileage': [mileage], 'engine': [engine],
        'max_power': [max_power], 'seats': [seats]
    })
    
    # You will need to handle the categorical encoding for the prediction model
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data['name'].replace(cars_data['name'].unique(), list(range(1, len(cars_data['name'].unique()) + 1)), inplace=True)

    # Predict the price using the model
    predicted_price = model.predict(input_data)
    st.success(f"The predicted car price is: RS {predicted_price[0]:,.2f}")

# Streamlit Elements for additional UI (Cards, Alerts, etc.)
with elements("demo"):
    mui.Card(
        mui.CardHeader(title="Prediction Summary"),
        mui.CardContent(
            html.div(f"Car Brand: {name}", style={"font-size": "1.2em", "color": "#007bff"}),
            html.div(f"Year: {year}"),
            html.div(f"Kilometers Driven: {km_driven}"),
            html.div(f"Fuel Type: {fuel}"),
            html.div(f"Transmission: {transmission}"),
            html.div(f"Owner Type: {owner}"),
            html.div(f"Engine: {engine} CC"),
            html.div(f"Max Power: {max_power} BHP"),
            html.div(f"Seats: {seats}"),
            style={"padding": "20px"}
        ),
        style={"width": "500px", "margin": "20px auto"}
    )
