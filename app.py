import streamlit as st
import requests


st.set_page_config(
    page_title="Swiggy ETA Predictor",
    page_icon="🍔"
)

st.title("🍔 Food Delivery ETA Predictor")
st.write("Predict estimated food delivery time using XGBoost.")

st.divider()

distance = st.number_input(
    "Distance (km)",
    min_value=0.1,
    max_value=50.0,
    value=5.0
)

order_hour = st.slider(
    "Order Hour",
    0,
    23,
    19
)

day_of_week = st.selectbox(
    "Day of Week",
    [0, 1, 2, 3, 4, 5, 6]
)

month = st.selectbox(
    "Month",
    [2, 3, 4],
    index=1
)

is_weekend = st.selectbox(
    "Weekend?",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

is_peak_hour = st.selectbox(
    "Peak Hour?",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

age = st.number_input(
    "Delivery Person Age",
    18,
    60,
    28
)

rating = st.number_input(
    "Delivery Person Rating",
    1.0,
    5.0,
    4.8,
    step=0.1
)

vehicle_condition = st.selectbox(
    "Vehicle Condition",
    [0, 1, 2, 3, 4, 5],
    index=2
)

multiple_deliveries = st.selectbox(
    "Multiple Deliveries",
    [0, 1, 2, 3]
)

weather = st.selectbox(
    "Weather",
    [
        "Sunny",
        "Cloudy",
        "Fog",
        "Stormy",
        "Sandstorms",
        "Windy"
    ]
)

traffic = st.selectbox(
    "Road Traffic",
    [
        "Low",
        "Medium",
        "High",
        "Jam"
    ]
)

order_type = st.selectbox(
    "Order Type",
    [
        "Meal",
        "Snack",
        "Drinks",
        "Buffet"
    ]
)

vehicle_type = st.selectbox(
    "Vehicle Type",
    [
        "motorcycle",
        "scooter",
        "electric_scooter"
    ]
)

festival = st.selectbox(
    "Festival",
    ["No", "Yes"]
)

city = st.selectbox(
    "City",
    [
        "Metropolitian",
        "Urban",
        "Semi-Urban"
    ]
)

st.divider()


if st.button("🚀 Predict Delivery Time", use_container_width=True):

    payload = {
        "Distance_km": distance,
        "order_hour": order_hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city
    }

    try:
        response = requests.post(
            "https://food-delivery-eta-prediction.onrender.com/predict",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "predicted_delivery_time_minutes"
            ]

            st.success(
                f"🍔 Estimated Delivery Time: {prediction} minutes"
            )

        else:
            st.error(
                f"API Error: {response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI Docker container."
        )
