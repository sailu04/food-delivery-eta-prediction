import numpy as np
import pandas as pd
def clean_coordinates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Correct malformed geographic coordinates.

    This dataset represents Indian food deliveries.
    Negative latitude/longitude values are treated as
    sign errors and converted to their absolute values.
    """

    df = df.copy()

    coordinate_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    for column in coordinate_columns:
        df[column] = df[column].abs()

    return df

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate the great-circle distance between
    two geographic coordinates.

    Returns distance in kilometers.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius_km * c
def create_time_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create useful time-based features from
    order date and order time.
    """

    df = df.copy()

    # Convert order date
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    # Convert order time into datetime
    order_time = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M:%S",
        errors="coerce"
    )

    # Extract hour
    df["order_hour"] = order_time.dt.hour

    # Extract day of week
    df["day_of_week"] = (
        df["Order_Date"].dt.dayofweek
    )

    # Extract month
    df["month"] = (
        df["Order_Date"].dt.month
    )

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df
def create_pickup_delay_feature(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the number of minutes between
    order placement and order pickup.

    Handles orders where pickup happens after midnight.
    """

    df = df.copy()

    order_time = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M:%S",
        errors="coerce"
    )

    pickup_time = pd.to_datetime(
        df["Time_Order_picked"],
        format="%H:%M:%S",
        errors="coerce"
    )

    # Convert order time into minutes after midnight
    order_minutes = (
        order_time.dt.hour * 60
        + order_time.dt.minute
    )

    # Convert pickup time into minutes after midnight
    pickup_minutes = (
        pickup_time.dt.hour * 60
        + pickup_time.dt.minute
    )

    # Calculate initial delay
    pickup_delay = (
        pickup_minutes - order_minutes
    )

    # If pickup time is earlier than order time,
    # assume pickup happened after midnight.
    pickup_delay = pickup_delay.where(
        pickup_delay >= 0,
        pickup_delay + 1440
    )

    df["pickup_delay_minutes"] = pickup_delay

    return df
def create_peak_hour_feature(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a binary feature indicating whether
    the order was placed during a peak period.

    Peak periods:
        Lunch  : 12:00 - 14:00
        Dinner : 19:00 - 22:00
    """

    df = df.copy()

    df["is_peak_hour"] = (
        df["order_hour"].between(12, 14)
        |
        df["order_hour"].between(19, 22)
    ).astype(int)

    return df
if __name__ == "__main__":

    clean_df = pd.read_csv(
        "data/processed/swiggy_cleaned.csv"
    )

    # Distance
    clean_df = clean_coordinates(clean_df)

    clean_df["Distance_km"] = haversine_distance(
    clean_df["Restaurant_latitude"],
    clean_df["Restaurant_longitude"],
    clean_df["Delivery_location_latitude"],
    clean_df["Delivery_location_longitude"]
)

    # Time features
    clean_df = create_time_features(
        clean_df
    )

    # Pickup delay
    clean_df = create_pickup_delay_feature(
        clean_df
    )
    clean_df = create_peak_hour_feature(
    clean_df
)

    print(
        clean_df[
            [
                "Time_Orderd",
                "Time_Order_picked",
                "order_hour",
                "pickup_delay_minutes",
                "Distance_km"
            ]
        ].head(10)
    )