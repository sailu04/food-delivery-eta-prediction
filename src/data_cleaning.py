from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw Swiggy delivery dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("Raw dataset loaded.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary spaces from column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
    )

    return df


# ============================================================
# 3. CLEAN STRING VALUES
# ============================================================

def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean whitespace and convert string representations
    of missing values into real NaN values.
    """

    df = df.copy()

    string_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in string_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            {
                "NaN": pd.NA,
                "nan": pd.NA,
                "": pd.NA,
                "None": pd.NA
            }
        )

    return df


# ============================================================
# 4. CLEAN WEATHER
# ============================================================

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Weatherconditions values.

    Example:
        'conditions Sunny' -> 'Sunny'
        'conditions Rainy' -> 'Rainy'
    """

    df = df.copy()

    df["Weatherconditions"] = (
        df["Weatherconditions"]
        .str.replace(
            "conditions ",
            "",
            regex=False
        )
        .str.strip()
    )

    return df


# ============================================================
# 5. CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert columns that should contain numeric values.
    """

    df = df.copy()

    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# 6. CLEAN DELIVERY PERSON RATINGS
# ============================================================

def clean_ratings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean courier ratings.

    Valid ratings are between 1 and 5.
    Invalid values are converted to NaN.
    Missing values are filled with the median.
    """

    df = df.copy()

    df["Delivery_person_Ratings"] = (
        pd.to_numeric(
            df["Delivery_person_Ratings"],
            errors="coerce"
        )
    )

    # Ratings outside the valid 1-5 range
    # are considered invalid.
    df.loc[
        ~df["Delivery_person_Ratings"].between(1, 5),
        "Delivery_person_Ratings"
    ] = np.nan

    median_rating = (
        df["Delivery_person_Ratings"]
        .median()
    )

    df["Delivery_person_Ratings"] = (
        df["Delivery_person_Ratings"]
        .fillna(median_rating)
    )

    return df


# ============================================================
# 7. CLEAN TARGET
# ============================================================

def clean_target(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert '(min) 24' into numeric value 24.
    """

    df = df.copy()

    df["Time_taken(min)"] = (
        df["Time_taken(min)"]
        .astype("string")
        .str.replace(
            "(min)",
            "",
            regex=False
        )
        .str.strip()
    )

    df["Time_taken(min)"] = pd.to_numeric(
        df["Time_taken(min)"],
        errors="coerce"
    )

    return df


# ============================================================
# 8. CLEAN DATE
# ============================================================

def clean_date(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert Order_Date from string to datetime.
    """

    df = df.copy()

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    return df


# ============================================================
# 9. CLEAN TIME COLUMNS
# ============================================================

def clean_time_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert order and pickup times into datetime.time
    compatible strings.

    Invalid or missing values become NaN.
    """

    df = df.copy()

    for column in [
        "Time_Orderd",
        "Time_Order_picked"
    ]:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            {
                "NaN": pd.NA,
                "nan": pd.NA,
                "": pd.NA
            }
        )

    return df


# ============================================================
# 10. HANDLE CATEGORICAL MISSING VALUES
# ============================================================

def handle_categorical_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing categorical values with 'Unknown'
    instead of inventing a specific category.
    """

    df = df.copy()

    categorical_columns = [
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City"
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("string")
            .fillna("Unknown")
        )

    return df


# ============================================================
# 11. HANDLE NUMERIC MISSING VALUES
# ============================================================

def handle_numeric_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing numeric values using the median.
    """

    df = df.copy()

    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries"
    ]

    for column in numeric_columns:

        median_value = df[column].median()

        df[column] = (
            df[column]
            .fillna(median_value)
        )

    return df


# ============================================================
# 12. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove completely duplicated rows.
    """

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"Duplicates removed: {before - after}"
    )

    return df


# ============================================================
# 13. MAIN CLEANING PIPELINE
# ============================================================

def clean_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run the complete data-cleaning pipeline.
    """

    df = clean_column_names(df)

    df = clean_string_columns(df)

    df = clean_weather(df)

    df = convert_numeric_columns(df)

    df = clean_ratings(df)

    df = clean_target(df)

    df = clean_date(df)

    df = clean_time_columns(df)

    df = handle_categorical_missing_values(df)

    df = handle_numeric_missing_values(df)

    df = remove_duplicates(df)

    return df


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    input_path = (
        "data/raw/swiggy.csv"
    )

    output_path = (
        "data/processed/"
        "swiggy_cleaned.csv"
    )

    # Load raw dataset
    df = load_raw_data(input_path)

    print("\nStarting data cleaning...")

    # Clean dataset
    clean_df = clean_data(df)

    # Save cleaned dataset
    Path(
        "data/processed"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    clean_df.to_csv(
        output_path,
        index=False
    )

    print("\nCleaning completed!")

    print(
        f"Cleaned dataset shape: "
        f"{clean_df.shape}"
    )

    print(
        f"Saved to: {output_path}"
    )

    print("\nCleaned data types:")

    print(
        clean_df.dtypes
    )

    print("\nMissing values after cleaning:")

    print(
        clean_df.isnull().sum()
    )