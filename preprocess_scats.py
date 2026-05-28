"""
Preprocess the raw SCATS dataset from October 2006.
The raw dataset has 96 columns for traffic flow (V00 to V95), representing 15-minute intervals throughout the day.
This script reshapes the data into a long format with one row per timestamp and flow value,and saves the processed dataset as a new CSV file.

Example command:
    python preprocess_scats.py
    
"""

import pandas as pd
import os

# Load raw SCATS dataset
INPUT_FILE = "data/Scats Data October 2006.csv"
OUTPUT_FILE = "data/processed_scats_15min.csv"

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    header=1
)

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

print("Raw data loaded successfully")
print("Raw shape:", df.shape)

# Select traffic columns
# V00 to V95 = 96 readings per day, one every 15 minutes
volume_cols = [f"V{i:02d}" for i in range(96)]

print("Number of traffic interval columns:", len(volume_cols))

# Convert wide data to long format
df_long = df.melt(
    id_vars=[
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "Date"
    ],
    value_vars=volume_cols,
    var_name="Interval",
    value_name="Flow"
)

print("Data reshaped successfully")
print("Long shape:", df_long.shape)

# Convert V00-V95 into real time
df_long["Interval_Number"] = (
    df_long["Interval"]
    .str.replace("V", "", regex=False)
    .astype(int)
)

df_long["Minutes"] = df_long["Interval_Number"] * 15

# Create datetime column
df_long["Date"] = pd.to_datetime(
    df_long["Date"],
    dayfirst=True,
    errors="coerce"
)

df_long["datetime"] = df_long["Date"] + pd.to_timedelta(
    df_long["Minutes"],
    unit="m"
)


# Clean flow values
df_long["Flow"] = pd.to_numeric(df_long["Flow"], errors="coerce")

# Remove rows with invalid date/time or flow
df_long = df_long.dropna(subset=["datetime", "Flow"])

# Remove negative traffic values if any exist
df_long = df_long[df_long["Flow"] >= 0]


# Sort data by SCATS Number, Location, and datetime
df_long = df_long.sort_values(
    ["SCATS Number", "Location", "datetime"]
)

# Keep final useful columns
processed = df_long[
    [
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "datetime",
        "Flow"
    ]
]

# Save processed dataset
os.makedirs("data", exist_ok=True)
processed.to_csv(OUTPUT_FILE, index=False)

print("Processed data saved successfully")
print("Output file:", OUTPUT_FILE)
print("Processed shape:", processed.shape)
print(processed.head()) 