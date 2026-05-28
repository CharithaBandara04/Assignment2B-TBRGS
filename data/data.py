"""
This file loads the processed SCATS dataset and prepares
train/test data for LSTM, GRU, and MLP models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_processed_data(file_path="data/processed_scats_15min.csv"):
    """
    Load the processed SCATS time-series dataset.
    """

    df = pd.read_csv(file_path)

    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def create_sequences(data, lag=12):
    """
    Create sliding window sequences.

    Example:
    previous 12 traffic flow values -> next traffic flow value
    """

    X = []
    y = []

    for i in range(len(data) - lag):
        X.append(data[i:i + lag])
        y.append(data[i + lag])

    return np.array(X), np.array(y)


def process_data(file_path="data/processed_scats_15min.csv", site_id=2200, lag=12, test_ratio=0.2):
    """
    Prepare train/test data for ML models.

    Args:
        file_path: path to processed CSV
        site_id: SCATS site number
        lag: number of previous time steps used for prediction
        test_ratio: percentage of data used for testing

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """

    df = load_processed_data(file_path)

    site_data = df[df["SCATS Number"] == site_id].copy()

    if site_data.empty:
        raise ValueError(f"No data found for SCATS site {site_id}")

    site_data = site_data.sort_values("datetime")

    flow_values = site_data["Flow"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    flow_scaled = scaler.fit_transform(flow_values).flatten()

    X, y = create_sequences(flow_scaled, lag)

    split_index = int(len(X) * (1 - test_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path="data/processed_scats_15min.csv",
        site_id=2200,
        lag=12
    )

    print("Data module test successful")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)