"""
Evaluation script for Assignment 2B.

Usage:
python main.py --model mlp --site 2200
python main.py --model lstm --site 2200
python main.py --model gru --site 2200
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

from data.data import process_data


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero = y_true != 0

    return np.mean(
        np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    ) * 100


def evaluate_model(model_name, site_id, lag=12):
    """
    Load trained model and evaluate predictions.
    """

    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path="data/processed_scats_15min.csv",
        site_id=site_id,
        lag=lag
    )

    model_path = f"model/{model_name}_site_{site_id}.keras"

    model = load_model(model_path)

    if model_name in ["lstm", "gru"]:
        X_test = np.reshape(
            X_test,
            (X_test.shape[0], X_test.shape[1], 1)
        )

    predictions = model.predict(X_test)

    pred_flow = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()

    actual_flow = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(actual_flow, pred_flow)
    mse = mean_squared_error(actual_flow, pred_flow)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual_flow, pred_flow)
    evs = explained_variance_score(actual_flow, pred_flow)
    mape = calculate_mape(actual_flow, pred_flow)

    print(f"\n{model_name.upper()} Evaluation Results")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2 Score: {r2:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    results_path = f"results/{model_name}_site_{site_id}_metrics.txt"

    with open(results_path, "w") as f:
        f.write(f"{model_name.upper()} Evaluation Results\n")
        f.write(f"SCATS Site: {site_id}\n\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAPE: {mape:.4f}%\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"Explained Variance Score: {evs:.4f}\n")

    plt.figure(figsize=(12, 5))

    plt.plot(actual_flow[:288], label="Actual Flow")
    plt.plot(pred_flow[:288], label=f"{model_name.upper()} Prediction")

    plt.title(
        f"{model_name.upper()} Traffic Prediction - SCATS {site_id}"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    image_path = f"images/{model_name}_site_{site_id}.png"

    plt.savefig(image_path, dpi=300)

    print("\nResults saved to:", results_path)
    print("Prediction graph saved to:", image_path)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Model to evaluate: mlp, lstm, gru"
    )

    parser.add_argument(
        "--site",
        type=int,
        default=2200,
        help="SCATS site number"
    )

    args = parser.parse_args()

    evaluate_model(args.model.lower(), args.site)