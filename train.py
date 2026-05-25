"""
Training script for Assignment 2B.

Usage:
python train.py --model mlp --site 2200
python train.py --model lstm --site 2200
python train.py --model gru --site 2200
"""

import os
import argparse
import numpy as np
import pandas as pd

from data.data import process_data
from model.model import get_mlp, get_lstm, get_gru


def train_model(model_name, site_id, lag=12):
    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path="data/processed_scats_15min.csv",
        site_id=site_id,
        lag=lag
    )

    if model_name == "mlp":
        model = get_mlp(lag)

    elif model_name == "lstm":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = get_lstm(lag)

    elif model_name == "gru":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = get_gru(lag)

    else:
        raise ValueError("Model must be one of: mlp, lstm, gru")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = f"model/{model_name}_site_{site_id}.keras"
    loss_path = f"results/{model_name}_site_{site_id}_loss.csv"

    model.save(model_path)

    loss_df = pd.DataFrame(history.history)
    loss_df.to_csv(loss_path, index=False)

    print(f"{model_name.upper()} model trained successfully")
    print("Model saved to:", model_path)
    print("Training history saved to:", loss_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Model to train: mlp, lstm, or gru"
    )

    parser.add_argument(
        "--site",
        type=int,
        default=2200,
        help="SCATS site number"
    )

    args = parser.parse_args()

    train_model(args.model.lower(), args.site)