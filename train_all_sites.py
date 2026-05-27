"""
Batch trainer for COS30019 Assignment 2B.

This script trains three models for every SCATS site:
1. MLP
2. LSTM
3. GRU

Expected project structure:

ASSIGNMENT2B-TUTOR-STYLE/
│
├── train_all_sites.py
├── train.py
├── main.py
├── routing.py
├── traveltime.py
│
├── data/
│   ├── data.py
│   ├── processed_scats_15min.csv
│   └── Scats Data October 2006.csv
│
├── model/
│   ├── model.py
│   ├── mlp_site_2200.keras
│   ├── lstm_site_2200.keras
│   └── gru_site_2200.keras
│
└── results/

Example commands:

Train all SCATS sites using all three models:
    python train_all_sites.py

Quick test on only a few sites:
    python train_all_sites.py --sites 2200 2825 3001 --epochs 5

Skip models that already exist:
    python train_all_sites.py --skip-existing

Train only selected model types:
    python train_all_sites.py --models mlp lstm
"""

import os
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from data.data import process_data
from model.model import get_mlp, get_lstm, get_gru


DATA_PATH = "data/processed_scats_15min.csv"
MODEL_DIR = "model"
RESULTS_DIR = "results"


def get_all_scats_sites(file_path=DATA_PATH):
    """
    Read the processed SCATS dataset and return every unique SCATS site number.
    """
    df = pd.read_csv(file_path)

    if "SCATS Number" not in df.columns:
        raise ValueError("The dataset must contain a 'SCATS Number' column.")

    sites = (
        df["SCATS Number"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    return sites


def build_model(model_name, lag):
    """
    Build one of the three required models.
    """
    model_name = model_name.lower()

    if model_name == "mlp":
        return get_mlp(lag)

    if model_name == "lstm":
        return get_lstm(lag)

    if model_name == "gru":
        return get_gru(lag)

    raise ValueError("model_name must be one of: mlp, lstm, gru")


def reshape_for_sequence_model(model_name, X_train, X_test):
    """
    LSTM and GRU need 3D input:
        samples, time_steps, features

    MLP uses 2D input:
        samples, lag
    """
    if model_name.lower() in ["lstm", "gru"]:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test


def train_one_model_for_one_site(
    model_name,
    site_id,
    lag=12,
    epochs=50,
    batch_size=32,
    skip_existing=False
):
    """
    Train one model type for one SCATS site and save the model + history.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = Path(MODEL_DIR) / f"{model_name}_site_{site_id}.keras"
    history_path = Path(RESULTS_DIR) / f"{model_name}_site_{site_id}_loss.csv"

    if skip_existing and model_path.exists():
        print(f"[SKIP] {model_name.upper()} site {site_id} already exists.")
        return {
            "site": site_id,
            "model": model_name,
            "status": "skipped_existing",
            "final_loss": None,
            "final_mae": None,
            "final_val_loss": None,
            "final_val_mae": None,
            "model_path": str(model_path),
            "history_path": str(history_path),
            "training_seconds": None,
            "error": ""
        }

    print("=" * 70)
    print(f"Training {model_name.upper()} model for SCATS site {site_id}")
    print("=" * 70)

    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path=DATA_PATH,
        site_id=site_id,
        lag=lag
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Not enough data for site {site_id} with lag={lag}")

    X_train, X_test = reshape_for_sequence_model(model_name, X_train, X_test)

    model = build_model(model_name, lag)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    training_seconds = round(time.time() - start_time, 2)

    model.save(model_path)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)

    final_loss = float(history.history["loss"][-1])
    final_mae = float(history.history["mae"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_val_mae = float(history.history["val_mae"][-1])

    print(f"[DONE] Saved model to: {model_path}")
    print(f"[DONE] Saved loss history to: {history_path}")
    print(f"[TIME] {training_seconds} seconds")

    return {
        "site": site_id,
        "model": model_name,
        "status": "trained",
        "final_loss": final_loss,
        "final_mae": final_mae,
        "final_val_loss": final_val_loss,
        "final_val_mae": final_val_mae,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "training_seconds": training_seconds,
        "error": ""
    }


def train_all_sites(
    sites=None,
    models=None,
    lag=12,
    epochs=50,
    batch_size=32,
    skip_existing=False
):
    """
    Train selected models for selected sites.
    If no sites are provided, train every SCATS site in the dataset.
    If no models are provided, train MLP, LSTM, and GRU.
    """
    if sites is None or len(sites) == 0:
        sites = get_all_scats_sites(DATA_PATH)

    if models is None or len(models) == 0:
        models = ["mlp", "lstm", "gru"]

    models = [m.lower() for m in models]

    allowed_models = {"mlp", "lstm", "gru"}
    invalid_models = [m for m in models if m not in allowed_models]

    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Use mlp, lstm, gru.")

    print("\nBatch training started")
    print("SCATS sites:", sites)
    print("Models:", models)
    print("Lag:", lag)
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Skip existing:", skip_existing)
    print()

    summary_rows = []

    for site_id in sites:
        for model_name in models:
            try:
                row = train_one_model_for_one_site(
                    model_name=model_name,
                    site_id=int(site_id),
                    lag=lag,
                    epochs=epochs,
                    batch_size=batch_size,
                    skip_existing=skip_existing
                )

            except Exception as e:
                print(f"[ERROR] Failed training {model_name.upper()} for site {site_id}")
                print(str(e))
                traceback.print_exc()

                row = {
                    "site": site_id,
                    "model": model_name,
                    "status": "failed",
                    "final_loss": None,
                    "final_mae": None,
                    "final_val_loss": None,
                    "final_val_mae": None,
                    "model_path": "",
                    "history_path": "",
                    "training_seconds": None,
                    "error": str(e)
                }

            summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(Path(RESULTS_DIR) / "batch_training_summary.csv", index=False)

    print("\nBatch training finished.")
    print("Summary saved to: results/batch_training_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sites",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of SCATS sites to train. If omitted, trains all sites."
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["mlp", "lstm", "gru"],
        help="Models to train: mlp lstm gru"
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=12,
        help="Number of previous 15-minute values used for prediction."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size."
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if the model file already exists."
    )

    args = parser.parse_args()

    train_all_sites(
        sites=args.sites,
        models=args.models,
        lag=args.lag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing
    )
