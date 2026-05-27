"""
Evaluate all trained SCATS traffic prediction models.

This script evaluates MLP, LSTM, and GRU models for every SCATS site
and creates comparison CSV files for the Assignment 2B report.

Example commands:

Evaluate all sites and all models:
    python evaluate_all_sites.py

Evaluate selected sites only:
    python evaluate_all_sites.py --sites 2200 2825 3001

Evaluate selected model types only:
    python evaluate_all_sites.py --models mlp lstm
"""

import os
import math
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

from data.data import process_data


DATA_PATH = "data/processed_scats_15min.csv"
MODEL_DIR = "model"
RESULTS_DIR = "results"


def get_all_scats_sites(file_path=DATA_PATH):
    df = pd.read_csv(file_path)

    if "SCATS Number" not in df.columns:
        raise ValueError("The dataset must contain a 'SCATS Number' column.")

    return (
        df["SCATS Number"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero = y_true != 0

    if np.sum(non_zero) == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
    ) * 100


def evaluate_one_model(model_name, site_id, lag=12):
    model_name = model_name.lower()

    model_path = Path(MODEL_DIR) / f"{model_name}_site_{site_id}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path=DATA_PATH,
        site_id=site_id,
        lag=lag
    )

    if len(X_test) == 0:
        raise ValueError(f"No test data available for SCATS site {site_id}")

    if model_name in ["lstm", "gru"]:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = load_model(model_path)

    predictions = model.predict(X_test, verbose=0)

    predicted_flow = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()

    actual_flow = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(actual_flow, predicted_flow)
    mse = mean_squared_error(actual_flow, predicted_flow)
    rmse = math.sqrt(mse)
    mape = calculate_mape(actual_flow, predicted_flow)
    r2 = r2_score(actual_flow, predicted_flow)
    evs = explained_variance_score(actual_flow, predicted_flow)

    return {
        "site": int(site_id),
        "model": model_name.upper(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2,
        "Explained_Variance": evs,
        "test_samples": len(actual_flow),
        "model_path": str(model_path),
        "status": "evaluated",
        "error": ""
    }


def evaluate_all_sites(sites=None, models=None, lag=12):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if sites is None or len(sites) == 0:
        sites = get_all_scats_sites(DATA_PATH)

    if models is None or len(models) == 0:
        models = ["mlp", "lstm", "gru"]

    models = [m.lower() for m in models]

    allowed_models = {"mlp", "lstm", "gru"}
    invalid_models = [m for m in models if m not in allowed_models]

    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Use mlp, lstm, gru.")

    print("\nEvaluation started")
    print("SCATS sites:", sites)
    print("Models:", models)
    print("Lag:", lag)
    print()

    rows = []

    for site_id in sites:
        for model_name in models:
            print(f"Evaluating {model_name.upper()} for SCATS site {site_id}...")

            try:
                row = evaluate_one_model(
                    model_name=model_name,
                    site_id=int(site_id),
                    lag=lag
                )

                print(
                    f"  DONE | MAE={row['MAE']:.4f}, "
                    f"RMSE={row['RMSE']:.4f}, "
                    f"MAPE={row['MAPE_percent']:.2f}%, "
                    f"R2={row['R2']:.4f}"
                )

            except Exception as e:
                print(f"  ERROR | {str(e)}")
                traceback.print_exc()

                row = {
                    "site": int(site_id),
                    "model": model_name.upper(),
                    "MAE": None,
                    "MSE": None,
                    "RMSE": None,
                    "MAPE_percent": None,
                    "R2": None,
                    "Explained_Variance": None,
                    "test_samples": None,
                    "model_path": "",
                    "status": "failed",
                    "error": str(e)
                }

            rows.append(row)

            # Save progress after every model evaluation.
            pd.DataFrame(rows).to_csv(
                Path(RESULTS_DIR) / "model_comparison_all_sites.csv",
                index=False
            )

    results_df = pd.DataFrame(rows)
    successful_df = results_df[results_df["status"] == "evaluated"].copy()

    if not successful_df.empty:
        average_df = (
            successful_df
            .groupby("model", as_index=False)
            .agg(
                sites_evaluated=("site", "count"),
                average_MAE=("MAE", "mean"),
                average_MSE=("MSE", "mean"),
                average_RMSE=("RMSE", "mean"),
                average_MAPE_percent=("MAPE_percent", "mean"),
                average_R2=("R2", "mean"),
                average_Explained_Variance=("Explained_Variance", "mean")
            )
            .sort_values("average_RMSE")
        )

        average_df.to_csv(
            Path(RESULTS_DIR) / "model_average_comparison.csv",
            index=False
        )

        best_by_site_df = (
            successful_df
            .sort_values(["site", "RMSE"])
            .groupby("site", as_index=False)
            .first()
        )

        best_by_site_df.to_csv(
            Path(RESULTS_DIR) / "best_model_by_site.csv",
            index=False
        )

        print("\nAverage model comparison:")
        print(average_df.to_string(index=False))

        print("\nBest model count by SCATS site:")
        print(best_by_site_df["model"].value_counts().to_string())

    print("\nEvaluation finished.")
    print("Saved full comparison to: results/model_comparison_all_sites.csv")
    print("Saved average comparison to: results/model_average_comparison.csv")
    print("Saved best model by site to: results/best_model_by_site.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sites",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of SCATS sites to evaluate. If omitted, evaluates all sites."
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["mlp", "lstm", "gru"],
        help="Models to evaluate: mlp lstm gru"
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=12,
        help="Number of previous 15-minute values used for prediction."
    )

    args = parser.parse_args()

    evaluate_all_sites(
        sites=args.sites,
        models=args.models,
        lag=args.lag
    )
