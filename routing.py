"""
Routing module for Assignment 2B.

This file:
1. Builds a simple SCATS graph using nearest neighbours
2. Loads a trained ML model
3. Predicts traffic flow
4. Converts predicted flow into travel time
5. Finds top-k routes using travel time as edge cost
"""

import heapq
import argparse
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from data.data import process_data
from traveltime import calculate_distance_km, calculate_travel_time


def load_scats_nodes(file_path="data/processed_scats_15min.csv"):
    """
    Load unique SCATS sites and their coordinates.
    """

    df = pd.read_csv(file_path)

    nodes = (
        df[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
        .groupby("SCATS Number", as_index=False)
        .mean()
    )

    return nodes


def build_graph(nodes, max_neighbors=8):
    """
    Build a simple graph by connecting each SCATS site
    to its nearest neighbouring SCATS sites.
    """

    graph = {}

    for _, row_a in nodes.iterrows():
        scats_a = int(row_a["SCATS Number"])
        graph[scats_a] = []

        distances = []

        for _, row_b in nodes.iterrows():
            scats_b = int(row_b["SCATS Number"])

            if scats_a == scats_b:
                continue

            distance = calculate_distance_km(
                row_a["NB_LATITUDE"],
                row_a["NB_LONGITUDE"],
                row_b["NB_LATITUDE"],
                row_b["NB_LONGITUDE"]
            )

            distances.append((distance, scats_b))

        distances.sort(key=lambda x: x[0])

        for distance, scats_b in distances[:max_neighbors]:
            graph[scats_a].append((scats_b, distance))

    return graph


def predict_next_flow(model_name, site_id, lag=12):
    """
    Load trained ML model and predict next traffic flow
    for the selected SCATS site.
    """

    X_train, y_train, X_test, y_test, scaler = process_data(
        file_path="data/processed_scats_15min.csv",
        site_id=site_id,
        lag=lag
    )

    model_path = f"model/{model_name}_site_{site_id}.keras"
    model = load_model(model_path)

    latest_sequence = X_test[-1].reshape(1, lag)

    if model_name in ["lstm", "gru"]:
        latest_sequence = latest_sequence.reshape(1, lag, 1)

    predicted_scaled = model.predict(latest_sequence, verbose=0)

    predicted_flow = scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    )[0][0]

    return predicted_flow


def find_top_k_routes(graph, origin, destination, predicted_flow, k=5):
    """
    Find top-k routes using travel time as the edge cost.
    """

    routes = []
    seen_paths = set()

    queue = [(0, origin, [origin])]

    while queue and len(routes) < k:
        current_time, current_node, path = heapq.heappop(queue)

        if current_node == destination:
            path_tuple = tuple(path)

            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                routes.append((current_time, path))

            continue

        for neighbor, distance_km in graph.get(current_node, []):
            if neighbor not in path:
                edge_time = calculate_travel_time(
                    distance_km,
                    predicted_flow
                )

                new_time = current_time + edge_time

                heapq.heappush(
                    queue,
                    (new_time, neighbor, path + [neighbor])
                )

    return routes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Model to use: mlp, lstm, or gru"
    )

    parser.add_argument(
        "--site",
        type=int,
        default=2200,
        help="SCATS site used for traffic flow prediction"
    )

    parser.add_argument(
        "--origin",
        type=int,
        default=2200,
        help="Origin SCATS site"
    )

    parser.add_argument(
        "--destination",
        type=int,
        default=2825,
        help="Destination SCATS site"
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbours connected to each SCATS site"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of routes to return"
    )

    args = parser.parse_args()

    model_name = args.model.lower()

    nodes = load_scats_nodes()
    graph = build_graph(nodes, max_neighbors=args.neighbors)

    predicted_flow = predict_next_flow(
        model_name=model_name,
        site_id=args.site
    )

    routes = find_top_k_routes(
        graph=graph,
        origin=args.origin,
        destination=args.destination,
        predicted_flow=predicted_flow,
        k=args.k
    )

    print(f"Model used: {model_name.upper()}")
    print(f"Prediction SCATS site: {args.site}")
    print(f"Predicted traffic flow: {round(predicted_flow, 2)}")
    print(f"Top {args.k} routes from {args.origin} to {args.destination}")
    print()

    if not routes:
        print("No route found. Try increasing --neighbors.")
    else:
        for i, (time, path) in enumerate(routes, start=1):
            print(f"Route {i}:")
            print("Path:", path)
            print("Estimated travel time:", round(time, 2), "minutes")
            print()