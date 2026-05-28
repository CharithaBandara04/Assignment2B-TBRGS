"""
Traffic-Based Route Guidance System (TBRGS)
COS30019 Introduction to AI - Assignment 2B

This file is the main controller for the route guidance system.

It connects together:
    1. data/processed_scats_15min.csv      -> processed SCATS traffic data
    2. data/scats_connections.csv          -> manually created road graph
    3. model/<model>_site_<site>.keras     -> trained ML models for each SCATS site
    4. traveltime.py                       -> travel-time conversion functions
    5. Uniform-cost search                 -> returns top-k lowest-time routes

Command examples:
    python tbrgs.py --origin 2000 --destination 3002 --datetime "2006-10-15 08:00" --model mlp

    python tbrgs.py --origin 2000 --destination 3002 --datetime "2006-10-15 08:00" --model lstm --k 5

    python tbrgs.py --origin 970 --destination 3685 --datetime "2006-10-20 17:30" --model gru


IMPORTANT GRAPH RULE:
---------------------
The program only allows travel between SCATS sites listed in:

    data/scats_connections.csv
This file is a manually created list of valid road connections between SCATS sites.
The TBRGS program will only consider routes that follow these connections.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import heapq
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from search_algorithms import top_k_uniform_cost_search

from traveltime import (
    calculate_distance_km,
    calculate_edge_travel_time_from_15min_flow,
    convert_15min_flow_to_hourly,
    flow_to_speed
)


# ---------------------------------------------------------------------------
# File paths and constants
# ---------------------------------------------------------------------------

DATA_PATH = "data/processed_scats_15min.csv"
CONNECTIONS_PATH = "data/scats_connections.csv"
MODEL_DIR = "model"
RESULTS_DIR = "results"

DEFAULT_LAG = 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scats_data(file_path=DATA_PATH):
    """
    Load the processed SCATS traffic dataset.

    Expected columns:
        SCATS Number, Location, NB_LATITUDE, NB_LONGITUDE, datetime, Flow
    """

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Processed SCATS data file not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "datetime",
        "Flow"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in processed SCATS data: {missing_columns}")

    df["SCATS Number"] = df["SCATS Number"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["Flow"] = pd.to_numeric(df["Flow"], errors="coerce")

    df = df.dropna(subset=["datetime", "Flow", "NB_LATITUDE", "NB_LONGITUDE"])

    return df


def load_scats_nodes(df):
    """
    Create one node per SCATS site.

    Returns a dataframe with:
        SCATS Number, Location, NB_LATITUDE, NB_LONGITUDE
    """

    nodes = (
        df[["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]]
        .groupby("SCATS Number", as_index=False)
        .agg({
            "Location": "first",
            "NB_LATITUDE": "mean",
            "NB_LONGITUDE": "mean"
        })
    )

    nodes["SCATS Number"] = nodes["SCATS Number"].astype(int)

    return nodes


def load_connections(file_path=CONNECTIONS_PATH):
    """
    Load the manually created SCATS road connection CSV.

    Required columns:
        from_site,to_site

    Optional columns are allowed, but ignored by the route algorithm.
    """

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"SCATS connections file not found: {file_path}\n"
            "Create data/scats_connections.csv with columns: from_site,to_site"
        )

    connections = pd.read_csv(file_path)

    required_columns = ["from_site", "to_site"]
    missing_columns = [col for col in required_columns if col not in connections.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in SCATS connections file: {missing_columns}\n"
            "The file must contain: from_site,to_site"
        )

    connections = connections.dropna(subset=["from_site", "to_site"])
    connections["from_site"] = connections["from_site"].astype(int)
    connections["to_site"] = connections["to_site"].astype(int)

    # Remove accidental duplicate rows.
    connections = connections.drop_duplicates(subset=["from_site", "to_site"])

    return connections


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph_from_connections(connections, nodes):
    """
    Build a graph using only the manual road links in scats_connections.csv.

    The CSV controls which edges exist.
    Distance is calculated automatically from the site coordinates.

    Graph format:
        {
            2000: [
                {"to": 2200, "distance_km": 1.25},
                {"to": 3126, "distance_km": 2.10}
            ],
            ...
        }
    """

    node_map = {}

    for _, row in nodes.iterrows():
        site = int(row["SCATS Number"])

        node_map[site] = {
            "location": row["Location"],
            "lat": float(row["NB_LATITUDE"]),
            "lon": float(row["NB_LONGITUDE"])
        }

    graph = {}

    skipped_edges = []

    for _, row in connections.iterrows():
        from_site = int(row["from_site"])
        to_site = int(row["to_site"])

        if from_site not in node_map or to_site not in node_map:
            skipped_edges.append((from_site, to_site))
            continue

        distance_km = calculate_distance_km(
            node_map[from_site]["lat"],
            node_map[from_site]["lon"],
            node_map[to_site]["lat"],
            node_map[to_site]["lon"]
        )

        if from_site not in graph:
            graph[from_site] = []

        graph[from_site].append({
            "to": to_site,
            "distance_km": distance_km
        })

    if skipped_edges:
        print("Warning: Some edges were skipped because their SCATS sites were not found in the dataset:")
        for edge in skipped_edges:
            print(f"  {edge[0]} -> {edge[1]}")

    return graph, node_map


def validate_site(site_id, node_map, graph=None, role="site"):
    """
    Validate that a site exists in the processed data.
    Optionally check if it has outgoing graph connections.
    """

    site_id = int(site_id)

    if site_id not in node_map:
        available = sorted(node_map.keys())
        raise ValueError(
            f"{role.capitalize()} SCATS site {site_id} does not exist in the processed dataset.\n"
            f"Available sites include: {available[:15]} ..."
        )

    if graph is not None and site_id not in graph:
        print(
            f"Warning: {role.capitalize()} SCATS site {site_id} exists in the dataset, "
            "but it has no outgoing road connections in scats_connections.csv."
        )


# ---------------------------------------------------------------------------
# ML prediction helper functions
# ---------------------------------------------------------------------------

def get_previous_flow_sequence(df, site_id, requested_datetime, lag=DEFAULT_LAG):
    """
    Get the previous lag traffic-flow values before the requested time.

    Example:
        If requested time is 08:00 and lag=12,
        the model uses the previous 12 x 15-minute values.

    This function is used to create the model input sequence.
    """

    site_id = int(site_id)

    site_data = df[df["SCATS Number"] == site_id].copy()

    if site_data.empty:
        raise ValueError(f"No traffic data found for SCATS site {site_id}")

    site_data = site_data.sort_values("datetime")

    previous_data = site_data[site_data["datetime"] < requested_datetime].copy()

    if len(previous_data) < lag:
        raise ValueError(
            f"Not enough previous data for SCATS site {site_id} before {requested_datetime}. "
            f"Need at least {lag} previous values."
        )

    sequence = previous_data.tail(lag)["Flow"].values.astype(float)

    return site_data, sequence


def minmax_scale_sequence(site_data, sequence):
    """
    Apply the same simple MinMax scaling idea used during training.

    In data.py, each SCATS site is scaled using MinMaxScaler based on that site's
    full flow history. Here we reproduce that scaling manually:

        scaled = (value - min) / (max - min)

    The model output is then inverse-scaled back to real flow:

        real_value = scaled_value * (max - min) + min
    """

    site_min = float(site_data["Flow"].min())
    site_max = float(site_data["Flow"].max())

    if site_max == site_min:
        scaled_sequence = np.zeros_like(sequence, dtype=float)
    else:
        scaled_sequence = (sequence - site_min) / (site_max - site_min)

    return scaled_sequence, site_min, site_max


def inverse_minmax_scale(scaled_value, site_min, site_max):
    """
    Convert one scaled prediction back into real traffic-flow units.
    """

    return float(scaled_value * (site_max - site_min) + site_min)


def predict_15min_flow(model_name, site_id, requested_datetime, df, lag=DEFAULT_LAG):
    """
    Predict 15-minute traffic flow for a SCATS site at the requested time.

    The model for the selected site is loaded from:
        model/<model_name>_site_<site_id>.keras

    Example:
        model/mlp_site_2200.keras
        model/lstm_site_2200.keras
        model/gru_site_2200.keras
    """

    model_name = model_name.lower()
    site_id = int(site_id)

    model_path = Path(MODEL_DIR) / f"{model_name}_site_{site_id}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    site_data, sequence = get_previous_flow_sequence(
        df=df,
        site_id=site_id,
        requested_datetime=requested_datetime,
        lag=lag
    )

    scaled_sequence, site_min, site_max = minmax_scale_sequence(site_data, sequence)

    X = scaled_sequence.reshape(1, lag)

    if model_name in ["lstm", "gru"]:
        X = X.reshape(1, lag, 1)

    model = load_model(model_path)

    predicted_scaled = model.predict(X, verbose=0).reshape(-1)[0]

    predicted_15min_flow = inverse_minmax_scale(
        scaled_value=predicted_scaled,
        site_min=site_min,
        site_max=site_max
    )

    # Traffic flow should not be negative.
    predicted_15min_flow = max(0.0, predicted_15min_flow)

    return predicted_15min_flow


# ---------------------------------------------------------------------------
# Edge cost calculation
# ---------------------------------------------------------------------------

def calculate_edge_cost(
    from_site,
    to_site,
    distance_km,
    model_name,
    requested_datetime,
    df,
    prediction_cache,
    lag=DEFAULT_LAG
):
    """
    Calculate the travel-time cost for one valid edge.

    For edge:
        from_site -> to_site

    The assignment says travel time from A to B can be approximated using:
        - distance between A and B
        - accumulated traffic volume at SCATS site B
        - 30-second average intersection delay

    Therefore:
        edge cost uses the predicted traffic flow at the destination site.
    """

    to_site = int(to_site)

    # Cache predictions so the same site is not repeatedly predicted many times.
    if to_site not in prediction_cache:
        predicted_15min_flow = predict_15min_flow(
            model_name=model_name,
            site_id=to_site,
            requested_datetime=requested_datetime,
            df=df,
            lag=lag
        )

        predicted_hourly_flow = convert_15min_flow_to_hourly(predicted_15min_flow)

        estimated_speed = flow_to_speed(predicted_hourly_flow)

        prediction_cache[to_site] = {
            "predicted_15min_flow": predicted_15min_flow,
            "predicted_hourly_flow": predicted_hourly_flow,
            "estimated_speed_kmh": estimated_speed
        }

    edge_time_minutes = calculate_edge_travel_time_from_15min_flow(
        distance_km=distance_km,
        predicted_15min_flow=prediction_cache[to_site]["predicted_15min_flow"]
    )

    return edge_time_minutes, prediction_cache[to_site]


# ---------------------------------------------------------------------------
# Top-k route search
# ---------------------------------------------------------------------------

def find_top_k_routes(
    graph,
    origin,
    destination,
    model_name,
    requested_datetime,
    df,
    k=5,
    lag=DEFAULT_LAG,
    max_expansions=50000
):
    """
    Find up to k lowest-time routes.

    This function connects Assignment 2A to Assignment 2B.

    Assignment 2A:
        CUS1 used Uniform Cost Search to minimise path cost.

    Assignment 2B:
        The same Uniform Cost Search idea is reused, but the edge cost is now
        calculated dynamically as predicted travel time.

    The actual search algorithm is imported from:
        search_algorithms.py
    """

    prediction_cache = {}

    def edge_cost_function(current_site, next_site, edge):
        """
        Calculate edge travel time for the search algorithm.

        For an edge A -> B:
            - B is next_site
            - the trained model for B predicts traffic flow at B
            - traveltime.py converts that flow and distance into edge time
        """

        distance_km = float(edge["distance_km"])

        edge_time, prediction_info = calculate_edge_cost(
            from_site=current_site,
            to_site=next_site,
            distance_km=distance_km,
            model_name=model_name,
            requested_datetime=requested_datetime,
            df=df,
            prediction_cache=prediction_cache,
            lag=lag
        )

        edge_detail = {
            "from_site": int(current_site),
            "to_site": int(next_site),
            "distance_km": distance_km,
            "predicted_15min_flow_at_to_site": prediction_info["predicted_15min_flow"],
            "predicted_hourly_flow_at_to_site": prediction_info["predicted_hourly_flow"],
            "estimated_speed_kmh": prediction_info["estimated_speed_kmh"],
            "edge_time_minutes": edge_time
        }

        return edge_time, edge_detail

    routes = top_k_uniform_cost_search(
        graph=graph,
        origin=origin,
        destination=destination,
        edge_cost_function=edge_cost_function,
        k=k,
        max_expansions=max_expansions
    )

    return routes


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------

def print_routes(routes, node_map):
    """
    Print route results in a simple assignment-friendly format.
    """

    if not routes:
        print("No route found.")
        return

    print("Routes:")
    for route in routes:
        path_text = " -> ".join(str(site) for site in route["path"])
        total_time = route["total_time_minutes"]

        print(
            f"Route {route['route_number']}: "
            f"{path_text} | Travel time: {total_time:.2f} minutes"
        )

def save_routes_to_csv(routes, output_path="results/tbrgs_routes.csv"):
    """
    Save route summary and edge details to CSV files.

    Files created:
        results/tbrgs_routes.csv
        results/tbrgs_routes_edge_details.csv
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    route_rows = []
    edge_rows = []

    for route in routes:
        route_number = route["route_number"]

        route_rows.append({
            "route_number": route_number,
            "path": " -> ".join(str(site) for site in route["path"]),
            "total_time_minutes": route["total_time_minutes"]
        })

        for edge_index, edge in enumerate(route["edge_details"], start=1):
            edge_rows.append({
                "route_number": route_number,
                "edge_number": edge_index,
                "from_site": edge["from_site"],
                "to_site": edge["to_site"],
                "distance_km": edge["distance_km"],
                "predicted_15min_flow_at_to_site": edge["predicted_15min_flow_at_to_site"],
                "predicted_hourly_flow_at_to_site": edge["predicted_hourly_flow_at_to_site"],
                "estimated_speed_kmh": edge["estimated_speed_kmh"],
                "edge_time_minutes": edge["edge_time_minutes"]
            })

    pd.DataFrame(route_rows).to_csv(output_path, index=False)

    edge_output_path = output_path.replace(".csv", "_edge_details.csv")
    pd.DataFrame(edge_rows).to_csv(edge_output_path, index=False)

    print()
    print(f"Results saved to: {output_path}")


def print_graph_summary(graph):
    """
    Print a small summary of the manually built graph.
    """

    edge_count = sum(len(edges) for edges in graph.values())
    node_count = len(graph)

    print(f"Manual graph nodes with outgoing connections: {node_count}")
    print(f"Manual graph directed edges: {edge_count}")


# ---------------------------------------------------------------------------
# Main command-line program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Traffic-Based Route Guidance System for SCATS sites"
    )

    parser.add_argument(
        "--origin",
        type=int,
        required=True,
        help="Origin SCATS site number, e.g. 2000"
    )

    parser.add_argument(
        "--destination",
        type=int,
        required=True,
        help="Destination SCATS site number, e.g. 3002"
    )

    parser.add_argument(
        "--datetime",
        type=str,
        required=True,
        help='Requested date/time, e.g. "2006-10-15 08:00"'
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "lstm", "gru"],
        help="ML model type to use: mlp, lstm, or gru"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of best routes to return"
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=DEFAULT_LAG,
        help="Number of previous 15-minute intervals used for prediction"
    )

    parser.add_argument(
        "--connections",
        type=str,
        default=CONNECTIONS_PATH,
        help="Path to SCATS connection CSV file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/tbrgs_routes.csv",
        help="CSV output path for route results"
    )

    args = parser.parse_args()

    requested_datetime = pd.to_datetime(args.datetime)

    # Load data.
    df = load_scats_data(DATA_PATH)
    nodes = load_scats_nodes(df)
    connections = load_connections(args.connections)

    # Build manual graph.
    graph, node_map = build_graph_from_connections(connections, nodes)

    validate_site(args.origin, node_map, graph, role="origin")
    validate_site(args.destination, node_map, graph=None, role="destination")

    print()
    print("Traffic-Based Route Guidance System")
    print("-----------------------------------")
    print(f"Origin: {args.origin}")
    print(f"Destination: {args.destination}")
    print(f"Time: {requested_datetime}")
    print(f"Model: {args.model.upper()}")
    print()

    routes = find_top_k_routes(
        graph=graph,
        origin=args.origin,
        destination=args.destination,
        model_name=args.model,
        requested_datetime=requested_datetime,
        df=df,
        k=args.k,
        lag=args.lag
    )

    print_routes(routes, node_map)
    save_routes_to_csv(routes, output_path=args.output)


if __name__ == "__main__":
    main()
