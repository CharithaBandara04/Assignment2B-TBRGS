"""
Routing module for Assignment 2B.

This file builds a simple SCATS graph and finds routes using travel time as edge cost.
"""

import heapq
import pandas as pd

from traveltime import calculate_distance_km, calculate_travel_time


def load_scats_nodes(file_path="data/processed_scats_15min.csv"):
    df = pd.read_csv(file_path)

    nodes = (
        df[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
        .groupby("SCATS Number", as_index=False)
        .mean()
    )

    return nodes


def build_graph(nodes, max_neighbors=3):
    """
    Build simple graph by connecting each SCATS site to nearest neighbours.
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


def dijkstra(graph, origin, destination, predicted_flow):
    """
    Find shortest-time path using Dijkstra.
    """

    queue = [(0, origin, [origin])]
    visited = set()

    while queue:
        current_time, current_node, path = heapq.heappop(queue)

        if current_node == destination:
            return current_time, path

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, distance_km in graph.get(current_node, []):
            if neighbor not in visited:
                edge_time = calculate_travel_time(distance_km, predicted_flow)
                new_time = current_time + edge_time
                heapq.heappush(queue, (new_time, neighbor, path + [neighbor]))

    return None, []


def find_top_k_routes(graph, origin, destination, predicted_flow, k=5):
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
                edge_time = calculate_travel_time(distance_km, predicted_flow)
                new_time = current_time + edge_time
                heapq.heappush(queue, (new_time, neighbor, path + [neighbor]))

    return routes


if __name__ == "__main__":
    nodes = load_scats_nodes()
    graph = build_graph(nodes, max_neighbors=8)

    origin = 2200
    destination = 2825

    # temporary example predicted flow
    # later this should come from MLP/LSTM/GRU
    predicted_flow = 500

    routes = find_top_k_routes(
        graph,
        origin,
        destination,
        predicted_flow,
        k=5
    )

    if not routes:
        print("No route found. Try increasing max_neighbors.")
    else:
        for i, (time, path) in enumerate(routes, start=1):
            print(f"Route {i}:")
            print("Path:", path)
            print("Estimated travel time:", round(time, 2), "minutes")
            print()