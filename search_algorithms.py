"""
Search algorithms adapted from Assignment 2A for Assignment 2B TBRGS.

This file shows the Part A integration clearly.

In Assignment 2A, CUS1 used Uniform Cost Search to find the lowest-cost path.
In Assignment 2B, the same idea is used to find the lowest-travel-time route.

The main difference is:
    Assignment 2A edge cost = fixed cost from text file
    Assignment 2B edge cost = dynamic travel time from ML traffic prediction
"""

import heapq


def top_k_uniform_cost_search(
    graph,
    origin,
    destination,
    edge_cost_function,
    k=5,
    max_expansions=50000
):
    """
    Top-k Uniform Cost Search for TBRGS.

    This is adapted from the Assignment 2A CUS1 / Uniform Cost Search idea.

    Args:
        graph:
            Dictionary where each node has outgoing edges.

            Expected format:
                {
                    2000: [
                        {"to": 2200, "distance_km": 1.2},
                        {"to": 3126, "distance_km": 2.0}
                    ]
                }

        origin:
            Origin SCATS site.

        destination:
            Destination SCATS site.

        edge_cost_function:
            Function that calculates travel time for one edge.

            It must accept:
                current_site, next_site, edge

            It must return:
                edge_time_minutes, edge_detail_dictionary

        k:
            Number of best routes to return.

        max_expansions:
            Safety limit to prevent endless searching in very large/incomplete graphs.

    Returns:
        List of routes.

        Each route has:
            route_number
            total_time_minutes
            path
            edge_details
    """

    origin = int(origin)
    destination = int(destination)

    routes = []
    seen_complete_paths = set()

    # Priority queue item:
    # total_cost, insertion_order, current_node, path, edge_details
    frontier = []
    insertion_order = 0

    heapq.heappush(frontier, (0.0, insertion_order, origin, [origin], []))
    insertion_order += 1

    expansions = 0

    while frontier and len(routes) < k:
        expansions += 1

        if expansions > max_expansions:
            print(
                f"Search stopped after {max_expansions} expansions. "
                "The graph may be too large or incomplete."
            )
            break

        current_cost, _, current_site, path, edge_details = heapq.heappop(frontier)

        if current_site == destination:
            path_tuple = tuple(path)

            if path_tuple not in seen_complete_paths:
                seen_complete_paths.add(path_tuple)

                routes.append({
                    "route_number": len(routes) + 1,
                    "total_time_minutes": current_cost,
                    "path": path,
                    "edge_details": edge_details
                })

            continue

        # Expand neighbours in ascending SCATS number order,
        # similar to Assignment 2A tie-breaking.
        neighbours = sorted(
            graph.get(current_site, []),
            key=lambda edge: int(edge["to"])
        )

        for edge in neighbours:
            next_site = int(edge["to"])

            # Avoid loops in a single route.
            if next_site in path:
                continue

            edge_time_minutes, edge_detail = edge_cost_function(
                current_site,
                next_site,
                edge
            )

            new_path = path + [next_site]
            new_edge_details = edge_details + [edge_detail]
            new_cost = current_cost + edge_time_minutes

            heapq.heappush(
                frontier,
                (
                    new_cost,
                    insertion_order,
                    next_site,
                    new_path,
                    new_edge_details
                )
            )

            insertion_order += 1

    return routes
