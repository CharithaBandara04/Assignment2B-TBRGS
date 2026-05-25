"""
Travel time estimation module for Assignment 2B.

This file converts predicted traffic flow into speed and travel time.
It can be used by MLP, LSTM, and GRU predictions.
"""

import math


SPEED_LIMIT_KMH = 60
INTERSECTION_DELAY_SECONDS = 30


def flow_to_speed(flow):
    """
    Convert traffic flow into estimated speed.

    Formula:
    flow = -1.4648375 * speed^2 + 93.75 * speed
    """

    flow = max(0, float(flow))

    if flow <= 351:
        return SPEED_LIMIT_KMH

    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return 32

    speed_1 = (-b + math.sqrt(discriminant)) / (2 * a)
    speed_2 = (-b - math.sqrt(discriminant)) / (2 * a)

    speed = max(speed_1, speed_2)

    speed = min(speed, SPEED_LIMIT_KMH)
    speed = max(speed, 1)

    return speed


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two SCATS sites
    using latitude and longitude.
    """

    earth_radius_km = 6371

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def calculate_travel_time(distance_km, predicted_flow):
    """
    Calculate travel time in minutes using distance and predicted flow.
    """

    speed_kmh = flow_to_speed(predicted_flow)

    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60

    delay_minutes = INTERSECTION_DELAY_SECONDS / 60

    total_time_minutes = time_minutes + delay_minutes

    return total_time_minutes


if __name__ == "__main__":
    # Example test
    predicted_flow = 500

    lat1, lon1 = -37.81655, 145.09831
    lat2, lon2 = -37.82300, 145.10500

    distance = calculate_distance_km(lat1, lon1, lat2, lon2)
    speed = flow_to_speed(predicted_flow)
    travel_time = calculate_travel_time(distance, predicted_flow)

    print("Predicted flow:", predicted_flow)
    print("Estimated speed:", round(speed, 2), "km/h")
    print("Distance:", round(distance, 3), "km")
    print("Estimated travel time:", round(travel_time, 2), "minutes")