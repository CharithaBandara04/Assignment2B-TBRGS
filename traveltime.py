"""
Travel time estimation module for COS30019 Assignment 2B.

Purpose of this file:

This file only does the mathematical conversion used after a valid edge
has already been selected by the routing/TBRGS program.

For a valid edge:
    SCATS A -> SCATS B

The route program should:
    1. Predict the traffic flow at SCATS site B using the trained ML model.
    2. Convert the predicted 15-minute flow into hourly flow if needed.
    3. Use this file to convert flow + distance into travel time.
    4. Use that travel time as the edge cost.

Main formula idea:
------------------
The ML model predicts traffic flow.
Traffic flow is converted into estimated speed.
Distance and speed are used to calculate travel time.
An average 30 second delay is added for each controlled intersection.
"""

import math


SPEED_LIMIT_KMH = 60
INTERSECTION_DELAY_SECONDS = 30


def convert_15min_flow_to_hourly(predicted_15min_flow):
    """
    Convert a predicted 15-minute traffic flow value into an approximate hourly flow.

    The SCATS dataset has 96 values per day:
        24 hours * 4 values per hour = 96

    Therefore, each model prediction is for one 15-minute interval.

    Example:
        predicted_15min_flow = 120 vehicles per 15 minutes
        predicted_hourly_flow = 120 * 4 = 480 vehicles per hour
    """

    predicted_15min_flow = max(0, float(predicted_15min_flow))

    return predicted_15min_flow * 4


def flow_to_speed(hourly_flow):
    """
    Convert hourly traffic flow into estimated speed in km/h.

    The simplified assignment formula is:

        flow = -1.4648375 * speed^2 + 93.75 * speed

    In this function, flow means accumulated hourly traffic volume.

    If the hourly flow is low, the speed is assumed to be the speed limit:
        60 km/h

    If the quadratic equation cannot produce a valid result, a fallback
    speed of 32 km/h is used.
    """

    hourly_flow = max(0, float(hourly_flow))

    if hourly_flow <= 351:
        return SPEED_LIMIT_KMH

    a = -1.4648375
    b = 93.75
    c = -hourly_flow

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
    Calculate the straight-line distance between two SCATS sites using latitude
    and longitude.

    This function only calculates distance. It does not decide whether two SCATS
    sites are connected by road.

    The routing program should only call this function for valid road links from:
        data/scats_connections.csv
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


def calculate_travel_time(distance_km, predicted_hourly_flow):
    """
    Calculate travel time in minutes for one valid road edge.

    Args:
        distance_km:
            Distance between two connected SCATS sites.

        predicted_hourly_flow:
            Predicted accumulated hourly traffic volume at the destination
            SCATS site of the edge.

    Returns:
        Total edge travel time in minutes.

    Important:
        If your ML model prediction is a 15-minute flow value, convert it first:

            predicted_hourly_flow = convert_15min_flow_to_hourly(predicted_15min_flow)

        Then call:

            calculate_travel_time(distance_km, predicted_hourly_flow)
    """

    distance_km = max(0, float(distance_km))
    predicted_hourly_flow = max(0, float(predicted_hourly_flow))

    speed_kmh = flow_to_speed(predicted_hourly_flow)

    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60

    delay_minutes = INTERSECTION_DELAY_SECONDS / 60

    total_time_minutes = time_minutes + delay_minutes

    return total_time_minutes


def calculate_edge_travel_time_from_15min_flow(distance_km, predicted_15min_flow):
    """
    Convenience function for the TBRGS program.

    Use this when the ML model gives a 15-minute flow prediction.

    It performs:
        15-minute flow -> hourly flow -> speed -> travel time

    This is useful for a valid edge:
        SCATS A -> SCATS B

    where predicted_15min_flow is the model prediction for SCATS site B.
    """

    predicted_hourly_flow = convert_15min_flow_to_hourly(predicted_15min_flow)

    return calculate_travel_time(
        distance_km=distance_km,
        predicted_hourly_flow=predicted_hourly_flow
    )


if __name__ == "__main__":
    # Example test for one valid road edge.
    # This does not check road connectivity. It only tests the travel time maths.

    predicted_15min_flow = 125

    lat1, lon1 = -37.81655, 145.09831
    lat2, lon2 = -37.82300, 145.10500

    distance = calculate_distance_km(lat1, lon1, lat2, lon2)

    predicted_hourly_flow = convert_15min_flow_to_hourly(predicted_15min_flow)

    speed = flow_to_speed(predicted_hourly_flow)

    travel_time = calculate_travel_time(
        distance_km=distance,
        predicted_hourly_flow=predicted_hourly_flow
    )

    print("Predicted 15-minute flow:", predicted_15min_flow)
    print("Predicted hourly flow:", predicted_hourly_flow)
    print("Estimated speed:", round(speed, 2), "km/h")
    print("Distance:", round(distance, 3), "km")
    print("Estimated edge travel time:", round(travel_time, 2), "minutes")
