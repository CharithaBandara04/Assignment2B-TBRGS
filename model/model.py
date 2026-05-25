"""
Model definitions for Assignment 2B.

This file defines MLP, LSTM, and GRU models.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU


def get_mlp(lag):
    """
    Build MLP model.
    Input shape: previous lag traffic flow values.
    """

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(lag,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    return model


def get_lstm(lag):
    """
    Build LSTM model.
    Input shape: previous lag traffic values as sequence.
    """

    model = Sequential()
    model.add(LSTM(64, input_shape=(lag, 1), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model


def get_gru(lag):
    """
    Build GRU model.
    Input shape: previous lag traffic values as sequence.
    """

    model = Sequential()
    model.add(GRU(64, input_shape=(lag, 1), return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model