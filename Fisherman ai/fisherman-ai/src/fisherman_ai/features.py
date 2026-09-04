from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WINDOWS_MINUTES = (10, 30, 60)


def _circular_change(series: pd.Series, periods: int) -> pd.Series:
    radians = np.deg2rad(series)
    delta = np.arctan2(np.sin(radians - radians.shift(periods)), np.cos(radians - radians.shift(periods)))
    return np.rad2deg(delta)


def engineer_features(
    observations: pd.DataFrame,
    windows_minutes: tuple[int, ...] = DEFAULT_WINDOWS_MINUTES,
    sample_minutes: int = 1,
) -> pd.DataFrame:
    """Derive causal rolling and trend features from chronologically ordered observations."""
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    periods = {minutes: max(1, round(minutes / sample_minutes)) for minutes in windows_minutes}

    for column in ("pressure_hpa", "temperature_c", "humidity_pct", "wind_speed_ms", "wind_gust_ms"):
        if column not in frame:
            continue
        for minutes, period in periods.items():
            frame[f"{column}_change_{minutes}m"] = frame[column].diff(period)
        frame[f"{column}_rolling_mean_30m"] = frame[column].rolling(periods.get(30, 30), min_periods=2).mean()
        frame[f"{column}_rolling_std_30m"] = frame[column].rolling(periods.get(30, 30), min_periods=2).std()

    if "wind_direction_deg" in frame:
        for minutes, period in periods.items():
            frame[f"wind_direction_change_{minutes}m"] = _circular_change(frame["wind_direction_deg"], period)

    if {"acceleration_x", "acceleration_y", "acceleration_z"}.issubset(frame):
        acceleration = frame[["acceleration_x", "acceleration_y", "acceleration_z"]]
        frame["acceleration_magnitude"] = np.sqrt((acceleration**2).sum(axis=1))
        frame["acceleration_variability_30m"] = frame["acceleration_magnitude"].rolling(periods.get(30, 30), min_periods=2).std()
    if {"gyro_x", "gyro_y", "gyro_z"}.issubset(frame):
        gyro = frame[["gyro_x", "gyro_y", "gyro_z"]]
        frame["gyro_magnitude"] = np.sqrt((gyro**2).sum(axis=1))
        frame["gyro_variability_30m"] = frame["gyro_magnitude"].rolling(periods.get(30, 30), min_periods=2).std()
    return frame
