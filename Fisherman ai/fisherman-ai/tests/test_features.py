import pandas as pd

from fisherman_ai.features import engineer_features


def test_features_include_causal_trends_and_circular_direction():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=61, freq="min"),
        "latitude":  [10.0] * 61,
        "longitude": [20.0] * 61,
        "pressure_hpa": range(1000, 1061),
        "wind_direction_deg": [359.0] * 30 + [1.0] * 31,
    })
    features = engineer_features(frame)
    assert "pressure_hpa_change_10m" in features
    assert features.loc[60, "pressure_hpa_change_10m"] == 10
    assert abs(features.loc[30, "wind_direction_change_10m"]) < 10
