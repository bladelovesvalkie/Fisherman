from pathlib import Path

import pandas as pd

from fisherman_ai.schema import validate_columns


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_columns(frame.columns)
    if "danger_label" not in frame.columns:
        raise ValueError("Training data must contain danger_label")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)
