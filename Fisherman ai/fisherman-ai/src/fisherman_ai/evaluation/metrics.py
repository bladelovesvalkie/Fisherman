from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from fisherman_ai.models.training import TrainedModel, prepare_training_data


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    split_index = max(1, int(len(frame) * (1 - test_fraction)))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def evaluate(model: TrainedModel, frame: pd.DataFrame) -> dict:
    features, labels = prepare_training_data(frame)
    predicted = model.pipeline.predict(features)
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "classification_report": classification_report(labels, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
    }
