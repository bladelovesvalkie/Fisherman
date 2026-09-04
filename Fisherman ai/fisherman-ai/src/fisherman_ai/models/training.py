from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fisherman_ai.features import engineer_features

LABEL_COLUMN = "danger_label"
EXCLUDED_COLUMNS = {"timestamp", LABEL_COLUMN}


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_names: list[str]
    model_name: str
    version: str = "0.1.0"

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        return joblib.load(path)


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    featured = engineer_features(frame).dropna(subset=[LABEL_COLUMN])
    feature_names = [column for column in featured.columns if column not in EXCLUDED_COLUMNS]
    return featured[feature_names], featured[LABEL_COLUMN].astype(int)


def train(frame: pd.DataFrame, model_name: str = "random_forest") -> TrainedModel:
    features, labels = prepare_training_data(frame)
    if labels.nunique() < 2:
        raise ValueError("Training data must contain at least two danger classes")
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        pipeline = Pipeline([( "imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", estimator)])
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        pipeline = Pipeline([( "imputer", SimpleImputer()), ("model", estimator)])
    else:
        raise ValueError("model_name must be logistic_regression or random_forest")
    pipeline.fit(features, labels)
    return TrainedModel(pipeline, list(features.columns), model_name)
