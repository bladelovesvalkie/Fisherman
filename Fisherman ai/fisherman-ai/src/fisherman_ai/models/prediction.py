from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from fisherman_ai.features import engineer_features
from fisherman_ai.models.training import EXCLUDED_COLUMNS, TrainedModel
from fisherman_ai.risk import RiskPolicy, expected_risk_score


@dataclass(frozen=True)
class Prediction:
    risk_score: float
    danger_level: str
    probabilities: dict[str, float]
    top_risk_factors: list[str]
    model: str
    version: str

    def to_dict(self) -> dict:
        return asdict(self)


def predict(model: TrainedModel, observations: pd.DataFrame, policy: RiskPolicy | None = None) -> Prediction:
    featured = engineer_features(observations).tail(1)
    values = featured.reindex(columns=model.feature_names)
    probabilities_array = model.pipeline.predict_proba(values)[0]
    classes = model.pipeline.classes_
    probabilities = {str(int(label)): float(probability) for label, probability in zip(classes, probabilities_array)}
    named = {name.lower(): probabilities.get(str(index), 0.0) for index, name in enumerate(("NORMAL", "CAUTION", "DANGER", "CRITICAL"))}
    score = expected_risk_score(named, policy)
    factors = explain(model, values)
    return Prediction(score, (policy or RiskPolicy()).classify(score), named, factors, model.model_name, model.version)


def explain(model: TrainedModel, values: pd.DataFrame, limit: int = 5) -> list[str]:
    estimator = model.pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        weights = abs(estimator.coef_).mean(axis=0)
    else:
        return []
    ranked = sorted(zip(model.feature_names, weights), key=lambda item: item[1], reverse=True)
    return [name for name, weight in ranked[:limit] if weight > 0]
