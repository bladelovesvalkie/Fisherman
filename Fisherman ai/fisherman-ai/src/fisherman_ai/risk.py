from __future__ import annotations

from dataclasses import dataclass

LEVEL_NAMES = ("NORMAL", "CAUTION", "DANGER", "CRITICAL")


@dataclass(frozen=True)
class RiskPolicy:
    """Decision-layer policy, intentionally independent of model training."""

    level_points: tuple[float, float, float] = (25.0, 50.0, 75.0)

    def classify(self, risk_score: float) -> str:
        for index, threshold in enumerate(self.level_points):
            if risk_score < threshold:
                return LEVEL_NAMES[index]
        return LEVEL_NAMES[-1]


def expected_risk_score(probabilities: dict[str, float], policy: RiskPolicy | None = None) -> float:
    policy = policy or RiskPolicy()
    scores = (0.0, 33.333333, 66.666667, 100.0)
    return round(sum(probabilities.get(name.lower(), 0.0) * score for name, score in zip(LEVEL_NAMES, scores)), 2)
