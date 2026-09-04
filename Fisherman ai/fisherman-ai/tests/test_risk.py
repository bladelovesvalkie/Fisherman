from fisherman_ai.risk import RiskPolicy, expected_risk_score


def test_risk_policy_is_separate_from_probability_output():
    probabilities = {"normal": 0.0, "caution": 0.0, "danger": 1.0, "critical": 0.0}
    assert expected_risk_score(probabilities) == 66.67
    assert RiskPolicy().classify(66.67) == "DANGER"
