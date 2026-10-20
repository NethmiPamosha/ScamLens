"""Quick checks. Run with: python -m pytest -q"""
import joblib
import pytest

from src.explain import coefficient_explanation, predict
from src.stability import check_stability
from src.tactics import detect_tactics

MODEL = joblib.load("models/scamlens_model.joblib")


def test_obvious_scam_is_flagged():
    label, p = predict(MODEL, "Congratulations! You have won a cash prize. Claim now, click here bit.ly/x")
    assert label == "Scam" and p > 0.8


def test_normal_message_is_legit():
    label, p = predict(MODEL, "Are we still meeting for lunch tomorrow?")
    assert label == "Legitimate" and p < 0.3


@pytest.mark.parametrize("text,tactic", [
    ("Act immediately!", "Urgency"),
    ("Your account will be blocked.", "Fear / Threat"),
    ("This is your bank's security team.", "Authority Impersonation"),
    ("You have won a cash prize!", "Reward / Greed"),
    ("Click this link to verify.", "Suspicious Call to Action"),
])
def test_slide_examples_detect_tactics(text, tactic):
    assert tactic in [t["tactic"] for t in detect_tactics(text)]


def test_explanation_and_stability_run():
    msg = "Claim your free prize now"
    assert coefficient_explanation(MODEL, msg)
    res = check_stability(MODEL, msg)
    assert 0 <= res["score"] <= 100 and res["rows"]


def test_overall_risk_raises_bank_threat():
    from src.explain import overall_risk
    msg = "Your account will be blocked immediately. Verify now!"
    _, p = predict(MODEL, msg)
    assert overall_risk(p, detect_tactics(msg)) > p
