"""Quick checks. Run with: python -m pytest -q"""
import json

import joblib
import pytest

from src.explain import coefficient_explanation
from src.stability import check_stability
from src.tactics import detect_tactics

MODEL = joblib.load("models/scamlens_model.joblib")


def test_obvious_scam_is_flagged():
    out = MODEL.analyze("Congratulations! You have won a cash prize. Claim now, click here bit.ly/x")
    assert out["label"] == "Scam" and out["risk"] > 0.8


def test_normal_message_is_legit():
    out = MODEL.analyze("Are we still meeting for lunch tomorrow?")
    assert out["label"] == "Legitimate" and out["risk"] < 0.1


def test_promotion_is_spam_not_scam():
    out = MODEL.analyze("Get 50% off all shoes this weekend only at our store! Reply STOP to opt out")
    assert out["label"] == "Spam" and out["risk"] < 0.6


def test_class_probabilities_sum_to_one():
    out = MODEL.analyze("Your parcel is waiting, pay the fee here")
    assert abs(sum(out["class_proba"].values()) - 1) < 1e-6


@pytest.mark.parametrize("text,tactic", [
    ("Act immediately!", "Urgency"),
    ("Your account will be blocked.", "Fear / Threat"),
    ("This is your bank's security team.", "Authority Impersonation"),
    ("You have won a cash prize!", "Reward / Greed"),
    ("Click this link to verify.", "Suspicious Call to Action"),
])
def test_slide_examples_detect_tactics(text, tactic):
    assert tactic in [t["tactic"] for t in detect_tactics(text)]


def test_calibrated_score_beats_heuristic():
    m = json.load(open("models/metrics.json"))["scam_detection"]
    assert m["calibrated_risk_score"]["brier"] <= m["old_heuristic_formula"]["brier"]
    assert m["calibrated_risk_score"]["ece"] <= m["old_heuristic_formula"]["ece"]


def test_tactics_fire_more_on_scams():
    v = json.load(open("models/tactic_validation.json"))["Any tactic"]
    assert v["Scam"] > v["Legitimate"]


def test_explanation_and_stability_run():
    msg = "Claim your free prize now"
    assert coefficient_explanation(MODEL, msg)
    res = check_stability(MODEL, msg)
    assert 0 <= res["score"] <= 100 and res["rows"]