"""The ScamLens model object.

It has two layers, trained in train.py:

1. text_model: TF-IDF + multinomial Logistic Regression that predicts
   three classes: Legitimate, Spam (unwanted promotion) and Scam (smishing).

2. combiner: a small Logistic Regression that takes
       - the text model's scam and spam scores (as log-odds), and
       - five yes/no flags from the rule-based tactic detector
   and outputs the final "scam risk score".
   It is trained on out-of-fold predictions, so it learns how much each
   signal really matters instead of us guessing the weights by hand.

Important: the scam risk score is calibrated on OUR dataset's test split.
It is a well-behaved score for that data, not a guaranteed probability that
a real message in the wild is fraud.
"""

import numpy as np

from src.tactics import TACTICS, detect_tactics

CLASSES = ["Legitimate", "Spam", "Scam"]
SCAM, SPAM = 2, 1
TACTIC_NAMES = list(TACTICS)


def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def tactic_flags(texts):
    """5 yes/no columns, one per tactic, for each message."""
    rows = []
    for t in texts:
        found = {x["tactic"] for x in detect_tactics(t)}
        rows.append([1.0 if name in found else 0.0 for name in TACTIC_NAMES])
    return np.array(rows)


def combiner_features(class_proba, texts):
    """Feature matrix for the combiner: [logit P(scam), logit P(spam), 5 tactic flags]."""
    class_proba = np.asarray(class_proba)
    return np.column_stack([
        _logit(class_proba[:, SCAM]),
        _logit(class_proba[:, SPAM]),
        tactic_flags(texts),
    ])


FEATURE_NAMES = ["AI model: scam score", "AI model: spam score"] + [f"Tactic: {n}" for n in TACTIC_NAMES]


class ScamLensModel:
    def __init__(self, text_model, combiner):
        self.text_model = text_model
        self.combiner = combiner

    def predict_proba(self, texts):
        """Class probabilities from the text model, columns = CLASSES."""
        return self.text_model.predict_proba(list(texts))

    def scam_proba(self, texts):
        """Text model's probability of the Scam class only."""
        return self.predict_proba(texts)[:, SCAM]

    def scam_risk(self, texts):
        """Final calibrated scam risk score (text model + tactics)."""
        texts = list(texts)
        proba = self.predict_proba(texts)
        return self.combiner.predict_proba(combiner_features(proba, texts))[:, 1]

    def analyze(self, text):
        """Everything the dashboard needs for one message."""
        proba = self.predict_proba([text])[0]
        risk = float(self.combiner.predict_proba(combiner_features([proba], [text]))[0, 1])
        return {
            "class_proba": {c: float(p) for c, p in zip(CLASSES, proba)},
            "label": CLASSES[int(np.argmax(proba))],
            "p_scam": float(proba[SCAM]),
            "p_spam": float(proba[SPAM]),
            "risk": risk,
        }

    @property
    def vectorizer(self):
        return self.text_model.named_steps["tfidf"]

    @property
    def classifier(self):
        return self.text_model.named_steps["clf"]