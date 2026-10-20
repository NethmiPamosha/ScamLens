"""Prediction Stability Checker.

Generates small, meaning-preserving variations of a message (typos, leetspeak,
extra punctuation, casing...) and re-runs the classifier on each one.
If the label flips or the probability swings a lot, the decision is fragile.
"""

import random

import numpy as np

LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "l": "1", "t": "7"}
SYNONYMS = {
    "free": "complimentary", "win": "get", "won": "received", "prize": "gift",
    "claim": "collect", "urgent": "important", "now": "today", "click": "tap",
    "account": "profile", "verify": "check", "call": "ring", "cash": "money",
}


def _words(text):
    return text.split(" ")


def leetspeak(text, rng):
    words = _words(text)
    i = rng.randrange(len(words))
    w = words[i]
    words[i] = "".join(LEET.get(c.lower(), c) if rng.random() < 0.5 else c for c in w)
    return " ".join(words)


def swap_chars(text, rng):
    words = _words(text)
    long_idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not long_idx:
        return text
    i = rng.choice(long_idx)
    w = list(words[i])
    j = rng.randrange(len(w) - 1)
    w[j], w[j + 1] = w[j + 1], w[j]
    words[i] = "".join(w)
    return " ".join(words)


def drop_char(text, rng):
    words = _words(text)
    long_idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not long_idx:
        return text
    i = rng.choice(long_idx)
    w = words[i]
    j = rng.randrange(len(w))
    words[i] = w[:j] + w[j + 1:]
    return " ".join(words)


def insert_dots(text, rng):
    words = _words(text)
    long_idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not long_idx:
        return text
    i = rng.choice(long_idx)
    words[i] = ".".join(words[i])
    return " ".join(words)


def change_case(text, rng):
    return text.upper() if rng.random() < 0.5 else text.lower()


def add_punctuation(text, rng):
    return text + rng.choice(["!!!", " ...", " :)", "??"])


def synonym(text, rng):
    words = _words(text)
    idx = [i for i, w in enumerate(words) if w.lower().strip(".,!?") in SYNONYMS]
    if not idx:
        return text
    i = rng.choice(idx)
    key = words[i].lower().strip(".,!?")
    words[i] = SYNONYMS[key]
    return " ".join(words)


PERTURBATIONS = {
    "Leetspeak": leetspeak,
    "Swapped letters": swap_chars,
    "Missing letter": drop_char,
    "Dotted word": insert_dots,
    "Casing change": change_case,
    "Extra punctuation": add_punctuation,
    "Synonym swap": synonym,
}


def generate_variants(text, n_per_type=3, seed=42):
    rng = random.Random(seed)
    variants = []
    for name, fn in PERTURBATIONS.items():
        for _ in range(n_per_type):
            v = fn(text, rng)
            if v != text and v not in [x[1] for x in variants]:
                variants.append((name, v))
    return variants


def check_stability(model, text, n_per_type=3, seed=42):
    """Return a summary dict plus the per-variant results."""
    base_p = float(model.predict_proba([text])[0, 1])
    base_label = int(base_p >= 0.5)
    variants = generate_variants(text, n_per_type, seed)
    if not variants:
        return {"score": 100.0, "verdict": "Stable", "base_p": base_p,
                "flip_rate": 0.0, "mean_shift": 0.0, "max_shift": 0.0, "rows": []}

    probs = model.predict_proba([v for _, v in variants])[:, 1]
    rows = []
    for (kind, v), p in zip(variants, probs):
        rows.append({
            "perturbation": kind,
            "variant": v,
            "scam_probability": round(float(p), 4),
            "shift": round(float(p - base_p), 4),
            "label_flipped": int(p >= 0.5) != base_label,
        })

    shifts = np.abs(probs - base_p)
    flip_rate = float(np.mean([r["label_flipped"] for r in rows]))
    mean_shift = float(shifts.mean())
    # Score: 100 = perfectly stable. Flips cost the most, then average drift.
    score = max(0.0, 100.0 * (1 - 0.7 * flip_rate - 0.3 * min(1.0, mean_shift / 0.5)))
    if score >= 80:
        verdict = "Stable"
    elif score >= 55:
        verdict = "Somewhat fragile"
    else:
        verdict = "Fragile"
    return {
        "score": round(score, 1), "verdict": verdict, "base_p": base_p,
        "flip_rate": flip_rate, "mean_shift": mean_shift,
        "max_shift": float(shifts.max()), "rows": rows,
    }
