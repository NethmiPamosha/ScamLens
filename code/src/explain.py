"""Word-level explainability (XAI) for ScamLens.

Two methods:
1. Coefficient contributions: exact and instant for a linear model.
   contribution(word) = tfidf_value(word) * model_weight(word)
2. LIME: model-agnostic, perturbs the message and fits a local surrogate.
   Slower, but the same method would still work if we swap the model later.
"""

from lime.lime_text import LimeTextExplainer

from src.preprocess import pretty_token

CLASS_NAMES = ["Legitimate", "Scam"]


def predict(model, text):
    """Return (label, scam_probability) for one message."""
    p = float(model.predict_proba([text])[0, 1])
    return ("Scam" if p >= 0.5 else "Legitimate"), p


def coefficient_explanation(model, text, top_k=12):
    """Exact per-feature contribution for a linear model. Positive = towards Scam."""
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    row = vec.transform([text])
    vocab = vec.get_feature_names_out()
    weights = clf.coef_[0]
    items = []
    for idx, value in zip(row.indices, row.data):
        items.append((vocab[idx], float(value * weights[idx])))
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [(pretty_token(w), c) for w, c in items[:top_k]]


def lime_explanation(model, text, num_features=10, num_samples=800):
    """LIME word weights. Positive = pushes towards Scam."""
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
    exp = explainer.explain_instance(
        text, model.predict_proba, num_features=num_features,
        num_samples=num_samples, labels=(1,))
    return exp.as_list(label=1)


def highlight_html(text, word_weights):
    """Wrap words of the original message in coloured spans.

    Red = pushes towards Scam, green = pushes towards Legitimate.
    Opacity scales with the size of the weight.
    """
    weights = {w.lower(): v for w, v in word_weights}
    if not weights:
        return text
    max_abs = max(abs(v) for v in weights.values()) or 1.0
    out = []
    for token in text.split(" "):
        core = "".join(ch for ch in token.lower() if ch.isalnum())
        v = weights.get(core)
        if v is None:
            out.append(token)
            continue
        alpha = 0.15 + 0.75 * abs(v) / max_abs
        color = f"rgba(220,38,38,{alpha:.2f})" if v > 0 else f"rgba(22,163,74,{alpha:.2f})"
        out.append(f'<span style="background:{color};padding:1px 4px;border-radius:4px" '
                   f'title="weight {v:+.3f}">{token}</span>')
    return " ".join(out)


def risk_level(p):
    """Map probability to a simple risk band for non-technical users."""
    if p >= 0.8:
        return "High risk", "#dc2626"
    if p >= 0.5:
        return "Medium risk", "#ea580c"
    if p >= 0.25:
        return "Low risk", "#ca8a04"
    return "Very low risk", "#16a34a"



def overall_risk(model_probability, tactics, per_tactic=0.15):
    """Combine the ML probability with the rule-based tactic layer.

    Each detected tactic independently adds a 15% chance of being a scam:
        overall = 1 - (1 - p) * (1 - 0.15) ** number_of_tactics
    This keeps the model as the main signal but stops it from under-warning
    on phishing styles that the training data covers poorly.
    """
    n = len(tactics)
    return 1 - (1 - model_probability) * (1 - per_tactic) ** n
