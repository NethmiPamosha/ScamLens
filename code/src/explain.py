"""Word-level explainability (XAI) for ScamLens.
 
Two methods, both explaining the SCAM class of the text model:
1. Coefficient contributions: exact and instant for a linear model.
   contribution(word) = tfidf_value(word) * (scam weight - average weight of the other classes)
   Positive means the word pushes towards Scam, negative pushes away from it.
2. LIME: model-agnostic, perturbs the message and fits a local surrogate.
   Slower, but it would still work if we swap the model later.
"""
 
import html
 
from lime.lime_text import LimeTextExplainer
 
from src.model import CLASSES, SCAM
from src.preprocess import pretty_token
 
 
def coefficient_explanation(model, text, top_k=12):
    vec = model.vectorizer
    W = model.classifier.coef_
    scam_dir = W[SCAM] - W[[0, 1]].mean(axis=0)
    row = vec.transform([text])
    vocab = vec.get_feature_names_out()
    items = [(vocab[i], float(v * scam_dir[i])) for i, v in zip(row.indices, row.data)]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [(pretty_token(w), c) for w, c in items[:top_k]]
 
 
def lime_explanation(model, text, num_features=10, num_samples=800):
    """LIME word weights for the Scam class. Positive = pushes towards Scam."""
    explainer = LimeTextExplainer(class_names=CLASSES, random_state=42)
    exp = explainer.explain_instance(text, model.predict_proba, num_features=num_features,
                                     num_samples=num_samples, labels=(SCAM,))
    return exp.as_list(label=SCAM)
 
 
def highlight_html(text, word_weights):
    """Wrap words of the message in coloured spans (red = towards Scam, green = away)."""
    weights = {w.lower(): v for w, v in word_weights}
    if not weights:
        return html.escape(text)
    max_abs = max(abs(v) for v in weights.values()) or 1.0
    out = []
    for token in text.split(" "):
        core = "".join(ch for ch in token.lower() if ch.isalnum())
        v = weights.get(core)
        safe = html.escape(token)
        if v is None:
            out.append(safe)
            continue
        alpha = 0.15 + 0.75 * abs(v) / max_abs
        color = f"rgba(220,38,38,{alpha:.2f})" if v > 0 else f"rgba(22,163,74,{alpha:.2f})"
        out.append(f'<span style="background:{color};padding:1px 4px;border-radius:4px" '
                   f'title="weight {v:+.3f}">{safe}</span>')
    return " ".join(out)
 

