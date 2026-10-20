"""Text preprocessing for ScamLens.

Cleaning is kept simple on purpose so the explanations stay readable:
URLs, emails, phone numbers, money amounts and plain numbers are replaced
with placeholder tokens. That way the model learns "this message has a link"
instead of memorising one specific link.
"""

import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|info|biz|ly|co|lk)\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"\+?\d[\d\s-]{7,}\d")
MONEY_RE = re.compile(r"(£|\$|€|rs\.?|lkr)\s?\d[\d,.]*|\d[\d,.]*\s?(usd|lkr|rs|pounds|dollars)", re.I)
NUM_RE = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Lowercase the message and replace noisy parts with placeholder tokens."""
    if text is None:
        return ""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


# Friendly names for the placeholder tokens, used when we show explanations.
TOKEN_LABELS = {
    "urltoken": "[a link]",
    "emailtoken": "[an email address]",
    "moneytoken": "[a money amount]",
    "phonetoken": "[a phone number]",
    "numtoken": "[a number]",
}


def pretty_token(token):
    """Turn 'urltoken' into '[a link]' etc. so users understand the feature."""
    parts = token.split()
    return " ".join(TOKEN_LABELS.get(p, p) for p in parts)
