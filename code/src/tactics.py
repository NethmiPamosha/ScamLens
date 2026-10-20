"""Rule-based Manipulation Tactic Explainer.

Each tactic has a list of regex patterns. When a pattern matches, we keep the
exact phrase so the dashboard can say WHICH words triggered WHICH tactic.
This part is independent of the ML model, so it still works on messages the
model has never seen.
"""

import re

TACTICS = {
    "Urgency": {
        "icon": "⏰",
        "why": "Creates time pressure so you act before you think or check.",
        "patterns": [
            r"\b(act|respond|reply|call|verify|confirm|update|claim)\s+(now|immediately|today|asap|urgently)\b",
            r"\b(immediately|urgent(ly)?|asap|right away|hurry|quick(ly)?)\b",
            r"\b(within|in)\s+\d+\s*(hours?|hrs?|minutes?|mins?|days?)\b",
            r"\b(expires?|expiring|last chance|final (notice|warning|reminder)|limited time|only today|ends (today|tonight|soon))\b",
            r"\bnow!+",
        ],
    },
    "Fear / Threat": {
        "icon": "⚠️",
        "why": "Threatens a loss or punishment to make you panic.",
        "patterns": [
            r"\b(account|card|number|service|sim)\s+(will be|has been|is)\s+(blocked|suspended|locked|closed|deactivated|terminated|disabled)\b",
            r"\b(blocked|suspended|locked|deactivated|terminated|frozen)\b",
            r"\b(unauthori[sz]ed|suspicious) (activity|login|access|transaction)\b",
            r"\b(legal action|arrest|penalty|fine|police|lawsuit|court)\b",
            r"\b(security (alert|breach|warning)|compromised|hacked)\b",
        ],
    },
    "Authority Impersonation": {
        "icon": "🏛️",
        "why": "Pretends to be a trusted organisation so you lower your guard.",
        "patterns": [
            r"\b(your )?bank('s)?\b",
            r"\b(security|fraud|support|customer care|it|helpdesk) (team|department|centre|center|desk)\b",
            r"\b(paypal|amazon|apple|microsoft|google|netflix|dhl|fedex|ups|post office|courier)\b",
            r"\b(government|tax (office|department)|irs|hmrc|inland revenue|customs|police)\b",
            r"\b(official|administrator|admin|ceo|manager) (notice|message|team)?\b",
        ],
    },
    "Reward / Greed": {
        "icon": "🎁",
        "why": "Promises something valuable for free so excitement beats caution.",
        "patterns": [
            r"\b(you('ve| have)? (won|been selected|been chosen))\b",
            r"\b(winner|congratulations|congrats|prize|reward|jackpot|lottery|bonus|cash ?back)\b",
            r"\b(free|gift|voucher|giveaway)\b",
            r"\b(claim|collect) (your|now|the)\b",
            r"\b(guaranteed|risk[- ]free|100% (free|guaranteed)|double your)\b",
        ],
    },
    "Suspicious Call to Action": {
        "icon": "🔗",
        "why": "Pushes you to click, reply or share details through an unverified channel.",
        "patterns": [
            r"\b(click|tap|visit|open|follow) (here|this|the|on|below|link)\b",
            r"(https?://\S+|www\.\S+|bit\.ly\S*|tinyurl\S*)",
            r"\b(verify|confirm|update|validate) (your )?(account|identity|details|information|password|pin|otp)\b",
            r"\b(send|share|enter|provide|give) (us )?(your )?(password|pin|otp|code|card|bank|details|cvv)\b",
            r"\b(text|txt|reply|call|send)\s+[a-z]*\s*(to )?\d{4,}\b",
        ],
    },
}

# Pre-compile once for speed.
_COMPILED = {name: [re.compile(p, re.I) for p in info["patterns"]] for name, info in TACTICS.items()}


def detect_tactics(text):
    """Return a list of detected tactics, each with the phrases that matched."""
    found = []
    for name, regexes in _COMPILED.items():
        phrases = []
        for rx in regexes:
            for m in rx.finditer(text):
                phrase = m.group(0).strip()
                if phrase and phrase.lower() not in [p.lower() for p in phrases]:
                    phrases.append(phrase)
        if phrases:
            found.append({
                "tactic": name,
                "icon": TACTICS[name]["icon"],
                "why": TACTICS[name]["why"],
                "phrases": phrases,
            })
    return found


def explain_in_plain_language(tactics, scam_probability):
    """Build the short plain-language summary shown at the top of the dashboard."""
    if not tactics:
        if scam_probability >= 0.5:
            return ("The model finds this message suspicious based on its wording, "
                    "but no classic manipulation tactic was matched. Check the highlighted words.")
        return "No common manipulation tactics were found in this message."
    lines = []
    for t in tactics:
        quoted = ", ".join(f'"{p}"' for p in t["phrases"][:3])
        lines.append(f"**{t['tactic']}**: {t['why']} Found in {quoted}.")
    return "\n\n".join(lines)


SAFETY_TIPS = {
    "Urgency": "Slow down. Real organisations give you time and never need an answer within minutes.",
    "Fear / Threat": "Do not panic. Contact the organisation using the number on their official website or your card.",
    "Authority Impersonation": "Verify the sender independently. Never trust a name or logo inside a message.",
    "Reward / Greed": "If you did not enter a competition, you did not win one.",
    "Suspicious Call to Action": "Do not click links or share passwords, PINs or OTPs. Type the official address yourself.",
}
