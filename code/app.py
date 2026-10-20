"""ScamLens: Explain the Scam, Not Just the Spam.

Streamlit dashboard. Run from the project root:
    streamlit run app.py
"""

import html
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import coefficient_explanation, highlight_html, lime_explanation
from src.preprocess import pretty_token
from src.stability import check_stability
from src.tactics import SAFETY_TIPS, detect_tactics

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "scamlens_model.joblib"

EXAMPLES = {
    "Bank threat": "URGENT: This is your bank's security team. Your account will be blocked within 24 hours. Verify your details now at www.secure-bank-verify.com",
    "Prize scam": "Congratulations! You have won a cash prize of $5000. Claim your free reward now, click here: bit.ly/claimprize",
    "Courier scam": "DHL: Your parcel is on hold due to unpaid customs fee. Pay immediately to avoid return: http://dhl-track-pay.info",
    "Shop promotion": "Get 50% off all shoes this weekend only at our store! Reply STOP to opt out",
    "Normal message": "Hey, are we still meeting for lunch tomorrow at 12? Let me know if you want to change the place.",
}

# Sidebar menu: (page id, label, material icon)
PAGES = [
    ("home", "Home", ":material/home:"),
    ("analyze", "Analyze Message", ":material/search:"),
    ("whatif", "What-If Lab", ":material/science:"),
    ("stability", "Stability Check", ":material/verified_user:"),
    ("history", "History", ":material/history:"),
    ("awareness", "Tips & Awareness", ":material/lightbulb:"),
    ("insights", "Model Insights", ":material/insights:"),
    ("settings", "Settings", ":material/settings:"),
]

# Short descriptions for the "Why is this a scam?" list
TACTIC_CARDS = {
    "Urgency": ("⏰", "Creates urgency", "Pushes you to act fast so you do not stop and check."),
    "Fear / Threat": ("⚠️", "Uses fear or threats", "Threatens to block, fine or punish you."),
    "Authority Impersonation": ("🏛️", "Pretends to be an authority", "Claims to be your bank, a company or the government."),
    "Reward / Greed": ("🎁", "Offers a reward", "Promises prizes or free money to excite you."),
    "Suspicious Call to Action": ("🔗", "Suspicious link or request", "Asks you to click a link or share private details."),
}

st.set_page_config(page_title="ScamLens", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

# Styling
st.markdown("""
<style>
:root { --accent:#5b4cf0; --accent-soft:#eeecff; --ink:#111827; --muted:#6b7280; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* Dark sidebar like the mockup */
[data-testid="stSidebar"] { background:#0f172a; }
[data-testid="stSidebar"] * { color:#e2e8f0; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background:transparent; border:none; justify-content:flex-start; color:#cbd5e1;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover { background:#1e293b; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background:var(--accent); border:none; justify-content:flex-start;
}
[data-testid="stSidebar"] button { justify-content:flex-start !important; padding-left:1rem; }
[data-testid="stSidebar"] button > div { justify-content:flex-start !important; width:100%; }
.brand { display:flex; align-items:center; gap:.5rem; font-size:1.35rem; font-weight:700; margin-bottom:1.2rem; }
.brand-badge { background:var(--accent); border-radius:10px; padding:2px 8px; }

/* Cards */
.card { background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:1.2rem 1.3rem; }
.hero { background:linear-gradient(120deg,#ffffff 0%,#eeecff 100%); border-radius:20px;
        padding:2rem 2.2rem; border:1px solid #e5e7eb; margin-bottom:1rem; }
.hero h1 { margin:0; padding:0; font-size:2.3rem; color:var(--ink) !important; }
.hero h1 span { color:var(--ink) !important; }
.hero h1 .hl { color:var(--accent) !important; }
.hero p { color:var(--muted); margin:.4rem 0 0; font-size:1.05rem; }
.feature { border-radius:16px; padding:1.1rem; text-align:center; height:100%; }
.feature .ico { font-size:1.7rem; }
.feature b { display:block; margin-top:.3rem; color:var(--ink); }
.feature small { color:var(--muted); }

/* Result banner + ring */
.banner { display:flex; justify-content:space-between; align-items:center;
          border-radius:16px; padding:1.2rem 1.5rem; margin:.6rem 0 1rem; }
.banner h2 { margin:0; font-size:1.6rem; }
.banner p { margin:.2rem 0 0; color:#374151; }
.ring { width:96px; height:96px; border-radius:50%; display:grid; place-items:center; }
.ring div { width:74px; height:74px; border-radius:50%; background:#fff; display:grid;
            place-items:center; font-weight:800; font-size:1.25rem; }

/* Reason rows */
.reason { display:flex; gap:.8rem; align-items:flex-start; background:#fff5f5;
          border:1px solid #fde2e2; border-radius:12px; padding:.7rem .9rem; margin-bottom:.5rem; }
.reason.safe { background:#f0fdf4; border-color:#dcfce7; }
.reason .ico { font-size:1.3rem; }
.reason b { color:var(--ink); }
.reason small { color:var(--muted); display:block; }
.reason code { font-size:.8rem; color:#b91c1c; background:#fee2e2; margin-right:4px; }
.reason.safe code { color:#15803d; background:#dcfce7; }
.todo { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:14px; padding:1rem 1.2rem; }
.todo h4 { margin:0 0 .4rem; color:#166534; }

.badge { padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; }
.badge.scam { background:#fee2e2; color:#b91c1c; }
.badge.warn { background:#ffedd5; color:#c2410c; }
.badge.safe { background:#dcfce7; color:#15803d; }
.badge.spam { background:#fef9c3; color:#a16207; }
.tag { font-size:.7rem; font-weight:600; padding:1px 7px; border-radius:999px; margin-left:6px; vertical-align:middle; }
.tag.rule { background:#e0e7ff; color:#3730a3; }
.tag.ai { background:#f3e8ff; color:#7e22ce; }
.note { font-size:.8rem; color:#6b7280; }

.tipcard { background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:1.1rem; height:100%; }
.tipcard .ico { font-size:1.6rem; }
.tipcard b { display:block; margin:.3rem 0; color:var(--ink); }
.tipcard ul { padding-left:1.1rem; margin:.3rem 0 0; color:#374151; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    # First start on a fresh server (for example Streamlit Cloud): train the model once.
    from src.train import main as train_model
    with st.spinner("First start: training the ScamLens model (about 30 seconds)..."):
        train_model()

model = load_model()

# Session state defaults
defaults = {"page": "home", "draft": "", "result": None, "history": [],
            "use_lime": False, "n_variants": 3, "show_tech": True}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)
    # Re-assigning keeps widget values alive when their page is not shown
    st.session_state[k] = st.session_state[k]


# Helpers
def go_to(page):
    st.session_state["page"] = page


def analyze(text):
    """Run the full analysis, save it as the current result and in history."""
    text = text.strip()
    if not text:
        st.session_state["result"] = None
        return
    out = model.analyze(text)
    result = {
        "message": text, **out, "tactics": detect_tactics(text),
        "words": coefficient_explanation(model, text),
        "time": datetime.now().strftime("%b %d, %Y  %I:%M %p"),
    }
    st.session_state["result"] = result
    st.session_state["history"].insert(0, result)


def on_analyze_click():
    analyze(st.session_state["draft"])
    go_to("analyze")


def on_clear_click():
    st.session_state["draft"] = ""
    st.session_state["result"] = None


def load_example(text):
    st.session_state["draft"] = text


def verdict(risk, label=None):
    """Title, subtitle, colours and badge class for a risk score and predicted class."""
    if risk >= 0.6:
        return "Scam Detected!", "This message is likely a scam.", "#fef2f2", "#dc2626", "scam"
    if risk >= 0.3:
        return "Be Careful", "This message shows some warning signs.", "#fff7ed", "#ea580c", "warn"
    if label == "Spam":
        return ("Promotional Spam", "Unwanted advertising. Not detected as a scam.",
                "#fefce8", "#ca8a04", "spam")
    return "Looks Safe", "No strong scam signals were found.", "#f0fdf4", "#16a34a", "safe"


def ring_html(risk, color):
    pct = round(risk * 100)
    return (f"<div style='text-align:center'><div class='ring' "
            f"style='background:conic-gradient({color} {pct * 3.6}deg,#e5e7eb 0deg)'>"
            f"<div style='color:{color}'>{pct}%</div></div>"
            f"<small style='color:#6b7280'>Scam risk score</small></div>")


def contribution_bar(items, title):
    if not items:
        return None
    items = sorted(items, key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[w for w, _ in items], orientation="h",
        marker_color=["#dc2626" if v > 0 else "#16a34a" for _, v in items]))
    fig.update_layout(title=title, height=60 + 26 * len(items),
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="towards Safe  |  towards Scam",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def message_input(height=130):
    st.text_area("Message", key="draft", height=height, label_visibility="collapsed",
                 placeholder="Type or paste a message here...")


# Sidebar
with st.sidebar:
    st.markdown("<div class='brand'><span class='brand-badge'>🔍</span> ScamLens</div>",
                unsafe_allow_html=True)
    for pid, label, icon in PAGES:
        st.button(label, key=f"nav_{pid}", icon=icon, width="stretch",
                  type="primary" if st.session_state["page"] == pid else "secondary",
                  on_click=go_to, args=(pid,))
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("VectorX · AI Innovation Hackathon 2026")

page = st.session_state["page"]

# HOME
if page == "home":
    st.markdown("""
    <div class='hero'>
      <h1>Stay Safe from <span class='hl'>Scams</span></h1>
      <p>Analyze messages instantly. ScamLens tells you if a message is suspicious,
      <b>why</b>, and how much you can trust the answer.</p>
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 💬 Check a Message")
        st.caption("Paste a suspicious SMS or email below.")
        message_input()
        c1, c2 = st.columns([1, 3])
        c1.button("Analyze Message", icon=":material/search:", type="primary",
                  width="stretch", on_click=on_analyze_click)
        with c2:
            st.caption("Or try an example:")
            ex_cols = st.columns(len(EXAMPLES))
            for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
                col.button(name, key=f"ex_{name}", width="stretch",
                           on_click=load_example, args=(text,))

    st.markdown("<br>", unsafe_allow_html=True)
    features = [
        ("⚡", "Instant Detection", "Get results in seconds", "#fef2f2"),
        ("🧠", "Explainable AI", "See exactly which words matter", "#f0fdf4"),
        ("🛡️", "Trust Check", "Tests if the AI can be fooled", "#eeecff"),
        ("📚", "Stay Informed", "Learn common scam tricks", "#eff6ff"),
    ]
    for col, (ico, title, sub, bg) in zip(st.columns(4), features):
        col.markdown(f"<div class='feature' style='background:{bg}'><div class='ico'>{ico}</div>"
                     f"<b>{title}</b><small>{sub}</small></div>", unsafe_allow_html=True)

# ANALYZE
elif page == "analyze":
    st.markdown("## 🔎 Analyze Message")
    with st.container(border=True):
        message_input(height=110)
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("Analyze Again", icon=":material/refresh:", type="primary",
                  width="stretch", on_click=on_analyze_click)
        c2.button("Clear", icon=":material/delete:", width="stretch", on_click=on_clear_click)

    r = st.session_state["result"]
    if r is None:
        st.info("Type a message above and press **Analyze Again**.")
    else:
        title, sub, bg, color, _ = verdict(r["risk"], r["label"])
        icon = {"Scam Detected!": "⛔", "Be Careful": "⚠️", "Promotional Spam": "📢",
                "Looks Safe": "✅"}[title]
        cp = r["class_proba"]
        st.markdown(f"""
        <div class='banner' style='background:{bg};border:1px solid {color}33'>
          <div><h2 style='color:{color}'>{icon} {title}</h2><p>{sub}</p>
          <p style='font-size:.85rem;color:#6b7280'>AI text model: Legitimate {cp['Legitimate'] * 100:.0f}% ·
          Spam {cp['Spam'] * 100:.0f}% · Scam {cp['Scam'] * 100:.0f}% · Tactic rules matched: {len(r['tactics'])}</p></div>
          {ring_html(r['risk'], color)}
        </div>""", unsafe_allow_html=True)
        st.markdown("<p class='note'>The scam risk score combines the AI text model with the tactic "
                    "rules and is calibrated on our test data. It is a risk score, not a guaranteed "
                    "probability that a message is fraud.</p>", unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            if title == "Promotional Spam" and not r["tactics"]:
                st.markdown("#### Why is this flagged as spam?")
                st.markdown("<div class='reason' style='background:#fefce8;border-color:#fef08a'>"
                            "<span class='ico'>📢</span><div><b>Promotional wording</b>"
                            "<span class='tag ai'>AI model</span><small>The AI rates this "
                            f"{cp['Spam'] * 100:.0f}% likely to be advertising spam. No manipulation "
                            "tactics were found, so it is not treated as a scam.</small></div></div>",
                            unsafe_allow_html=True)
            elif r["risk"] >= 0.3 or r["tactics"]:
                heading = ("Why is this flagged as spam?" if title == "Promotional Spam"
                           else "Why is this suspicious?")
                st.markdown(f"#### {heading}")
                rows = []
                for t in r["tactics"]:
                    ico, head, desc = TACTIC_CARDS[t["tactic"]]
                    phrases = " ".join(f"<code>{html.escape(x)}</code>" for x in t["phrases"][:3])
                    rows.append(f"<div class='reason'><span class='ico'>{ico}</span><div>"
                                f"<b>{head}</b><span class='tag rule'>Pattern rule</span>"
                                f"<small>{desc}</small>{phrases}</div></div>")
                scam_words = [w for w, v in r["words"] if v > 0 and len(w) >= 3][:4]
                if scam_words:
                    words = " ".join(f"<code>{html.escape(w)}</code>" for w in scam_words)
                    rows.append("<div class='reason'><span class='ico'>🧠</span><div>"
                                "<b>AI flagged these words</b><span class='tag ai'>AI model</span>"
                                "<small>Words the model has "
                                f"learned are common in scams.</small>{words}</div></div>")
                st.markdown("".join(rows), unsafe_allow_html=True)
            else:
                st.markdown("#### Why does this look safe?")
                safe_words = [w for w, v in r["words"] if v < 0 and len(w) >= 4][:5]
                words = " ".join(f"<code>{html.escape(w)}</code>" for w in safe_words)
                st.markdown("<div class='reason safe'><span class='ico'>✅</span><div>"
                            "<b>No manipulation tactics</b><small>No urgency, threats, "
                            "rewards or suspicious requests were found.</small></div></div>"
                            "<div class='reason safe'><span class='ico'>🧠</span><div>"
                            "<b>Normal wording</b><small>The AI sees everyday language.</small>"
                            f"{words}</div></div>", unsafe_allow_html=True)

            st.markdown("#### Highlighted message")
            st.caption("Red words push towards scam, green words towards safe.")
            st.markdown(f"<div class='card' style='line-height:2'>"
                        f"{highlight_html(r['message'], r['words'])}</div>",
                        unsafe_allow_html=True)

        with right:
            if r["risk"] >= 0.3 or r["tactics"]:
                tips = [SAFETY_TIPS[t["tactic"]] for t in r["tactics"]] or [
                    "Do not click links or reply until you verify the sender."]
                if r["risk"] >= 0.3:
                    tips.append("Report the message to your bank or the relevant authority.")
                else:
                    tips.append("If you do not want these offers, block the sender.")
            elif title == "Promotional Spam":
                tips = ["This looks like advertising, not a scam.",
                        "If you did not sign up, block the sender or reply STOP only to known companies.",
                        "Never share personal or bank details in reply to an offer."]
            else:
                tips = ["It looks fine, but still never share passwords, PINs or OTPs.",
                        "If something feels off, contact the sender another way."]
            items = "".join(f"<li>{html.escape(t)}</li>" for t in tips)
            st.markdown(f"<div class='todo'><h4>🛡️ What should you do?</h4><ul>{items}</ul></div>",
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.button("Try What-If", icon=":material/science:", width="stretch",
                      on_click=go_to, args=("whatif",))
            c2.button("Check Stability", icon=":material/verified_user:", width="stretch",
                      on_click=go_to, args=("stability",))

        if st.session_state["show_tech"]:
            with st.expander("🔬 Technical details: word-level evidence"):
                fig = contribution_bar(r["words"], "Model contribution per word / phrase")
                if fig:
                    st.plotly_chart(fig, width="stretch")
                if st.session_state["use_lime"]:
                    with st.spinner("Running LIME..."):
                        st.plotly_chart(contribution_bar(lime_explanation(model, r["message"]),
                                                         "LIME explanation"), width="stretch")
                else:
                    st.caption("Turn on LIME in Settings for a second, model-agnostic explanation.")

# WHAT-IF
elif page == "whatif":
    st.markdown("## 🧪 What-If Lab")
    st.caption("Change the wording and see how the AI's decision changes.")
    r = st.session_state["result"]
    if r is None:
        st.info("Analyze a message first, then come back here.")
        st.button("Go to Home", icon=":material/home:", on_click=go_to, args=("home",))
    else:
        with st.form("whatif_form"):
            edited = st.text_area("Modified message", value=r["message"], height=110)
            submitted = st.form_submit_button("Compare", icon=":material/compare_arrows:",
                                              type="primary")
        if submitted or edited:
            out2 = model.analyze(edited)
            label2, risk2 = out2["label"], out2["risk"]
            t2 = detect_tactics(edited)
            c1, c2, c3 = st.columns(3)
            for col, name, risk, lab in [(c1, "Original", r["risk"], r["label"]),
                                         (c2, "Modified", risk2, label2)]:
                t, _, bg, color, _ = verdict(risk, lab)
                col.markdown(f"<div class='card' style='text-align:center'><b>{name}</b><br><br>"
                             f"{ring_html(risk, color)}<br><span style='color:{color};"
                             f"font-weight:700'>{t}</span></div>", unsafe_allow_html=True)
            with c3:
                st.metric("Change in scam risk", f"{risk2 * 100:.0f}%",
                          delta=f"{(risk2 - r['risk']) * 100:+.0f} pts", delta_color="inverse")
                before = {t["tactic"] for t in r["tactics"]}
                after = {t["tactic"] for t in t2}
                if before - after:
                    st.success("Removed: " + ", ".join(sorted(before - after)))
                if after - before:
                    st.error("Added: " + ", ".join(sorted(after - before)))

        st.markdown("#### Which single word matters most?")
        words = r["message"].split()
        if 1 < len(words) <= 60:
            variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
            risks = model.scam_risk(variants)
            rows = [{"Removed word": w, "New risk score": round(float(pw), 3),
                     "Change": round(float(pw) - r["risk"], 3)} for w, pw in zip(words, risks)]
            st.dataframe(pd.DataFrame(rows).sort_values("Change").head(8),
                         width="stretch", hide_index=True)

# STABILITY
elif page == "stability":
    st.markdown("## 🛡️ Stability Check")
    st.caption("Can the AI be fooled by typos, leetspeak or small rewording?")
    r = st.session_state["result"]
    if r is None:
        st.info("Analyze a message first, then come back here.")
        st.button("Go to Home", icon=":material/home:", on_click=go_to, args=("home",))
    else:
        st.markdown(f"<div class='card'><small>Testing:</small><br>{html.escape(r['message'])}</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        res = check_stability(model, r["message"], n_per_type=st.session_state["n_variants"])
        color = {"Stable": "#16a34a", "Somewhat fragile": "#ea580c", "Fragile": "#dc2626"}[res["verdict"]]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(ring_html(res["score"] / 100, color).replace("Scam risk score", "Stability"),
                    unsafe_allow_html=True)
        c2.metric("Verdict", res["verdict"])
        c3.metric("Label flips", f"{res['flip_rate'] * 100:.0f}%")
        c4.metric("Largest swing", f"{res['max_shift'] * 100:.1f} pts")
        if res["rows"]:
            df = pd.DataFrame(res["rows"])
            fig = go.Figure()
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="decision line")
            fig.add_hline(y=res["base_p"], line_color="#5b4cf0", annotation_text="original")
            fig.add_trace(go.Scatter(
                x=df["perturbation"], y=df["risk_score"], mode="markers",
                marker=dict(size=12, color=["#dc2626" if f else "#94a3b8"
                                            for f in df["label_flipped"]]),
                text=df["variant"], hovertemplate="%{text}<br>score=%{y:.3f}<extra></extra>"))
            fig.update_layout(height=340, yaxis_range=[0, 1], yaxis_title="Scam risk score",
                              margin=dict(l=10, r=10, t=20, b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            flipped = df[df["label_flipped"]]
            if len(flipped):
                st.warning(f"{len(flipped)} variant(s) fooled the AI. Red dots show them.")
                st.dataframe(flipped[["perturbation", "variant", "risk_score"]],
                             width="stretch", hide_index=True)
            else:
                st.success("No variant changed the AI's decision.")
            with st.expander("All variants"):
                st.dataframe(df, width="stretch", hide_index=True)

# HISTORY
elif page == "history":
    st.markdown("## 🕘 Message History")
    hist = st.session_state["history"]
    top_l, top_r = st.columns([4, 1])
    top_l.caption("Your recent message analyses in this session.")
    if hist:
        top_r.button("Clear History", icon=":material/delete:", width="stretch",
                     on_click=lambda: st.session_state.update(history=[]))
    if not hist:
        st.info("No messages analyzed yet.")
    else:
        with st.container(border=True):
            head = st.columns([5, 1.2, 1.2, 2, 0.8])
            for col, h in zip(head, ["Message", "Result", "Risk score", "Date & time", ""]):
                col.markdown(f"**{h}**")
            for i, h in enumerate(hist):
                title, _, _, _, cls = verdict(h["risk"], h.get("label"))
                short = {"scam": "Scam", "warn": "Careful", "spam": "Spam", "safe": "Safe"}[cls]
                row = st.columns([5, 1.2, 1.2, 2, 0.8])
                preview = h["message"][:70] + ("..." if len(h["message"]) > 70 else "")
                row[0].write(preview)
                row[1].markdown(f"<span class='badge {cls}'>{short}</span>", unsafe_allow_html=True)
                row[2].write(f"{h['risk'] * 100:.0f}%")
                row[3].caption(h["time"])
                if row[4].button("", key=f"open_{i}", icon=":material/visibility:",
                                 help="Open this result"):
                    st.session_state["result"] = h
                    st.session_state["draft"] = h["message"]
                    st.session_state["page"] = "analyze"
                    st.rerun()

# AWARENESS
elif page == "awareness":
    st.markdown("## 💡 Scam Awareness")
    st.caption("Learn to spot the most common scams.")
    cards = [
        ("🎣", "Phishing Scams", ["Fake login pages that steal passwords",
                                  "Links that look almost like the real site"]),
        ("🏦", "Fake Bank Alerts", ["Claims your account is blocked",
                                    "Asks you to verify card or OTP details"]),
        ("🎁", "Prize Scams", ["Says you won something you never entered",
                               "Asks for a fee to release the prize"]),
        ("💼", "Job Offer Scams", ["Too good to be true salary for easy work",
                                   "Asks for a registration or training fee"]),
        ("📦", "Delivery Scams", ["Parcel on hold, pay a small fee",
                                  "Tracking links from unknown numbers"]),
        ("🔗", "Suspicious Links", ["Short links like bit.ly hide the real address",
                                    "Misspelled domains such as paypa1.com"]),
    ]
    for start in range(0, len(cards), 3):
        for col, (ico, title, pts) in zip(st.columns(3), cards[start:start + 3]):
            items = "".join(f"<li>{p}</li>" for p in pts)
            col.markdown(f"<div class='tipcard'><div class='ico'>{ico}</div><b>{title}</b>"
                         f"<ul>{items}</ul></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### The 5 manipulation tactics ScamLens looks for")
    st.caption("These are detected with transparent pattern-matching rules, not a separately trained AI model.")
    for name, (ico, head, desc) in TACTIC_CARDS.items():
        st.markdown(f"<div class='reason'><span class='ico'>{ico}</span><div><b>{name}</b>"
                    f"<small>{desc} {SAFETY_TIPS[name]}</small></div></div>",
                    unsafe_allow_html=True)

# MODEL INSIGHTS
elif page == "insights":
    st.markdown("## 📊 Model Insights")
    st.caption("How ScamLens was trained, how well it works, and where its limits are.")
    mdir = ROOT / "models"
    m = json.loads((mdir / "metrics.json").read_text())
    cal = json.loads((mdir / "calibration.json").read_text())
    tv = json.loads((mdir / "tactic_validation.json").read_text())
    g = json.loads((mdir / "global_features.json").read_text())

    # Dataset
    st.markdown("#### Dataset")
    d = m["dataset"]
    c = st.columns(4)
    c[0].metric("Messages", f"{d['total']:,}")
    c[1].metric("Legitimate", f"{d['per_class']['Legitimate']:,}")
    c[2].metric("Spam (promotional)", f"{d['per_class']['Spam']:,}")
    c[3].metric("Scam (smishing)", f"{d['per_class']['Scam']:,}")
    st.caption(f"SMS Phishing Dataset (Mishra and Soni, Mendeley Data, 2022). After removing duplicates: "
               f"{d['train']:,} training and {d['test']:,} test messages (stratified 80/20).")

    # Text model
    st.markdown("#### 1. AI text model: Legitimate vs Spam vs Scam")
    pc = m["per_class"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    c[1].metric("Macro F1", f"{m['macro_f1']:.3f}")
    c[2].metric("Scam precision", f"{pc['Scam']['precision']:.3f}")
    c[3].metric("Scam recall", f"{pc['Scam']['recall']:.3f}")
    left, right = st.columns([1, 1])
    left.dataframe(pd.DataFrame([{"Class": k, "Precision": v["precision"], "Recall": v["recall"],
                                  "F1": v["f1-score"], "Test messages": int(v["support"])}
                                 for k, v in pc.items()]), width="stretch", hide_index=True)
    cm = m["confusion_matrix"]
    labels = ["Legitimate", "Spam", "Scam"]
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {x}" for x in labels], y=[f"True {x}" for x in labels],
                               text=cm, texttemplate="%{text}", colorscale="Purples", showscale=False))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_autorange="reversed", paper_bgcolor="rgba(0,0,0,0)")
    right.plotly_chart(fig, width="stretch")

    # Calibration
    st.markdown("#### 2. Scam risk score: learned and calibrated")
    st.write("The final score comes from a small logistic regression that combines the text model's "
             "scam and spam scores with the five tactic flags. Its weights are **learned from data** "
             "(on out-of-fold predictions, so no test data is used), not chosen by hand.")
    names = {"text_model_only": "AI text model only",
             "old_heuristic_formula": "Old hand-made formula",
             "calibrated_risk_score": "Calibrated risk score (used)"}
    sd = m["scam_detection"]
    st.dataframe(pd.DataFrame([{"Method": names[k], "ROC AUC (higher better)": v["roc_auc"],
                                "Brier score (lower better)": v["brier"],
                                "Calibration error (lower better)": v["ece"]} for k, v in sd.items()]),
                 width="stretch", hide_index=True)
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(dash="dash", color="#94a3b8")))
    for key, colr in [("old_heuristic_formula", "#ea580c"), ("calibrated_risk_score", "#5b4cf0")]:
        fig.add_trace(go.Scatter(x=cal[key]["mean_predicted"], y=cal[key]["actual_scam_rate"],
                                 mode="lines+markers", name=names[key], line=dict(color=colr)))
    fig.update_layout(title="Reliability diagram (test set)", height=340,
                      xaxis_title="Predicted risk", yaxis_title="Actual share of scams",
                      legend=dict(orientation="h", y=-0.3), margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    left.plotly_chart(fig, width="stretch")
    right.plotly_chart(contribution_bar(list(cal["combiner_weights"].items()),
                                       "Learned weight of each signal"), width="stretch")
    st.caption("Points close to the dashed line mean that when ScamLens says 80%, about 80% of such "
               "test messages really were scams. This holds for our dataset only.")

    # Tactic validation
    st.markdown("#### 3. Manipulation tactics: rule validation")
    st.write("Tactics are found with **transparent pattern-matching rules**, not a trained model, "
             "because no public dataset labels manipulation tactics. To check the rules are "
             "meaningful, we measure how often each one fires in each class:")
    st.dataframe(pd.DataFrame([{"Tactic": k, "Legitimate": f"{v['Legitimate'] * 100:.1f}%",
                                "Spam": f"{v['Spam'] * 100:.1f}%", "Scam": f"{v['Scam'] * 100:.1f}%"}
                               for k, v in tv.items()]), width="stretch", hide_index=True)
    st.caption("A useful rule fires often on scams and rarely on legitimate messages. This shows the "
               "rules separate the classes; it does not prove each match is a real manipulation attempt.")

    # Global words
    st.markdown("#### 4. Words the AI relies on most")
    c1, c2 = st.columns(2)
    c1.plotly_chart(contribution_bar([(pretty_token(w), v) for w, v in g["scam"][:15]],
                                     "Top scam indicators"), width="stretch")
    c2.plotly_chart(contribution_bar([(pretty_token(w), v) for w, v in g["legitimate"][:15]],
                                     "Top non-scam indicators"), width="stretch")

    # Limitations
    st.markdown("#### Limitations and design choices")
    st.markdown("""
- **Dataset:** English SMS only, collected online and partly built from older spam collections.
  Spam and scam are separate classes, but real local scams (Sinhala, Tamil, Singlish) are not covered,
  so real-world accuracy will be lower than the test scores.
- **Tactics are rules, not AI:** they are transparent and easy to audit, but can miss creative wording
  or fire on harmless messages that mention a bank or a prize.
- **Risk score is not a probability of fraud:** it is calibrated on our test split. On messages unlike
  our data, the number can be off. Treat it as a risk indicator.
- **Robustness:** the Stability Check shows the model can be fooled by leetspeak and spacing tricks.
""")

# SETTINGS
elif page == "settings":
    st.markdown("## ⚙️ Settings")
    with st.container(border=True):
        st.toggle("Show technical details on the Analyze page", key="show_tech")
        st.toggle("Run LIME explanation (slower, more thorough)", key="use_lime")
        st.slider("Stability check: variations per perturbation type", 1, 8, key="n_variants")
    st.caption("Settings last for this browser session.")