"""ScamLens: Explain the Scam, Not Just the Spam.

Streamlit dashboard. Run from the project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import (coefficient_explanation, highlight_html, lime_explanation,
                         overall_risk, predict, risk_level)
from src.preprocess import pretty_token
from src.stability import check_stability
from src.tactics import SAFETY_TIPS, detect_tactics, explain_in_plain_language

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "scamlens_model.joblib"

EXAMPLES = {
    "Bank threat": "URGENT: This is your bank's security team. Your account will be blocked within 24 hours. Verify your details now at www.secure-bank-verify.com",
    "Prize scam": "Congratulations! You have won a cash prize of $5000. Claim your free reward now, click here: bit.ly/claimprize",
    "Courier scam": "DHL: Your parcel is on hold due to unpaid customs fee. Pay immediately to avoid return: http://dhl-track-pay.info",
    "Normal message": "Hey, are we still meeting for lunch tomorrow at 12? Let me know if you want to change the place.",
    "Work message": "Hi team, the project review has moved to Thursday afternoon. Please update your slides before then.",
}

st.set_page_config(page_title="ScamLens", page_icon="🔍", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def gauge(p, title="Scam probability"):
    label, color = risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100, 1),
        number={"suffix": "%"},
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{color}'>{label}</span>"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 25], "color": "#dcfce7"},
                         {"range": [25, 50], "color": "#fef9c3"},
                         {"range": [50, 80], "color": "#ffedd5"},
                         {"range": [80, 100], "color": "#fee2e2"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=10))
    return fig


def contribution_bar(items, title):
    if not items:
        return None
    items = sorted(items, key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[w for w, _ in items], orientation="h",
        marker_color=["#dc2626" if v > 0 else "#16a34a" for _, v in items]))
    fig.update_layout(title=title, height=40 + 28 * len(items),
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="towards Legitimate  |  towards Scam")
    return fig


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first (see README).")
    st.stop()

model = load_model()

# Sidebar
st.sidebar.title("🔍 ScamLens")
st.sidebar.caption("Explain the Scam, Not Just the Spam")
example = st.sidebar.selectbox("Load an example", ["(type your own)"] + list(EXAMPLES))
use_lime = st.sidebar.checkbox("Also run LIME (slower)", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("VectorX · AI Innovation Hackathon 2026 · Challenge 05")

default_text = EXAMPLES.get(example, "")
if "message" not in st.session_state or example != st.session_state.get("last_example"):
    st.session_state["message"] = default_text
    st.session_state["last_example"] = example

st.title("ScamLens")
st.write("Paste an SMS or email. ScamLens tells you **if** it looks suspicious, **why**, "
         "and **how much you can trust** that answer.")

message = st.text_area("Message", key="message", height=120,
                       placeholder="Paste a message here...")

tab_analyze, tab_whatif, tab_stability, tab_model = st.tabs(
    ["🧠 Analyze", "🧪 What-If", "🛡️ Stability", "📊 Model Insights"])

if not message.strip():
    with tab_analyze:
        st.info("Enter a message above or pick an example from the sidebar.")
else:
    label, p = predict(model, message)
    tactics = detect_tactics(message)
    word_items = coefficient_explanation(model, message)
    risk = overall_risk(p, tactics)

    # Analyze tab
    with tab_analyze:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(gauge(risk, "Overall risk"), width="stretch")
            m1, m2 = st.columns(2)
            m1.metric("Model says", label, f"{p * 100:.0f}% scam", delta_color="off")
            m2.metric("Tactics found", len(tactics))
            st.caption("Overall risk combines the ML model with the tactic rules.")
        with c2:
            st.subheader("Why this result")
            st.markdown(explain_in_plain_language(tactics, p))
            st.markdown("**Highlighted message** (red pushes to Scam, green to Legitimate):")
            st.markdown(f"<div style='line-height:2;font-size:1.05em'>"
                        f"{highlight_html(message, word_items)}</div>",
                        unsafe_allow_html=True)

        st.subheader("Manipulation tactics")
        if tactics:
            cols = st.columns(min(len(tactics), 3))
            for i, t in enumerate(tactics):
                with cols[i % len(cols)]:
                    st.markdown(f"#### {t['icon']} {t['tactic']}")
                    st.caption(t["why"])
                    st.write(", ".join(f"`{x}`" for x in t["phrases"]))
        else:
            st.success("No common manipulation tactics detected.")

        st.subheader("Word-level evidence")
        fig = contribution_bar(word_items, "Model contribution per word / phrase")
        if fig:
            st.plotly_chart(fig, width="stretch")
        if use_lime:
            with st.spinner("Running LIME..."):
                lime_items = lime_explanation(model, message)
            st.plotly_chart(contribution_bar(lime_items, "LIME explanation"),
                            width="stretch")

        if risk >= 0.5 or tactics:
            st.subheader("Safety guidance")
            tips = [SAFETY_TIPS[t["tactic"]] for t in tactics] or [
                "Do not click links or reply until you verify the sender through an official channel."]
            for tip in tips:
                st.markdown(f"- {tip}")

    # What-If tab
    with tab_whatif:
        st.write("Edit the message and see how the model's decision changes.")
        edited = st.text_area("Modified message", value=message, height=120, key=f"edited_{abs(hash(message))}")
        if edited.strip():
            label2, p2 = predict(model, edited)
            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(gauge(p, "Original"), width="stretch")
            c2.plotly_chart(gauge(p2, "Modified"), width="stretch")
            with c3:
                st.metric("Change in scam probability", f"{p2 * 100:.1f}%",
                          delta=f"{(p2 - p) * 100:+.1f} pts", delta_color="inverse")
                st.write(f"Original: **{label}**  →  Modified: **{label2}**")
                t1 = {t["tactic"] for t in tactics}
                t2 = {t["tactic"] for t in detect_tactics(edited)}
                if t1 - t2:
                    st.write("Tactics removed: " + ", ".join(sorted(t1 - t2)))
                if t2 - t1:
                    st.write("Tactics added: " + ", ".join(sorted(t2 - t1)))
            st.markdown("**Modified message evidence:**")
            st.markdown(highlight_html(edited, coefficient_explanation(model, edited)),
                        unsafe_allow_html=True)

        st.markdown("---")
        st.write("**Remove one word at a time** (which single word matters most?)")
        words = message.split()
        if 1 < len(words) <= 60:
            rows = []
            for i, w in enumerate(words):
                _, pw = predict(model, " ".join(words[:i] + words[i + 1:]))
                rows.append({"removed word": w, "new scam probability": round(pw, 3),
                             "change": round(pw - p, 3)})
            df = pd.DataFrame(rows).sort_values("change")
            st.dataframe(df.head(10), width="stretch", hide_index=True)

    # Stability tab
    with tab_stability:
        st.write("ScamLens creates small variations of the message (typos, leetspeak, "
                 "casing, synonyms) and checks whether the prediction holds.")
        n = st.slider("Variations per perturbation type", 1, 8, 3)
        res = check_stability(model, message, n_per_type=n)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stability score", f"{res['score']:.0f} / 100")
        c2.metric("Verdict", res["verdict"])
        c3.metric("Label flip rate", f"{res['flip_rate'] * 100:.0f}%")
        c4.metric("Largest probability swing", f"{res['max_shift'] * 100:.1f} pts")
        if res["rows"]:
            df = pd.DataFrame(res["rows"])
            fig = go.Figure()
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="decision threshold")
            fig.add_hline(y=res["base_p"], line_color="#2563eb",
                          annotation_text="original")
            fig.add_trace(go.Scatter(
                x=df["perturbation"], y=df["scam_probability"], mode="markers",
                marker=dict(size=11, color=["#dc2626" if f else "#64748b"
                                            for f in df["label_flipped"]]),
                text=df["variant"], hovertemplate="%{text}<br>p=%{y:.3f}<extra></extra>"))
            fig.update_layout(height=360, yaxis_range=[0, 1], yaxis_title="Scam probability",
                              margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Red points = variants where the label flipped.")
            st.dataframe(df, width="stretch", hide_index=True)

# Model insights tab (works even without a message)
with tab_model:
    metrics_path = ROOT / "models" / "metrics.json"
    feats_path = ROOT / "models" / "global_features.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(key.upper() if key == "roc_auc" else key.title(), f"{m[key]:.3f}")
        cm = m["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Pred Legit", "Pred Scam"], y=["True Legit", "True Scam"],
            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix (test set)", height=320,
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    if feats_path.exists():
        g = json.loads(feats_path.read_text())
        st.subheader("Global explanation: strongest words in the model")
        c1, c2 = st.columns(2)
        c1.plotly_chart(contribution_bar([(pretty_token(w), v) for w, v in g["scam"][:15]],
                                         "Top scam indicators"), width="stretch")
        c2.plotly_chart(contribution_bar([(pretty_token(w), v) for w, v in g["legitimate"][:15]],
                                         "Top legitimate indicators"), width="stretch")
