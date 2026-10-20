# ScamLens MVP

**Explain the Scam, Not Just the Spam.** VectorX, AI Innovation Hackathon 2026, Challenge 05.

ScamLens classifies an SMS or email as Scam or Legitimate, explains which words drove the decision, names the manipulation tactics in it, lets you run what-if edits, and tells you how stable the prediction is.

## What is in this MVP

| Slide feature | Where it lives | How it works |
|---|---|---|
| AI classification | `src/train.py` | TF-IDF (1 and 2 word phrases) + Logistic Regression, class balanced |
| Word-level explainability | `src/explain.py` | Exact linear contributions (instant) + optional LIME |
| Manipulation tactic explainer | `src/tactics.py` | Regex rules for Urgency, Fear/Threat, Authority, Reward/Greed, Call to Action |
| What-if analysis | `app.py`, What-If tab | Edit the text, compare gauges, plus "remove one word" ranking |
| Stability checker | `src/stability.py` | 7 perturbation types (leetspeak, typos, synonyms...), flip rate, score 0 to 100 |
| Overall risk | `src/explain.py` | Model probability combined with tactic count: `1 - (1 - p) * 0.85 ** tactics` |
| Dashboard | `app.py` | Streamlit + Plotly, 4 tabs |
| Global XAI | Model Insights tab | Top model weights, metrics, confusion matrix |

Test set results on the SMS Spam Collection (5,171 unique messages, 80/20 stratified split): accuracy 0.990, precision 0.969, recall 0.954, F1 0.962, ROC AUC 0.997.

## Project structure

```
code/
  app.py                 Streamlit dashboard
  requirements.txt
  data/sms.tsv           dataset (label TAB message)
  models/                created by training
  src/
    preprocess.py        cleaning, placeholder tokens for links / numbers / money
    train.py             training + metrics + global features
    explain.py           prediction, word contributions, LIME, highlighting
    tactics.py           manipulation rules and safety tips
    stability.py         perturbations and stability score
  tests/test_scamlens.py
```

## Step by step setup

You need Python 3.10 or newer. Check with `python --version`.

### 1. Open a terminal in the project folder

```bash
cd scamlens
```

### 2. Create a virtual environment

Use a venv. On some systems `pip install lime` fails outside a venv because of an old setuptools; inside a fresh venv it installs cleanly.

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the libraries

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Get the dataset (skip if `data/sms.tsv` is already there)

It is included in the zip. If you need it again:

```bash
curl -L -o data/sms.tsv https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
```

On Windows PowerShell, use `Invoke-WebRequest -Uri <url> -OutFile data/sms.tsv`.

### 5. Train the model

```bash
python -m src.train
```

You should see the metrics printed and three files appear in `models/`. Run it as `python -m src.train` from the project root, not `python src/train.py`, otherwise the `src` imports fail.

### 6. Run the tests

```bash
python -m pytest -q
```

Expect `9 passed`. The tests include all five example messages from the slides.

### 7. Launch the dashboard

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501.

## Demo script for judges (about 3 minutes)

1. **Analyze tab**: pick "Bank threat" in the sidebar. Show the gauge, the red/green highlighted words, and the tactic cards (Urgency, Fear, Authority, Call to Action) with the exact phrases. Point at the safety guidance.
   Then type the slide message `Your account will be blocked immediately. Verify now!`. The model alone says only 44% scam (the SMS dataset has few phishing threats), but the tactic layer finds Urgency and Fear and lifts the overall risk to 60%. This is why ScamLens uses both layers.
2. **What-If tab**: replace the text with `Please review your account information.` (the slide example). The probability drops; the tactics removed are listed. Show the "remove one word" table.
3. **Stability tab**: type `Claim your free prize` (model: 91% scam). Show the score, flip rate and the scatter. Red dots are variants that fooled the model, for example `f.r.e.e`. Then type the slide's own example `Cl@im your fr3e prize`: the model drops to 47% and calls it Legitimate. This is the honest part: the AI shows its weak spots.
4. **Model Insights tab**: metrics, confusion matrix, the global top scam words.
5. Tick "Also run LIME" to show a model agnostic explanation that agrees with the linear one.

## How to extend (roadmap from the slides)

- **Robustness**: add a leetspeak normalizer in `src/preprocess.py` (map `3` to `e`, `@` to `a`, `0` to `o`), retrain, and show the stability score going up. Good "before vs after" slide.
- **Better data**: the SMS Spam Collection is spam, not purely scams. Add phishing / smishing samples in the same `label TAB message` format and retrain.
- **Multilingual (Sinhala, Tamil, Singlish)**: collect labelled messages, switch the TF-IDF to `analyzer="char_wb", ngram_range=(2, 5)` which handles non-Latin scripts and code mixing better, and add Sinhala / Tamil keyword lists to `tactics.py`. Report metrics per language.
- **Stronger model**: swap Logistic Regression for a fine-tuned DistilBERT / XLM-R. `lime_explanation` keeps working because LIME only needs `predict_proba`.

## Deploy (optional)

Push the folder to GitHub (include `models/` or train on startup), then deploy on Streamlit Community Cloud by pointing it at `app.py`.

## Known limitations

- Rule-based tactics can miss creative wording or fire on normal messages that mention a bank.
- Trained on English SMS from one public dataset, so accuracy on real local scams will be lower than the test score.
- Placeholder tokens mean the model reacts to "has a link" rather than whether the link is actually malicious.
