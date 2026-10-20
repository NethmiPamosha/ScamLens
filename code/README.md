# ScamLens MVP

**Explain the Scam, Not Just the Spam.** VectorX, AI Innovation Hackathon 2026, Challenge 05.

ScamLens sorts an SMS into **Legitimate, Spam (promotional) or Scam (smishing)**, explains which words drove the decision, names the manipulation tactics it contains, lets you run what-if edits, and shows how stable the prediction is.

## What is in this MVP

| Feature | Where it lives | How it works |
|---|---|---|
| AI text classification | `src/train.py`, `src/model.py` | TF-IDF (1 and 2 word phrases) + multinomial Logistic Regression, 3 classes |
| Word-level explainability | `src/explain.py` | Exact linear contributions (instant) + optional LIME |
| Manipulation tactic detector | `src/tactics.py` | **Transparent pattern-matching rules** for Urgency, Fear/Threat, Authority, Reward/Greed, Call to Action |
| Scam risk score | `src/model.py` | **Learned** logistic regression combining the AI scores and the 5 tactic flags, trained on out-of-fold predictions |
| What-if analysis | `app.py`, What-If Lab | Edit the text and compare, plus a "remove one word" ranking |
| Stability checker | `src/stability.py` | 7 perturbation types, flip rate, score 0 to 100 |
| Dashboard | `app.py` | Streamlit + Plotly, 8 pages |
| Validation | Model Insights page | Per-class metrics, calibration comparison, tactic validation, limitations |

## Dataset

**SMS Phishing Dataset for Machine Learning and Pattern Recognition**, Sandhya Mishra and Devpriya Soni, Mendeley Data, 2022, doi:10.17632/f45bkkt8pr.1.

* 5,971 messages labelled ham (4,844), spam (489) and smishing (638).
* We map these to Legitimate, Spam and Scam. Keeping spam and scam separate matters: promotional spam is annoying, while smishing tries to steal money or data.
* Cleaning: 34 texts that appear with two different labels are dropped (ambiguous), then duplicates are removed. That leaves **5,797 messages**: 4,834 Legitimate, 435 Spam, 528 Scam.
* Split: stratified 80/20 with a fixed seed (4,637 train, 1,160 test).

## Results (test set, 1,160 messages)

| | Precision | Recall | F1 |
|---|---|---|---|
| Legitimate | 0.989 | 1.000 | 0.994 |
| Spam | 0.922 | 0.816 | 0.866 |
| Scam | 0.943 | 0.934 | 0.938 |

Accuracy 0.980, macro F1 0.933. No legitimate message was flagged as spam or scam.

### Scam risk score: why it is learned, not hand-made

| Method | ROC AUC (higher better) | Brier (lower better) | Calibration error (lower better) |
|---|---|---|---|
| AI text model only | 0.9951 | 0.0100 | 0.0094 |
| Old hand-made formula | 0.9974 | 0.0118 | 0.0249 |
| **Calibrated risk score (used)** | **0.9976** | **0.0095** | **0.0064** |

The combiner is a logistic regression over 7 inputs: the text model's scam and spam scores (as log-odds) and the 5 tactic flags. It is trained on 5-fold out-of-fold predictions from the training set, so the test set is never seen. Its learned weights are shown in the dashboard.

### Tactic rule validation (share of messages where each rule fires)

| Tactic | Legitimate | Spam | Scam |
|---|---|---|---|
| Urgency | 1.3% | 13.3% | 34.1% |
| Fear / Threat | 1.3% | 0.2% | 2.5% |
| Authority Impersonation | 0.6% | 2.3% | 4.0% |
| Reward / Greed | 1.8% | 31.0% | 47.0% |
| Suspicious Call to Action | 0.2% | 44.6% | 67.4% |
| Any tactic | 4.9% | 67.4% | 86.4% |

## Limitations and design choices (how to explain them to judges)

1. **Dataset.** The data is English SMS collected online, and part of it comes from older spam collections. Unlike the first version, spam and scam are now separate classes. It still does not cover Sinhala, Tamil or Singlish scams, so real-world accuracy will be lower than the test scores.
2. **Tactics are rules, not a trained AI model.** No public dataset labels manipulation tactics, so we use transparent pattern-matching rules that anyone can audit. We validate them by showing they fire far more often on scams (86%) than on legitimate messages (5%). That shows the rules separate the classes; it does not prove every match is a real manipulation attempt. Fear/Threat and Authority fire rarely in this dataset, so those rules have the weakest evidence. The AI part of ScamLens is the text classifier and the learned risk combiner.
3. **The risk score is not a probability of fraud.** It is a calibrated score: on our test split, messages scored around 80% were scams about 80% of the time. On messages unlike our data, that relationship may not hold, so the dashboard calls it a "scam risk score".
4. **Robustness.** The Stability Check shows the model can be fooled by leetspeak, missing letters and dotted words.

## Project structure

```
code/
  app.py                    Streamlit dashboard
  requirements.txt
  .streamlit/config.toml    theme
  data/sms_phishing.csv     dataset (LABEL, TEXT, URL, EMAIL, PHONE)
  models/                   created by training
  src/
    __init__.py             empty file, required
    preprocess.py           cleaning, placeholder tokens for links / numbers / money
    model.py                ScamLensModel: text model + calibrated combiner
    train.py                training, evaluation, calibration, tactic validation
    explain.py              word contributions, LIME, highlighting
    tactics.py              tactic rules and safety tips
    stability.py            perturbations and stability score
  tests/test_scamlens.py
```

## Setup

You need Python 3.10 or newer.

1. Open a terminal in the `scamlens` folder.
2. Create and activate a virtual environment:
   * Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   * Mac / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install: `python -m pip install --upgrade pip setuptools wheel` then `pip install -r requirements.txt`
4. Get the dataset into `data/sms_phishing.csv`:
   * Mac / Linux / Git Bash: `curl -L -o data/sms_phishing.csv https://raw.githubusercontent.com/arinargh/sms-spam-detector/main/Dataset_5971.csv`
   * PowerShell: `Invoke-WebRequest -Uri https://raw.githubusercontent.com/arinargh/sms-spam-detector/main/Dataset_5971.csv -OutFile data/sms_phishing.csv`
   * Or download it from the official Mendeley page: https://data.mendeley.com/datasets/f45bkkt8pr/1
5. Train: `python -m src.train`
6. Test: `python -m pytest -q` (expect `12 passed`)
7. Run: `streamlit run app.py`

## Demo script (about 3 minutes)

1. **Home**, click **Bank threat**, then **Analyze Message**. Result: Scam Detected, 96%. Point out the class breakdown, the "Pattern rule" and "AI model" tags that show which layer found what, and the advice box.
2. **Home**, click **Shop promotion**. Result: Promotional Spam, 25% scam risk. This shows ScamLens separates annoying advertising from real scams.
3. **What-If Lab** with the bank threat: change it to `This is your bank. Your monthly statement is now available in the app.` The risk drops from 96% to 28% and the removed tactics are listed.
4. **Stability Check** with `Claim your free prize` (risk 87%): score about 60, Somewhat fragile. Variants like `Claim your f.r.e.e prize` drop below 50% and fool the model. This is the honest part.
5. **Model Insights**: per-class results, the calibration table (learned score beats the hand-made formula), the tactic validation table, and the limitations section.

Note: the slide example `Please review your account information.` now scores 51%, because phrasing like this is common in real smishing in the dataset. Avoid it as a "safe" example.

## Future work

* Leetspeak and spacing normalizer, then show the stability score improving.
* Sinhala, Tamil and Singlish data, character n-grams, local tactic keywords, per-language evaluation.
* A small hand-labelled set of tactic examples to measure the rules' precision and recall directly, or to train a tactic classifier.
* A multilingual transformer (for example XLM-R). LIME, the combiner and the stability checker keep working because they only need `predict_proba`.
