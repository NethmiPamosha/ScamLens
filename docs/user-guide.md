# ScamLens User Guide

ScamLens checks a text message and tells you three things: **is it a scam, why, and how much you can trust that answer.**

## Quick start

1. Open the live demo, or run the app locally (see the [code README](../code/README.md)).
2. On **Home**, paste a message into the box, or click an example button.
3. Click **Analyze Message**.

## Pages

### Home

* Paste an SMS or email text and click **Analyze Message**.
* Example buttons fill the box with ready-made messages: Bank threat, Prize scam, Courier scam, Shop promotion and Normal message.

### Analyze Message

The result page has five parts.

**1. Verdict banner and risk ring**

| Verdict | When it appears | Colour |
|---|---|---|
| Scam Detected! | Scam risk score of 60% or more | Red |
| Be Careful | Scam risk score from 30% to 60% | Orange |
| Promotional Spam | Score under 30% and the AI's top class is Spam | Yellow |
| Looks Safe | Score under 30% and the AI's top class is Legitimate | Green |

Under the title, the banner shows the AI text model's three class probabilities (Legitimate, Spam, Scam) and how many tactic rules matched.

**2. Why is this suspicious?** Each reason has a tag showing which part of ScamLens found it:
* **Pattern rule:** a manipulation tactic found by a transparent rule, with the exact phrases that matched.
* **AI model:** words that the trained model has learned are common in scams.

**3. Highlighted message:** red words push towards Scam, green words push away from it. Darker means stronger.

**4. What should you do?** Practical advice that matches the tactics found.

**5. Technical details** (expandable): a bar chart of each word's contribution, and an optional LIME chart (turn it on in Settings).

### What-If Lab

Edit the analyzed message and click **Compare**. You see the original and modified risk side by side, the change in points, and which tactics were removed or added.

Below that, **Which single word matters most?** removes each word one at a time and shows how the risk changes.

### Stability Check

ScamLens makes small changes to the message that keep its meaning (leetspeak, swapped or missing letters, dotted words, casing, punctuation, synonyms) and re-scores each one.

* **Stability score (0 to 100):** higher means the result does not change easily.
* **Verdict:** Stable (80 or more), Somewhat fragile (55 to 80) or Fragile (under 55).
* **Label flips:** the share of variants that crossed the 50% line.
* **Red dots** on the chart are variants that fooled the model.

### History

Every analysis in the current browser session is listed with its result badge, risk score and time. Click the eye icon to reopen a result. **Clear History** removes the list. History is not saved after the tab is closed.

### Tips & Awareness

Cards describing six common scam types, and the five manipulation tactics ScamLens looks for, each with a safety tip.

### Model Insights

Dataset size, per-class results, confusion matrix, the calibration comparison, learned weights of the risk combiner, tactic validation, the most important words, and the limitations.

### Settings

* **Show technical details** on the Analyze page.
* **Run LIME explanation** (slower).
* **Stability check variations** per perturbation type (1 to 8).

## How to read the scam risk score

The score combines the AI text model and the tactic rules. It is calibrated on the test data: messages scored around 80% were scams about 80% of the time **in our dataset**. It is a risk indicator, not a guaranteed probability that a message is fraud. A low score does not prove a message is safe.

## Safety reminder

ScamLens supports your judgement; it does not replace it. Never share passwords, PINs or OTPs, and verify any request through the organisation's official website or phone number.