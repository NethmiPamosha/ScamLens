"""Train ScamLens.
 
Run from the project root:
    python -m src.train
 
Dataset: SMS Phishing Dataset for Machine Learning and Pattern Recognition
(Mishra and Soni, Mendeley Data, 2022, doi:10.17632/f45bkkt8pr.1).
Labels: ham, spam, smishing  ->  Legitimate, Spam, Scam.
 
Steps:
1. Clean the data (drop conflicting duplicates and repeats).
2. Split 80 / 20 (stratified, fixed seed).
3. Train the text model (TF-IDF + multinomial Logistic Regression).
4. Get out-of-fold predictions on the training set (5-fold).
5. Train the combiner on those out-of-fold predictions + tactic flags.
6. Evaluate on the untouched test set: classes, calibration, tactics.
 
Outputs in models/:
    scamlens_model.joblib, metrics.json, global_features.json,
    calibration.json, tactic_validation.json
"""
 
import json
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
 
from src.model import (CLASSES, FEATURE_NAMES, SCAM, TACTIC_NAMES, ScamLensModel,
                       combiner_features, tactic_flags)
from src.preprocess import clean_text
 
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sms_phishing.csv"
MODEL_DIR = ROOT / "models"
LABEL_MAP = {"ham": 0, "spam": 1, "smishing": 2}
SEED = 42
 
 
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["text"] = df["TEXT"].astype(str).str.strip()
    df["label"] = df["LABEL"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABEL_MAP)]
    # The same text with two different labels is ambiguous, so drop every copy.
    n_labels = df.groupby("text")["label"].transform("nunique")
    df = df[n_labels == 1]
    df = df.drop_duplicates(subset="text")
    df["y"] = df["label"].map(LABEL_MAP)
    return df[["text", "y"]].reset_index(drop=True)
 
 
def build_text_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=3000)),
    ])
 
 
def expected_calibration_error(y_true, p, n_bins=10):
    """Average gap between predicted risk and the real scam rate, weighted by bin size."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y_true[mask].mean())
    return float(ece)
 
 
def score_block(y_bin, p):
    return {
        "roc_auc": round(roc_auc_score(y_bin, p), 4),
        "brier": round(brier_score_loss(y_bin, p), 4),
        "ece": round(expected_calibration_error(y_bin, p), 4),
    }
 
 
def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    counts = df["y"].value_counts().sort_index().tolist()
    print(f"Loaded {len(df)} messages: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts)))
 
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["y"].values, test_size=0.2, random_state=SEED, stratify=df["y"])
 
    # 1) text model
    text_model = build_text_model().fit(X_train, y_train)
 
    # 2) combiner trained on out-of-fold predictions (no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_text_model(), X_train, y_train, cv=cv, method="predict_proba")
    combiner = LogisticRegression(C=1.0, max_iter=2000)
    combiner.fit(combiner_features(oof, X_train), (y_train == SCAM).astype(int))
 
    model = ScamLensModel(text_model, combiner)
 
    # 3) evaluation on the untouched test set
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    y_bin = (y_test == SCAM).astype(int)
    p_model = proba[:, SCAM]
    p_risk = model.scam_risk(X_test)
    n_tactics = tactic_flags(X_test).sum(axis=1)
    p_heuristic = 1 - (1 - p_model) * 0.85 ** n_tactics  # old hand-made formula, for comparison
 
    report = classification_report(y_test, pred, target_names=CLASSES, output_dict=True)
    metrics = {
        "dataset": {"total": len(df), "train": len(X_train), "test": len(X_test),
                    "per_class": dict(zip(CLASSES, counts))},
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {c: {k: round(report[c][k], 4) for k in ["precision", "recall", "f1-score", "support"]}
                      for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "scam_detection": {
            "text_model_only": score_block(y_bin, p_model),
            "old_heuristic_formula": score_block(y_bin, p_heuristic),
            "calibrated_risk_score": score_block(y_bin, p_risk),
        },
    }
    print(json.dumps(metrics, indent=2))
 
    # Reliability diagram data
    calib = {}
    for name, p in [("text_model_only", p_model), ("old_heuristic_formula", p_heuristic),
                    ("calibrated_risk_score", p_risk)]:
        frac, mean_pred = calibration_curve(y_bin, p, n_bins=10, strategy="quantile")
        calib[name] = {"mean_predicted": mean_pred.round(4).tolist(), "actual_scam_rate": frac.round(4).tolist()}
    calib["combiner_weights"] = dict(zip(FEATURE_NAMES, combiner.coef_[0].round(3).tolist()))
    calib["combiner_intercept"] = round(float(combiner.intercept_[0]), 3)
 
    # Tactic validation: how often each rule fires in each class (whole dataset)
    flags = tactic_flags(df["text"].tolist())
    validation = {}
    for j, name in enumerate(TACTIC_NAMES):
        validation[name] = {c: round(float(flags[df["y"].values == k, j].mean()), 4)
                            for k, c in enumerate(CLASSES)}
    any_flag = flags.max(axis=1)
    validation["Any tactic"] = {c: round(float(any_flag[df["y"].values == k].mean()), 4)
                                for k, c in enumerate(CLASSES)}
    print(json.dumps(validation, indent=2))
 
    # Global explanation for the Scam class (scam weight minus average of other classes)
    vocab = model.vectorizer.get_feature_names_out()
    W = model.classifier.coef_
    scam_dir = W[SCAM] - W[[0, 1]].mean(axis=0)
    order = scam_dir.argsort()
    global_features = {
        "scam": [[vocab[i], round(float(scam_dir[i]), 3)] for i in order[::-1][:25]],
        "legitimate": [[vocab[i], round(float(scam_dir[i]), 3)] for i in order[:25]],
    }
 
    joblib.dump(model, MODEL_DIR / "scamlens_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "calibration.json").write_text(json.dumps(calib, indent=2))
    (MODEL_DIR / "tactic_validation.json").write_text(json.dumps(validation, indent=2))
    (MODEL_DIR / "global_features.json").write_text(json.dumps(global_features, indent=2))
    print(f"Saved model and reports to {MODEL_DIR}")
 
 
if __name__ == "__main__":
    main()
 

