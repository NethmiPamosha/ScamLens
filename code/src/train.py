"""Train the ScamLens classifier: TF-IDF + Logistic Regression.

Run from the project root:
    python -m src.train

Outputs (in models/):
    scamlens_model.joblib   the full sklearn pipeline (vectorizer + classifier)
    metrics.json            accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    global_features.json    top words pushing towards scam and towards legitimate
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sms.tsv"
MODEL_DIR = ROOT / "models"


def load_data(path=DATA_PATH):
    """Load the SMS dataset. Expected format: label<TAB>message, labels ham/spam."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    df = df.dropna()
    df = df.drop_duplicates(subset="text")
    # 1 = scam / suspicious, 0 = legitimate
    df["y"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return df


def build_pipeline():
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),      # single words and two word phrases ("click here")
        min_df=2,                # ignore words seen only once
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = LogisticRegression(
        C=10,
        class_weight="balanced",  # scam messages are the minority class
        max_iter=2000,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} messages ({df['y'].sum()} scam, {(df['y'] == 0).sum()} legitimate)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

    model = build_pipeline()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(json.dumps(metrics, indent=2))

    # Global explainability: the largest logistic regression weights.
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    global_features = {
        "scam": [[vocab[i], round(float(coefs[i]), 3)] for i in order[::-1][:25]],
        "legitimate": [[vocab[i], round(float(coefs[i]), 3)] for i in order[:25]],
    }

    joblib.dump(model, MODEL_DIR / "scamlens_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "global_features.json").write_text(json.dumps(global_features, indent=2))
    print(f"Saved model and reports to {MODEL_DIR}")


if __name__ == "__main__":
    main()
