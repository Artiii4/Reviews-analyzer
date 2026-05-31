from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ASPECTS = {
    "delivery": ["delivery", "arrived", "late", "package", "shipping", "courier", "order", "delivered"],
    "support": ["support", "staff", "team", "response", "answered", "request", "agent", "ticket", "chat"],
    "price": ["price", "cost", "money", "overpriced", "value", "charged"],
    "app": ["app", "application", "website", "interface", "system", "feature", "dashboard", "page", "account", "profile", "update"],
    "quality": ["quality", "product", "result", "reliable", "damaged", "broken", "scratched", "parts"],
    "payment": ["payment", "paid", "checkout", "transaction", "invoice", "refund", "receipt"],
}

MARKERS = {
    "positive": ["excellent", "great", "helpful", "satisfied", "fast", "reliable", "polite", "smooth", "convenient", "good", "professional", "pleasant", "easy"],
    "negative": ["bad", "slow", "broken", "failed", "poor", "wrong", "late", "damaged", "confusing", "frustrating", "unacceptable", "crashed", "ignored"],
    "neutral": ["shows", "contains", "available", "created", "submitted", "scheduled", "displays", "status", "registered", "assigned", "pending"],
}


def load_reviews(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {"sentiment", "stars", "review"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain sentiment, stars and review columns")
    frame = frame.dropna(subset=["sentiment", "stars", "review"]).copy()
    frame["sentiment"] = frame["sentiment"].astype(str).str.lower().str.strip()
    frame["review"] = frame["review"].astype(str).str.strip()
    frame["stars"] = frame["stars"].astype(int)
    frame = frame[frame["review"] != ""]
    return frame


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=1, max_features=11000, analyzer="word")),
            ("char", TfidfVectorizer(lowercase=True, ngram_range=(3, 5), min_df=1, max_features=9000, analyzer="char_wb")),
        ]
    )
    classifier = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=17), cv=3)
    return Pipeline([("features", features), ("classifier", classifier)])


def train_model(frame: pd.DataFrame) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        frame["review"],
        frame["sentiment"],
        test_size=0.22,
        random_state=17,
        stratify=frame["sentiment"],
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
        "dataset_size": int(len(frame)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "classes": list(model.named_steps["classifier"].classes_),
    }
    return model, metrics


def save_model(model: Pipeline, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_model(path: str | Path) -> Pipeline:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return joblib.load(source)


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    if not value:
        raise ValueError("Review text is empty")
    return value


def find_aspects(text: str) -> list[str]:
    lowered = normalize_text(text).lower()
    result = []
    for aspect, words in ASPECTS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words):
            result.append(aspect)
    return result or ["general"]


def estimate_stars(sentiment: str, confidence: float) -> int:
    if sentiment == "positive":
        return 5 if confidence >= 0.7 else 4
    if sentiment == "negative":
        return 1 if confidence >= 0.7 else 2
    return 3


def detect_urgency(text: str, sentiment: str, confidence: float) -> str:
    lowered = normalize_text(text).lower()
    triggers = ["urgent", "failed", "crashed", "broken", "unacceptable", "refund", "not working", "payment failed", "lost"]
    if sentiment == "negative" and (confidence >= 0.65 or any(item in lowered for item in triggers)):
        return "high"
    if sentiment == "negative":
        return "medium"
    return "normal"


def find_markers(text: str) -> dict[str, list[str]]:
    lowered = normalize_text(text).lower()
    groups = {}
    for name, words in MARKERS.items():
        matched = [word for word in words if re.search(rf"\b{re.escape(word)}\b", lowered)]
        if matched:
            groups[name] = matched
    return groups


def analyze_review(model: Pipeline, text: str) -> dict:
    value = normalize_text(text)
    sentiment = str(model.predict([value])[0])
    probabilities = model.predict_proba([value])[0]
    classes = list(model.named_steps["classifier"].classes_)
    distribution = {name: round(float(score), 4) for name, score in zip(classes, probabilities)}
    confidence = distribution[sentiment]
    return {
        "review": value,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": distribution,
        "aspects": find_aspects(value),
        "estimated_stars": estimate_stars(sentiment, confidence),
        "urgency": detect_urgency(value, sentiment, confidence),
        "markers": find_markers(value),
    }


def analyze_frame(model: Pipeline, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Column not found: {column}")
    result = []
    for value in frame[column].fillna("").astype(str):
        try:
            item = analyze_review(model, value)
            result.append(
                {
                    "review": item["review"],
                    "sentiment": item["sentiment"],
                    "confidence": item["confidence"],
                    "estimated_stars": item["estimated_stars"],
                    "urgency": item["urgency"],
                    "aspects": ", ".join(item["aspects"]),
                }
            )
        except ValueError:
            result.append(
                {
                    "review": value,
                    "sentiment": "empty",
                    "confidence": 0.0,
                    "estimated_stars": 0,
                    "urgency": "unknown",
                    "aspects": "",
                }
            )
    return pd.DataFrame(result)
