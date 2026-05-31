from __future__ import annotations

import json
from pathlib import Path

from src.review_engine import load_reviews, save_model, train_model


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "reviews_dataset.csv"
MODEL = ROOT / "models" / "reviews_model.joblib"
METRICS = ROOT / "models" / "reviews_metrics.json"


def main() -> None:
    frame = load_reviews(DATASET)
    model, metrics = train_model(frame)
    save_model(model, MODEL)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Reviews Analyzer model is ready")
    print(f"Dataset size: {metrics['dataset_size']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Model: {MODEL}")


if __name__ == "__main__":
    main()
