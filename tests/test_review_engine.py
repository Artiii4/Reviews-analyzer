import pandas as pd
import pytest

from src.review_engine import analyze_frame, analyze_review, find_aspects, load_reviews, train_model


@pytest.fixture(scope="module")
def model():
    frame = load_reviews("data/reviews_dataset.csv")
    trained, _ = train_model(frame)
    return trained


def test_negative_review_urgency(model):
    result = analyze_review(model, "the app crashed and payment failed again")
    assert result["sentiment"] == "negative"
    assert result["urgency"] in {"high", "medium"}


def test_aspect_detection_support():
    assert "support" in find_aspects("support answered my request quickly")


def test_frame_analysis(model):
    frame = pd.DataFrame({"review": ["great support and fast delivery", "the app is broken"]})
    result = analyze_frame(model, frame, "review")
    assert len(result) == 2
    assert "sentiment" in result.columns


def test_empty_review_error(model):
    with pytest.raises(ValueError):
        analyze_review(model, " ")
