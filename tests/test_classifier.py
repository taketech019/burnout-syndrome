import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_classify_text_shape_with_text():
    from src.classifier import classify_text
    r = classify_text("우울하고 잠 안 와요")
    assert isinstance(r, dict)
    for k in ("ok", "status", "message", "backend", "classification", "scores", "raw_scores"):
        assert k in r, f"missing key: {k}"
    for k in ("depression", "anxiety", "addiction"):
        assert k in r["classification"]
        assert r["classification"][k] in (0, 1)


def test_classify_text_empty_input():
    from src.classifier import classify_text
    r = classify_text("")
    assert r["ok"] is False
    assert r["classification"] == {"depression": 0, "anxiety": 0, "addiction": 0}


def test_classify_text_whitespace_only():
    from src.classifier import classify_text
    r = classify_text("   \n  \t  ")
    assert r["ok"] is False
