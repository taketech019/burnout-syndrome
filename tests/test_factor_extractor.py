import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_factor_keys_length_28():
    from src.factor_extractor import FACTOR_KEYS, FACTOR_LABELS
    assert len(FACTOR_KEYS) == 28
    assert all(k in FACTOR_LABELS for k in FACTOR_KEYS)


def test_extract_factors_empty_input():
    from src.factor_extractor import extract_factors, FACTOR_KEYS
    r = extract_factors("", {"depression": 0, "anxiety": 0, "addiction": 0}, backend="gemini_api")
    assert r["ok"] is False
    assert all(v == 0 for v in r["factors"].values())
    assert set(r["factors"].keys()) == set(FACTOR_KEYS)
    assert len(r["factors"]) == 28


def test_extract_factors_shape_with_text():
    from src.factor_extractor import extract_factors, FACTOR_KEYS
    cls = {"depression": 1, "anxiety": 0, "addiction": 0}
    r = extract_factors("내담자: 너무 우울하고 잠도 안 와요. 자살하고 싶어요.", cls, backend="gemini_api")
    assert isinstance(r, dict)
    for k in ("ok", "status", "message", "backend", "factors"):
        assert k in r
    assert set(r["factors"].keys()) == set(FACTOR_KEYS)
    for v in r["factors"].values():
        assert isinstance(v, int) and 0 <= v <= 3
