import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_factor_keys_length_28():
    from src.factor_extractor import FACTOR_KEYS
    assert len(FACTOR_KEYS) == 28


def test_factor_categories_match_user_design():
    from src.factor_extractor import FACTOR_CATEGORIES, FACTOR_KEYS
    counts = {}
    for k in FACTOR_KEYS:
        counts[FACTOR_CATEGORIES[k]] = counts.get(FACTOR_CATEGORIES[k], 0) + 1
    assert counts == {"우울": 9, "우울/위험": 1, "불안": 8, "중독": 7, "중독/기능": 3}


def test_factor_keys_korean():
    from src.factor_extractor import FACTOR_KEYS
    for k in FACTOR_KEYS:
        assert any("가" <= c <= "힣" for c in k), f"{k} 한글 키 아님"


def test_extract_factors_empty_input():
    from src.factor_extractor import extract_factors, FACTOR_KEYS
    r = extract_factors("", {"depression": 0, "anxiety": 0, "addiction": 0})
    assert r["ok"] is False
    assert set(r["factors"].keys()) == set(FACTOR_KEYS)
    assert all(v == 0 for v in r["factors"].values())


def test_extract_factors_shape_with_text():
    from src.factor_extractor import extract_factors, FACTOR_KEYS
    r = extract_factors(
        "내담자: 너무 우울하고 잠도 안 와요. 자살하고 싶어요.",
        {"depression": 1, "anxiety": 0, "addiction": 0},
    )
    for k in ("ok", "status", "message", "backend", "factors"):
        assert k in r
    assert set(r["factors"].keys()) == set(FACTOR_KEYS)
    for v in r["factors"].values():
        assert isinstance(v, int) and 0 <= v <= 3
