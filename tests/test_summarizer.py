import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_summarize_empty_input():
    from src.summarizer import summarize
    r = summarize("")
    assert r["ok"] is False
    assert r["text"] == ""
    assert r["sections"] == {
        "symptoms": "", "risk_factors": "", "improvement_factors": "", "intervention_factors": ""
    }


def test_summarize_short_input_uses_gemma_fallback():
    """5000자 미만은 KoAlpaca skip → Gemma 폴백 시도."""
    from src.summarizer import summarize
    r = summarize("상담사: 안녕\n내담자: 너무 우울하고 잠을 못 자요. 자살하고 싶어요.\n"
                  "상담사: 언제부터요?\n내담자: 한 달 전부터 술도 매일 마셔요.")
    for k in ("ok", "status", "message", "text", "sections", "source"):
        assert k in r, f"missing key: {k}"
    assert isinstance(r["text"], str)
    assert isinstance(r["sections"], dict)
    for s in ("symptoms", "risk_factors", "improvement_factors", "intervention_factors"):
        assert s in r["sections"]
