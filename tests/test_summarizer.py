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


def test_summarize_returns_brief_field():
    """짧은 1단락 요약 brief 필드 존재."""
    from src.summarizer import summarize
    r = summarize("상담사: 안녕\n내담자: 우울하고 잠 안 와요.\n상담사: 언제부터?\n내담자: 한 달.")
    assert "brief" in r
    assert isinstance(r["brief"], str)


def test_summarize_empty_brief_field():
    """빈 입력에도 brief 키 존재."""
    from src.summarizer import summarize
    r = summarize("")
    assert "brief" in r
    assert r["brief"] == ""


def test_summarize_calls_koalpaca_even_short(monkeypatch):
    """짧은 입력에서도 KoAlpaca를 호출해야 한다."""
    from src import summarizer
    calls = {"koalpaca": 0, "gemma": 0}

    def fake_ka(t):
        calls["koalpaca"] += 1
        return {
            "sections": {k: "" for k in summarizer._EMPTY_SECTIONS},
            "brief": "", "ok": True, "source": "koalpaca",
        }

    def fake_gemma(t):
        calls["gemma"] += 1
        return {
            "sections": {k: f"gemma_{k}" for k in summarizer._EMPTY_SECTIONS},
            "brief": "g_brief", "ok": True, "source": "gemma_fallback",
        }

    monkeypatch.setattr(summarizer, "_try_koalpaca", fake_ka)
    monkeypatch.setattr(summarizer, "_try_gemma_fallback", fake_gemma)
    r = summarizer.summarize("내담자: 짧은 입력입니다.")
    assert calls["koalpaca"] == 1, "KoAlpaca 호출 안 됨"
    assert r["koalpaca_attempted"] is True


def test_summarize_koalpaca_partial_uses_gemma_full(monkeypatch):
    """KoAlpaca 1섹션만 채우면 무시하고 Gemma 4섹션 사용."""
    from src import summarizer
    ka_sections = {"symptoms": "ka_phq9_catalog", "risk_factors": "",
                   "improvement_factors": "", "intervention_factors": ""}
    gemma_sections = {"symptoms": "g_sym", "risk_factors": "g_risk",
                      "improvement_factors": "g_imp", "intervention_factors": "g_int"}
    monkeypatch.setattr(
        summarizer, "_try_koalpaca",
        lambda t: {"sections": ka_sections, "brief": "", "ok": True, "source": "koalpaca"},
    )
    monkeypatch.setattr(
        summarizer, "_try_gemma_fallback",
        lambda t: {"sections": gemma_sections, "brief": "g_brief", "ok": True, "source": "gemma_fallback"},
    )
    r = summarizer.summarize("내담자: 테스트")
    assert r["source"] == "gemma", f"KoAlpaca partial은 무시되고 Gemma 사용해야: source={r['source']}"
    assert r["sections"]["symptoms"] == "g_sym"
    assert r["sections"]["risk_factors"] == "g_risk"
    assert r["sections"]["improvement_factors"] == "g_imp"
    assert r["sections"]["intervention_factors"] == "g_int"
    assert r["koalpaca_attempted"] is True
    assert r["koalpaca_sections_filled"] == 1
    assert r["gemma_sections_filled"] == 4


def test_summarize_koalpaca_3_sections_also_uses_gemma(monkeypatch):
    """KoAlpaca가 3섹션 채워도 4섹션 미만이면 신뢰 못 해서 Gemma 사용."""
    from src import summarizer
    ka_sections = {"symptoms": "ka", "risk_factors": "ka",
                   "improvement_factors": "ka", "intervention_factors": ""}
    gemma_sections = {k: f"g_{k}" for k in summarizer._EMPTY_SECTIONS}
    monkeypatch.setattr(
        summarizer, "_try_koalpaca",
        lambda t: {"sections": ka_sections, "brief": "", "ok": True, "source": "koalpaca"},
    )
    monkeypatch.setattr(
        summarizer, "_try_gemma_fallback",
        lambda t: {"sections": gemma_sections, "brief": "g_brief", "ok": True, "source": "gemma_fallback"},
    )
    r = summarizer.summarize("내담자: 테스트")
    assert r["source"] == "gemma"
    assert r["sections"]["symptoms"] == "g_symptoms"
    assert r["koalpaca_sections_filled"] == 3


def test_summarize_koalpaca_full_no_gemma_call(monkeypatch):
    """KoAlpaca가 4섹션 다 채우면 Gemma 호출 안 함."""
    from src import summarizer
    full_sections = {k: f"ka_{k}" for k in summarizer._EMPTY_SECTIONS}
    gemma_called = [False]

    monkeypatch.setattr(
        summarizer, "_try_koalpaca",
        lambda t: {"sections": full_sections, "brief": "ka_brief", "ok": True, "source": "koalpaca"},
    )

    def gemma_spy(t):
        gemma_called[0] = True
        return None

    monkeypatch.setattr(summarizer, "_try_gemma_fallback", gemma_spy)
    r = summarizer.summarize("내담자: 충분히 긴 텍스트입니다. " * 100)
    assert r["source"] == "koalpaca"
    assert r["koalpaca_sections_filled"] == 4
    assert gemma_called[0] is False, "KoAlpaca 완전 성공 시 Gemma 호출되면 안 됨"


def test_summarize_koalpaca_none_falls_back_to_gemma(monkeypatch):
    """KoAlpaca endpoint 미설정/실패 시 Gemma만."""
    from src import summarizer
    monkeypatch.setattr(summarizer, "_try_koalpaca", lambda t: None)
    monkeypatch.setattr(
        summarizer, "_try_gemma_fallback",
        lambda t: {"sections": {k: "g" for k in summarizer._EMPTY_SECTIONS},
                   "brief": "g_brief", "ok": True, "source": "gemma_fallback"},
    )
    r = summarizer.summarize("내담자: 짧은 입력입니다.")
    assert r["source"] == "gemma_only"
    assert r["koalpaca_attempted"] is False
    assert r["gemma_sections_filled"] == 4


def test_summarize_strips_scale_before_model(monkeypatch):
    """척도 평가 응답은 모델 입력에서 제외되어야 한다."""
    from src import summarizer
    captured_inputs = []

    def fake_ka(t):
        captured_inputs.append(("ka", t))
        return None

    def fake_gemma(t):
        captured_inputs.append(("gemma", t))
        return {
            "sections": {k: "g" for k in summarizer._EMPTY_SECTIONS},
            "brief": "g_brief", "ok": True, "source": "gemma_fallback",
        }

    monkeypatch.setattr(summarizer, "_try_koalpaca", fake_ka)
    monkeypatch.setattr(summarizer, "_try_gemma_fallback", fake_gemma)
    text = ("상담사: 오늘 어떠셨어요?\n"
            "내담자: 잠을 잘 못 자요.\n"
            "상담사: 1번 이전보다 너무 많이 먹거나 적게 먹는다.\n"
            "내담자: 2\n"
            "상담사: 2번 기분이 가라앉는다.\n"
            "내담자: 1\n"
            "상담사: 3번 죽고 싶다.\n"
            "내담자: 0")
    summarizer.summarize(text)
    # KoAlpaca / Gemma 어느 쪽이든 모델 입력에 척도가 들어가면 안 됨
    for _, inp in captured_inputs:
        assert "1번 이전보다" not in inp
        assert "기분이 가라앉는다" not in inp
        # 본문은 살아 있어야
        assert "잠을 잘 못 자요" in inp
