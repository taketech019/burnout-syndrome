"""src/classifier.py — F1 1단계: KlueBERT 3-binary 판별 (HF Spaces /predict 호출).

⚠️ 현재 상태: KlueBERT 우회 모드 (BYPASS)
배포된 KlueBERT 모델이 입력과 무관하게 동일한 회귀값을 출력 (변별력 0,
2026-05-23 검증). 모델 재학습 전까지 1차 판별을 우회하고 모든 입력을 양성으로
간주하여 2차 Gemini 28요인 분석이 실제 분류 수행.

원래 Space 호출 코드는 아래 _call_kluebert_space()로 보존 — 모델 재학습 후
classify() 본문을 거기로 교체하면 1차 판별 복원.
"""
import requests

from config import KLUEBERT_ENDPOINT_URL, KLUEBERT_API_KEY

_LABELS = ("anxiety", "depression", "addiction")
_TIMEOUT = 90

# F1 1차 우회 여부. False로 바꾸면 다시 KlueBERT Space 호출.
KLUEBERT_BYPASS = True


def _empty(extra: dict | None = None) -> dict:
    out = {k: 0 for k in _LABELS} | {"is_normal": True}
    if extra:
        out.update(extra)
    return out


def classify(text: str) -> dict:
    """F1 1단계: 정신질환 3-binary 판별. 반환: {anxiety, depression, addiction, is_normal}.

    BYPASS 모드: 모든 입력 양성 (is_normal=False) → 2차 Gemini 28요인이 실제 판별.
    NORMAL 모드(KLUEBERT_BYPASS=False): KlueBERT Space 호출 결과 사용.
    """
    if KLUEBERT_BYPASS:
        return {
            "anxiety": 1,
            "depression": 1,
            "addiction": 1,
            "is_normal": False,
            "_note": "F1 stage 1 bypassed — KlueBERT 모델 변별력 부족, 2차 Gemini가 실제 분류",
        }
    return _call_kluebert_space(text)


def _call_kluebert_space(text: str) -> dict:
    """KlueBERT HF Space /predict 호출 (모델 재학습 후 사용)."""
    if not KLUEBERT_ENDPOINT_URL:
        return _empty({"error": "KLUEBERT_ENDPOINT_URL 미설정"})

    url = KLUEBERT_ENDPOINT_URL.rstrip("/") + "/predict"
    try:
        resp = requests.post(
            url,
            json={"text": text},
            headers={
                "X-API-Key": KLUEBERT_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {k: int(data.get(k, 0)) for k in _LABELS}
        result["is_normal"] = all(v == 0 for v in result.values())
        return result
    except requests.exceptions.ConnectionError:
        return _empty({"error": "KlueBERT Space에 연결할 수 없습니다."})
    except requests.exceptions.Timeout:
        return _empty({"error": f"응답 시간 초과 ({_TIMEOUT}초). Space cold-start 가능성 — 잠시 후 재시도."})
    except Exception as e:
        return _empty({"error": str(e)})
