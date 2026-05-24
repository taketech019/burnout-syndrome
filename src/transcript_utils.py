"""src/transcript_utils.py — transcript 정제 유틸.

척도 영역 detect/분리: transcript 마지막에 "상담사: N번 …" + "내담자: <int>"
패턴이 연속 2회 이상이면 척도 평가 블록으로 판정, 본문과 분리.

분류·28요인·요약 모듈이 모델 호출 전에 호출해서 PHQ-9 같은 척도 응답이
상담 내용으로 오인되는 것을 막는다.
"""
import re

# "상담사: 5번 ..." 또는 "상담사\t5번 ..." 매치
_SCALE_QUESTION = re.compile(
    r"^[\s]*상담사[\s:.\t]+\s*(\d+)\s*번\s+(.+?)\s*$",
    re.MULTILINE,
)
# "내담자: 2" 단일 정수 응답 (한 자리)
_SCALE_ANSWER = re.compile(
    r"^[\s]*내담자[\s:.\t]+\s*(\d)\s*$",
    re.MULTILINE,
)


def split_transcript_and_scale(text: str) -> tuple[str, list[dict]]:
    """transcript 끝 척도 평가 영역 분리.

    반환:
      body: 척도 제거된 본문 (분석 대상)
      scale: [{"item": int, "question": str, "answer": int}, ...] (없으면 [])
    """
    if not text or not text.strip():
        return text, []

    q_matches = list(_SCALE_QUESTION.finditer(text))
    a_matches = list(_SCALE_ANSWER.finditer(text))
    if len(q_matches) < 2:
        return text, []

    # 질문과 그 직후 응답 페어링 (질문/응답 사이에 다른 질문이 없어야 함)
    pairs: list[tuple[re.Match, re.Match]] = []
    for q in q_matches:
        for a in a_matches:
            if a.start() > q.end():
                in_between = [
                    qq for qq in q_matches
                    if q.end() < qq.start() < a.start()
                ]
                if not in_between:
                    pairs.append((q, a))
                break

    if len(pairs) < 2:
        return text, []

    first_q_start = pairs[0][0].start()
    last_a_end = pairs[-1][1].end()

    scale = []
    for q, a in pairs:
        try:
            item = int(q.group(1))
            question = q.group(2).strip()
            answer = int(a.group(1))
            scale.append({"item": item, "question": question, "answer": answer})
        except (ValueError, IndexError):
            continue

    if not scale:
        return text, []

    body = text[:first_q_start].rstrip()
    # 본문이 비면 척도 분리하지 않고 원본 반환 (안전 fallback)
    if not body:
        return text, scale

    # 척도 블록 이후 남은 텍스트도 있을 수 있음 (마무리 인사 등)
    tail = text[last_a_end:].strip()
    if tail:
        body = body + "\n\n" + tail

    return body, scale
