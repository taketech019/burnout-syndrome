import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_split_with_scale_at_end():
    from src.transcript_utils import split_transcript_and_scale
    text = """상담사: 안녕하세요. 오늘 어떠셨어요?
내담자: 잠을 못 자고 우울해요.
상담사: 1번 이전보다 너무 많이 먹거나 너무 적게 먹는다.
내담자: 2
상담사: 2번 기운이 없고 기분이 가라앉는다.
내담자: 1
상담사: 3번 죽고 싶은 생각이 든다.
내담자: 0"""
    body, scale = split_transcript_and_scale(text)
    assert "1번" not in body
    assert "기운이 없고 기분이 가라앉는다" not in body
    assert len(scale) == 3
    assert scale[0]["item"] == 1 and scale[0]["answer"] == 2
    assert scale[2]["item"] == 3 and scale[2]["answer"] == 0
    assert "잠을 못 자고 우울해요" in body


def test_split_without_scale_returns_original():
    from src.transcript_utils import split_transcript_and_scale
    text = "상담사: 안녕하세요\n내담자: 우울해요\n상담사: 언제부터요?\n내담자: 한 달 전부터."
    body, scale = split_transcript_and_scale(text)
    assert body == text
    assert scale == []


def test_split_single_item_not_treated_as_scale():
    """단일 번호 매김(예: '1번 과제')은 척도로 오인하지 않음."""
    from src.transcript_utils import split_transcript_and_scale
    text = "상담사: 1번 과제 어떠셨어요?\n내담자: 잘 했어요.\n상담사: 좋아요."
    body, scale = split_transcript_and_scale(text)
    assert body == text
    assert scale == []


def test_split_handles_speaker_tab_or_colon():
    """`상담사 : ` `상담사\\t` `상담사: ` 모두 매치."""
    from src.transcript_utils import split_transcript_and_scale
    text = ("내담자: 본 내용입니다.\n"
            "상담사 : 1번 질문 하나.\n"
            "내담자 : 0\n"
            "상담사 : 2번 질문 둘.\n"
            "내담자 : 1\n"
            "상담사 : 3번 질문 셋.\n"
            "내담자 : 2")
    body, scale = split_transcript_and_scale(text)
    assert "1번" not in body
    assert len(scale) == 3
