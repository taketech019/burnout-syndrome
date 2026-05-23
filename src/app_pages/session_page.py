"""F1 회기 분석 페이지 — transcript 입력 → KlueBERT 판별 + Gemini 28요인 분류 → 저장."""
from datetime import date

import streamlit as st

from src.classifier import classify
from src.factor_extractor import extract_factors
from src.storage import add_session, get_patient, list_sessions


def render() -> None:
    st.title("회기 분석")
    st.caption("상담 스크립트 → F1 1차(KlueBERT 판별) + F1 2차(Gemini 28요인 분류)")

    pid = st.session_state.get("current_patient_id")
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return
    patient = get_patient(pid)
    if not patient:
        st.error("선택된 내담자를 찾을 수 없습니다.")
        return

    st.markdown(f"**내담자**: {patient['alias']} ({patient['gender']}/{patient['age']}/{patient['region']})")

    st.info(
        "**가드레일**: MVP는 데모 데이터 전용입니다. 실제 환자 식별정보(이름·전화·주소)를 입력하지 마세요. "
        "발화자는 `상담사: ...` / `내담자: ...` 형식으로 줄바꿈해 주세요."
    )

    session_date = st.date_input("회기일", value=date.today())
    transcript = st.text_area(
        "상담 스크립트",
        height=400,
        placeholder="상담사: 안녕하세요. 오늘 어떻게 지내셨어요?\n내담자: ...",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        primary_disease = st.selectbox(
            "2차 분류 우선 질환 (증상요인 라벨 세트)",
            ["depression", "anxiety", "addiction"],
            help="발화별 28요인 분류에서 증상요인 28개 중 어떤 질환 하위세트를 사용할지 선택.",
        )
    with col2:
        run_btn = st.button("분석 실행", type="primary", use_container_width=True)

    if not run_btn:
        st.divider()
        _render_history(pid)
        return

    if not transcript.strip():
        st.error("상담 스크립트가 비어 있습니다.")
        return

    # F1 1차 — Gemma 4 31B 0~3 정도값 (KlueBERT 변별력 부족 — vibe-coding-report.md 참조)
    with st.spinner("1차 판별 (Gemma 4 31B)..."):
        cls = classify(transcript)
    st.markdown("### 1차 판별 결과")
    if "error" in cls:
        st.error(cls["error"])
    else:
        levels = cls.get("level") or {}
        cols = st.columns(3)
        for i, label in enumerate(("anxiety", "depression", "addiction")):
            level = levels.get(label)
            if level is not None:
                cols[i].metric(label, f"{level} / 3", help=f"binary={cls.get(label, 0)}")
            else:
                cols[i].metric(label, cls.get(label, 0))
        if cls.get("_note"):
            st.caption(f"ℹ️ {cls['_note']}")
        if cls.get("_source"):
            st.caption(f"판별 소스: `{cls['_source']}`")

    if cls.get("is_normal"):
        st.info("정상 판별 — 2차 분석을 건너뜁니다. 필요 시 그래도 진행할 수 있습니다.")

    # F1 2차 — Gemini 28요인 분류
    with st.spinner("2차 발화별 28+20+5+11 라벨링 (Gemini)..."):
        factors = extract_factors(transcript, primary_disease=primary_disease)

    st.markdown(f"### 2차 분류 결과 (Gemini — primary: **{primary_disease}**)")
    if "error" in factors:
        st.error(factors["error"])
    else:
        st.success(f"발화 {factors.get('utterance_count', 0)}개 라벨링 완료")
        freq = factors.get("frequency", {})
        cols = st.columns(4)
        cat_names = {
            "symptom_factor": "증상요인",
            "risk_factor": "위험요인",
            "improvement_factor": "개선요인",
            "intervention_factor": "개입요인",
        }
        for i, (cat, items) in enumerate(freq.items()):
            top = sorted(items, key=lambda x: x.get("count", 0), reverse=True)[:3]
            with cols[i]:
                st.markdown(f"**{cat_names.get(cat, cat)} 상위 3**")
                for it in top:
                    st.write(f"- {it['label']}: {it['count']}회")

    # 저장
    s = add_session(
        patient_id=pid,
        session_date=str(session_date),
        transcript=transcript,
        classifier_result=cls,
        factor_result=factors,
    )
    st.success(f"회기 저장 완료 (ID: `{s['id']}`)")


def _render_history(pid: str) -> None:
    sessions = list_sessions(pid)
    if not sessions:
        st.caption("저장된 회기가 없습니다.")
        return
    st.markdown(f"### 저장된 회기 ({len(sessions)}개)")
    for s in sessions[:5]:
        cls = s.get("classifier", {})
        binary = ", ".join(f"{k}={cls.get(k, 0)}" for k in ("anxiety", "depression", "addiction"))
        n_utt = s.get("factors", {}).get("utterance_count", 0)
        st.markdown(f"- **{s['session_date']}** (ID: `{s['id']}`) — 1차: {binary} · 2차 발화 {n_utt}개")
