"""CounsHelper — Streamlit 진입점.

F1~F5 통합: 5페이지 (내담자 관리 / 상담내역 기록·추가 / 분석 대시보드 / AI 보고서 / 챗봇)

- 사이드바: src.db.list_patients() 동적 로드, 검색·선택
- 분석 실행 시 run_analysis(script, patient_id, session_date) → DB 3행(classifier/factors/summary) 저장
- 대시보드: 0~3 정도값 카드 + 28요인 가로 막대 + 회기별 추이(DB 실데이터) + HIRA 카드
- 보고서: .md / .docx / .pdf 다운로드 (sections 직접 사용, 재파싱 없음)
- 챗봇: src.rag.answer_query() 실연결, 출처 표시
"""
import json
import os
from datetime import date, datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import db
from src.classifier import classify_text
from src.factor_extractor import FACTOR_KEYS, FACTOR_LABELS, extract_factors
from src.hira import lookup as hira_lookup
from src.rag import answer_query, healthcheck as rag_healthcheck
from src.report import build_docx, build_md, build_pdf
from src.summarizer import summarize


# ── 설정 ──────────────────────────────────────────────────────────────────────

APP_NAME = "CounsHelper — 상담 기록 분석 & 보고서 자동화 플랫폼"
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "gemma")
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "gemini_api")
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "koalpaca_api")

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

# DB 초기화 (idempotent)
db.init_db()


# ── 분석 파이프라인 ───────────────────────────────────────────────────────────

DEFAULT_DIALOGUE = pd.DataFrame({
    "화자": ["상담사", "내담자", "상담사", "내담자"],
    "발화": [
        "오늘은 어떤 이야기를 나누고 싶으세요?",
        "요즘 잠을 잘 못 자고, 아침에 일어나기가 너무 힘들어요.",
        "수면 문제는 언제부터 시작되었나요?",
        "회사 일이 많아진 뒤부터 계속 피곤하고 불안해요. 출근 전부터 가슴이 답답합니다.",
    ],
})


def build_dialogue_text(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        speaker = str(row.get("화자", "")).strip()
        utterance = str(row.get("발화", "")).strip()
        if speaker and utterance and utterance.lower() != "nan":
            lines.append(f"{speaker}: {utterance}")
    return "\n".join(lines)


def soften_diagnostic_expression(text: str) -> str:
    """진단 단정 표현 방지용 후처리."""
    repl = {
        "우울증입니다": "우울 관련 호소가 확인됩니다",
        "불안장애입니다": "불안 관련 호소가 확인됩니다",
        "중독입니다": "중독 관련 호소가 확인됩니다",
        "진단됩니다": "가능성이 표시됩니다",
        "확진": "라벨상 표시",
    }
    for s, d in repl.items():
        text = text.replace(s, d)
    return text


def run_analysis(script: str, patient_id: str, session_date: str) -> Dict[str, Any]:
    """F1 1차 + F1 2차 + F3 요약 → SQLite 저장 (sessions + analyses 3행)."""
    cls_result = classify_text(script)
    fact_result = extract_factors(script, cls_result["classification"], backend=FACTOR_BACKEND)
    summ_result = summarize(script)
    summ_result["text"] = soften_diagnostic_expression(summ_result["text"])

    sess = db.add_session(patient_id, session_date, script)
    db.add_analysis(sess["id"], "classifier", cls_result["backend"], cls_result)
    db.add_analysis(sess["id"], "factors", fact_result["backend"], fact_result)
    db.add_analysis(sess["id"], "summary", summ_result["source"], summ_result)

    return {
        "session_id": sess["id"],
        "classifier": cls_result,
        "factors": fact_result,
        "summary": summ_result,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_factor_dataframe(factors: Dict[str, int]) -> pd.DataFrame:
    rows = []
    cat_map = (
        {k: "우울/증상" for k in FACTOR_KEYS[:10]}
        | {k: "불안" for k in FACTOR_KEYS[10:14]}
        | {k: "중독" for k in FACTOR_KEYS[14:18]}
        | {k: "상담사 개입" for k in FACTOR_KEYS[18:27]}
        | {FACTOR_KEYS[27]: "변화/기타"}
    )
    for k in FACTOR_KEYS:
        rows.append({
            "요인코드": k,
            "요인": FACTOR_LABELS.get(k, k),
            "카테고리": cat_map.get(k, "기타"),
            "점수": int(factors.get(k, 0)),
        })
    return pd.DataFrame(rows)


# ── Session State ─────────────────────────────────────────────────────────────


def init_session_state() -> None:
    defaults = {
        "page": "상담내역 기록·추가",
        "selected_client": None,
        "client_search": "",
        "dialogue_rows": DEFAULT_DIALOGUE.copy(),
        "chat_history": [],
        "analysis_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_page(page_name: str) -> None:
    st.session_state.page = page_name


# ── 사이드바 ──────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.title("CounsHelper")
        st.caption("상담 기록 분석 & 보고서 자동화")

        st.info(
            f"분류: `{CLASSIFIER_BACKEND}` · 28요인: `{FACTOR_BACKEND}` · 요약: `{SUMMARIZER_BACKEND}`"
        )

        st.divider()

        patients = db.list_patients()
        if not patients:
            st.warning("등록된 내담자가 없습니다.")
            st.caption("'내담자 관리' 페이지에서 신규 등록하세요.")
            st.session_state.selected_client = None
        else:
            keyword = st.text_input(
                "내담자 검색",
                value=st.session_state.client_search,
                placeholder="alias / 성별 / 지역",
            ).strip()
            st.session_state.client_search = keyword

            filtered = patients
            if keyword:
                k = keyword.lower()
                filtered = [
                    p for p in patients
                    if any(k in str(p.get(f, "")).lower()
                           for f in ("alias", "gender", "region", "note"))
                ]

            if not filtered:
                st.info("검색 결과 없음.")
                st.session_state.selected_client = None
            else:
                options = {
                    f"{p['alias']} ({p['gender']}/{p['age']}/{p['region']})": p["id"]
                    for p in filtered
                }
                default_idx = 0
                if st.session_state.selected_client:
                    for i, pid in enumerate(options.values()):
                        if pid == st.session_state.selected_client:
                            default_idx = i
                            break
                label = st.selectbox("내담자 선택", list(options.keys()), index=default_idx)
                st.session_state.selected_client = options[label]

        st.divider()
        st.caption("MVP Demo · 본 시스템은 상담사의 임상적 판단을 대체하지 않습니다.")


# ── 헤더 + 네비 ───────────────────────────────────────────────────────────────


def render_header() -> None:
    st.markdown(f"## {APP_NAME}")
    pid = st.session_state.selected_client
    if pid:
        p = db.get_patient(pid)
        if p:
            tag = (
                f"**{p['alias']}** · {p['age']}세 {p['gender']} · {p['region']}"
                + (f" · {p['note']}" if p.get("note") else "")
            )
            st.caption(tag)


def render_main_nav() -> None:
    nav = [
        ("내담자 관리", "nav_patients"),
        ("상담내역 기록·추가", "nav_records"),
        ("분석 대시보드", "nav_dashboard"),
        ("AI 보고서", "nav_report"),
        ("챗봇", "nav_chat"),
    ]
    cols = st.columns(len(nav))
    for col, (label, key) in zip(cols, nav):
        with col:
            if st.button(
                label,
                key=key,
                use_container_width=True,
                type="primary" if st.session_state.page == label else "secondary",
            ):
                go_page(label)
                st.rerun()
    st.divider()


# ── 페이지 1: 내담자 관리 ─────────────────────────────────────────────────────


def render_patients_page() -> None:
    st.subheader("내담자 관리")
    st.caption("MVP 데모 전용 — 실제 식별정보(실명·연락처) 입력 금지.")

    tab_list, tab_add = st.tabs(["목록", "신규 등록"])

    with tab_list:
        patients = db.list_patients()
        if not patients:
            st.info("등록된 내담자가 없습니다. '신규 등록' 탭에서 추가하세요.")
        for p in patients:
            with st.expander(f"**{p['alias']}** — {p['gender']}/{p['age']}/{p['region']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"ID: `{p['id']}`")
                    st.write(f"등록일: {p['created_at']}")
                    if p.get("note"):
                        st.write(f"메모: {p['note']}")
                    n_sess = len(db.list_sessions(p["id"]))
                    st.write(f"회기 수: **{n_sess}회**")
                with col2:
                    exp = db.export_patient(p["id"])
                    st.download_button(
                        "JSON 내보내기",
                        data=json.dumps(exp, ensure_ascii=False, indent=2),
                        file_name=f"{p['alias']}_{p['id']}.json",
                        mime="application/json",
                        key=f"exp_{p['id']}",
                    )
                with col3:
                    if st.button("삭제", key=f"del_{p['id']}", type="secondary"):
                        db.delete_patient(p["id"])
                        st.rerun()

    with tab_add:
        with st.form("add_patient_form"):
            alias = st.text_input("익명 식별자 (alias)", placeholder="예: 내담자A, P-001")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("성별", ["여성", "남성", "기타"])
            with c2:
                age = st.number_input("연령", min_value=10, max_value=100, value=30)
            with c3:
                region = st.text_input("지역", placeholder="예: 서울")
            note = st.text_area("메모 (선택)", height=80)
            if st.form_submit_button("등록"):
                if not alias.strip():
                    st.error("alias는 필수입니다.")
                else:
                    p = db.add_patient(alias, gender, age, region, note)
                    st.success(f"등록 완료: {p['alias']} (ID: {p['id']})")
                    st.rerun()


# ── 페이지 2: 상담내역 기록·추가 ──────────────────────────────────────────────


def render_record_page() -> None:
    pid = st.session_state.selected_client
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return

    st.subheader("상담내역 기록·추가")

    st.session_state.dialogue_rows = st.data_editor(
        st.session_state.dialogue_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "화자": st.column_config.SelectboxColumn(
                "화자", options=["상담사", "내담자"], required=True
            ),
            "발화": st.column_config.TextColumn("발화", width="large"),
        },
        key="dialogue_editor",
    )

    script = build_dialogue_text(st.session_state.dialogue_rows)

    st.markdown("#### 미리보기 (모델 입력)")
    st.text_area("스크립트", value=script, height=180, label_visibility="collapsed", disabled=True)

    sess_date = st.date_input("회기일", value=date.today())

    c1, c2 = st.columns([0.25, 0.75])
    with c1:
        if st.button("AI 분석 실행", type="primary", use_container_width=True):
            if not script.strip():
                st.warning("발화를 입력하세요.")
            else:
                with st.spinner("F1 1차 + 2차 + F3 요약 + DB 저장..."):
                    st.session_state.analysis_result = run_analysis(
                        script=script, patient_id=pid, session_date=str(sess_date)
                    )
                sid = st.session_state.analysis_result["session_id"]
                st.success(f"분석 + DB 저장 완료. session_id={sid}")
                go_page("분석 대시보드")
                st.rerun()

    st.divider()
    st.markdown("#### 저장된 회기")
    sessions = db.list_sessions(pid)
    if not sessions:
        st.caption("저장된 회기 없음.")
    else:
        for s in sessions[:10]:
            cls = db.get_latest_analysis(s["id"], "classifier")
            cls_str = "(분석 없음)"
            if cls and cls["payload"].get("classification"):
                c = cls["payload"]["classification"]
                cls_str = (
                    f"dep={c.get('depression', 0)} "
                    f"anx={c.get('anxiety', 0)} "
                    f"add={c.get('addiction', 0)}"
                )
            transcript_preview = (s.get("transcript") or "")[:60].replace("\n", " ")
            st.write(f"- **{s['session_date']}** (id=`{s['id']}`) — {cls_str}")
            st.caption(transcript_preview + ("..." if len(s.get("transcript", "")) > 60 else ""))


# ── 페이지 3: 분석 대시보드 ───────────────────────────────────────────────────


def render_dashboard() -> None:
    st.subheader("분석 대시보드")

    pid = st.session_state.selected_client
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 분석된 회기가 없습니다.")
        return

    options = {f"{s['session_date']} (id={s['id']})": s for s in sessions}
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]

    cls = db.get_latest_analysis(s["id"], "classifier")
    fact = db.get_latest_analysis(s["id"], "factors")

    if not (cls and fact):
        st.warning("이 회기에 분석 결과 없음. '상담내역 기록·추가'에서 AI 분석을 먼저 실행하세요.")
        return

    classification = cls["payload"].get("classification", {})
    scores = cls["payload"].get("scores", {})
    factors = fact["payload"].get("factors", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "우울 (0~3)",
        f"{int(scores.get('depression', 0))} / 3",
        delta="양성" if classification.get("depression") else "정상",
    )
    c2.metric(
        "불안 (0~3)",
        f"{int(scores.get('anxiety', 0))} / 3",
        delta="양성" if classification.get("anxiety") else "정상",
    )
    c3.metric(
        "중독 (0~3)",
        f"{int(scores.get('addiction', 0))} / 3",
        delta="양성" if classification.get("addiction") else "정상",
    )
    c4.metric("백엔드", cls["payload"].get("backend", "?"))

    st.caption("주의: 모델 출력 참고값. 임상 진단/표준화 검사 점수로 단정하지 않음.")

    st.divider()
    st.markdown("#### 28요인 점수")
    df = build_factor_dataframe(factors)
    fig = px.bar(
        df.sort_values("점수", ascending=True),
        x="점수", y="요인", color="카테고리",
        orientation="h", range_x=[0, 3], height=720,
    )
    st.plotly_chart(fig, use_container_width=True)

    if factors.get("suicidal", 0) > 0:
        st.error("⚠️ 자해/자살 관련 라벨이 표시되었습니다. 상담사가 별도 안전 평가를 수행하세요.")

    st.divider()
    st.markdown("#### 회기별 핵심 요인 추이")
    _render_trend_chart(pid)

    st.divider()
    st.markdown("#### HIRA 인구통계 비교")
    p = db.get_patient(pid)
    primary = (
        "depression" if classification.get("depression")
        else "anxiety" if classification.get("anxiety")
        else "addiction"
    )
    h = hira_lookup(p, primary)
    if h["available"]:
        st.info(h["summary_text"])
    else:
        st.caption(h["summary_text"])


def _render_trend_chart(patient_id: str) -> None:
    """전체 회기의 핵심 요인 5개 시계열 (db.analyses 'factors' stage)."""
    sessions = sorted(
        db.list_sessions(patient_id), key=lambda s: s.get("session_date", "")
    )
    if len(sessions) < 2:
        st.caption("회기가 2개 이상 누적되면 추이가 표시됩니다.")
        return

    rows = []
    for s in sessions:
        f = db.get_latest_analysis(s["id"], "factors")
        if not f:
            continue
        factors = f["payload"].get("factors", {})
        for key in (
            "depressive_mood", "anxiety", "sleep_disturbance", "fatigue", "suicidal"
        ):
            rows.append({
                "회기일": s["session_date"],
                "요인": FACTOR_LABELS.get(key, key),
                "점수": int(factors.get(key, 0)),
            })

    if not rows:
        st.caption("회기에 분석 결과가 없습니다.")
        return

    tdf = pd.DataFrame(rows)
    fig = px.line(
        tdf, x="회기일", y="점수", color="요인",
        markers=True, range_y=[0, 3], height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 페이지 4: AI 보고서 ───────────────────────────────────────────────────────


def render_report() -> None:
    st.subheader("AI 보고서")
    pid = st.session_state.selected_client
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 회기가 없습니다.")
        return

    options = {f"{s['session_date']} (id={s['id']})": s for s in sessions}
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]
    p = db.get_patient(pid)

    summ = db.get_latest_analysis(s["id"], "summary")
    if not summ:
        st.warning("이 회기에 요약 결과 없음. 상담내역 페이지에서 AI 분석을 먼저 실행하세요.")
        return

    sections = summ["payload"].get("sections", {})
    base_text = summ["payload"].get("text", "")
    source = summ["payload"].get("source", "?")

    st.caption(f"요약 소스: `{source}`")

    edited = st.text_area("보고서 본문 (편집 가능)", value=base_text, height=420)
    next_plan = st.text_area(
        "다음 회기 계획 (수동 작성)",
        height=100,
        placeholder="목표 / 기법 / 과제 ...",
    )

    st.divider()
    st.markdown("#### 다운로드")
    cols = st.columns(3)

    md_bytes = build_md(p, s, sections, next_plan)
    cols[0].download_button(
        ".md 다운로드", md_bytes,
        file_name=f"report_{s['id']}.md", mime="text/markdown",
        use_container_width=True,
    )

    try:
        docx_bytes = build_docx(p, s, sections, next_plan, [])
        cols[1].download_button(
            ".docx 다운로드", docx_bytes,
            file_name=f"report_{s['id']}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception as e:
        cols[1].error(f".docx 실패: {e}")

    pdf_bytes = build_pdf(p, s, sections)
    if pdf_bytes:
        cols[2].download_button(
            ".pdf 다운로드", pdf_bytes,
            file_name=f"report_{s['id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        cols[2].caption(".pdf 비활성\n(weasyprint Windows GTK 의존성)")


# ── 페이지 5: RAG 챗봇 ────────────────────────────────────────────────────────


def render_chatbot() -> None:
    st.subheader("RAG 챗봇")
    st.caption(
        "Gemma 4 31B + KoSBERT + ChromaDB "
        "(AI Hub 라벨링 · 윤리규정 · 임상 가이드라인 · HIRA)"
    )

    status = rag_healthcheck()
    cols = st.columns(2)
    cols[0].metric("LLM API", "✓" if status.get("llm") else "✗", help=status.get("llm_model", ""))
    cols[1].metric("ChromaDB", "✓" if status.get("chroma") else "✗")

    if not status.get("llm"):
        st.warning("GEMINI_API_KEY 미설정 — .env 확인.")
        return
    if not status.get("chroma"):
        st.warning("RAG 인덱스 미생성. `python -m src.rag.ingest` 실행.")
        return

    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("#### 빠른 질문")
    quick = [
        "상담심리사의 정신장애 진단에 대한 윤리 규정은?",
        "수면 문제와 불안을 동반한 내담자에게 추천되는 개입은?",
        "자살 사고가 있는 내담자에게 상담사가 즉시 해야 할 것은?",
    ]
    qc = st.columns(len(quick))
    for col, q in zip(qc, quick):
        with col:
            if st.button(q, use_container_width=True, key=f"qq_{hash(q)}"):
                _handle_rag(q)
                st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("출처"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src['title']}`")
                        st.caption(src.get("desc", ""))

    prompt = st.chat_input("질문을 입력하세요.")
    if prompt:
        _handle_rag(prompt)
        st.rerun()


def _handle_rag(prompt: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    result = answer_query(prompt, k=5)
    if result.get("error"):
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"⚠️ {result['error']}",
            "sources": [],
        })
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": [
                {"title": s["source"], "desc": s["snippet"]}
                for s in result["sources"]
            ],
        })


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    init_session_state()
    render_sidebar()
    render_header()
    render_main_nav()

    page = st.session_state.page
    if page == "내담자 관리":
        render_patients_page()
    elif page == "상담내역 기록·추가":
        render_record_page()
    elif page == "분석 대시보드":
        render_dashboard()
    elif page == "AI 보고서":
        render_report()
    elif page == "챗봇":
        render_chatbot()


if __name__ == "__main__":
    main()
