"""CounsHelper — Streamlit 진입점.

사용자 제공 디자인 v2 + SQLite 백엔드 통합.
top-nav 4페이지(상담내역 기록·추가 / 분석 대시보드 / AI 보고서 / 챗봇),
사이드바 프로필 카드 + "➕ 신규 등록" expander.
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
from src.factor_extractor import FACTOR_CATEGORIES, FACTOR_KEYS, FACTOR_LABELS, extract_factors
from src.hira import lookup as hira_lookup
from src.rag import answer_query, healthcheck as rag_healthcheck
from src.report import build_docx, build_md, build_pdf
from src.summarizer import summarize


# ── 설정 ──────────────────────────────────────────────────────────────────────

APP_NAME = "CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼"
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "gemma")
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "gemini_api")
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "koalpaca_api")

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")
db.init_db()


# ── 분석 파이프라인 ───────────────────────────────────────────────────────────

DEFAULT_DIALOGUE = pd.DataFrame({
    "화자": ["상담사", "내담자", "상담사", "내담자", "상담사", "내담자"],
    "발화": [
        "오늘은 어떤 이야기를 나누고 싶으세요?",
        "요즘 잠을 잘 못 자고, 아침에 일어나기가 너무 힘들어요.",
        "수면 문제는 언제부터 시작되었나요?",
        "회사 일이 많아진 뒤부터 계속 피곤하고 불안해요. 출근하기 전부터 가슴이 답답하고, 아무것도 하기 싫다는 생각이 자주 들어요.",
        "그럴 때 주로 어떤 생각이 드나요?",
        "내가 일을 잘 못하고 있는 것 같고, 사람들을 만나는 것도 조금 피하게 돼요.",
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


def run_analysis(
    script: str,
    patient_id: str,
    session_date: str,
    session_no: str = "",
    scope: str = "",
    topic: str = "",
) -> Dict[str, Any]:
    """F1 1차 + 2차 + F3 요약 → SQLite 저장."""
    cls = classify_text(script)
    fact = extract_factors(script, cls["classification"], backend=FACTOR_BACKEND)
    summ = summarize(script)
    summ["text"] = soften_diagnostic_expression(summ["text"])

    sess = db.add_session(
        patient_id, session_date, script,
        session_no=session_no or None, scope=scope or None, topic=topic or None,
    )
    db.add_analysis(sess["id"], "classifier", cls["backend"], cls)
    db.add_analysis(sess["id"], "factors", fact["backend"], fact)
    db.add_analysis(sess["id"], "summary", summ["source"], summ)

    return {
        "session_id": sess["id"],
        "classifier": cls,
        "factors": fact,
        "summary": summ,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_factor_dataframe(factors: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for k in FACTOR_KEYS:
        rows.append({
            "요인": k,
            "카테고리": FACTOR_CATEGORIES.get(k, "기타"),
            "점수": int(factors.get(k, 0)),
        })
    return pd.DataFrame(rows)


# ── Session State ─────────────────────────────────────────────────────────────


def init_session_state() -> None:
    defaults = {
        "page": "상담내역 기록·추가",
        "selected_client": None,
        "selected_session": None,
        "client_search": "",
        "record_mode": "existing",
        "dialogue_rows": DEFAULT_DIALOGUE.copy(),
        "chat_history": [],
        "analysis_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_page(page_name: str) -> None:
    st.session_state.page = page_name


def select_session(session_id: str) -> None:
    st.session_state.selected_session = session_id
    st.session_state.record_mode = "existing"


def start_new_session() -> None:
    st.session_state.record_mode = "new"
    st.session_state.selected_session = "새 상담"


# ── 전역 스타일 (사용자 제공 CSS) ─────────────────────────────────────────────

PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E40AF"
PRIMARY_LIGHT = "#EFF6FF"
PRIMARY_SOFT = "#DBEAFE"
CARD_BLUE = "#F3F8FF"
CARD_BLUE_BORDER = "#D9EAFE"
TEXT = "#0F172A"
SUBTEXT = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#F1F5F9"


def apply_global_style() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .stApp {{ background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 44%); }}
        .main .block-container {{
            padding-top: 1.0rem; padding-bottom: 2.5rem;
            max-width: 1640px; padding-left: 2.2rem; padding-right: 2.2rem;
        }}
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG}; border-right: 1px solid {BORDER};
        }}
        .app-title {{
            font-size: 1.72rem; font-weight: 700; color: {TEXT};
            letter-spacing: -0.045em; margin-bottom: 0.35rem;
        }}
        .section-title {{
            font-size: 1.18rem; font-weight: 650; color: {TEXT};
            letter-spacing: -0.035em; margin-top: 0.1rem; margin-bottom: 0.55rem;
        }}
        .chart-section-title {{
            font-size: 1.02rem; font-weight: 620; color: {TEXT};
            margin-top: 0.8rem; margin-bottom: 0.25rem;
        }}
        .page-desc {{ color: {SUBTEXT}; font-size: 0.9rem; margin-bottom: 1rem; }}
        .tag {{
            display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
            background: {PRIMARY_LIGHT}; color: {PRIMARY_DARK};
            font-size: 0.76rem; font-weight: 620;
            margin-right: 0.35rem; margin-bottom: 0.2rem;
            border: 1px solid {PRIMARY_SOFT};
        }}
        .hero-card {{
            background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 65%, #FFFFFF 100%);
            border: 1px solid #BFDBFE; border-radius: 1.15rem;
            padding: 1.15rem 1.3rem; margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.035);
        }}
        .hero-title {{
            font-size: 1.04rem; font-weight: 650;
            color: {PRIMARY_DARK}; margin-bottom: 0.25rem;
        }}
        .hero-desc {{ color: {SUBTEXT}; font-size: 0.9rem; line-height: 1.55; }}
        .nav-spacer {{ height: 2.4rem; }}
        .summary-card {{
            background: {CARD_BLUE}; border: 1px solid {CARD_BLUE_BORDER};
            border-radius: 0.95rem; padding: 0.95rem 1.05rem;
            min-height: 132px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.025);
        }}
        .summary-card-title {{
            font-size: 0.86rem; font-weight: 720; color: {PRIMARY_DARK};
            letter-spacing: -0.015em; margin-bottom: 0.72rem;
            padding-bottom: 0.35rem; border-bottom: 1px solid #D6E6FF;
        }}
        .summary-card-body {{
            font-size: 0.82rem; font-weight: 480; color: #334155; line-height: 1.65;
        }}
        div.stButton > button:first-child {{
            border-radius: 999px; min-height: 2.45rem;
            font-size: 0.88rem; line-height: 1.2; font-weight: 600;
            border: 1px solid #CBD5E1; color: {TEXT}; background: #FFFFFF;
            white-space: nowrap;
        }}
        div.stButton > button:hover {{
            border-color: {PRIMARY}; color: {PRIMARY_DARK};
            background-color: {PRIMARY_LIGHT};
        }}
        div.stButton > button[kind="primary"] {{
            background: {PRIMARY}; border-color: {PRIMARY}; color: white;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
        }}
        div.stDownloadButton > button:first-child {{
            border-radius: 999px; min-height: 2.35rem;
            font-size: 0.88rem; font-weight: 600; border: 1px solid #CBD5E1;
        }}
        div[data-testid="stMetric"] {{
            background-color: #FFFFFF; padding: 0.85rem 0.9rem;
            border-radius: 1rem; border: 1px solid {BORDER};
            box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.035);
        }}
        .quick-question-title {{
            font-size: 1rem; font-weight: 650; margin-bottom: 0.3rem; color: {TEXT};
        }}
        .chat-helper-caption {{
            color: {SUBTEXT}; font-size: 0.84rem; margin-bottom: 0.7rem;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] p,
        div[data-baseweb="tab-list"] button[aria-selected="true"] div {{
            color: {PRIMARY} !important;
        }}
        div[data-baseweb="tab-highlight"] {{ background-color: {PRIMARY} !important; }}
        textarea:focus, input:focus, div[data-baseweb="select"]:focus-within {{
            border-color: {PRIMARY} !important;
            box-shadow: 0 0 0 1px {PRIMARY} !important;
        }}
        .profile-card {{
            background: #FFFFFF; border: 1px solid {BORDER};
            border-radius: 1rem; padding: 0.9rem 0.95rem; margin-bottom: 0.75rem;
        }}
        .profile-row {{ display: flex; align-items: center; gap: 0.7rem; }}
        .avatar-circle {{
            width: 42px; height: 42px; border-radius: 999px;
            background: {PRIMARY}; color: #FFFFFF;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.95rem;
        }}
        .profile-name {{
            font-weight: 700; color: {TEXT};
            font-size: 0.94rem; line-height: 1.3;
        }}
        .profile-sub {{
            color: {SUBTEXT}; font-size: 0.75rem; line-height: 1.3;
        }}
        .subscription-card {{
            background: {PRIMARY_LIGHT}; border: 1px solid {PRIMARY_SOFT};
            color: {PRIMARY_DARK}; border-radius: 999px;
            padding: 0.6rem 0.85rem; font-size: 0.82rem;
            font-weight: 650; text-align: center;
        }}
        hr {{ border-color: {BORDER}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── 사이드바 ──────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.title("CounsHelper")
        st.caption("상담 기록 분석 & 보고서 자동화")

        st.markdown(
            """
            <div class="profile-card">
                <div class="profile-row">
                    <div class="avatar-circle">보</div>
                    <div>
                        <div class="profile-name">보아즈</div>
                        <div class="profile-sub">boaz@counshelper.ai</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="subscription-card">구독 상태 · MVP Demo 플랜</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"분류 `{CLASSIFIER_BACKEND}` · 28요인 `{FACTOR_BACKEND}` · 요약 `{SUMMARIZER_BACKEND}`"
        )

        st.divider()

        st.markdown("#### 내담자 선택")
        patients = db.list_patients()

        if not patients:
            st.warning("등록된 내담자 없음")
            st.caption("아래 '➕ 신규 등록'에서 추가하세요.")
            st.session_state.selected_client = None
        else:
            keyword = st.text_input(
                "내담자 검색",
                value=st.session_state.client_search,
                placeholder="예: P-001, alias, 서울",
            ).strip()
            st.session_state.client_search = keyword

            filtered = patients
            if keyword:
                k = keyword.lower()
                filtered = [
                    p for p in patients
                    if any(k in str(p.get(f, "")).lower()
                           for f in ("id", "alias", "gender", "region", "note"))
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
                label = st.selectbox(
                    "내담자 선택", list(options.keys()), index=default_idx,
                    label_visibility="collapsed",
                )
                st.session_state.selected_client = options[label]

        with st.expander("➕ 신규 등록"):
            with st.form("add_patient_form_sidebar"):
                new_alias = st.text_input("alias (필수)", placeholder="예: 내담자A, P-005")
                new_gender = st.selectbox("성별", ["여성", "남성", "기타"])
                new_age = st.number_input("연령", min_value=10, max_value=100, value=30)
                new_region = st.text_input("지역", placeholder="예: 서울")
                new_note = st.text_area("메모", height=68)
                if st.form_submit_button("등록"):
                    if not new_alias.strip():
                        st.error("alias 필수")
                    else:
                        p = db.add_patient(new_alias, new_gender, new_age, new_region, new_note)
                        st.success(f"등록: {p['alias']} (`{p['id']}`)")
                        st.session_state.selected_client = p["id"]
                        st.rerun()

        st.divider()
        st.button("설정", key="settings_disabled", use_container_width=True, disabled=True)


# ── 헤더 ──────────────────────────────────────────────────────────────────────


def render_header() -> None:
    st.markdown(
        f'<div class="app-title">{APP_NAME}</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.selected_client
    if pid:
        p = db.get_patient(pid)
        if p:
            sessions = db.list_sessions(pid)
            session_tag = (
                f"{sessions[0].get('session_no') or '미분류'}"
                if sessions else "회기 없음"
            )
            # 분류 결과 기반 태그 (있으면)
            scope_tag = p.get("note", "")
            if sessions:
                cls = db.get_latest_analysis(sessions[0]["id"], "classifier")
                if cls and cls["payload"].get("classification"):
                    c = cls["payload"]["classification"]
                    tags = []
                    if c.get("depression"): tags.append("우울")
                    if c.get("anxiety"): tags.append("불안")
                    if c.get("addiction"): tags.append("중독")
                    scope_tag = "/".join(tags) if tags else "정상"
                # session_no override
                if sessions[0].get("scope"):
                    scope_tag = sessions[0]["scope"]

            st.markdown(
                f"""
                <span class="tag">{p['alias']}</span>
                <span class="tag">{session_tag}</span>
                <span class="tag">{p['age']}대 {p['gender']} · {p['region']}</span>
                <span class="tag">{scope_tag or '-'}</span>
                """,
                unsafe_allow_html=True,
            )


# ── top-nav ───────────────────────────────────────────────────────────────────


def render_main_nav() -> None:
    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    nav_items = [
        (n1, "상담내역 기록·추가", "top_nav_records"),
        (n2, "분석 대시보드", "top_nav_dashboard"),
        (n3, "AI 보고서", "top_nav_report"),
        (n4, "챗봇", "top_nav_chat"),
    ]
    for col, label, key in nav_items:
        with col:
            if st.button(
                label, key=key, use_container_width=True,
                type="primary" if st.session_state.page == label else "secondary",
            ):
                go_page(label)
                st.rerun()
    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)


# ── 페이지 1: 상담내역 기록·추가 ──────────────────────────────────────────────


def render_session_cards() -> None:
    pid = st.session_state.selected_client
    st.markdown('<div class="section-title">상담 내역</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">선택한 내담자의 기존 상담 기록을 확인하거나 새 상담 내역을 추가합니다.</div>',
        unsafe_allow_html=True,
    )

    sessions = db.list_sessions(pid) if pid else []

    if not sessions:
        st.info("기존 상담 내역이 없습니다. 새 상담 내역을 추가해 주세요.")
    else:
        for s in sessions:
            selected = (
                st.session_state.record_mode == "existing"
                and st.session_state.selected_session == s["id"]
            )
            summ = db.get_latest_analysis(s["id"], "summary")
            status = "작성 완료" if summ else "검토 필요"

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([0.14, 0.18, 0.34, 0.18, 0.16])
                c1.markdown(f"**{s.get('session_no') or '미분류'}**")
                c2.write(s.get("session_date", "-"))
                c3.write(s.get("topic") or "(주제 미입력)")
                c4.write(status)
                with c5:
                    label = "선택됨" if selected else "기록 보기"
                    if st.button(
                        label, key=f"sel_{s['id']}",
                        use_container_width=True, disabled=selected,
                    ):
                        select_session(s["id"])
                        st.rerun()

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.18, 0.60, 0.22])
        c1.markdown("**+ 신규**")
        c2.write("새 상담 내역 추가")
        c2.caption("회기 정보와 상담 내용을 입력해 새 기록을 생성합니다.")
        with c3:
            if st.button("추가하기", key="add_new_session", use_container_width=True):
                start_new_session()
                st.rerun()


def render_existing_record_preview() -> None:
    sid = st.session_state.selected_session
    if not sid or sid == "새 상담":
        st.info("좌측 상단 회기 카드의 '기록 보기'를 눌러 회기를 선택하세요.")
        return

    s = db.get_session(sid)
    if not s:
        st.error("선택된 회기를 찾을 수 없습니다.")
        return

    summ = db.get_latest_analysis(sid, "summary")
    status = "작성 완료" if summ else "검토 필요"

    st.markdown('<div class="section-title">선택한 상담 기록 요약</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([0.18, 0.22, 0.22, 0.38])
    f1.metric("회기", s.get("session_no") or "-")
    f2.metric("상담일", s.get("session_date", "-"))
    f3.metric("보고서 상태", status)
    f4.metric("상담 주제", s.get("topic") or "-")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">기존 기록이 선택되었습니다</div>
            <div class="hero-desc">
                상단의 분석 대시보드 또는 AI 보고서 탭을 눌러 선택한 상담 기록의 분석 결과와 보고서 초안을 확인할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 스크립트 미리보기
    with st.expander("상담 스크립트 미리보기"):
        st.text_area("스크립트", value=s.get("transcript", ""), height=240, disabled=True,
                     label_visibility="collapsed")


def render_new_record_form() -> None:
    pid = st.session_state.selected_client
    st.markdown('<div class="section-title">새 상담 내역 추가</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">새 회기 정보와 상담자·내담자 발화가 구분된 상담 내용을 입력합니다.</div>',
        unsafe_allow_html=True,
    )

    n_sessions = len(db.list_sessions(pid)) if pid else 0
    f1, f2, f3, f4 = st.columns([0.18, 0.22, 0.22, 0.38])
    new_session_no = f1.text_input("회기 번호", value=f"{n_sessions + 1}회기")
    new_session_date = f2.text_input("회기 일시", value=date.today().isoformat())
    new_scope = f3.selectbox("상담 범위", ["우울/불안", "우울", "불안", "중독"])
    new_topic = f4.text_input("상담 주제", value="", placeholder="예: 업무 스트레스 및 불안")

    st.markdown("")
    input_tab1, input_tab2 = st.tabs(["전사 텍스트 붙여넣기", "발화 단위 입력"])

    transcript_text = ""
    with input_tab1:
        transcript_text = st.text_area(
            "상담 내용",
            value=build_dialogue_text(st.session_state.dialogue_rows),
            height=330,
            key="new_write_text",
            help="이미 정리된 상담 전사문이 있으면 그대로 붙여넣습니다.",
        )

    with input_tab2:
        st.caption("상담사/내담자를 선택하고 발화를 한 줄씩 입력하면 자동으로 상담 텍스트로 합쳐집니다.")
        edited_rows = st.data_editor(
            st.session_state.dialogue_rows,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "화자": st.column_config.SelectboxColumn(
                    "화자", options=["상담사", "내담자"], required=True, width="small",
                ),
                "발화": st.column_config.TextColumn("발화", width="large", required=True),
            },
            key="new_dialogue_editor",
            height=300,
        )
        st.session_state.dialogue_rows = edited_rows

    # 두 탭 모두 동기화 — text가 비어있으면 data_editor에서 build
    final_script = transcript_text.strip() or build_dialogue_text(st.session_state.dialogue_rows)

    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
    else:
        if st.button(
            "상담 내역 저장 및 AI 분석 실행", key="new_save_analyze",
            type="primary", use_container_width=True,
        ):
            if not final_script.strip():
                st.error("상담 발화를 입력하세요.")
            else:
                with st.spinner("F1 1차 + 2차 + F3 요약 + DB 저장... (Gemma 호출 ~1분)"):
                    result = run_analysis(
                        script=final_script,
                        patient_id=pid,
                        session_date=new_session_date,
                        session_no=new_session_no,
                        scope=new_scope,
                        topic=new_topic,
                    )
                st.session_state.analysis_result = result
                st.session_state.selected_session = result["session_id"]
                st.session_state.record_mode = "existing"
                st.success(f"분석 + DB 저장 완료. session=`{result['session_id']}`")
                go_page("분석 대시보드")
                st.rerun()


def render_record_page() -> None:
    if not st.session_state.selected_client:
        st.warning("좌측 사이드바에서 내담자를 선택하거나 '➕ 신규 등록'에서 추가하세요.")
        return

    render_session_cards()
    st.divider()

    if st.session_state.record_mode == "new":
        render_new_record_form()
    else:
        render_existing_record_preview()


# ── 페이지 2: 분석 대시보드 ───────────────────────────────────────────────────


def _summary_card(title: str, items: list[str]) -> None:
    body = "".join(f"<div>{item}</div>" for item in items) if items else "<div>(없음)</div>"
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-card-title">{title}</div>
            <div class="summary-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_section_to_items(text: str, max_items: int = 4) -> list[str]:
    """sections 텍스트 → bullet 리스트. 줄바꿈/마침표 기준 분할."""
    if not text:
        return []
    import re
    parts = re.split(r"[\n。\.](?:\s|$)", text)
    items = [p.strip(" -*•").strip() for p in parts if p.strip()]
    return items[:max_items]


def render_dashboard() -> None:
    st.markdown('<div class="section-title">분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">선택한 상담 기록의 우울·불안·중독 위험도, 세부 증상 요인, 회기별 변화 추이, 인구통계 맥락을 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.selected_client
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 분석된 회기가 없습니다. '상담내역 기록·추가'에서 분석을 실행하세요.")
        return

    # 회기 선택
    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']} (id={s['id']})": s
        for s in sessions
    }
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]

    cls = db.get_latest_analysis(s["id"], "classifier")
    fact = db.get_latest_analysis(s["id"], "factors")
    summ = db.get_latest_analysis(s["id"], "summary")

    if not (cls and fact):
        st.warning("이 회기에 분석 결과 없음. '상담내역 기록·추가'에서 AI 분석을 먼저 실행하세요.")
        return

    classification = cls["payload"].get("classification", {})
    scores = cls["payload"].get("scores", {})
    factors = fact["payload"].get("factors", {})

    # 4 metric
    review_pct = int(sum(factors.values()) / (28 * 3) * 100) if factors else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("우울 위험도", f"{int(scores.get('depression', 0))} / 3",
              "주의" if classification.get("depression") else "낮음")
    m2.metric("불안 위험도", f"{int(scores.get('anxiety', 0))} / 3",
              "관찰" if classification.get("anxiety") else "낮음")
    m3.metric("중독 위험도", f"{int(scores.get('addiction', 0))} / 3",
              "확인" if classification.get("addiction") else "낮음")
    m4.metric("검토 필요도", f"{review_pct}%",
              "상담사 확인 필요" if review_pct >= 30 else "안정")

    st.markdown("")

    # AI 분석 요약 (2x2) + HIRA 차트 (우측)
    summary_col, hira_col = st.columns([0.68, 0.32], gap="large")

    with summary_col:
        st.markdown("#### AI 분석 요약")
        sections = summ["payload"].get("sections", {}) if summ else {}
        cards = [
            ("주요 증상", _split_section_to_items(sections.get("symptoms", ""))),
            ("위험 요인", _split_section_to_items(sections.get("risk_factors", ""))),
            ("개선 요인", _split_section_to_items(sections.get("improvement_factors", ""))),
            ("개입 요인", _split_section_to_items(sections.get("intervention_factors", ""))),
        ]
        for row_start in range(0, len(cards), 2):
            row_cols = st.columns(2)
            for col, (title, items) in zip(row_cols, cards[row_start:row_start + 2]):
                with col:
                    _summary_card(title, items)
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    with hira_col:
        st.markdown("#### HIRA 인구통계 비교")
        if not any(classification.values()):
            st.caption("분류 결과 모두 0(정상군) — HIRA 비교 생략.")
        else:
            primary = (
                "depression" if classification.get("depression")
                else "anxiety" if classification.get("anxiety")
                else "addiction"
            )
            p = db.get_patient(pid)
            h = hira_lookup(p, primary)
            if h["available"]:
                m = h.get("matched", {})
                rp = m.get("region_patients") or 0
                np_ = m.get("national_patients") or 1
                # 정규화된 막대 (지역 비율 vs 평균)
                hira_chart_data = pd.DataFrame({
                    "구분": ["내담자 지역", "전국 평균"],
                    "환자수": [int(rp), int(np_ / max(m.get("n_rows", 1), 1))],
                })
                fig_hira = px.bar(
                    hira_chart_data, x="구분", y="환자수", text="환자수",
                    color="구분",
                    color_discrete_sequence=[PRIMARY, "#94A3B8"],
                )
                fig_hira.update_traces(textposition="outside", width=0.46)
                fig_hira.update_layout(
                    height=280, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    yaxis_title="환자수(2024)", xaxis_title="",
                    font=dict(color=TEXT, size=11),
                )
                st.plotly_chart(fig_hira, use_container_width=True)
                st.caption(f"건강보험심사평가원 2024 · {p['age']//10*10}대 {p['gender']}")
            else:
                st.caption(h["summary_text"])

    st.markdown("")

    # 28요인 Top 10
    color_map = {
        "우울": PRIMARY,
        "우울/위험": PRIMARY_DARK,
        "불안": "#38BDF8",
        "중독": "#94A3B8",
        "중독/기능": "#64748B",
    }

    st.markdown(
        '<div class="chart-section-title">상위 세부 증상 요인 Top 10</div>',
        unsafe_allow_html=True,
    )
    selected_cats = st.multiselect(
        "카테고리 필터",
        ["우울", "우울/위험", "불안", "중독", "중독/기능"],
        default=["우울", "우울/위험", "불안", "중독", "중독/기능"],
    )
    df = build_factor_dataframe(factors)
    if selected_cats:
        df = df[df["카테고리"].isin(selected_cats)]
    top_scores = df.sort_values("점수", ascending=False).head(10)

    fig_bar = px.bar(
        top_scores.sort_values("점수"),
        x="점수", y="요인", color="카테고리",
        color_discrete_map=color_map, orientation="h", range_x=[0, 3],
    )
    fig_bar.update_traces(width=0.42)
    fig_bar.update_layout(
        height=420, margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT), legend_title_text="카테고리",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 자살 위험 경고
    if factors.get("자살생각", 0) > 0:
        st.error("⚠️ 자살 사고 라벨이 표시되었습니다. 상담사가 별도 안전 평가를 수행하세요.")

    # 회기별 추이
    st.markdown(
        '<div class="chart-section-title">회기별 변화 추이</div>',
        unsafe_allow_html=True,
    )
    _render_trend_chart(pid)


def _render_trend_chart(patient_id: str) -> None:
    """전체 회기의 핵심 요인 5개 시계열."""
    sessions = sorted(
        db.list_sessions(patient_id), key=lambda s: s.get("session_date", "")
    )
    if len(sessions) < 2:
        st.caption("회기가 2개 이상 누적되면 추이가 표시됩니다.")
        return

    line_color_map = {
        "우울한 기분": PRIMARY_DARK,
        "불안감": PRIMARY,
        "수면문제": "#38BDF8",
        "피로감": "#64748B",
        "자살생각": "#DC2626",
    }

    rows = []
    for s in sessions:
        f = db.get_latest_analysis(s["id"], "factors")
        if not f:
            continue
        factors = f["payload"].get("factors", {})
        for key in ("우울한 기분", "불안감", "수면문제", "피로감", "자살생각"):
            rows.append({
                "회기": s.get("session_no") or s["session_date"],
                "요인": key,
                "점수": int(factors.get(key, 0)),
            })

    if not rows:
        st.caption("회기에 분석 결과가 없습니다.")
        return

    tdf = pd.DataFrame(rows)
    fig = px.line(
        tdf, x="회기", y="점수", color="요인",
        color_discrete_map=line_color_map,
        markers=True, range_y=[0, 3],
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT), legend_title_text="요인",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 페이지 3: AI 보고서 ───────────────────────────────────────────────────────


def render_report() -> None:
    st.markdown('<div class="section-title">AI 보고서</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI가 생성한 요약본과 보고서 형식의 텍스트 초안을 상담사가 검토·수정합니다.</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.selected_client
    if not pid:
        st.warning("좌측 사이드바에서 내담자를 먼저 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 회기가 없습니다.")
        return

    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']} (id={s['id']})": s
        for s in sessions
    }
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]
    p = db.get_patient(pid)

    summ = db.get_latest_analysis(s["id"], "summary")
    if not summ:
        st.warning("이 회기에 요약 결과 없음. '상담내역 기록·추가'에서 AI 분석을 먼저 실행하세요.")
        return

    payload = summ["payload"]
    brief = payload.get("brief", "") or "(요약본 생성 실패)"
    text = payload.get("text", "")
    sections = payload.get("sections", {})
    source = payload.get("source", "?")
    ka_filled = payload.get("koalpaca_sections_filled", 0)
    gemma_filled = payload.get("gemma_sections_filled", 0)
    source_label = {
        "koalpaca": "KoAlpaca (4/4 섹션 완전 응답)",
        "koalpaca+gemma": f"KoAlpaca {ka_filled}/4 + Gemma 보강 {gemma_filled}/4",
        "koalpaca_partial": f"KoAlpaca {ka_filled}/4 (부분 응답, Gemma 보강 실패)",
        "gemma_fallback": "Gemma 폴백 (KoAlpaca 시도 → 빈 응답)",
        "gemma_only": "Gemma 단독 (KoAlpaca endpoint 미설정/네트워크 실패)",
        "none": "응답 없음",
    }.get(source, source)

    st.caption(f"요약 소스: `{source_label}`")

    st.markdown("#### 요약본")
    edited_brief = st.text_area(
        "요약본", value=brief, height=180, key="report_brief",
        label_visibility="collapsed",
    )

    st.markdown("#### 보고서 형식 텍스트")
    edited_text = st.text_area(
        "보고서 형식 텍스트", value=text, height=430, key="report_text",
        label_visibility="collapsed",
    )

    final_md_text = f"[요약본]\n{edited_brief}\n\n{edited_text}"

    st.divider()
    d1, d2, d3 = st.columns(3)

    md_bytes = build_md(p, s, sections, edited_brief)
    d1.download_button(
        ".md 다운로드", md_bytes,
        file_name=f"report_{s['id']}.md", mime="text/markdown",
        use_container_width=True,
    )

    try:
        docx_bytes = build_docx(p, s, sections, edited_brief, [])
        d2.download_button(
            ".docx 다운로드", docx_bytes,
            file_name=f"report_{s['id']}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception as e:
        d2.error(f".docx 실패: {e}")

    pdf_bytes = build_pdf(p, s, sections)
    if pdf_bytes:
        d3.download_button(
            ".pdf 다운로드", pdf_bytes,
            file_name=f"report_{s['id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        d3.button(".pdf 비활성", disabled=True, use_container_width=True,
                  help="weasyprint Windows GTK 의존성")

    st.caption("※ 본 보고서는 AI 생성 초안이며, 상담사의 검토와 수정 후 사용하는 것을 전제로 합니다.")


# ── 페이지 4: RAG 챗봇 ────────────────────────────────────────────────────────


def render_chat_messages() -> None:
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="💬"):
            st.write(
                "안녕하세요. 현재 내담자의 상담 기록을 바탕으로 "
                "유사 사례, 임상 가이드, 다음 회기 계획을 도와드릴 수 있습니다."
            )
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="💬"):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("참고 출처 보기"):
                        for src in msg["sources"]:
                            with st.container(border=True):
                                st.markdown(f"**{src['title']}**")
                                st.caption(src.get("desc", ""))
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])


def render_quick_question_buttons() -> None:
    st.markdown(
        '<div class="quick-question-title">예상 질문</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chat-helper-caption">상담사가 자주 확인하는 질문을 바로 실행할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    questions = [
        ("유사 상담 사례", "이 내담자와 비슷한 상담 사례가 있나요?", "chip_case"),
        ("임상 가이드", "수면 장애가 동반된 우울 사례의 임상 가이드를 알려주세요.", "chip_guide"),
        ("지난 회기 변화", "지난 3회기 동안 어떤 변화가 있었나요?", "chip_history"),
        ("다음 회기 질문", "다음 회기 질문을 추천해줘.", "chip_next"),
        ("보고서 요약", "이번 회기 상담 내용을 보고서 형식으로 요약해줘.", "chip_report"),
        ("위험 요인", "이 내담자의 주요 위험 요인을 정리해줘.", "chip_risk"),
        ("상담 목표", "다음 회기 상담 목표를 추천해줘.", "chip_goal"),
        ("개입 방향", "이 내담자에게 적합한 상담 개입 방향을 추천해줘.", "chip_intervention"),
    ]

    for row_start in range(0, len(questions), 4):
        cols = st.columns(4)
        for col, (label, prompt, key) in zip(cols, questions[row_start:row_start + 4]):
            with col:
                if st.button(label, key=key, use_container_width=True):
                    _handle_rag(prompt)
                    st.rerun()


def _handle_rag(prompt: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    result = answer_query(prompt, k=5)
    if result.get("error"):
        st.session_state.chat_history.append({
            "role": "assistant", "content": f"⚠️ {result['error']}", "sources": [],
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


def render_chatbot() -> None:
    pid = st.session_state.selected_client
    sid = st.session_state.selected_session
    p = db.get_patient(pid) if pid else None
    s = db.get_session(sid) if sid and sid != "새 상담" else None
    ctx = (
        f"{p['alias'] if p else '내담자 미선택'} · "
        f"{(s.get('session_no') or s['session_date']) if s else '회기 미선택'}"
    )

    top_col1, top_col2 = st.columns([0.78, 0.22])
    with top_col1:
        st.markdown('<div class="section-title">챗봇</div>', unsafe_allow_html=True)
        st.caption(f"현재 컨텍스트: {ctx}")
    with top_col2:
        if st.button("대화 초기화", key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    status = rag_healthcheck()
    if not status.get("llm"):
        st.warning("GEMINI_API_KEY 미설정 — .env 확인.")
        return
    if not status.get("chroma"):
        st.warning("RAG 인덱스 미생성. `python -m src.rag.ingest` 실행.")
        return

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">RAG 기반 AI 도우미</div>
            <div class="hero-desc">
                현재 상담기록을 바탕으로 유사 상담 케이스, 임상 가이드라인, 상담 이론 문서를 검색해 답변을 제공합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_chat_messages()
    st.markdown("")
    render_quick_question_buttons()

    prompt = st.chat_input("AI 도우미에게 질문하세요. 예: 이 내담자의 다음 회기 질문을 추천해줘")
    if prompt:
        _handle_rag(prompt)
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    init_session_state()
    apply_global_style()
    render_sidebar()
    render_header()
    render_main_nav()

    page = st.session_state.page
    if page == "상담내역 기록·추가":
        render_record_page()
    elif page == "분석 대시보드":
        render_dashboard()
    elif page == "AI 보고서":
        render_report()
    elif page == "챗봇":
        render_chatbot()


if __name__ == "__main__":
    main()
