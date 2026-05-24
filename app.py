"""CounsHelper — Streamlit 진입점 (Phase 7: WeeNote 레퍼런스 기반 UI).

indigo 디자인 토큰 + Pretendard 한국어 폰트 + 사이드바 메뉴 통합 + AI 도우미 우측 패널.
6 페이지: 내담자 홈 / 상담내역 기록·추가 / 분석 대시보드 / 통계 / AI 보고서 / 챗봇.
"""
import json
import os
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import db
from src.classifier import classify_text
from src.factor_extractor import FACTOR_CATEGORIES, FACTOR_KEYS, FACTOR_LABELS, extract_factors
from src.hira import lookup as hira_lookup
from src.insight import dashboard_insight, hira_summary_one_line, stats_insight
from src.rag import answer_query, healthcheck as rag_healthcheck
from src.report import build_docx, build_md, build_pdf
from src.stats import aggregate_global_stats, classification_distribution, factor_top_n
from src.summarizer import summarize


# ── 설정 ──────────────────────────────────────────────────────────────────────

APP_NAME = "CounsHelper"
APP_SUB = "상담 기록 분석 v0.7"
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "gemma")
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "gemini_api")
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "koalpaca_api")

st.set_page_config(
    page_title=f"{APP_NAME} — 상담 기록 분석 & 보고서 자동화",
    layout="wide",
    initial_sidebar_state="expanded",
)
db.init_db()


# ── 디자인 토큰 ───────────────────────────────────────────────────────────────

PRIMARY = "#4F46E5"          # indigo-600
PRIMARY_DARK = "#3730A3"     # indigo-800
PRIMARY_LIGHT = "#EEF2FF"    # indigo-50
PRIMARY_SOFT = "#E0E7FF"     # indigo-100
PRIMARY_DEEPER = "#312E81"   # indigo-900
ACCENT = "#0891B2"           # cyan-600
DANGER = "#DC2626"
WARNING = "#D97706"
SUCCESS = "#059669"
TEXT = "#1E1B4B"             # indigo-950
SUBTEXT = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#F8FAFC"
PAGE_BG = "#FAFAFB"

CHART_PALETTE = {
    "우울": PRIMARY,
    "우울/위험": PRIMARY_DARK,
    "불안": "#06B6D4",
    "중독": "#A78BFA",
    "중독/기능": "#7C3AED",
    "정상군": "#94A3B8",
}


def apply_global_style() -> None:
    st.markdown(
        f"""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont,
                'Noto Sans KR', 'Segoe UI', sans-serif;
        }}
        .stApp {{ background: {PAGE_BG}; }}
        .main .block-container {{
            padding-top: 1.4rem; padding-bottom: 2.5rem;
            max-width: 1640px; padding-left: 2rem; padding-right: 2rem;
        }}

        /* ── 사이드바 ── */
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG}; border-right: 1px solid {BORDER};
            min-width: 16rem !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem; padding-bottom: 1.5rem;
        }}
        .brand {{
            font-size: 1.4rem; font-weight: 700; color: {PRIMARY};
            letter-spacing: -0.04em; line-height: 1.1;
        }}
        .brand-sub {{
            font-size: 0.72rem; color: {SUBTEXT}; margin-top: 0.2rem;
        }}
        .user-info {{
            font-size: 0.8rem; color: {TEXT}; line-height: 1.5;
            margin-top: 0.9rem; padding: 0.55rem 0;
        }}
        .user-info-name {{
            font-weight: 600; color: {TEXT};
        }}
        .nav-spacer {{ height: 1.5rem; }}

        /* ── 헤더 (페이지 공통) ── */
        .page-title {{
            font-size: 1.65rem; font-weight: 700; color: {TEXT};
            letter-spacing: -0.045em; margin: 0;
            display: flex; align-items: center; gap: 0.5rem;
        }}
        .page-subtitle {{
            color: {SUBTEXT}; font-size: 0.88rem; margin-top: 0.25rem;
        }}
        .page-tags {{
            margin-top: 0.6rem; display: flex; flex-wrap: wrap; gap: 0.3rem;
        }}
        .tag {{
            display: inline-block; padding: 0.22rem 0.7rem;
            border-radius: 999px; background: {PRIMARY_LIGHT};
            color: {PRIMARY_DARK}; font-size: 0.74rem; font-weight: 620;
            border: 1px solid {PRIMARY_SOFT};
        }}

        /* ── 카드 ── */
        .metric-card, .chart-card, .insight-card {{
            background: #FFFFFF; border: 1px solid {BORDER};
            border-radius: 1rem; padding: 1.05rem 1.15rem;
            box-shadow: 0 1px 3px 0 rgb(79 70 229 / 0.04);
        }}
        .insight-card {{
            background: linear-gradient(180deg, #FFFFFF 0%, {PRIMARY_LIGHT} 130%);
            border-color: {PRIMARY_SOFT};
        }}
        .insight-card h4 {{
            color: {PRIMARY_DARK}; font-weight: 700; letter-spacing: -0.02em;
            margin-top: 0;
        }}
        .insight-body {{
            font-size: 0.86rem; line-height: 1.7; color: {TEXT};
        }}
        .summary-card {{
            background: {PRIMARY_LIGHT}; border: 1px solid {PRIMARY_SOFT};
            border-radius: 0.9rem; padding: 0.9rem 1rem;
            min-height: 120px;
            box-shadow: 0 1px 2px 0 rgb(79 70 229 / 0.03);
        }}
        .summary-card-title {{
            font-size: 0.82rem; font-weight: 720; color: {PRIMARY_DARK};
            letter-spacing: -0.015em; margin-bottom: 0.55rem;
            padding-bottom: 0.3rem; border-bottom: 1px solid {PRIMARY_SOFT};
        }}
        .summary-card-body {{
            font-size: 0.8rem; color: #334155; line-height: 1.6;
        }}
        .alert-card {{
            background: #FEF2F2; border: 1px solid #FECACA;
            border-radius: 0.85rem; padding: 0.85rem 1rem;
            color: #991B1B; font-weight: 600; font-size: 0.88rem;
        }}

        /* ── AI 도우미 패널 ── */
        .ai-panel-header {{
            font-size: 1.05rem; font-weight: 700; color: {PRIMARY};
            display: flex; align-items: center; gap: 0.5rem;
            margin-bottom: 0.4rem;
        }}
        .context-chip {{
            display: inline-block; padding: 0.3rem 0.7rem;
            border-radius: 999px; background: {PRIMARY_LIGHT};
            color: {PRIMARY_DARK}; font-size: 0.74rem; font-weight: 620;
            border: 1px solid {PRIMARY_SOFT}; margin-bottom: 0.6rem;
        }}

        /* ── 버튼 ── */
        div.stButton > button:first-child {{
            border-radius: 0.7rem; min-height: 2.3rem;
            font-size: 0.85rem; line-height: 1.2; font-weight: 600;
            border: 1px solid #CBD5E1; color: {TEXT}; background: #FFFFFF;
            transition: all 0.15s ease;
        }}
        div.stButton > button:hover {{
            border-color: {PRIMARY}; color: {PRIMARY_DARK};
            background-color: {PRIMARY_LIGHT};
        }}
        div.stButton > button[kind="primary"] {{
            background: {PRIMARY}; border-color: {PRIMARY}; color: #FFFFFF;
            box-shadow: 0 4px 10px -2px rgb(79 70 229 / 0.25);
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK};
        }}
        div.stDownloadButton > button:first-child {{
            border-radius: 0.7rem; min-height: 2.3rem;
            font-size: 0.85rem; font-weight: 600;
            border: 1px solid {PRIMARY_SOFT}; color: {PRIMARY_DARK};
            background: {PRIMARY_LIGHT};
        }}

        /* ── 메트릭 ── */
        div[data-testid="stMetric"] {{
            background: #FFFFFF; padding: 0.9rem 1rem;
            border-radius: 0.95rem; border: 1px solid {BORDER};
            box-shadow: 0 1px 3px 0 rgb(79 70 229 / 0.04);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {SUBTEXT} !important; font-weight: 600 !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT} !important; font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }}

        /* ── 입력/탭 ── */
        textarea:focus, input:focus, div[data-baseweb="select"]:focus-within {{
            border-color: {PRIMARY} !important;
            box-shadow: 0 0 0 1px {PRIMARY} !important;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] p,
        div[data-baseweb="tab-list"] button[aria-selected="true"] div {{
            color: {PRIMARY} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {PRIMARY} !important;
        }}

        /* ── 사이드바 메뉴 ── */
        section[data-testid="stSidebar"] div.stButton > button {{
            text-align: left; justify-content: flex-start;
            padding-left: 0.85rem;
        }}

        hr {{ border-color: {BORDER}; margin: 1rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── 기본 데이터 ───────────────────────────────────────────────────────────────

DEFAULT_DIALOGUE = pd.DataFrame({
    "화자": ["상담사", "내담자", "상담사", "내담자", "상담사", "내담자"],
    "발화": [
        "오늘은 어떤 이야기를 나누고 싶으세요?",
        "요즘 잠을 잘 못 자고, 아침에 일어나기가 너무 힘들어요.",
        "수면 문제는 언제부터 시작되었나요?",
        "회사 일이 많아진 뒤부터 계속 피곤하고 불안해요.",
        "그럴 때 주로 어떤 생각이 드나요?",
        "내가 일을 잘 못하고 있는 것 같고, 사람들 만나는 것도 피하게 돼요.",
    ],
})

NAV_ITEMS = [
    ("내담자 홈", "🏠"),
    ("상담내역 기록·추가", "📋"),
    ("분석 대시보드", "📊"),
    ("통계", "📈"),
    ("AI 보고서", "📄"),
    ("챗봇", "💬"),
]


# ── Session State ─────────────────────────────────────────────────────────────


def init_session_state() -> None:
    defaults = {
        "page": "내담자 홈",
        "selected_client": None,
        "selected_session": None,
        "client_search": "",
        "record_mode": "existing",
        "dialogue_rows": DEFAULT_DIALOGUE.copy(),
        "chat_history": [],
        "ai_panel_open": False,
        "ai_panel_history": [],
        "analysis_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_page(page_name: str) -> None:
    st.session_state.page = page_name
    st.session_state.ai_panel_open = False  # 페이지 전환 시 패널 닫음


def toggle_ai_panel() -> None:
    st.session_state.ai_panel_open = not st.session_state.get("ai_panel_open", False)


# ── 분석 파이프라인 ───────────────────────────────────────────────────────────


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


# ── 사이드바 ──────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f'<div class="brand">{APP_NAME}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="brand-sub">{APP_SUB}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="user-info">'
            '<span class="user-info-name">전문상담사 (데모)</span><br>'
            '2026학년도 · MVP Demo'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # 내담자 선택
        st.markdown("**내담자 선택**")
        patients = db.list_patients()
        if not patients:
            st.caption("등록된 내담자 없음 — 아래 '➕ 신규 등록'에서 추가")
            st.session_state.selected_client = None
        else:
            keyword = st.text_input(
                "검색",
                value=st.session_state.client_search,
                placeholder="alias / 성별 / 지역",
                label_visibility="collapsed",
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
                st.caption("검색 결과 없음")
                st.session_state.selected_client = None
            else:
                options = {
                    f"{p['alias']} · {p['gender']}/{p['age']}/{p['region']}": p["id"]
                    for p in filtered
                }
                default_idx = 0
                if st.session_state.selected_client:
                    for i, pid in enumerate(options.values()):
                        if pid == st.session_state.selected_client:
                            default_idx = i
                            break
                label = st.selectbox(
                    "select_client", list(options.keys()),
                    index=default_idx, label_visibility="collapsed",
                )
                st.session_state.selected_client = options[label]

        with st.expander("➕ 신규 등록"):
            with st.form("add_patient_form_sidebar"):
                new_alias = st.text_input("alias (필수)", placeholder="예: 내담자A")
                new_gender = st.selectbox("성별", ["여성", "남성", "기타"])
                new_age = st.number_input("연령", min_value=10, max_value=100, value=30)
                new_region = st.text_input("지역", placeholder="예: 서울")
                new_note = st.text_area("메모", height=68)
                if st.form_submit_button("등록", use_container_width=True):
                    if not new_alias.strip():
                        st.error("alias 필수")
                    else:
                        p = db.add_patient(new_alias, new_gender, new_age, new_region, new_note)
                        st.success(f"등록: {p['alias']}")
                        st.session_state.selected_client = p["id"]
                        st.rerun()

        st.divider()

        # 메뉴
        for label, icon in NAV_ITEMS:
            active = st.session_state.page == label
            btn_type = "primary" if active else "secondary"
            if st.button(f"{icon}  {label}", key=f"nav_{label}",
                         use_container_width=True, type=btn_type):
                go_page(label)
                st.rerun()

        st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
        st.divider()
        st.button("⚙️  설정", key="nav_settings",
                  disabled=True, use_container_width=True)
        st.button("💬  문의", key="nav_inquiry",
                  disabled=True, use_container_width=True)
        st.button("❓  도움말", key="nav_help",
                  disabled=True, use_container_width=True)


# ── 페이지 헤더 ───────────────────────────────────────────────────────────────


def render_page_header(
    title: str,
    *,
    subtitle: str = "",
    show_tags: bool = False,
    actions: Optional[list] = None,
) -> None:
    """공통 헤더 — 좌측 타이틀, 우측 액션 버튼들.

    actions: [(label, callback, type), ...] — type은 "primary"/"secondary".
    """
    head_cols = st.columns([0.6, 0.4])
    with head_cols[0]:
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="page-subtitle">{subtitle}</div>',
                        unsafe_allow_html=True)

    with head_cols[1]:
        if actions:
            action_cols = st.columns(len(actions))
            for i, (label, callback, btn_type) in enumerate(actions):
                with action_cols[i]:
                    if st.button(label, key=f"action_{title}_{label}",
                                 type=btn_type or "secondary",
                                 use_container_width=True):
                        if callback:
                            callback()
                            st.rerun()

    # 헤더 태그 — 내담자 선택 상태에 따라
    if show_tags:
        pid = st.session_state.selected_client
        if pid:
            p = db.get_patient(pid)
            if p:
                sessions = db.list_sessions(pid)
                session_tag = (sessions[0].get("session_no") or sessions[0]["session_date"]) if sessions else "회기 없음"
                cls_tag = "-"
                if sessions:
                    cls = db.get_latest_analysis(sessions[0]["id"], "classifier")
                    if cls and cls["payload"].get("classification"):
                        c = cls["payload"]["classification"]
                        flags = []
                        if c.get("depression"): flags.append("우울")
                        if c.get("anxiety"): flags.append("불안")
                        if c.get("addiction"): flags.append("중독")
                        cls_tag = "/".join(flags) if flags else "정상"
                st.markdown(
                    f'<div class="page-tags">'
                    f'<span class="tag">{p["alias"]}</span>'
                    f'<span class="tag">{session_tag}</span>'
                    f'<span class="tag">{p["age"]}대 {p["gender"]} · {p["region"]}</span>'
                    f'<span class="tag">{cls_tag}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()


# ── AI 도우미 우측 패널 ───────────────────────────────────────────────────────


def render_ai_panel(context_session_id: Optional[str] = None) -> None:
    st.markdown(
        '<div class="ai-panel-header">🤖 AI 도우미</div>',
        unsafe_allow_html=True,
    )

    if context_session_id:
        sess = db.get_session(context_session_id)
        if sess:
            label = sess.get("session_no") or sess["session_date"]
            st.markdown(
                f'<span class="context-chip">📎 회기 첨부됨 — {label}</span>',
                unsafe_allow_html=True,
            )

    status = rag_healthcheck()
    if not status.get("llm") or not status.get("chroma"):
        st.warning("RAG 인덱스/LLM 키 점검 필요")

    # 빠른 질문 (회기 컨텍스트 기반)
    if context_session_id:
        st.caption("이 회기에 대해:")
        for q, key_suffix in [
            ("주요 호소 문제", "qq_main"),
            ("다음 회기 계획 추천", "qq_next"),
            ("유사 사례 검색", "qq_similar"),
        ]:
            if st.button(q, key=f"ai_{key_suffix}", use_container_width=True):
                _handle_panel_query(q, context_session_id)
                st.rerun()

    st.markdown("")
    # 대화 이력 (간략)
    for msg in st.session_state.get("ai_panel_history", [])[-6:]:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div style="font-size:0.85rem">{msg["content"]}</div>',
                            unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f'<div style="font-size:0.85rem">{msg["content"][:600]}</div>',
                            unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("출처", expanded=False):
                        for src in msg["sources"][:3]:
                            st.caption(f"`{src.get('title', '?')}`")

    prompt = st.chat_input("무엇이든 물어보세요", key="ai_panel_input")
    if prompt:
        _handle_panel_query(prompt, context_session_id)
        st.rerun()

    if st.button("🗑 대화 초기화", key="ai_panel_clear", use_container_width=True):
        st.session_state.ai_panel_history = []
        st.rerun()


def _handle_panel_query(prompt: str, session_id: Optional[str]) -> None:
    enriched = prompt
    if session_id:
        summ = db.get_latest_analysis(session_id, "summary")
        if summ:
            brief = (summ["payload"].get("brief") or "")[:600]
            if brief:
                enriched = f"[현재 회기 요약]\n{brief}\n\n[질문]\n{prompt}"

    result = answer_query(enriched, k=5)
    history = st.session_state.setdefault("ai_panel_history", [])
    history.append({"role": "user", "content": prompt})
    if result.get("error"):
        history.append({
            "role": "assistant",
            "content": f"⚠️ {result['error']}",
            "sources": [],
        })
    else:
        history.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": [
                {"title": s["source"], "desc": s["snippet"]}
                for s in result.get("sources", [])
            ],
        })


def page_with_optional_panel(
    main_render_fn: Callable[[], None],
    context_session_id: Optional[str] = None,
) -> None:
    """페이지 본문 + (옵션) AI 도우미 패널.

    Streamlit의 1-level column nesting 제약으로 인해 우측 컬럼 대신 expander 형태.
    헤더의 'AI 도우미' 버튼 클릭 시 ai_panel_open=True → expander 자동 펼침.
    """
    main_render_fn()

    if st.session_state.get("ai_panel_open"):
        st.divider()
        with st.expander("🤖 AI 도우미", expanded=True):
            render_ai_panel(context_session_id)


# ── 페이지 1: 내담자 홈 ──────────────────────────────────────────────────────


def render_patient_home() -> None:
    render_page_header("내담자 홈", subtitle="환자 요약과 최근 회기를 확인합니다.", show_tags=True)

    pid = st.session_state.selected_client
    if not pid:
        st.info("좌측 사이드바에서 내담자를 선택하거나 '➕ 신규 등록'에서 추가하세요.")
        return

    p = db.get_patient(pid)
    sessions = db.list_sessions(pid)

    # 4 메트릭
    m = st.columns(4)
    m[0].metric("내담자 ID", p["id"][:8])
    m[1].metric("성별 / 연령", f"{p['gender']} / {p['age']}세")
    m[2].metric("지역", p.get("region", "-"))
    m[3].metric("회기 수", f"{len(sessions)} 회")

    st.markdown("#### 최근 회기")
    if not sessions:
        st.caption("아직 등록된 회기가 없습니다. '상담내역 기록·추가'에서 새 회기를 추가하세요.")
    else:
        for s in sessions[:5]:
            cls = db.get_latest_analysis(s["id"], "classifier")
            cls_str = "(미분석)"
            if cls and cls["payload"].get("classification"):
                c = cls["payload"]["classification"]
                flags = []
                if c.get("depression"): flags.append("우울")
                if c.get("anxiety"): flags.append("불안")
                if c.get("addiction"): flags.append("중독")
                cls_str = "/".join(flags) if flags else "정상군"
            with st.container(border=True):
                cols = st.columns([0.18, 0.18, 0.34, 0.16, 0.14])
                cols[0].markdown(f"**{s.get('session_no') or '미분류'}**")
                cols[1].write(s["session_date"])
                cols[2].write(s.get("topic") or "(주제 미입력)")
                cols[3].write(cls_str)
                with cols[4]:
                    if st.button("보기", key=f"home_view_{s['id']}",
                                 use_container_width=True):
                        st.session_state.selected_session = s["id"]
                        go_page("분석 대시보드")
                        st.rerun()

    if p.get("note"):
        st.divider()
        st.markdown("#### 메모")
        st.write(p["note"])


# ── 페이지 2: 상담내역 기록·추가 ──────────────────────────────────────────────


def _render_record_page_body() -> None:
    render_page_header(
        "상담내역 기록·추가",
        subtitle="기존 회기를 확인하거나 새 회기를 분석합니다.",
        show_tags=True,
    )

    pid = st.session_state.selected_client
    if not pid:
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    sessions = db.list_sessions(pid)

    st.markdown("**상담 내역**")
    if not sessions:
        st.caption("기존 상담 내역이 없습니다. 아래 '+ 신규' 카드에서 새 회기를 추가하세요.")
    else:
        for s in sessions:
            selected = (
                st.session_state.record_mode == "existing"
                and st.session_state.selected_session == s["id"]
            )
            summ = db.get_latest_analysis(s["id"], "summary")
            status = "작성 완료" if summ else "분석 필요"

            with st.container(border=True):
                cols = st.columns([0.14, 0.18, 0.34, 0.18, 0.16])
                cols[0].markdown(f"**{s.get('session_no') or '미분류'}**")
                cols[1].write(s["session_date"])
                cols[2].write(s.get("topic") or "(주제 미입력)")
                cols[3].write(status)
                with cols[4]:
                    label = "선택됨" if selected else "기록 보기"
                    if st.button(label, key=f"sel_{s['id']}",
                                 use_container_width=True, disabled=selected):
                        st.session_state.selected_session = s["id"]
                        st.session_state.record_mode = "existing"
                        st.rerun()

    # 신규 카드
    with st.container(border=True):
        c1, c2, c3 = st.columns([0.18, 0.60, 0.22])
        c1.markdown("**+ 신규**")
        c2.write("새 상담 내역 추가")
        c2.caption("회기 정보와 상담 내용을 입력해 새 기록을 생성합니다.")
        with c3:
            if st.button("추가하기", key="add_new_session_btn",
                         type="primary", use_container_width=True):
                st.session_state.record_mode = "new"
                st.session_state.selected_session = "새 상담"
                st.rerun()

    st.divider()

    if st.session_state.record_mode == "new":
        _render_new_record_form(pid)
    else:
        _render_existing_record_preview()


def _render_existing_record_preview() -> None:
    sid = st.session_state.selected_session
    if not sid or sid == "새 상담":
        st.info("위 회기 카드의 '기록 보기'를 눌러 회기를 선택하세요.")
        return

    s = db.get_session(sid)
    if not s:
        st.error("선택된 회기를 찾을 수 없습니다.")
        return

    summ = db.get_latest_analysis(sid, "summary")
    status = "작성 완료" if summ else "분석 필요"

    st.markdown("#### 선택한 상담 기록 요약")
    f = st.columns([0.18, 0.22, 0.22, 0.38])
    f[0].metric("회기", s.get("session_no") or "-")
    f[1].metric("상담일", s.get("session_date", "-"))
    f[2].metric("보고서 상태", status)
    f[3].metric("상담 주제", s.get("topic") or "-")

    with st.expander("상담 스크립트 미리보기"):
        st.text_area(
            "transcript_preview", value=s.get("transcript", ""),
            height=240, disabled=True, label_visibility="collapsed",
        )


def _render_new_record_form(pid: str) -> None:
    st.markdown("#### 새 상담 내역 추가")
    st.caption("새 회기 정보와 상담자·내담자 발화가 구분된 상담 내용을 입력합니다.")

    n_sessions = len(db.list_sessions(pid))
    f = st.columns([0.18, 0.22, 0.22, 0.38])
    new_session_no = f[0].text_input("회기 번호", value=f"{n_sessions + 1}회기")
    new_session_date = f[1].text_input("회기 일시", value=date.today().isoformat())
    new_scope = f[2].selectbox("상담 범위", ["우울/불안", "우울", "불안", "중독"])
    new_topic = f[3].text_input("상담 주제", placeholder="예: 업무 스트레스 및 불안")

    input_tab1, input_tab2 = st.tabs(["전사 텍스트 붙여넣기", "발화 단위 입력"])

    transcript_text = ""
    with input_tab1:
        transcript_text = st.text_area(
            "상담 내용",
            value=build_dialogue_text(st.session_state.dialogue_rows),
            height=330, key="new_write_text",
        )

    with input_tab2:
        st.caption("발화를 한 줄씩 입력 — 자동으로 상담 텍스트로 합쳐집니다.")
        edited = st.data_editor(
            st.session_state.dialogue_rows, num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={
                "화자": st.column_config.SelectboxColumn(
                    "화자", options=["상담사", "내담자"], required=True, width="small",
                ),
                "발화": st.column_config.TextColumn("발화", width="large", required=True),
            },
            key="new_dialogue_editor", height=300,
        )
        st.session_state.dialogue_rows = edited

    final_script = transcript_text.strip() or build_dialogue_text(st.session_state.dialogue_rows)

    if st.button("AI 분석 실행", key="new_save_analyze",
                 type="primary", use_container_width=True):
        if not final_script.strip():
            st.error("상담 발화를 입력하세요.")
        else:
            with st.spinner("F1 1차 + 2차 + F3 요약 + DB 저장... (Gemini 호출 약 60초)"):
                result = run_analysis(
                    script=final_script, patient_id=pid,
                    session_date=new_session_date, session_no=new_session_no,
                    scope=new_scope, topic=new_topic,
                )
            st.session_state.analysis_result = result
            st.session_state.selected_session = result["session_id"]
            st.session_state.record_mode = "existing"
            st.success(f"분석 + DB 저장 완료 (session=`{result['session_id']}`)")
            go_page("분석 대시보드")
            st.rerun()


def render_record_page() -> None:
    page_with_optional_panel(_render_record_page_body, st.session_state.selected_session)


# ── 페이지 3: 분석 대시보드 ───────────────────────────────────────────────────


def _render_dashboard_body() -> None:
    render_page_header(
        "분석 대시보드",
        subtitle="우울·불안·중독 위험도, 세부 요인, 회기별 변화 추이.",
        show_tags=True,
        actions=[("🤖 AI 도우미", toggle_ai_panel, "primary")],
    )

    pid = st.session_state.selected_client
    if not pid:
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        st.info("이 내담자의 분석된 회기가 없습니다.")
        return

    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']} (id={s['id']})": s
        for s in sessions
    }
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]
    st.session_state.selected_session = s["id"]

    cls = db.get_latest_analysis(s["id"], "classifier")
    fact = db.get_latest_analysis(s["id"], "factors")

    if not (cls and fact):
        st.warning("이 회기에 분석 결과 없음. '상담내역 기록·추가'에서 AI 분석을 먼저 실행하세요.")
        return

    classification = cls["payload"].get("classification", {})
    scores = cls["payload"].get("scores", {})
    factors = fact["payload"].get("factors", {})

    # 4 메트릭
    review_pct = int(sum(factors.values()) / (28 * 3) * 100) if factors else 0
    m = st.columns(4)
    m[0].metric("우울 위험도", f"{int(scores.get('depression', 0))} / 3",
                "양성" if classification.get("depression") else "낮음")
    m[1].metric("불안 위험도", f"{int(scores.get('anxiety', 0))} / 3",
                "양성" if classification.get("anxiety") else "낮음")
    m[2].metric("중독 위험도", f"{int(scores.get('addiction', 0))} / 3",
                "양성" if classification.get("addiction") else "낮음")
    m[3].metric("검토 필요도", f"{review_pct}%",
                "확인 필요" if review_pct >= 30 else "안정")
    st.caption("주의: 모델 출력 참고값 — 임상 진단/표준화 검사 점수로 단정하지 않음.")

    st.divider()

    # 좌: 차트 / 우: 분석 해석 카드
    left, right = st.columns([0.66, 0.34], gap="large")

    with left:
        st.markdown("#### 상위 28요인 Top 10")
        df = build_factor_dataframe(factors)
        selected_cats = st.multiselect(
            "카테고리 필터",
            list(CHART_PALETTE.keys())[:-1],  # "정상군" 제외
            default=["우울", "우울/위험", "불안", "중독", "중독/기능"],
        )
        if selected_cats:
            df = df[df["카테고리"].isin(selected_cats)]
        top = df.sort_values("점수", ascending=False).head(10)
        if len(top) > 0:
            fig = px.bar(
                top.sort_values("점수"), x="점수", y="요인", color="카테고리",
                color_discrete_map=CHART_PALETTE,
                orientation="h", range_x=[0, 3],
            )
            fig.update_layout(
                height=400, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color=TEXT, family="Pretendard"),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("선택한 카테고리에 해당하는 요인이 없습니다.")

        st.markdown("#### 회기별 변화 추이")
        _render_trend_chart(pid)

    with right:
        # 분석 해석 카드
        insight = dashboard_insight(s["id"])
        st.markdown(
            f'<div class="insight-card">'
            f'<h4>📌 분석 해석</h4>'
            f'<div class="insight-body">{insight}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # HIRA
        if any(classification.values()):
            primary = (
                "depression" if classification.get("depression")
                else "anxiety" if classification.get("anxiety") else "addiction"
            )
            p = db.get_patient(pid)
            h = hira_lookup(p, primary)
            if h["available"]:
                one_line = hira_summary_one_line(h, p, primary)
                st.markdown(
                    f'<div class="insight-card">'
                    f'<h4>📊 HIRA 인구통계</h4>'
                    f'<div class="insight-body">{one_line or h["summary_text"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

        # 안전 경고
        if factors.get("자살생각", 0) > 0:
            st.markdown(
                '<div class="alert-card">⚠️ 자살 사고 라벨이 표시되었습니다. '
                '상담사가 별도 안전 평가를 수행하세요.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # AI 분석 요약 2x2 (4섹션 — 본문 하단)
    summ = db.get_latest_analysis(s["id"], "summary")
    if summ:
        st.markdown("#### AI 분석 요약 (4섹션)")
        sections = summ["payload"].get("sections", {})
        cards = [
            ("주요 증상", sections.get("symptoms", "")),
            ("위험 요인", sections.get("risk_factors", "")),
            ("개선 요인", sections.get("improvement_factors", "")),
            ("개입 요인", sections.get("intervention_factors", "")),
        ]
        for row_start in range(0, len(cards), 2):
            row_cols = st.columns(2)
            for col, (title, body) in zip(row_cols, cards[row_start:row_start + 2]):
                with col:
                    body_html = (body or "(분석 미완)").replace("\n", "<br>")
                    st.markdown(
                        f'<div class="summary-card">'
                        f'<div class="summary-card-title">{title}</div>'
                        f'<div class="summary-card-body">{body_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)


def _render_trend_chart(patient_id: str) -> None:
    sessions = sorted(
        db.list_sessions(patient_id), key=lambda s: s.get("session_date", "")
    )
    if len(sessions) < 2:
        st.caption("회기가 2개 이상 누적되면 추이가 표시됩니다.")
        return

    line_palette = {
        "우울한 기분": PRIMARY_DARK,
        "불안감": PRIMARY,
        "수면문제": "#06B6D4",
        "피로감": SUBTEXT,
        "자살생각": DANGER,
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
        color_discrete_map=line_palette,
        markers=True, range_y=[0, 3],
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT, family="Pretendard"),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard() -> None:
    page_with_optional_panel(_render_dashboard_body, st.session_state.selected_session)


# ── 페이지 4: 통계 ────────────────────────────────────────────────────────────


def render_stats_page() -> None:
    render_page_header(
        "통계",
        subtitle="전체 상담 활동 집계 및 분류 분포.",
    )

    stats = aggregate_global_stats()

    m = st.columns(4)
    m[0].metric("총 상담 건수", f"{stats['sessions']} 건")
    m[1].metric("총 상담 인원", f"{stats['patients']} 명")
    hours = stats["total_minutes"] // 60
    mins = stats["total_minutes"] % 60
    m[2].metric("총 상담 시간", f"{hours}시간 {mins}분")
    m[3].metric("건당 평균", f"{stats['avg_minutes']:.1f} 분")

    if stats["sessions"] == 0:
        st.info("아직 등록된 회기가 없습니다.")
        return

    st.divider()

    left, right = st.columns([0.66, 0.34], gap="large")

    with left:
        st.markdown("#### 분류 분포")
        dist = classification_distribution()
        dist_nonzero = dist[dist["건수"] > 0]
        if not dist_nonzero.empty:
            fig = px.pie(
                dist_nonzero, values="건수", names="분류",
                color="분류", color_discrete_map=CHART_PALETTE,
                hole=0.45,
            )
            fig.update_layout(
                height=360, margin=dict(l=10, r=10, t=10, b=10),
                font=dict(color=TEXT, family="Pretendard"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("분류 결과 없음.")

        st.markdown("#### 상위 요인 평균 점수 (전 회기)")
        top = factor_top_n(10)
        if not top.empty:
            fig = px.bar(
                top.sort_values("평균 점수"), x="평균 점수", y="요인",
                color="카테고리", color_discrete_map=CHART_PALETTE,
                orientation="h", range_x=[0, 3],
            )
            fig.update_layout(
                height=400, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color=TEXT, family="Pretendard"),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("28요인 분석 결과가 누적되면 표시됩니다.")

    with right:
        text = stats_insight(stats, dist)
        st.markdown(
            f'<div class="insight-card">'
            f'<h4>📌 통계 해석</h4>'
            f'<div class="insight-body">{text.replace(chr(10), "<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── 페이지 5: AI 보고서 ───────────────────────────────────────────────────────


def _render_report_body() -> None:
    pid = st.session_state.selected_client
    if not pid:
        render_page_header("AI 보고서", show_tags=True)
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        render_page_header("AI 보고서", show_tags=True)
        st.info("이 내담자의 회기가 없습니다.")
        return

    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']} (id={s['id']})": s
        for s in sessions
    }
    sel_label = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel_label]
    st.session_state.selected_session = s["id"]
    p = db.get_patient(pid)

    summ = db.get_latest_analysis(s["id"], "summary")
    if not summ:
        render_page_header(
            "AI 보고서",
            actions=[("🤖 AI 도우미", toggle_ai_panel, "primary")],
            show_tags=True,
        )
        st.warning("이 회기에 요약 결과 없음. 상담내역 페이지에서 AI 분석을 먼저 실행하세요.")
        return

    payload = summ["payload"]
    brief = payload.get("brief", "") or "(요약본 생성 실패)"
    text = payload.get("text", "")
    sections = payload.get("sections", {})
    source = payload.get("source", "?")
    ka_filled = payload.get("koalpaca_sections_filled", 0)
    gemma_filled = payload.get("gemma_sections_filled", 0)

    md_bytes = build_md(p, s, sections, brief)
    try:
        docx_bytes = build_docx(p, s, sections, brief, [])
    except Exception:
        docx_bytes = None
    pdf_bytes = build_pdf(p, s, sections)

    # 헤더 액션: 다운로드 + AI 도우미 토글
    head_cols = st.columns([0.55, 0.45])
    with head_cols[0]:
        st.markdown('<div class="page-title">AI 보고서</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="page-subtitle">요약 소스: '
            f'<span class="tag">{_source_label(source, ka_filled, gemma_filled)}</span></div>',
            unsafe_allow_html=True,
        )

    with head_cols[1]:
        a = st.columns(4)
        with a[0]:
            st.download_button(
                "📄 .md", md_bytes,
                file_name=f"report_{s['id']}.md",
                mime="text/markdown", use_container_width=True,
                key=f"dl_md_{s['id']}",
            )
        with a[1]:
            if docx_bytes:
                st.download_button(
                    "📝 .docx", docx_bytes,
                    file_name=f"report_{s['id']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key=f"dl_docx_{s['id']}",
                )
            else:
                st.button(".docx", disabled=True, use_container_width=True,
                          key=f"dl_docx_disabled_{s['id']}")
        with a[2]:
            if pdf_bytes:
                st.download_button(
                    "📕 .pdf", pdf_bytes,
                    file_name=f"report_{s['id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True, key=f"dl_pdf_{s['id']}",
                )
            else:
                st.button("📕 .pdf", disabled=True, use_container_width=True,
                          key=f"dl_pdf_disabled_{s['id']}",
                          help="weasyprint Windows GTK 의존성 미지원")
        with a[3]:
            if st.button("🤖 AI", key=f"toggle_ai_report_{s['id']}",
                         type="primary", use_container_width=True):
                toggle_ai_panel()
                st.rerun()

    st.divider()

    # 분석 해석 미리보기 — 대시보드와 동일 텍스트
    insight = dashboard_insight(s["id"])
    st.markdown(
        f'<div class="insight-card">'
        f'<h4>📌 분석 해석 (대시보드 연동)</h4>'
        f'<div class="insight-body">{insight}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("#### 보고서 본문 편집")

    edited_brief = st.text_area("요약본", value=brief, height=180, key="report_brief")
    edited_text = st.text_area("5섹션 텍스트", value=text, height=420, key="report_text")

    st.caption("※ 본 보고서는 AI 생성 초안이며, 상담사의 검토와 수정 후 사용하는 것을 전제로 합니다.")


def _source_label(source: str, ka_filled: int, gemma_filled: int) -> str:
    return {
        "koalpaca": "KoAlpaca (4/4)",
        "gemma": f"Gemma — KoAlpaca {ka_filled}/4 부분응답 무시",
        "gemma_fallback": "Gemma 폴백 (KoAlpaca 빈 응답)",
        "gemma_only": "Gemma 단독",
        "none": "응답 없음",
    }.get(source, source)


def render_report() -> None:
    page_with_optional_panel(_render_report_body, st.session_state.selected_session)


# ── 페이지 6: RAG 챗봇 (풀화면) ───────────────────────────────────────────────


def render_chatbot() -> None:
    render_page_header(
        "챗봇",
        subtitle="Gemma + KoSBERT + ChromaDB — AI Hub 라벨링·윤리규정·HIRA 검색.",
    )

    status = rag_healthcheck()
    c = st.columns(3)
    c[0].metric("LLM API", "✓" if status.get("llm") else "✗",
                help=status.get("llm_model", ""))
    c[1].metric("ChromaDB", "✓" if status.get("chroma") else "✗")
    c[2].metric("대화 수", f"{len(st.session_state.get('chat_history', []))} 건")

    if not status.get("llm"):
        st.warning("GEMINI_API_KEY 미설정 — .env 확인.")
        return
    if not status.get("chroma"):
        st.warning("RAG 인덱스 미생성. `python -m src.rag.ingest` 실행.")
        return

    if st.button("🗑 대화 초기화", key="chat_clear"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("#### 예상 질문")
    quick = [
        ("유사 상담 사례", "이 내담자와 비슷한 상담 사례가 있나요?"),
        ("임상 가이드", "수면 장애가 동반된 우울 사례의 임상 가이드를 알려주세요."),
        ("지난 회기 변화", "지난 3회기 동안 어떤 변화가 있었나요?"),
        ("다음 회기 질문", "다음 회기 질문을 추천해줘."),
        ("보고서 요약", "이번 회기 상담 내용을 보고서 형식으로 요약해줘."),
        ("위험 요인", "이 내담자의 주요 위험 요인을 정리해줘."),
        ("상담 목표", "다음 회기 상담 목표를 추천해줘."),
        ("개입 방향", "이 내담자에게 적합한 상담 개입 방향을 추천해줘."),
    ]
    for row_start in range(0, len(quick), 4):
        cols = st.columns(4)
        for col, (label, prompt) in zip(cols, quick[row_start:row_start + 4]):
            with col:
                if st.button(label, use_container_width=True,
                             key=f"qq_{hash(label)}"):
                    _handle_chatbot_query(prompt)
                    st.rerun()

    st.markdown("")
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("출처"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src['title']}`")
                        st.caption(src.get("desc", ""))

    prompt = st.chat_input("질문을 입력하세요.")
    if prompt:
        _handle_chatbot_query(prompt)
        st.rerun()


def _handle_chatbot_query(prompt: str) -> None:
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "user", "content": prompt})
    result = answer_query(prompt, k=5)
    if result.get("error"):
        history.append({
            "role": "assistant", "content": f"⚠️ {result['error']}", "sources": [],
        })
    else:
        history.append({
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
    apply_global_style()
    render_sidebar()

    page = st.session_state.page
    if page == "내담자 홈":
        render_patient_home()
    elif page == "상담내역 기록·추가":
        render_record_page()
    elif page == "분석 대시보드":
        render_dashboard()
    elif page == "통계":
        render_stats_page()
    elif page == "AI 보고서":
        render_report()
    elif page == "챗봇":
        render_chatbot()
    else:
        render_patient_home()


if __name__ == "__main__":
    main()
