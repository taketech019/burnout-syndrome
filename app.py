"""CounsHelper — Streamlit 진입점 (Phase 8: WeeNote/Mingl 정확 모방).

violet #7C3AED 톤 + WeeNote 상담일지 폼 + Mingl 프로필 + 통계 페이지 + AI 도우미 슬라이드 패널.
6 페이지: 내담자 홈 / 상담내역 기록·추가 / 분석 대시보드 / 통계 / AI 보고서 / 챗봇.
"""
import json
import os
from datetime import date, datetime
from typing import Any, Callable, Dict, Optional

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
APP_SUB = "학교상담시스템 v0.8"
USER_LINE_1 = "2026학년도"
USER_LINE_2 = "전문상담사 (데모)"
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "gemma")
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "gemini_api")
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "koalpaca_api")

st.set_page_config(
    page_title=f"{APP_NAME} — 상담 기록 분석 & 보고서 자동화",
    layout="wide",
    initial_sidebar_state="expanded",
)
db.init_db()


# ── 디자인 토큰 (WeeNote violet 톤) ───────────────────────────────────────────

PRIMARY = "#7C3AED"          # violet-600 (WeeNote 메인)
PRIMARY_DARK = "#5B21B6"     # violet-800
PRIMARY_LIGHT = "#F5F3FF"    # violet-50
PRIMARY_SOFT = "#EDE9FE"     # violet-100
PRIMARY_DEEPER = "#4C1D95"   # violet-900
ACCENT = "#06B6D4"           # cyan-500
DANGER = "#DC2626"
WARNING = "#D97706"
SUCCESS = "#10B981"
TEXT = "#1F2937"             # gray-800
SUBTEXT = "#6B7280"          # gray-500
BORDER = "#E5E7EB"
SIDEBAR_BG = "#FFFFFF"
PAGE_BG = "#FAFAFA"

CHART_PALETTE = {
    "우울": PRIMARY,
    "우울/위험": PRIMARY_DARK,
    "불안": "#06B6D4",
    "중독": "#F59E0B",
    "중독/기능": "#EF4444",
    "정상군": "#94A3B8",
}


def apply_global_style() -> None:
    st.markdown(
        f"""<style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

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
            font-size: 1.32rem; font-weight: 800; color: {PRIMARY};
            letter-spacing: -0.04em; line-height: 1.1;
        }}
        .brand-sub {{
            font-size: 0.7rem; color: {SUBTEXT}; margin-top: 0.2rem;
            font-weight: 500;
        }}
        .user-info {{
            font-size: 0.78rem; color: {TEXT}; line-height: 1.5;
            margin-top: 0.9rem;
        }}
        .user-info-line1 {{ color: {SUBTEXT}; font-size: 0.72rem; }}
        .user-info-name {{ font-weight: 600; color: {TEXT}; }}
        .sidebar-section-label {{
            font-size: 0.72rem; font-weight: 700; color: {SUBTEXT};
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 0.4rem; margin-top: 0.4rem;
        }}

        /* 사이드바 메뉴 버튼: 좌측 정렬 + 풀폭 */
        section[data-testid="stSidebar"] div.stButton > button {{
            text-align: left; justify-content: flex-start;
            padding-left: 0.95rem; padding-right: 0.95rem;
            font-size: 0.88rem; font-weight: 500;
            border-radius: 0.55rem;
            background: transparent; border: 1px solid transparent;
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] div.stButton > button:hover {{
            background: {PRIMARY_LIGHT}; color: {PRIMARY_DARK};
            border-color: transparent;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background: {PRIMARY_LIGHT}; color: {PRIMARY}; font-weight: 700;
            border: 1px solid {PRIMARY_SOFT};
            box-shadow: none;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
            background: {PRIMARY_SOFT};
        }}

        /* ── 페이지 타이틀 ── */
        .page-title {{
            font-size: 1.6rem; font-weight: 700; color: {TEXT};
            letter-spacing: -0.045em; margin: 0; line-height: 1.2;
        }}
        .page-title-chevron {{
            color: {SUBTEXT}; font-weight: 400; margin-right: 0.45rem;
        }}
        .page-subtitle {{
            color: {SUBTEXT}; font-size: 0.86rem; margin-top: 0.3rem;
            line-height: 1.4;
        }}

        /* ── 태그 ── */
        .tag {{
            display: inline-block; padding: 0.2rem 0.65rem;
            border-radius: 999px; background: {PRIMARY_LIGHT};
            color: {PRIMARY_DARK}; font-size: 0.72rem; font-weight: 620;
            border: 1px solid {PRIMARY_SOFT};
            margin-right: 0.3rem;
        }}
        .tag-soft {{
            background: #F3F4F6; color: {SUBTEXT};
            border-color: {BORDER};
        }}
        .tag-danger {{
            background: #FEF2F2; color: #991B1B; border-color: #FECACA;
        }}
        .tag-success {{
            background: #ECFDF5; color: #065F46; border-color: #A7F3D0;
        }}

        /* ── 카드 일반 ── */
        .info-card, .metric-card, .chart-card, .insight-card {{
            background: #FFFFFF; border: 1px solid {BORDER};
            border-radius: 0.85rem; padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.03);
        }}
        .insight-card {{
            background: #FFFFFF;
            border-color: {PRIMARY_SOFT};
        }}
        .insight-card h4 {{
            color: {TEXT}; font-weight: 700; letter-spacing: -0.02em;
            margin-top: 0; margin-bottom: 0.6rem; font-size: 1rem;
        }}
        .insight-body {{
            font-size: 0.85rem; line-height: 1.7; color: {TEXT};
        }}

        /* ── 회기 행 카드 (Mingl 스타일) ── */
        .session-row {{
            background: #FFFFFF; border: 1px solid {BORDER};
            border-radius: 0.65rem; padding: 0.9rem 1rem;
            margin-bottom: 0.55rem; display: flex; align-items: center;
            transition: all 0.15s ease;
        }}
        .session-row:hover {{
            border-color: {PRIMARY_SOFT}; background: {PRIMARY_LIGHT};
        }}

        /* ── 점선 첨부 영역 (WeeNote) ── */
        .attach-zone {{
            background: #FAFAFA;
            border: 1px dashed #D1D5DB;
            border-radius: 0.7rem;
            padding: 1rem 1.2rem; text-align: center;
            color: {SUBTEXT}; font-size: 0.86rem;
        }}

        /* ── 프로필 카드 (Mingl 내담자 홈) ── */
        .profile-card {{
            background: #FFFFFF; border: 1px solid {BORDER};
            border-radius: 1rem; padding: 1.3rem 1.5rem;
            display: flex; align-items: center; gap: 1.2rem;
        }}
        .profile-avatar {{
            width: 72px; height: 72px; border-radius: 999px;
            background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK});
            color: #FFFFFF; display: flex; align-items: center;
            justify-content: center; font-weight: 700; font-size: 1.5rem;
            letter-spacing: -0.04em; flex-shrink: 0;
        }}
        .profile-meta-label {{
            color: {SUBTEXT}; font-size: 0.74rem; font-weight: 600;
            margin-right: 0.45rem;
        }}
        .profile-meta-value {{
            color: {TEXT}; font-size: 0.86rem; font-weight: 500;
        }}

        /* ── 메트릭 ── */
        div[data-testid="stMetric"] {{
            background: #FFFFFF; padding: 0.95rem 1.1rem;
            border-radius: 0.85rem; border: 1px solid {BORDER};
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.03);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {SUBTEXT} !important; font-weight: 600 !important;
            font-size: 0.82rem !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT} !important; font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }}

        /* ── 일반 버튼 ── */
        div.stButton > button:first-child {{
            border-radius: 0.7rem; min-height: 2.3rem;
            font-size: 0.85rem; line-height: 1.2; font-weight: 600;
            border: 1px solid #D1D5DB; color: {TEXT}; background: #FFFFFF;
            transition: all 0.15s ease;
        }}
        div.stButton > button:hover {{
            border-color: {PRIMARY}; color: {PRIMARY_DARK};
            background-color: {PRIMARY_LIGHT};
        }}
        div.stButton > button[kind="primary"] {{
            background: {PRIMARY}; border-color: {PRIMARY}; color: #FFFFFF;
            box-shadow: 0 2px 6px -1px rgb(124 58 237 / 0.30);
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK};
            color: #FFFFFF;
        }}

        /* ── 다운로드 버튼 ── */
        div.stDownloadButton > button:first-child {{
            border-radius: 0.7rem; min-height: 2.3rem;
            font-size: 0.85rem; font-weight: 600;
            border: 1px solid {BORDER}; color: {TEXT};
            background: #FFFFFF;
        }}
        div.stDownloadButton > button:hover {{
            border-color: {PRIMARY}; color: {PRIMARY_DARK};
            background-color: {PRIMARY_LIGHT};
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

        /* ── AI 도우미 패널 ── */
        .ai-panel-header {{
            font-size: 1.02rem; font-weight: 700; color: {PRIMARY};
            display: flex; align-items: center; gap: 0.5rem;
            margin-bottom: 0.55rem;
        }}
        .context-chip {{
            display: inline-block; padding: 0.3rem 0.75rem;
            border-radius: 999px; background: {PRIMARY_LIGHT};
            color: {PRIMARY_DARK}; font-size: 0.74rem; font-weight: 620;
            border: 1px solid {PRIMARY_SOFT}; margin-bottom: 0.6rem;
        }}

        /* ── 4섹션 카드 ── */
        .summary-card {{
            background: {PRIMARY_LIGHT};
            border: 1px solid {PRIMARY_SOFT};
            border-radius: 0.85rem; padding: 0.95rem 1.05rem;
            min-height: 130px;
        }}
        .summary-card-title {{
            font-size: 0.84rem; font-weight: 720; color: {PRIMARY_DARK};
            letter-spacing: -0.015em; margin-bottom: 0.55rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid {PRIMARY_SOFT};
        }}
        .summary-card-body {{
            font-size: 0.82rem; color: #334155; line-height: 1.65;
        }}

        /* ── 위험 경고 ── */
        .alert-card {{
            background: #FEF2F2; border: 1px solid #FECACA;
            border-radius: 0.7rem; padding: 0.85rem 1rem;
            color: #991B1B; font-weight: 600; font-size: 0.86rem;
        }}

        /* ── 일지 삭제 텍스트 ── */
        .danger-text-btn {{
            color: {DANGER}; font-size: 0.84rem; font-weight: 600;
            cursor: pointer; padding: 0.5rem 0; display: inline-flex;
            align-items: center; gap: 0.3rem;
        }}

        /* ── 분리선 ── */
        hr {{ border-color: {BORDER}; margin: 1rem 0; }}

        /* ── 필터 칩 행 ── */
        .filter-chip-row {{
            display: flex; gap: 0.4rem; flex-wrap: wrap;
            margin: 0.7rem 0 0.3rem 0;
        }}

        /* Streamlit native chat message 색상 조정 */
        div[data-testid="stChatMessage"] {{
            background: #FFFFFF;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


# ── 기본 데이터 ───────────────────────────────────────────────────────────────

DEFAULT_DIALOGUE = pd.DataFrame({
    "화자": ["상담사", "내담자", "상담사", "내담자"],
    "발화": [
        "오늘은 어떤 이야기를 나누고 싶으세요?",
        "요즘 친구들과 어울리기가 어려워요.",
        "어떤 점에서 어려움을 느끼시나요?",
        "쉬는 시간에 혼자 있는 일이 많아요.",
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

SCOPE_OPTIONS = ["우울/불안", "우울", "불안", "중독", "관계", "학업", "위기"]


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
        "stats_filter": "학년도",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_page(page_name: str) -> None:
    st.session_state.page = page_name
    st.session_state.ai_panel_open = False


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
    }


def reanalyze_session(session_id: str) -> Dict[str, Any]:
    """기존 회기 transcript로 재분석."""
    s = db.get_session(session_id)
    if not s:
        return {}
    script = s["transcript"]
    cls = classify_text(script)
    fact = extract_factors(script, cls["classification"], backend=FACTOR_BACKEND)
    summ = summarize(script)
    summ["text"] = soften_diagnostic_expression(summ["text"])
    db.add_analysis(session_id, "classifier", cls["backend"], cls)
    db.add_analysis(session_id, "factors", fact["backend"], fact)
    db.add_analysis(session_id, "summary", summ["source"], summ)
    return {"classifier": cls, "factors": fact, "summary": summ}


def build_factor_dataframe(factors: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for k in FACTOR_KEYS:
        rows.append({
            "요인": k,
            "카테고리": FACTOR_CATEGORIES.get(k, "기타"),
            "점수": int(factors.get(k, 0)),
        })
    return pd.DataFrame(rows)


def _classification_flags(payload: dict) -> str:
    if not payload:
        return "(미분석)"
    c = payload.get("classification", {})
    flags = []
    if c.get("depression"): flags.append("우울")
    if c.get("anxiety"): flags.append("불안")
    if c.get("addiction"): flags.append("중독")
    return "/".join(flags) if flags else "정상군"


# ── 사이드바 ──────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f'<div class="brand">{APP_NAME}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="brand-sub">{APP_SUB}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="user-info">'
            f'<div class="user-info-line1">{USER_LINE_1}</div>'
            f'<div class="user-info-name">{USER_LINE_2}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # 내담자 선택
        st.markdown('<div class="sidebar-section-label">내담자 선택</div>',
                    unsafe_allow_html=True)
        patients = db.list_patients()
        if not patients:
            st.caption("등록된 내담자 없음")
            st.session_state.selected_client = None
        else:
            keyword = st.text_input(
                "검색",
                value=st.session_state.client_search,
                placeholder="alias / 성별 / 지역 / 메모",
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
                    f"{p['alias']} · {p['gender'][0]}/{p['age']}/{p['region']}": p["id"]
                    for p in filtered
                }
                default_idx = 0
                if st.session_state.selected_client:
                    for i, pid in enumerate(options.values()):
                        if pid == st.session_state.selected_client:
                            default_idx = i
                            break
                label = st.selectbox(
                    "내담자", list(options.keys()),
                    index=default_idx, label_visibility="collapsed",
                )
                st.session_state.selected_client = options[label]

        with st.expander("➕ 신규 등록"):
            with st.form("add_patient_form_sidebar"):
                new_alias = st.text_input("alias (필수)", placeholder="예: S-2407")
                new_gender = st.selectbox("성별", ["여성", "남성", "기타"])
                new_age = st.number_input("연령", min_value=6, max_value=20, value=14)
                new_region = st.text_input("지역", placeholder="예: 서울")
                new_note = st.text_area("메모", height=68,
                                        placeholder="예: 중2 · 학업 스트레스 호소")
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

        st.divider()
        st.button("⚙️  설정", key="nav_settings",
                  disabled=True, use_container_width=True)
        st.button("💬  문의", key="nav_inquiry",
                  disabled=True, use_container_width=True)
        st.button("❓  도움말", key="nav_help",
                  disabled=True, use_container_width=True)


# ── 페이지 헤더 (< 타이틀 + 우측 액션) ───────────────────────────────────────


def render_page_header(
    title: str,
    *,
    subtitle: str = "",
    actions: Optional[list] = None,
    show_chevron: bool = True,
) -> None:
    head_cols = st.columns([0.6, 0.4])
    with head_cols[0]:
        chevron = '<span class="page-title-chevron">‹</span>' if show_chevron else ""
        st.markdown(
            f'<div class="page-title">{chevron}{title}</div>',
            unsafe_allow_html=True,
        )
        if subtitle:
            st.markdown(f'<div class="page-subtitle">{subtitle}</div>',
                        unsafe_allow_html=True)

    with head_cols[1]:
        if actions:
            n = len(actions)
            action_cols = st.columns(n)
            for i, (label, callback, btn_type) in enumerate(actions):
                with action_cols[i]:
                    if st.button(label, key=f"action_{title}_{label}_{i}",
                                 type=btn_type or "secondary",
                                 use_container_width=True):
                        if callback:
                            callback()
                            st.rerun()

    st.markdown("")


# ── AI 도우미 패널 ────────────────────────────────────────────────────────────


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
                f'<span class="context-chip">📎 회기 첨부 — {label}</span>',
                unsafe_allow_html=True,
            )

    status = rag_healthcheck()
    if not (status.get("llm") and status.get("chroma")):
        st.warning("RAG 인덱스/LLM 키 점검 필요")

    if context_session_id:
        st.caption("이 회기에 대해:")
        for q, key_suffix in [
            ("주요 호소 문제 정리", "qq_main"),
            ("다음 회기 계획 추천", "qq_next"),
            ("유사 사례 검색", "qq_similar"),
            ("위험 신호 점검", "qq_risk"),
        ]:
            if st.button(q, key=f"ai_{key_suffix}", use_container_width=True):
                _handle_panel_query(q, context_session_id)
                st.rerun()

    st.markdown("")
    history = st.session_state.get("ai_panel_history", [])
    for msg in history[-8:]:
        role = msg["role"]
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("출처", expanded=False):
                        for src in msg["sources"][:3]:
                            st.caption(f"`{src.get('title', '?')}`")

    prompt = st.chat_input("무엇이든 물어보세요", key="ai_panel_input")
    if prompt:
        _handle_panel_query(prompt, context_session_id)
        st.rerun()

    cols = st.columns(2)
    with cols[0]:
        if st.button("➕ 새 대화", key="ai_panel_new",
                     use_container_width=True):
            st.session_state.ai_panel_history = []
            st.rerun()
    with cols[1]:
        if st.button("패널 닫기", key="ai_panel_close",
                     use_container_width=True):
            st.session_state.ai_panel_open = False
            st.rerun()


def _handle_panel_query(prompt: str, session_id: Optional[str]) -> None:
    enriched = prompt
    if session_id:
        summ = db.get_latest_analysis(session_id, "summary")
        if summ:
            brief = (summ["payload"].get("brief") or "")[:800]
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
    main_render_fn()
    if st.session_state.get("ai_panel_open"):
        st.divider()
        with st.expander("🤖 AI 도우미", expanded=True):
            render_ai_panel(context_session_id)


# ── 페이지 1: 내담자 홈 (Mingl 스타일) ────────────────────────────────────────


def render_patient_home() -> None:
    pid = st.session_state.selected_client
    if not pid:
        render_page_header("내담자 홈", subtitle="좌측에서 내담자를 선택하거나 ➕ 신규 등록")
        st.info("등록된 내담자를 선택하세요. 사이드바 '➕ 신규 등록'에서 추가 가능합니다.")
        return

    p = db.get_patient(pid)
    sessions = db.list_sessions(pid)

    render_page_header(
        f"{p['alias']}",
        subtitle=p.get("note", ""),
    )

    # 프로필 카드 (Mingl 스타일)
    avatar_text = p["alias"][:2].upper() if p["alias"] else "??"
    risk_tag = ""
    if sessions:
        latest_cls = db.get_latest_analysis(sessions[0]["id"], "classifier")
        if latest_cls and latest_cls["payload"].get("classification"):
            c = latest_cls["payload"]["classification"]
            flags = []
            if c.get("depression"): flags.append("우울")
            if c.get("anxiety"): flags.append("불안")
            if c.get("addiction"): flags.append("중독")
            if flags:
                risk_tag = f'<span class="tag tag-danger">{"/".join(flags)}</span>'
            else:
                risk_tag = '<span class="tag tag-success">정상군</span>'

    profile_meta = (
        f'<span class="profile-meta-label">등록일</span>'
        f'<span class="profile-meta-value">{p.get("created_at", "")[:10]}</span> &nbsp;·&nbsp; '
        f'<span class="profile-meta-label">성별</span>'
        f'<span class="profile-meta-value">{p["gender"]}</span> &nbsp;·&nbsp; '
        f'<span class="profile-meta-label">연령</span>'
        f'<span class="profile-meta-value">{p["age"]}세</span> &nbsp;·&nbsp; '
        f'<span class="profile-meta-label">지역</span>'
        f'<span class="profile-meta-value">{p["region"]}</span>'
    )

    st.markdown(
        f'<div class="profile-card">'
        f'<div class="profile-avatar">{avatar_text}</div>'
        f'<div style="flex:1;">'
        f'<div style="font-size:1.15rem; font-weight:700; color:{TEXT};">{p["alias"]} {risk_tag}</div>'
        f'<div style="margin-top:0.35rem;">{profile_meta}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # 탭: 정보 / 상담관리 / 검사관리 / 문서관리
    tabs = st.tabs(["📌 내담자 정보", "💬 상담관리", "🧪 검사관리", "📁 문서관리"])

    with tabs[0]:
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.markdown("**메모**")
            if p.get("note"):
                st.markdown(f'<div class="info-card">{p["note"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.caption("메모 없음")

            st.markdown("**상담 통계**")
            stat_cols = st.columns(3)
            stat_cols[0].metric("총 회기", f"{len(sessions)} 회")
            analyzed = sum(1 for s in sessions if db.get_latest_analysis(s["id"], "summary"))
            stat_cols[1].metric("분석 완료", f"{analyzed} 회")
            risks = sum(
                1 for s in sessions
                if (cls := db.get_latest_analysis(s["id"], "classifier"))
                and any((cls["payload"].get("classification") or {}).values())
            )
            stat_cols[2].metric("주의 회기", f"{risks} 회")

        with cols[1]:
            st.markdown("**위험 신호**")
            sui_count = 0
            for s in sessions:
                f = db.get_latest_analysis(s["id"], "factors")
                if f and (f["payload"].get("factors") or {}).get("자살생각", 0) > 0:
                    sui_count += 1
            if sui_count > 0:
                st.markdown(
                    f'<div class="alert-card">⚠️ 자살 사고 라벨이 {sui_count}회 표시되었습니다. '
                    f'안전 평가 우선.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="info-card" style="color:#10B981; font-weight:600;">'
                    '✓ 자살/자해 관련 신호 없음</div>',
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        # 필터 칩 + 회기 리스트 (Mingl 스타일)
        f_cols = st.columns([0.15, 0.85])
        with f_cols[0]:
            filter_choice = st.selectbox(
                "필터", ["전체", "분석 완료", "분석 필요"],
                label_visibility="collapsed", key="patient_home_filter",
            )

        filtered_sessions = sessions
        if filter_choice == "분석 완료":
            filtered_sessions = [s for s in sessions
                                 if db.get_latest_analysis(s["id"], "summary")]
        elif filter_choice == "분석 필요":
            filtered_sessions = [s for s in sessions
                                 if not db.get_latest_analysis(s["id"], "summary")]

        if not filtered_sessions:
            st.caption("회기 없음")
        else:
            for s in filtered_sessions:
                summ = db.get_latest_analysis(s["id"], "summary")
                status_tag = (
                    '<span class="tag tag-success">분석 완료</span>'
                    if summ else '<span class="tag tag-soft">분석 필요</span>'
                )
                cls = db.get_latest_analysis(s["id"], "classifier")
                cls_str = _classification_flags(cls["payload"]) if cls else "-"
                topic_display = s.get("topic") or "(주제 미입력)"

                row_cols = st.columns([0.14, 0.18, 0.38, 0.15, 0.15])
                row_cols[0].markdown(f"**{s.get('session_no') or '미분류'}**")
                row_cols[1].caption(s["session_date"])
                row_cols[2].markdown(
                    f"<div style='font-size:0.86rem;'>{topic_display}</div>"
                    f"<div style='font-size:0.74rem; color:{SUBTEXT};'>"
                    f"분류: {cls_str}</div>",
                    unsafe_allow_html=True,
                )
                row_cols[3].markdown(status_tag, unsafe_allow_html=True)
                with row_cols[4]:
                    if st.button("열기", key=f"home_open_{s['id']}",
                                 use_container_width=True):
                        st.session_state.selected_session = s["id"]
                        st.session_state.record_mode = "existing"
                        go_page("상담내역 기록·추가")
                        st.rerun()

    with tabs[2]:
        st.caption("심리 검사 결과 (예정 — 추후 통합)")

    with tabs[3]:
        st.caption("관련 문서 (예정 — 추후 통합)")


# ── 페이지 2: 상담내역 기록·추가 (WeeNote 일지 폼) ────────────────────────────


def _render_record_body() -> None:
    pid = st.session_state.selected_client
    if not pid:
        render_page_header("상담내역 기록·추가")
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    p = db.get_patient(pid)
    sessions = db.list_sessions(pid)
    sid = st.session_state.get("selected_session")
    is_new = st.session_state.record_mode == "new" or sid == "새 상담"

    if is_new:
        _render_new_record_form_weenote(pid, len(sessions))
    elif sid and any(s["id"] == sid for s in sessions):
        _render_existing_record_form_weenote(sid)
    else:
        # 디폴트 화면: 회기 리스트
        render_page_header(
            f"상담내역 — {p['alias']}",
            subtitle="회기를 선택하거나 신규 추가",
        )
        st.markdown("**상담 내역**")
        if not sessions:
            st.caption("기존 회기 없음. 아래 '+ 신규' 카드에서 추가하세요.")
        else:
            for s in sessions:
                row_cols = st.columns([0.13, 0.18, 0.40, 0.14, 0.15])
                row_cols[0].markdown(f"**{s.get('session_no') or '미분류'}**")
                row_cols[1].caption(s["session_date"])
                row_cols[2].markdown(
                    f"<div style='font-size:0.86rem;'>{s.get('topic') or '(주제 미입력)'}</div>",
                    unsafe_allow_html=True,
                )
                summ = db.get_latest_analysis(s["id"], "summary")
                status = "작성 완료" if summ else "분석 필요"
                row_cols[3].caption(status)
                with row_cols[4]:
                    if st.button("열기", key=f"rec_open_{s['id']}",
                                 use_container_width=True):
                        st.session_state.selected_session = s["id"]
                        st.session_state.record_mode = "existing"
                        st.rerun()

        st.markdown("")
        with st.container(border=True):
            c1, c2, c3 = st.columns([0.18, 0.60, 0.22])
            c1.markdown("**+ 신규**")
            c2.markdown("새 상담 내역 추가")
            c2.caption("회기 정보와 상담 내용을 입력해 새 기록을 생성합니다.")
            with c3:
                if st.button("추가하기", key="rec_add_new",
                             type="primary", use_container_width=True):
                    st.session_state.record_mode = "new"
                    st.session_state.selected_session = "새 상담"
                    st.rerun()


def _render_existing_record_form_weenote(sid: str) -> None:
    s = db.get_session(sid)
    if not s:
        st.error("회기를 찾을 수 없습니다.")
        return
    p = db.get_patient(s["patient_id"])

    # 헤더 + 액션
    summ = db.get_latest_analysis(sid, "summary")
    actions = [("🤖 AI 도우미", toggle_ai_panel, "primary")]
    if summ:
        actions = [("📊 대시보드", lambda: go_page("분석 대시보드"), "secondary")] + actions
    else:
        actions = [("🧠 AI 분석 실행", lambda: _trigger_reanalyze(sid), "primary")]

    render_page_header(
        "상담일지 열람",
        subtitle=f"{p['alias']} · {s.get('session_no') or '미분류'} · {s['session_date']}",
        actions=actions,
    )

    # WeeNote 폼 — 4x4 grid (read-only 표시)
    g1 = st.columns(4)
    g1[0].markdown(f"**일지번호**<br><span class='profile-meta-value'>{sid[:8]}</span>",
                   unsafe_allow_html=True)
    g1[1].markdown(f"**상담유형**<br><span class='profile-meta-value'>전문상담</span>",
                   unsafe_allow_html=True)
    g1[2].markdown(f"**상담분류**<br><span class='profile-meta-value'>{s.get('scope') or '미분류'}</span>",
                   unsafe_allow_html=True)
    g1[3].markdown(f"**상담자소속**<br><span class='profile-meta-value'>학교상담실</span>",
                   unsafe_allow_html=True)

    st.markdown("")
    g2 = st.columns(4)
    g2[0].markdown(f"**대분류**<br><span class='profile-meta-value'>상담</span>",
                   unsafe_allow_html=True)
    g2[1].markdown(f"**중분류**<br><span class='profile-meta-value'>개인상담</span>",
                   unsafe_allow_html=True)
    g2[2].markdown(f"**상담구분**<br><span class='profile-meta-value'>{s.get('scope') or '-'}</span>",
                   unsafe_allow_html=True)
    g2[3].markdown(f"**상담매체구분**<br><span class='profile-meta-value'>면담</span>",
                   unsafe_allow_html=True)

    st.markdown("")
    g3 = st.columns(4)
    g3[0].markdown(f"**상담일자**<br><span class='profile-meta-value'>{s['session_date']}</span>",
                   unsafe_allow_html=True)
    g3[1].markdown(f"**회기**<br><span class='profile-meta-value'>{s.get('session_no') or '-'}</span>",
                   unsafe_allow_html=True)
    g3[2].markdown(f"**등록일시**<br><span class='profile-meta-value'>{(s.get('created_at') or '')[:16]}</span>",
                   unsafe_allow_html=True)
    g3[3].markdown(f"**상태**<br><span class='profile-meta-value'>{'분석 완료' if summ else '분석 필요'}</span>",
                   unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**상담제목**")
    st.markdown(
        f'<div class="info-card">{s.get("topic") or "(주제 미입력)"}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    body_cols = st.columns([0.65, 0.35])
    with body_cols[0]:
        st.markdown("**상담내용**")
        st.text_area(
            "transcript", value=s.get("transcript", ""), height=420,
            label_visibility="collapsed",
        )
    with body_cols[1]:
        st.markdown("**비공개 메모**")
        st.text_area(
            "private_memo",
            value=p.get("note", "") if p else "",
            height=420,
            label_visibility="collapsed",
        )
        st.caption("인쇄·PDF·NEIS 엑셀에는 포함되지 않습니다.")

    st.markdown("")
    st.markdown("**첨부파일 (0)**")
    st.markdown(
        '<div class="attach-zone">📎 첨부파일 없음</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    del_cols = st.columns([0.15, 0.85])
    with del_cols[0]:
        if st.button("🗑 일지 삭제", key=f"del_{sid}"):
            db.delete_session(sid)
            st.session_state.selected_session = None
            st.session_state.record_mode = "existing"
            st.success("삭제됨")
            st.rerun()


def _render_new_record_form_weenote(pid: str, existing_count: int) -> None:
    p = db.get_patient(pid)
    render_page_header(
        "새 상담 내역 추가",
        subtitle=f"{p['alias']} · 새 회기 정보와 상담 내용을 입력합니다.",
    )

    g1 = st.columns(4)
    new_session_no = g1[0].text_input("회기 번호", value=f"{existing_count + 1}회기")
    new_session_date = g1[1].text_input("회기 일시", value=date.today().isoformat())
    new_scope = g1[2].selectbox("상담 범위", SCOPE_OPTIONS)
    new_topic = g1[3].text_input("상담 주제", placeholder="예: 친구 관계 어려움")

    st.markdown("")
    tab1, tab2 = st.tabs(["📝 전사 텍스트 붙여넣기", "💬 발화 단위 입력"])

    transcript_text = ""
    with tab1:
        transcript_text = st.text_area(
            "transcript_text",
            value=build_dialogue_text(st.session_state.dialogue_rows),
            height=380, label_visibility="collapsed",
            help="상담사: ... / 내담자: ... 형식으로 입력. PHQ-9 등 척도는 끝에 붙여도 자동 분리됨.",
        )

    with tab2:
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
            key="new_dialogue_editor", height=320,
        )
        st.session_state.dialogue_rows = edited

    final_script = transcript_text.strip() or build_dialogue_text(st.session_state.dialogue_rows)

    st.markdown("")
    btn_cols = st.columns([0.3, 0.3, 0.4])
    with btn_cols[0]:
        if st.button("💾 임시 저장", key="rec_save_only",
                     use_container_width=True):
            if not final_script.strip():
                st.error("상담 발화 입력 필요")
            else:
                sess = db.add_session(
                    pid, new_session_date, final_script,
                    session_no=new_session_no, scope=new_scope, topic=new_topic,
                )
                st.session_state.selected_session = sess["id"]
                st.session_state.record_mode = "existing"
                st.success(f"저장됨 (session={sess['id']})")
                st.rerun()
    with btn_cols[1]:
        if st.button("🧠 저장 + AI 분석 실행",
                     key="rec_save_analyze",
                     type="primary", use_container_width=True):
            if not final_script.strip():
                st.error("상담 발화 입력 필요")
            else:
                with st.spinner("F1 1차 + 28요인 + F3 요약 + DB 저장... (약 60초)"):
                    result = run_analysis(
                        script=final_script, patient_id=pid,
                        session_date=new_session_date, session_no=new_session_no,
                        scope=new_scope, topic=new_topic,
                    )
                st.session_state.selected_session = result["session_id"]
                st.session_state.record_mode = "existing"
                st.success(f"분석 + DB 저장 완료")
                go_page("분석 대시보드")
                st.rerun()
    with btn_cols[2]:
        if st.button("취소", key="rec_cancel", use_container_width=True):
            st.session_state.record_mode = "existing"
            st.session_state.selected_session = None
            st.rerun()


def _trigger_reanalyze(sid: str) -> None:
    """버튼 콜백 — st.rerun 후 분석 실행."""
    st.session_state["_pending_reanalyze"] = sid


def render_record_page() -> None:
    # pending 재분석 처리
    pending = st.session_state.pop("_pending_reanalyze", None)
    if pending:
        with st.spinner("AI 분석 실행 중... (약 60초)"):
            reanalyze_session(pending)
        st.success("분석 완료")

    sid = st.session_state.get("selected_session")
    context_sid = sid if sid and sid != "새 상담" else None
    page_with_optional_panel(_render_record_body, context_sid)


# ── 페이지 3: 분석 대시보드 ───────────────────────────────────────────────────


def _render_dashboard_body() -> None:
    pid = st.session_state.selected_client
    if not pid:
        render_page_header("분석 대시보드")
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        render_page_header("분석 대시보드")
        st.info("이 내담자의 회기가 없습니다.")
        return

    p = db.get_patient(pid)

    render_page_header(
        f"분석 대시보드",
        subtitle=f"{p['alias']} · 우울·불안·중독 위험도, 28요인, 회기별 변화 추이",
        actions=[("🤖 AI 도우미", toggle_ai_panel, "primary")],
    )

    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']}": s
        for s in sessions
    }
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]
    st.session_state.selected_session = s["id"]

    cls = db.get_latest_analysis(s["id"], "classifier")
    fact = db.get_latest_analysis(s["id"], "factors")
    summ = db.get_latest_analysis(s["id"], "summary")

    if not (cls and fact):
        st.warning("이 회기에 분석 결과 없음.")
        if st.button("🧠 지금 분석 실행", type="primary"):
            with st.spinner("분석 중..."):
                reanalyze_session(s["id"])
            st.rerun()
        return

    classification = cls["payload"].get("classification", {})
    scores = cls["payload"].get("scores", {})
    factors = fact["payload"].get("factors", {})

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

    left, right = st.columns([0.66, 0.34], gap="large")

    with left:
        st.markdown("**📊 상위 28요인 Top 10**")
        df = build_factor_dataframe(factors)
        selected_cats = st.multiselect(
            "카테고리 필터",
            list(CHART_PALETTE.keys())[:-1],
            default=["우울", "우울/위험", "불안", "중독", "중독/기능"],
            label_visibility="collapsed",
        )
        if selected_cats:
            df = df[df["카테고리"].isin(selected_cats)]
        top = df.sort_values("점수", ascending=False).head(10)
        if len(top) > 0:
            fig = px.bar(
                top.sort_values("점수"), x="점수", y="요인", color="카테고리",
                color_discrete_map=CHART_PALETTE, orientation="h", range_x=[0, 3],
            )
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color=TEXT, family="Pretendard"),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📈 회기별 변화 추이**")
        _render_trend_chart(pid)

    with right:
        insight_text = dashboard_insight(s["id"])
        st.markdown(
            f'<div class="insight-card">'
            f'<h4>📌 분석 해석</h4>'
            f'<div class="insight-body">{insight_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if any(classification.values()):
            primary = (
                "depression" if classification.get("depression")
                else "anxiety" if classification.get("anxiety") else "addiction"
            )
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

        if factors.get("자살생각", 0) > 0:
            st.markdown(
                '<div class="alert-card">⚠️ 자살 사고 라벨이 표시되었습니다. '
                '상담사가 별도 안전 평가를 수행하세요.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    if summ:
        st.markdown("**📋 AI 분석 요약 (4섹션)**")
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
            st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)


def _render_trend_chart(patient_id: str) -> None:
    sessions = sorted(db.list_sessions(patient_id), key=lambda s: s.get("session_date", ""))
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
        st.caption("분석된 회기가 없습니다.")
        return

    tdf = pd.DataFrame(rows)
    fig = px.line(
        tdf, x="회기", y="점수", color="요인",
        color_discrete_map=line_palette, markers=True, range_y=[0, 3],
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEXT, family="Pretendard"),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard() -> None:
    sid = st.session_state.get("selected_session")
    context_sid = sid if sid and sid != "새 상담" else None
    page_with_optional_panel(_render_dashboard_body, context_sid)


# ── 페이지 4: 통계 (WeeNote 정확 모방) ────────────────────────────────────────


def render_stats_page() -> None:
    render_page_header(
        "통계",
        subtitle="전체 상담 활동 집계 및 분류 분포",
        actions=[("📄 PDF", lambda: None, "secondary"),
                 ("📥 엑셀 다운로드", lambda: None, "primary")],
    )

    # 날짜 + 빠른 필터 칩
    date_cols = st.columns([0.18, 0.18, 0.64])
    with date_cols[0]:
        st.date_input("시작일", value=date(2026, 3, 1), key="stats_start")
    with date_cols[1]:
        st.date_input("종료일", value=date(2027, 2, 28), key="stats_end")

    # 빠른 필터 칩
    chip_cols = st.columns([0.08, 0.08, 0.08, 0.08, 0.08, 0.60])
    chip_labels = ["학년도", "지난달", "이번 달", "어제", "오늘"]
    for i, label in enumerate(chip_labels):
        with chip_cols[i]:
            active = st.session_state.get("stats_filter") == label
            if st.button(label, key=f"stats_chip_{label}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state["stats_filter"] = label
                st.rerun()

    st.markdown("")

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
        st.markdown("**🗂 분류 체계**")
        dist = classification_distribution()
        dist_nonzero = dist[dist["건수"] > 0]
        chart_cols = st.columns([0.5, 0.5])
        with chart_cols[0]:
            st.caption("대분류별 분포")
            if not dist_nonzero.empty:
                fig = px.pie(
                    dist_nonzero, values="건수", names="분류",
                    color="분류", color_discrete_map=CHART_PALETTE,
                    hole=0.5,
                )
                fig.update_layout(
                    height=330, margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(color=TEXT, family="Pretendard"),
                    showlegend=True,
                )
                fig.update_traces(textposition="outside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("분류 결과 없음")
        with chart_cols[1]:
            st.caption("상위 28요인 평균")
            top = factor_top_n(10)
            if not top.empty:
                fig = px.bar(
                    top.sort_values("평균 점수"), x="평균 점수", y="요인",
                    color="카테고리", color_discrete_map=CHART_PALETTE,
                    orientation="h", range_x=[0, 3],
                )
                fig.update_layout(
                    height=330, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color=TEXT, family="Pretendard"),
                    legend_title_text="",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("분석된 회기가 없습니다.")

        st.markdown("")
        st.markdown("**📋 분류 테이블**")
        st.dataframe(
            dist.rename(columns={"분류": "대분류", "건수": "건수"}),
            use_container_width=True, hide_index=True,
        )

    with right:
        text = stats_insight(stats, dist)
        st.markdown(
            f'<div class="insight-card">'
            f'<h4>📌 통계 해석</h4>'
            f'<div class="insight-body">{text.replace(chr(10), "<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── 페이지 5: AI 보고서 (WeeNote 생성 모달 패턴) ──────────────────────────────


def _render_report_body() -> None:
    pid = st.session_state.selected_client
    if not pid:
        render_page_header("AI 보고서")
        st.info("좌측 사이드바에서 내담자를 선택하세요.")
        return

    sessions = db.list_sessions(pid)
    if not sessions:
        render_page_header("AI 보고서")
        st.info("이 내담자의 회기가 없습니다.")
        return

    p = db.get_patient(pid)
    options = {
        f"{s.get('session_no') or '미분류'} · {s['session_date']}": s
        for s in sessions
    }
    sel = st.selectbox("회기 선택", list(options.keys()))
    s = options[sel]
    st.session_state.selected_session = s["id"]

    summ = db.get_latest_analysis(s["id"], "summary")
    if not summ:
        render_page_header(
            "AI 보고서",
            subtitle=f"{p['alias']} · {s.get('session_no') or '미분류'}",
            actions=[("🧠 분석 실행", lambda: _trigger_reanalyze(s["id"]), "primary")],
        )
        st.warning("이 회기에 요약 결과 없음.")
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

    # 헤더 + 다운로드 액션
    head_cols = st.columns([0.55, 0.45])
    with head_cols[0]:
        st.markdown(
            f'<div class="page-title"><span class="page-title-chevron">‹</span>AI 보고서</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="page-subtitle">{p["alias"]} · {s.get("session_no") or "미분류"} · '
            f'요약 소스: <span class="tag">{_source_label(source, ka_filled, gemma_filled)}</span></div>',
            unsafe_allow_html=True,
        )
    with head_cols[1]:
        ac = st.columns(4)
        with ac[0]:
            st.download_button(
                "📄 .md", md_bytes, file_name=f"report_{s['id']}.md",
                mime="text/markdown", use_container_width=True,
                key=f"dl_md_{s['id']}",
            )
        with ac[1]:
            if docx_bytes:
                st.download_button(
                    "📝 .docx", docx_bytes,
                    file_name=f"report_{s['id']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key=f"dl_docx_{s['id']}",
                )
            else:
                st.button(".docx", disabled=True, use_container_width=True,
                          key=f"dl_docx_d_{s['id']}")
        with ac[2]:
            if pdf_bytes:
                st.download_button(
                    "📕 .pdf", pdf_bytes, file_name=f"report_{s['id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True, key=f"dl_pdf_{s['id']}",
                )
            else:
                st.button("📕 .pdf", disabled=True, use_container_width=True,
                          key=f"dl_pdf_d_{s['id']}",
                          help="weasyprint 환경 의존성")
        with ac[3]:
            if st.button("🤖 AI", key=f"toggle_ai_report_{s['id']}",
                         type="primary", use_container_width=True):
                toggle_ai_panel()
                st.rerun()

    st.markdown("")

    insight = dashboard_insight(s["id"])
    st.markdown(
        f'<div class="insight-card">'
        f'<h4>📌 분석 해석 (대시보드 연동)</h4>'
        f'<div class="insight-body">{insight}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("**📝 보고서 본문 편집**")
    edit_cols = st.columns([0.5, 0.5])
    with edit_cols[0]:
        st.markdown("요약본 (한 단락)")
        st.text_area("brief", value=brief, height=180,
                     key=f"report_brief_{s['id']}", label_visibility="collapsed")
    with edit_cols[1]:
        st.markdown("보고서 형식 텍스트 (5섹션)")
        st.text_area("text", value=text, height=180,
                     key=f"report_text_{s['id']}", label_visibility="collapsed")

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
    sid = st.session_state.get("selected_session")
    context_sid = sid if sid and sid != "새 상담" else None
    page_with_optional_panel(_render_report_body, context_sid)


# ── 페이지 6: RAG 챗봇 (풀화면) ───────────────────────────────────────────────


def render_chatbot() -> None:
    render_page_header(
        "챗봇",
        subtitle="Gemini 2.5 Flash + KoSBERT + ChromaDB — AI Hub 라벨링·윤리규정·HIRA 검색",
    )

    status = rag_healthcheck()
    c = st.columns(3)
    c[0].metric("LLM API", "✓" if status.get("llm") else "✗",
                help=status.get("llm_model", ""))
    c[1].metric("ChromaDB", "✓" if status.get("chroma") else "✗")
    c[2].metric("대화 수", f"{len(st.session_state.get('chat_history', []))} 건")

    if not status.get("llm"):
        st.warning("GEMINI_API_KEY 미설정")
        return
    if not status.get("chroma"):
        st.warning("RAG 인덱스 미생성. `python -m src.rag.ingest` 실행.")
        return

    if st.button("🗑 대화 초기화", key="chat_clear"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("**📌 예상 질문**")
    quick = [
        ("유사 상담 사례", "이 내담자와 비슷한 상담 사례가 있나요?"),
        ("임상 가이드", "수면 장애가 동반된 우울 사례의 임상 가이드를 알려주세요."),
        ("위기 평가", "자해/자살 사고가 있는 학생에 대한 위기 평가 절차는?"),
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
        with st.chat_message(msg["role"],
                             avatar="🤖" if msg["role"] == "assistant" else "👤"):
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
