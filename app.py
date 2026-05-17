import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Dummy Data
# ─────────────────────────────────────────────────────────────
CLIENTS = pd.DataFrame(
    {
        "내담자 ID": ["C-001", "C-002", "C-003", "C-004"],
        "이름": ["김OO", "박OO", "이OO", "최OO"],
        "성별": ["여성", "남성", "여성", "남성"],
        "연령대": ["30대", "20대", "40대", "30대"],
        "지역": ["서울", "경기", "부산", "서울"],
        "상담 유형": ["우울/불안", "불안", "중독", "우울"],
        "최근 회기": ["3회기", "2회기", "5회기", "1회기"],
        "상태": ["검토 필요", "안정", "확인 필요", "초기 상담"],
    }
)

SESSIONS = pd.DataFrame(
    {
        "내담자 ID": ["C-001", "C-001", "C-001", "C-002", "C-002", "C-003", "C-004"],
        "회기": ["1회기", "2회기", "3회기", "1회기", "2회기", "5회기", "1회기"],
        "상담일": [
            "2026-05-02",
            "2026-05-09",
            "2026-05-16",
            "2026-05-04",
            "2026-05-12",
            "2026-05-15",
            "2026-05-10",
        ],
        "상담 주제": [
            "초기 상담",
            "수면 문제",
            "업무 스트레스 및 불안",
            "불안 호소",
            "대인관계 불안",
            "중독 관련 상담",
            "우울감 호소",
        ],
        "보고서 상태": [
            "작성 완료",
            "작성 완료",
            "검토 필요",
            "작성 완료",
            "검토 필요",
            "검토 필요",
            "작성 완료",
        ],
    }
)

DEFAULT_DIALOGUE = pd.DataFrame(
    {
        "화자": ["상담사", "내담자", "상담사", "내담자", "상담사", "내담자"],
        "발화": [
            "오늘은 어떤 이야기를 나누고 싶으세요?",
            "요즘 잠을 잘 못 자고, 아침에 일어나기가 너무 힘들어요.",
            "수면 문제는 언제부터 시작되었나요?",
            "회사 일이 많아진 뒤부터 계속 피곤하고 불안해요. 출근하기 전부터 가슴이 답답하고, 아무것도 하기 싫다는 생각이 자주 들어요.",
            "그럴 때 주로 어떤 생각이 드나요?",
            "내가 일을 잘 못하고 있는 것 같고, 사람들을 만나는 것도 조금 피하게 돼요.",
        ],
    }
)

SYMPTOM_SCORES = pd.DataFrame(
    {
        "요인": [
            "우울한 기분",
            "무가치감",
            "죄책감",
            "사고력 저하",
            "자살생각",
            "흥미감소",
            "정신운동변화",
            "체중/식욕변화",
            "수면문제",
            "피로감",
            "불안감",
            "비현실감",
            "통제력상실감",
            "불안조절곤란",
            "집중력저하",
            "사회적상황회피",
            "신체증상",
            "과민성",
            "조절실패",
            "갈망",
            "거짓말",
            "내성",
            "금단",
            "현저성",
            "자원투자",
            "자기관리",
            "사회적문제발생",
            "부정적 결과",
        ],
        "카테고리": [
            "우울",
            "우울",
            "우울",
            "우울",
            "우울/위험",
            "우울",
            "우울",
            "우울",
            "우울",
            "우울",
            "불안",
            "불안",
            "불안",
            "불안",
            "불안",
            "불안",
            "불안",
            "불안",
            "중독",
            "중독",
            "중독",
            "중독",
            "중독",
            "중독",
            "중독",
            "중독/기능",
            "중독/기능",
            "중독/기능",
        ],
        "점수": [
            3,
            2,
            1,
            2,
            0,
            2,
            1,
            1,
            3,
            3,
            3,
            0,
            2,
            2,
            2,
            2,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
        ],
    }
)

TREND = pd.DataFrame(
    {
        "회기": ["1회기", "2회기", "3회기", "4회기"],
        "우울": [2.8, 2.6, 2.4, 2.1],
        "불안": [2.7, 2.5, 2.4, 2.0],
        "수면문제": [3.0, 3.0, 3.0, 2.5],
        "피로감": [2.2, 2.6, 3.0, 2.4],
    }
)

GOAL_PROGRESS = pd.DataFrame(
    {
        "상담 목표": [
            "수면 일지 작성",
            "불안 유발 상황 파악",
            "자기비하적 사고 감소",
            "업무 스트레스 대처전략",
        ],
        "진행률": [70, 60, 45, 35],
    }
)

# ─────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "상담내역 기록·추가"

if "selected_client" not in st.session_state:
    st.session_state.selected_client = "C-001"

if "selected_session" not in st.session_state:
    st.session_state.selected_session = "3회기"

if "client_search" not in st.session_state:
    st.session_state.client_search = "C-001"

if "record_mode" not in st.session_state:
    st.session_state.record_mode = "existing"  # existing / new

if "dialogue_rows" not in st.session_state:
    st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def go_page(page_name: str):
    st.session_state.page = page_name


def select_session(session_name: str):
    st.session_state.selected_session = session_name
    st.session_state.record_mode = "existing"


def start_new_session():
    st.session_state.record_mode = "new"
    st.session_state.selected_session = "새 상담"


def build_dialogue_text(dialogue_df: pd.DataFrame) -> str:
    lines = []
    for _, row in dialogue_df.iterrows():
        speaker = str(row.get("화자", "")).strip()
        utterance = str(row.get("발화", "")).strip()
        if speaker and utterance and utterance.lower() != "nan":
            lines.append(f"{speaker}: {utterance}")
    return "\n".join(lines)


def get_client_row():
    row = CLIENTS[CLIENTS["내담자 ID"] == st.session_state.selected_client]
    return row.iloc[0]


def get_session_row():
    row = SESSIONS[
        (SESSIONS["내담자 ID"] == st.session_state.selected_client)
        & (SESSIONS["회기"] == st.session_state.selected_session)
    ]
    if row.empty:
        return pd.Series(
            {
                "내담자 ID": st.session_state.selected_client,
                "회기": "새 상담",
                "상담일": datetime.now().strftime("%Y-%m-%d"),
                "상담 주제": "",
                "보고서 상태": "신규 작성",
            }
        )
    return row.iloc[0]


def build_report_text():
    return """[상담보고서 초안]

1. 주요 증상
내담자는 최근 업무량 증가 이후 수면 문제, 피로감, 출근 전 불안을 호소하였다.
상담 중 자기비하적 사고와 사회적 회피 경향이 일부 관찰되었다.

2. 위험 요인
업무량 증가, 수면 부족, 직무 스트레스 누적, 사회적 회피 증가가 주요 위험 요인으로 정리된다.

3. 개선 요인
내담자는 상담 참여 의지가 있으며, 자신의 상태를 비교적 구체적으로 언어화할 수 있다.
다음 회기까지 수면 일지를 작성해보겠다고 동의하였다.

4. 상담사 개입 요인
이번 회기에서는 감정 명명, 자동사고 탐색, 수면 패턴 확인, 다음 회기 과제 설정이 이루어졌다.

5. 다음 회기 계획
다음 회기에서는 수면 일지 확인, 출근 전 불안 상황 탐색,
자기비하적 사고에 대한 인지 재구성을 중심으로 상담을 진행한다.
"""


def make_json_export():
    export_data = {
        "client": st.session_state.selected_client,
        "session": st.session_state.selected_session,
        "mode": st.session_state.record_mode,
        "dialogue": st.session_state.dialogue_rows.to_dict(orient="records"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Streamlit UI 목업용 더미 데이터입니다.",
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


def clear_chat():
    st.session_state.chat_history = []


def add_mock_answer(user_prompt: str):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                "현재는 목업 응답입니다. 실제 구현 시 RAG가 현재 상담기록을 바탕으로 "
                "유사 상담 사례, 임상 가이드라인, 상담 이론 문서를 검색하고 답변을 생성합니다.\n\n"
                "현재 회기에서는 수면 문제, 피로감, 출근 전 불안이 반복적으로 나타나며, "
                "다음 회기에서는 수면 양상, 불안 유발 상황, 회피 행동, 자기비하적 사고를 우선 확인하는 것이 적절합니다."
            ),
            "sources": [
                {
                    "title": "유사 상담 예시 #CASE-014",
                    "desc": "수면 문제·출근 전 불안·직무 스트레스가 함께 나타난 유사 상담 사례",
                },
                {
                    "title": "상담 가이드라인 PDF p.12",
                    "desc": "입면곤란, 중도각성, 회피행동, 안전확인 질문 참고",
                },
            ],
        }
    )


# ─────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────
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

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 44%);
    }}

    .main .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2.5rem;
        max-width: 1640px;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }}

    section[data-testid="stSidebar"] {{
        background: {SIDEBAR_BG};
        border-right: 1px solid {BORDER};
    }}

    .app-title {{
        font-size: 1.72rem;
        font-weight: 700;
        color: {TEXT};
        letter-spacing: -0.045em;
        margin-bottom: 0.35rem;
    }}

    .section-title {{
        font-size: 1.18rem;
        font-weight: 650;
        color: {TEXT};
        letter-spacing: -0.035em;
        margin-top: 0.1rem;
        margin-bottom: 0.55rem;
    }}

    .chart-section-title {{
        font-size: 1.02rem;
        font-weight: 620;
        color: {TEXT};
        margin-top: 0.8rem;
        margin-bottom: 0.25rem;
    }}

    .page-desc {{
        color: {SUBTEXT};
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }}

    .tag {{
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: {PRIMARY_LIGHT};
        color: {PRIMARY_DARK};
        font-size: 0.76rem;
        font-weight: 620;
        margin-right: 0.35rem;
        margin-bottom: 0.2rem;
        border: 1px solid {PRIMARY_SOFT};
    }}

    .hero-card {{
        background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 65%, #FFFFFF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 1.15rem;
        padding: 1.15rem 1.3rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.035);
    }}

    .hero-title {{
        font-size: 1.04rem;
        font-weight: 650;
        color: {PRIMARY_DARK};
        margin-bottom: 0.25rem;
    }}

    .hero-desc {{
        color: {SUBTEXT};
        font-size: 0.9rem;
        line-height: 1.55;
    }}

    .nav-spacer {{
        height: 2.4rem;
    }}

    .summary-card {{
        background: {CARD_BLUE};
        border: 1px solid {CARD_BLUE_BORDER};
        border-radius: 0.95rem;
        padding: 0.95rem 1.05rem;
        min-height: 132px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.025);
    }}

    .summary-card-title {{
        font-size: 0.86rem;
        font-weight: 720;
        color: {PRIMARY_DARK};
        letter-spacing: -0.015em;
        margin-bottom: 0.72rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #D6E6FF;
    }}

    .summary-card-body {{
        font-size: 0.82rem;
        font-weight: 480;
        color: #334155;
        line-height: 1.65;
    }}

    div.stButton > button:first-child {{
        border-radius: 999px;
        min-height: 2.45rem;
        font-size: 0.88rem;
        line-height: 1.2;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        color: {TEXT};
        background: #FFFFFF;
        white-space: nowrap;
    }}

    div.stButton > button:hover {{
        border-color: {PRIMARY};
        color: {PRIMARY_DARK};
        background-color: {PRIMARY_LIGHT};
    }}

    div.stButton > button[kind="primary"] {{
        background: {PRIMARY};
        border-color: {PRIMARY};
        color: white;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
    }}

    div.stDownloadButton > button:first-child {{
        border-radius: 999px;
        min-height: 2.35rem;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
    }}

    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        padding: 0.85rem 0.9rem;
        border-radius: 1rem;
        border: 1px solid {BORDER};
        box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.035);
    }}

    .hira-card {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 1rem;
        padding: 1.0rem 1.05rem;
        box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.03);
    }}

    .hira-title {{
        font-size: 0.98rem;
        font-weight: 650;
        color: {TEXT};
        margin-bottom: 0.65rem;
    }}

    .hira-line {{
        font-size: 0.8rem;
        color: {SUBTEXT};
        line-height: 1.6;
        margin-bottom: 0.28rem;
    }}

    .hira-caption {{
        font-size: 0.72rem;
        color: #94A3B8;
        line-height: 1.55;
        margin-top: 0.65rem;
    }}

    .quick-question-title {{
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
        color: {TEXT};
    }}

    .chat-helper-caption {{
        color: {SUBTEXT};
        font-size: 0.84rem;
        margin-bottom: 0.7rem;
    }}

    /* Streamlit 기본 포커스/탭 포인트 컬러를 블루 계열로 통일 */
    div[data-baseweb="tab-list"] button[aria-selected="true"] p,
    div[data-baseweb="tab-list"] button[aria-selected="true"] div {{
        color: {PRIMARY} !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: {PRIMARY} !important;
    }}

    textarea:focus, input:focus, div[data-baseweb="select"]:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 1px {PRIMARY} !important;
    }}

    .profile-card {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 1rem;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.75rem;
    }}

    .profile-row {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }}

    .avatar-circle {{
        width: 42px;
        height: 42px;
        border-radius: 999px;
        background: {PRIMARY};
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
    }}

    .profile-name {{
        font-weight: 700;
        color: {TEXT};
        font-size: 0.94rem;
        line-height: 1.3;
    }}

    .profile-sub {{
        color: {SUBTEXT};
        font-size: 0.75rem;
        line-height: 1.3;
    }}

    .subscription-card {{
        background: {PRIMARY_LIGHT};
        border: 1px solid {PRIMARY_SOFT};
        color: {PRIMARY_DARK};
        border-radius: 999px;
        padding: 0.6rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 650;
        text-align: center;
    }}

    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CounsHelper")
    st.caption("상담 기록 분석 & 보고서 자동화")

    # 로그인은 완료된 상태로 표시하고, 별도 클릭 버튼은 두지 않음
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

    st.markdown('<div class="subscription-card">구독 상태 · MVP Demo 플랜</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 내담자 선택")
    client_keyword = st.text_input(
        "내담자 검색",
        value=st.session_state.client_search,
        placeholder="예: C-001, 김OO, 서울",
        key="client_search_input",
    )

    st.session_state.client_search = client_keyword

    if client_keyword.strip():
        keyword = client_keyword.strip().lower()
        client_view = CLIENTS.copy()
        mask = client_view.apply(lambda row: keyword in " ".join(row.astype(str)).lower(), axis=1)
        client_view = client_view[mask]

        if not client_view.empty:
            selected_client_id = client_view.iloc[0]["내담자 ID"]
            if selected_client_id != st.session_state.selected_client:
                st.session_state.selected_client = selected_client_id
                client_sessions = SESSIONS[SESSIONS["내담자 ID"] == selected_client_id]
                if not client_sessions.empty:
                    st.session_state.selected_session = client_sessions.iloc[-1]["회기"]
                    st.session_state.record_mode = "existing"
                st.rerun()
        else:
            st.info("검색 결과가 없습니다.")

    st.divider()

    st.button("설정", key="settings_disabled", use_container_width=True, disabled=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
client_row = get_client_row()
st.markdown(
    '<div class="app-title">CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼</div>',
    unsafe_allow_html=True,
)
# 내담자가 선택된 상태에서만 타이틀 하단 태그 표시
if st.session_state.selected_client:
    st.markdown(
        f"""
        <span class="tag">{st.session_state.selected_client}</span>
        <span class="tag">{st.session_state.selected_session}</span>
        <span class="tag">{client_row['연령대']} {client_row['성별']} · {client_row['지역']}</span>
        <span class="tag">{client_row['상담 유형']}</span>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Top Navigation
# ─────────────────────────────────────────────────────────────
def render_main_nav():
    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)

    nav_items = [
        (n1, "상담내역 기록·추가", "상담내역 기록·추가", "top_nav_records"),
        (n2, "분석 대시보드", "분석 대시보드", "top_nav_dashboard"),
        (n3, "AI 보고서", "AI 보고서", "top_nav_report"),
        (n4, "챗봇", "챗봇", "top_nav_chat"),
    ]

    for col, label, page, key in nav_items:
        with col:
            if st.button(
                label,
                key=key,
                use_container_width=True,
                type="primary" if st.session_state.page == page else "secondary",
            ):
                go_page(page)
                st.rerun()

    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)


def render_session_cards():
    st.markdown('<div class="section-title">상담 내역</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">선택한 내담자의 기존 상담 기록을 확인하거나 새 상담 내역을 추가합니다.</div>',
        unsafe_allow_html=True,
    )

    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client].copy()
    client_sessions["_date"] = pd.to_datetime(client_sessions["상담일"], errors="coerce")
    client_sessions = client_sessions.sort_values("_date", ascending=False).drop(columns=["_date"])

    if client_sessions.empty:
        st.info("기존 상담 내역이 없습니다. 새 상담 내역을 추가해 주세요.")
    else:
        for _, row in client_sessions.iterrows():
            selected = st.session_state.record_mode == "existing" and st.session_state.selected_session == row["회기"]
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([0.14, 0.18, 0.34, 0.18, 0.16])
                with c1:
                    st.markdown(f"**{row['회기']}**")
                with c2:
                    st.write(row["상담일"])
                with c3:
                    st.write(row["상담 주제"])
                with c4:
                    st.write(row["보고서 상태"])
                with c5:
                    button_label = "선택됨" if selected else "기록 보기"
                    if st.button(button_label, key=f"select_{row['내담자 ID']}_{row['회기']}", use_container_width=True, disabled=selected):
                        select_session(row["회기"])
                        st.rerun()

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.18, 0.60, 0.22])
        with c1:
            st.markdown("**+ 신규**")
        with c2:
            st.write("새 상담 내역 추가")
            st.caption("회기 정보와 상담 내용을 입력해 새 기록을 생성합니다.")
        with c3:
            if st.button("추가하기", key="add_new_session", use_container_width=True):
                start_new_session()
                st.rerun()


def render_existing_record_preview():
    session_row = get_session_row()
    st.markdown('<div class="section-title">선택한 상담 기록 요약</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([0.18, 0.22, 0.22, 0.38])
    f1.metric("회기", session_row["회기"])
    f2.metric("상담일", session_row["상담일"])
    f3.metric("보고서 상태", session_row["보고서 상태"])
    f4.metric("상담 주제", session_row["상담 주제"])

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


def render_new_record_form():
    st.markdown('<div class="section-title">새 상담 내역 추가</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">새 회기 정보와 상담자·내담자 발화가 구분된 상담 내용을 입력합니다.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns([0.18, 0.22, 0.22, 0.38])
    f1.text_input("회기 번호", value="4회기", key="new_session_no")
    f2.text_input("회기 일시", value=datetime.now().strftime("%Y-%m-%d"), key="new_session_date")
    f3.selectbox("상담 범위", ["우울/불안", "우울", "불안", "중독"], key="new_scope")
    f4.text_input("상담 주제", value="", placeholder="예: 업무 스트레스 및 불안", key="new_topic")

    st.markdown("")
    input_tab1, input_tab2 = st.tabs(["전사 텍스트 붙여넣기", "발화 단위 입력"])

    with input_tab1:
        st.text_area(
            "상담 내용",
            value=build_dialogue_text(st.session_state.dialogue_rows),
            height=330,
            key="new_write_text",
            help="이미 정리된 상담 전사문이 있으면 그대로 붙여넣는 방식입니다.",
        )

    with input_tab2:
        st.caption("상담사/내담자를 선택하고 발화를 한 줄씩 입력하면 자동으로 상담 텍스트 형태로 합쳐집니다.")
        edited_rows = st.data_editor(
            st.session_state.dialogue_rows,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "화자": st.column_config.SelectboxColumn(
                    "화자",
                    options=["상담사", "내담자"],
                    required=True,
                    width="small",
                ),
                "발화": st.column_config.TextColumn(
                    "발화",
                    width="large",
                    required=True,
                ),
            },
            key="new_dialogue_editor",
            height=300,
        )
        st.session_state.dialogue_rows = edited_rows

    st.info("현재는 목업 화면입니다. 실제 저장 및 분석 실행은 모델 연동 후 활성화할 예정입니다.")
    st.button("상담 내역 저장 및 AI 분석 실행", key="new_save_analyze_disabled", use_container_width=True, disabled=True)


# ─────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────
def render_record_page():
    render_main_nav()
    render_session_cards()

    st.divider()

    if st.session_state.record_mode == "new":
        render_new_record_form()
    else:
        render_existing_record_preview()


def render_dashboard():
    render_main_nav()

    st.markdown('<div class="section-title">분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">선택한 상담 기록의 우울·불안·중독 위험도, 세부 증상 요인, 회기별 변화 추이, 인구통계 맥락을 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.record_mode == "new":
        st.info("현재는 새 상담 내역 입력 화면의 예시 분석 결과입니다. 실제 분석 값은 모델 연동 후 입력된 상담 내용을 기반으로 생성됩니다.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("우울 위험도", "2.0 / 3", "주의")
    m2.metric("불안 위험도", "1.7 / 3", "관찰")
    m3.metric("중독 위험도", "0.3 / 3", "낮음")
    m4.metric("검토 필요도", "67%", "상담사 확인 필요")

    st.markdown("")

    summary_col, hira_col = st.columns([0.68, 0.32], gap="large")

    with summary_col:
        st.markdown("#### AI 분석 요약")
        cards = [
            ("주요 증상", ["수면 문제", "피로감", "출근 전 불안"]),
            ("위험 요인", ["업무량 증가", "수면 부족", "직무 스트레스"]),
            ("개선 요인", ["상담 참여 의지", "수면 일지 작성 동의"]),
            ("개입 요인", ["감정 명명", "자동사고 탐색", "수면 패턴 확인"]),
        ]
        for row_start in range(0, len(cards), 2):
            row_cols = st.columns(2)
            for col, (title, items) in zip(row_cols, cards[row_start:row_start + 2]):
                with col:
                    body = "".join([f"<div>{item}</div>" for item in items])
                    st.markdown(
                        f"""
                        <div class="summary-card">
                            <div class="summary-card-title">{title}</div>
                            <div class="summary-card-body">{body}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    with hira_col:
        st.markdown("#### HIRA 인구통계 비교")
        hira_chart_data = pd.DataFrame(
            {
                "구분": ["내담자", "서울 30대 여성", "전국 30대 여성"],
                "비율": [67, 8.3, 6.1],
            }
        )
        fig_hira = px.bar(
            hira_chart_data,
            x="구분",
            y="비율",
            text="비율",
            color="구분",
            color_discrete_sequence=["#2563EB", "#60A5FA", "#CBD5E1"],
        )
        fig_hira.update_traces(texttemplate="%{text}%", textposition="outside", width=0.46)
        fig_hira.update_layout(
            height=315,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            yaxis_title="비율(%)",
            xaxis_title="",
            font=dict(color="#0F172A", size=11),
        )
        fig_hira.update_xaxes(tickangle=0, tickfont=dict(size=10))
        st.plotly_chart(fig_hira, use_container_width=True)
        st.caption("예시 데이터입니다. 실제 HIRA 통계 연동 후 비교 지표로 대체됩니다.")

    st.markdown("")

    color_map = {
        "우울": "#2563EB",
        "우울/위험": "#1E40AF",
        "불안": "#38BDF8",
        "중독": "#94A3B8",
        "중독/기능": "#64748B",
    }

    line_color_map = {
        "우울": "#1E40AF",
        "불안": "#2563EB",
        "수면문제": "#38BDF8",
        "피로감": "#64748B",
    }

    st.markdown('<div class="chart-section-title">상위 세부 증상 요인 Top 10</div>', unsafe_allow_html=True)
    selected_categories = st.multiselect(
        "카테고리 필터",
        ["우울", "불안", "중독"],
        default=["우울", "불안", "중독"],
    )

    if selected_categories:
        factor_source = SYMPTOM_SCORES[
            SYMPTOM_SCORES["카테고리"].apply(
                lambda value: any(category in str(value) for category in selected_categories)
            )
        ].copy()
    else:
        factor_source = SYMPTOM_SCORES.iloc[0:0].copy()

    top_scores = factor_source.sort_values("점수", ascending=False).head(10)
    fig_bar = px.bar(
        top_scores.sort_values("점수"),
        x="점수",
        y="요인",
        color="카테고리",
        color_discrete_map=color_map,
        orientation="h",
        range_x=[0, 3],
    )
    fig_bar.update_traces(width=0.42)
    fig_bar.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        legend_title_text="카테고리",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="chart-section-title">회기별 변화 추이</div>', unsafe_allow_html=True)
    trend_long = TREND.melt(id_vars="회기", var_name="요인", value_name="점수")
    fig_line = px.line(
        trend_long,
        x="회기",
        y="점수",
        color="요인",
        color_discrete_map=line_color_map,
        markers=True,
        range_y=[0, 3],
    )
    fig_line.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        legend_title_text="요인",
    )
    st.plotly_chart(fig_line, use_container_width=True)



def render_report():
    render_main_nav()

    st.markdown('<div class="section-title">AI 보고서</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI가 생성한 요약본과 보고서 형식의 텍스트 초안을 상담사가 검토·수정합니다.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.record_mode == "new":
        st.info("현재는 새 상담 내역 입력 화면의 예시 보고서입니다. 실제 보고서는 모델 연동 후 입력된 상담 내용을 기반으로 생성됩니다.")

    summary_text = """[요약본]
내담자는 최근 업무량 증가 이후 수면 문제, 피로감, 출근 전 불안을 반복적으로 호소하였다. 상담 중 자기비하적 사고와 사회적 회피 경향이 관찰되었으며, 현재는 우울·불안 관련 요인을 중심으로 지속적인 모니터링이 필요한 상태로 보인다. 다만 상담 참여 의지가 있고 자신의 상태를 비교적 구체적으로 표현할 수 있어, 다음 회기에서는 수면 패턴 확인과 자동사고 탐색을 중심으로 개입할 수 있다.
"""

    report_style_text = build_report_text()

    st.markdown("#### 요약본")
    edited_summary = st.text_area("요약본", value=summary_text, height=180, key="summary_text")

    st.markdown("#### 보고서 형식 텍스트")
    edited_report = st.text_area("보고서 형식 텍스트", value=report_style_text, height=430, key="report_text")

    final_report = edited_summary + chr(10) + chr(10) + edited_report

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        ".md 다운로드",
        final_report,
        "counshelper_report.md",
        mime="text/markdown",
        key="report_md_download",
        use_container_width=True,
    )
    d2.button(".pdf 다운로드", key="report_pdf_download_disabled", use_container_width=True, disabled=True)
    d3.button(".docx 다운로드", key="report_docx_download_disabled", use_container_width=True, disabled=True)

    st.caption("※ 본 보고서는 AI 생성 초안이며, 상담사의 검토와 수정 후 사용하는 것을 전제로 합니다.")


def render_chat_messages():
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="💬"):
            st.write("안녕하세요. 현재 내담자의 상담 기록을 바탕으로 유사 사례, 임상 가이드, 다음 회기 계획을 도와드릴 수 있습니다.")
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="💬"):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("참고 출처 보기"):
                        for source in msg["sources"]:
                            with st.container(border=True):
                                st.markdown(f"**{source['title']}**")
                                st.caption(source["desc"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])


def render_quick_question_buttons():
    st.markdown('<div class="quick-question-title">예상 질문</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-helper-caption">상담사가 자주 확인하는 질문을 바로 실행할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    questions = [
        ("# 유사 상담 사례", "이 내담자와 비슷한 상담 사례가 있나요?", "chip_case"),
        ("# 임상 가이드", "수면 장애가 동반된 우울 사례의 임상 가이드를 알려주세요.", "chip_guide"),
        ("# 지난 회기 변화", "지난 3회기 동안 어떤 변화가 있었나요?", "chip_history"),
        ("# 다음 회기 질문", "다음 회기 질문을 추천해줘.", "chip_next"),
        ("# 보고서 요약", "이번 회기 상담 내용을 보고서 형식으로 요약해줘.", "chip_report"),
        ("# 위험 요인", "이 내담자의 주요 위험 요인을 정리해줘.", "chip_risk"),
        ("# 상담 목표", "다음 회기 상담 목표를 추천해줘.", "chip_goal"),
        ("# 개입 방향", "이 내담자에게 적합한 상담 개입 방향을 추천해줘.", "chip_intervention"),
    ]

    for row_start in range(0, len(questions), 4):
        cols = st.columns(4)
        for col, (label, prompt, key) in zip(cols, questions[row_start:row_start + 4]):
            with col:
                if st.button(label, key=key, use_container_width=True):
                    add_mock_answer(prompt)
                    st.rerun()


def render_chatbot():
    render_main_nav()

    top_col1, top_col2 = st.columns([0.78, 0.22])
    with top_col1:
        st.markdown('<div class="section-title">챗봇</div>', unsafe_allow_html=True)
        st.caption(f"현재 컨텍스트: {st.session_state.selected_client} · {st.session_state.selected_session}")
    with top_col2:
        st.button("대화 초기화", key="chat_clear", on_click=clear_chat, use_container_width=True)

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
        add_mock_answer(prompt)
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "상담내역 기록·추가":
    render_record_page()
elif st.session_state.page == "분석 대시보드":
    render_dashboard()
elif st.session_state.page == "AI 보고서":
    render_report()
elif st.session_state.page == "챗봇":
    render_chatbot()
