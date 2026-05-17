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
        "내담자 ID": ["C-001", "C-002", "C-003"],
        "성별": ["여성", "남성", "여성"],
        "연령대": ["30대", "20대", "40대"],
        "지역": ["서울", "경기", "부산"],
        "상담 유형": ["우울/불안", "불안", "중독"],
        "최근 회기": ["3회기", "2회기", "5회기"],
        "상태": ["검토 필요", "안정", "확인 필요"],
    }
)

SESSIONS = pd.DataFrame(
    {
        "내담자 ID": ["C-001", "C-001", "C-001", "C-002", "C-002", "C-003"],
        "회기": ["1회기", "2회기", "3회기", "1회기", "2회기", "5회기"],
        "상담일": [
            "2026-05-02",
            "2026-05-09",
            "2026-05-16",
            "2026-05-04",
            "2026-05-12",
            "2026-05-15",
        ],
        "상담 주제": [
            "초기 상담",
            "수면 문제",
            "업무 스트레스 및 불안",
            "불안 호소",
            "대인관계 불안",
            "중독 관련 상담",
        ],
        "보고서 상태": [
            "작성 완료",
            "작성 완료",
            "검토 필요",
            "작성 완료",
            "검토 필요",
            "검토 필요",
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
    st.session_state.page = "상담기록 작성"

if "selected_client" not in st.session_state:
    st.session_state.selected_client = "C-001"

if "selected_session" not in st.session_state:
    st.session_state.selected_session = "3회기"

if "dialogue_rows" not in st.session_state:
    st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def go_page(page_name: str):
    st.session_state.page = page_name


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
        "dialogue": st.session_state.dialogue_rows.to_dict(orient="records"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Streamlit UI 목업용 더미 데이터입니다.",
    }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


def clear_chat():
    st.session_state.chat_history = []


def add_mock_answer(user_prompt: str):
    st.session_state.page = "챗봇"

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                "현재는 목업 응답입니다. 실제 구현 시 RAG가 현재 회기 기록을 바탕으로 "
                "유사 상담 사례, 상담 가이드라인, 상담 이론 문서를 검색하고 답변을 생성합니다.\n\n"
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

    section[data-testid="stSidebar"] h1 {{
        color: {TEXT};
        font-weight: 720;
        letter-spacing: -0.035em;
        font-size: 1.2rem;
    }}

    section[data-testid="stSidebar"] h4 {{
        font-weight: 620;
        font-size: 0.9rem;
        color: {TEXT};
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

    .goal-section-title {{
        font-size: 1.0rem;
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

    .summary-card-body div {{
        margin-bottom: 0.22rem;
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

    div.stButton > button p {{
        font-size: 0.88rem;
        line-height: 1.2;
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

    div.stButton > button[kind="primary"]:hover {{
        background: {PRIMARY_DARK};
        border-color: {PRIMARY_DARK};
        color: white;
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

    div[data-testid="stMetric"] label {{
        color: {SUBTEXT};
        font-weight: 600;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT};
        font-weight: 650;
    }}

    div[data-testid="stMetricDelta"] {{
        color: {PRIMARY_DARK} !important;
        background: {PRIMARY_LIGHT};
        width: fit-content;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        font-weight: 600;
    }}

    textarea {{
        border-radius: 0.9rem !important;
        border-color: #CBD5E1 !important;
    }}

    input, textarea, select {{
        font-size: 0.95rem !important;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 0.85rem;
        border: 1px solid #BFDBFE;
        background: #EFF6FF;
        color: {PRIMARY_DARK};
    }}

    button[data-baseweb="tab"] {{
        color: {TEXT};
        font-weight: 560;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY_DARK} !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] > div {{
        color: {PRIMARY_DARK} !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: {PRIMARY} !important;
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

    .hira-line strong {{
        color: {TEXT};
        font-weight: 600;
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

    st.divider()

    if st.button("내담자 목록", key="nav_clients", use_container_width=True):
        go_page("내담자 목록")
        st.rerun()

    if st.button("회기 목록", key="nav_sessions", use_container_width=True):
        go_page("회기 목록")
        st.rerun()

    st.divider()

    st.markdown("#### 현재 시연 케이스")
    st.write(f"**내담자:** {st.session_state.selected_client}")
    st.write(f"**회기:** {st.session_state.selected_session}")

    st.divider()

    if st.button("메인 화면으로 돌아가기", key="nav_main", use_container_width=True):
        go_page("상담기록 작성")
        st.rerun()

    st.download_button(
        "JSON 내보내기",
        data=make_json_export(),
        file_name="counshelper_demo.json",
        mime="application/json",
        key="json_export",
        use_container_width=True,
    )

    st.caption("v0.1 MVP 목업")


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
client_row = get_client_row()

if st.session_state.page == "챗봇":
    header_col = st.columns([1])[0]

    with header_col:
        st.markdown(
            '<div class="app-title">CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <span class="tag">{st.session_state.selected_client}</span>
            <span class="tag">{st.session_state.selected_session}</span>
            <span class="tag">{client_row['연령대']} {client_row['성별']} · {client_row['지역']}</span>
            <span class="tag">{client_row['상담 유형']}</span>
            """,
            unsafe_allow_html=True,
        )
else:
    header_col, action_col1, action_col2 = st.columns([5.7, 1.2, 1.1])

    with header_col:
        st.markdown(
            '<div class="app-title">CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <span class="tag">{st.session_state.selected_client}</span>
            <span class="tag">{st.session_state.selected_session}</span>
            <span class="tag">{client_row['연령대']} {client_row['성별']} · {client_row['지역']}</span>
            <span class="tag">{client_row['상담 유형']}</span>
            """,
            unsafe_allow_html=True,
        )

    with action_col1:
        if st.button("AI 분석 실행", key="header_analyze", use_container_width=True):
            go_page("분석 대시보드")
            st.rerun()

    with action_col2:
        if st.button("💬 AI 도우미", key="header_chat", use_container_width=True):
            go_page("챗봇")
            st.rerun()


# ─────────────────────────────────────────────────────────────
# View Components
# ─────────────────────────────────────────────────────────────
def render_main_nav():
    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)

    n1, n2, n3 = st.columns(3)

    with n1:
        if st.button(
            "상담 기록 작성",
            key="top_nav_write",
            use_container_width=True,
            type="primary" if st.session_state.page == "상담기록 작성" else "secondary",
        ):
            go_page("상담기록 작성")
            st.rerun()

    with n2:
        if st.button(
            "분석 대시보드",
            key="top_nav_dashboard",
            use_container_width=True,
            type="primary" if st.session_state.page == "분석 대시보드" else "secondary",
        ):
            go_page("분석 대시보드")
            st.rerun()

    with n3:
        if st.button(
            "보고서·계획",
            key="top_nav_report",
            use_container_width=True,
            type="primary" if st.session_state.page == "보고서·계획" else "secondary",
        ):
            go_page("보고서·계획")
            st.rerun()

    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)


def render_client_list():
    st.markdown('<div class="section-title">내담자 목록</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">발표용 조회 화면입니다. 실제 시연은 메인 상담기록 작성 화면에서 진행합니다.</div>',
        unsafe_allow_html=True,
    )

    for _, row in CLIENTS.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.22, 0.28, 0.25, 0.25])

            with c1:
                st.markdown(f"### {row['내담자 ID']}")
                st.caption(f"{row['성별']} · {row['연령대']} · {row['지역']}")

            with c2:
                st.markdown("**상담 유형**")
                st.write(row["상담 유형"])

            with c3:
                st.markdown("**최근 회기**")
                st.write(row["최근 회기"])

            with c4:
                st.markdown("**상태**")
                st.write(row["상태"])


def render_session_list():
    st.markdown('<div class="section-title">전체 회기 목록</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">전체 상담 회기를 최신 상담일 순서로 보여주는 조회용 화면입니다.</div>',
        unsafe_allow_html=True,
    )

    df = SESSIONS.copy()
    df["_date"] = pd.to_datetime(df["상담일"], errors="coerce")
    df = df.sort_values("_date", ascending=False).drop(columns=["_date"]).reset_index(drop=True)

    for _, row in df.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([0.13, 0.18, 0.18, 0.34, 0.17])

            with c1:
                st.markdown(f"### {row['회기']}")

            with c2:
                st.write(row["상담일"])

            with c3:
                st.write(row["내담자 ID"])

            with c4:
                st.write(row["상담 주제"])

            with c5:
                st.write(row["보고서 상태"])


def render_record_write():
    render_main_nav()

    session_row = get_session_row()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">상담 기록 작성</div>
            <div class="hero-desc">
                상담 내용을 입력하면 AI가 우울·불안·중독 위험도와 28개 세부 증상 요인을 분석하고,
                이후 분석 대시보드와 요약 보고서 초안을 생성합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">회기 정보</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([0.18, 0.22, 0.22, 0.38])
    f1.text_input("회기 번호", value=st.session_state.selected_session, key="write_session_no")
    f2.text_input("회기 일시", value=session_row["상담일"], key="write_session_date")
    f3.selectbox("상담 범위", ["우울/불안", "우울", "불안", "중독"], key="write_scope")
    f4.text_input("상담 주제", value=session_row["상담 주제"], key="write_topic")

    st.markdown("")

    st.markdown('<div class="section-title">상담 내용 입력</div>', unsafe_allow_html=True)

    input_tab1, input_tab2 = st.tabs(["전사 텍스트 붙여넣기", "발화 단위 입력"])

    with input_tab1:
        st.text_area(
            "상담 내용",
            value=build_dialogue_text(st.session_state.dialogue_rows),
            height=330,
            key="write_text",
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
            key="dialogue_editor",
            height=300,
        )

        st.session_state.dialogue_rows = edited_rows

        with st.expander("자동 생성된 상담 텍스트 미리보기"):
            st.text_area(
                "미리보기",
                value=build_dialogue_text(edited_rows),
                height=180,
                key="dialogue_preview",
            )

    st.caption("※ 데모 목업에서는 비식별 상담 텍스트를 사용합니다.")

    run1, run2, run3 = st.columns([1.1, 1.0, 2.6])

    with run1:
        if st.button("AI 분석 실행", key="write_analyze", use_container_width=True, type="primary"):
            go_page("분석 대시보드")
            st.rerun()

    with run2:
        st.button("임시 저장", key="write_save", use_container_width=True)

    with run3:
        st.info("분석 완료 후 대시보드와 보고서 화면에서 결과를 확인할 수 있습니다.")


def render_dashboard():
    render_main_nav()

    st.markdown('<div class="section-title">분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">우울·불안·중독 위험도, 28개 세부 증상 요인, 회기별 변화 추이, 인구통계 맥락을 한 화면에서 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("우울 위험도", "2.0 / 3", "주의")
    m2.metric("불안 위험도", "1.7 / 3", "관찰")
    m3.metric("중독 위험도", "0.3 / 3", "낮음")
    m4.metric("검토 필요도", "67%", "상담사 확인 필요")

    st.markdown("")

    summary_col, hira_col = st.columns([0.72, 0.28], gap="large")

    with summary_col:
        st.markdown("#### AI 분석 요약")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="summary-card-title">주요 증상</div>
                    <div class="summary-card-body">
                        <div>수면 문제</div>
                        <div>피로감</div>
                        <div>출근 전 불안</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="summary-card-title">위험 요인</div>
                    <div class="summary-card-body">
                        <div>업무량 증가</div>
                        <div>수면 부족</div>
                        <div>직무 스트레스</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s3:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="summary-card-title">개선 요인</div>
                    <div class="summary-card-body">
                        <div>상담 참여 의지</div>
                        <div>수면 일지 작성 동의</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s4:
            st.markdown(
                """
                <div class="summary-card">
                    <div class="summary-card-title">개입 요인</div>
                    <div class="summary-card-body">
                        <div>감정 명명</div>
                        <div>자동사고 탐색</div>
                        <div>수면 패턴 확인</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with hira_col:
        st.markdown(
            """
            <div class="hira-card">
                <div class="hira-title">HIRA 인구통계 비교</div>
                <div class="hira-line"><strong>비교군</strong> 30대 여성 · 서울</div>
                <div class="hira-line"><strong>진료 맥락</strong> 동일 인구집단 대비 높은 편</div>
                <div class="hira-line"><strong>활용 목적</strong> 상담사가 참고하는 공공 통계 맥락</div>
                <div class="hira-caption">
                    공공 통계 기반 참고 정보이며, 개인 진단 근거로 사용하지 않습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    top_scores = SYMPTOM_SCORES.sort_values("점수", ascending=False).head(10)

    fig_bar = px.bar(
        top_scores.sort_values("점수"),
        x="점수",
        y="요인",
        color="카테고리",
        color_discrete_map=color_map,
        orientation="h",
        range_x=[0, 3],
    )
    fig_bar.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        legend_title_text="카테고리",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("28개 세부 증상 요인 자세히 보기"):
        fig_all = px.bar(
            SYMPTOM_SCORES.sort_values("점수"),
            x="점수",
            y="요인",
            color="카테고리",
            color_discrete_map=color_map,
            orientation="h",
            range_x=[0, 3],
        )
        fig_all.update_layout(
            height=620,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0F172A"),
            margin=dict(l=10, r=10, t=15, b=10),
            legend_title_text="카테고리",
        )
        st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("")

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

    st.markdown("")

    history_col, goal_col = st.columns([0.56, 0.44], gap="large")

    with history_col:
        st.markdown('<div class="chart-section-title">회기 기록 요약</div>', unsafe_allow_html=True)
        client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client]
        st.dataframe(client_sessions, use_container_width=True, hide_index=True)

    with goal_col:
        st.markdown('<div class="goal-section-title">장기 상담 목표 진행률</div>', unsafe_allow_html=True)

        fig_goal = px.bar(
            GOAL_PROGRESS,
            x="진행률",
            y="상담 목표",
            orientation="h",
            range_x=[0, 100],
            text="진행률",
            color_discrete_sequence=["#2563EB"],
        )
        fig_goal.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            width=0.32,
        )
        fig_goal.update_layout(
            height=260,
            margin=dict(l=10, r=35, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0F172A", size=11),
            xaxis_title="진행률",
            yaxis_title="",
            bargap=0.55,
        )
        st.plotly_chart(fig_goal, use_container_width=True)


def render_report_plan():
    render_main_nav()

    st.markdown('<div class="section-title">보고서·계획</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI가 생성한 4개 섹션 요약 보고서 초안을 상담사가 검토·수정합니다.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 요약 보고서 초안")

    edited_report = st.text_area(
        "보고서 초안",
        value=build_report_text(),
        height=540,
        key="report_text",
    )

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        ".md 다운로드",
        edited_report,
        "counshelper_report.md",
        mime="text/markdown",
        key="report_md_download",
        use_container_width=True,
    )
    d2.button(".pdf 다운로드", key="report_pdf_download", use_container_width=True)
    d3.button(".docx 다운로드", key="report_docx_download", use_container_width=True)

    st.caption("※ 본 보고서는 AI 생성 초안이며, 상담사의 검토와 수정 후 사용하는 것을 전제로 합니다.")


def render_chat_messages():
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="💬"):
            st.write(
                "안녕하세요. 현재 내담자의 상담 기록을 바탕으로 유사 사례, 임상 가이드, 다음 회기 계획을 도와드릴 수 있습니다."
            )
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
                    st.session_state.page = "챗봇"
                    st.rerun()


def render_chatbot():
    top_col1, top_col2, top_col3 = st.columns([0.66, 0.14, 0.20])

    with top_col1:
        st.markdown('<div class="section-title">AI 도우미 챗봇</div>', unsafe_allow_html=True)
        st.caption(f"현재 컨텍스트: {st.session_state.selected_client} · {st.session_state.selected_session}")

    with top_col2:
        st.button("대화 초기화", key="chat_clear", on_click=clear_chat, use_container_width=True)

    with top_col3:
        if st.button("메인으로 돌아가기", key="chat_back", use_container_width=True):
            go_page("상담기록 작성")
            st.rerun()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">RAG 기반 AI 도우미</div>
            <div class="hero-desc">
                유사 상담 케이스, 임상 가이드라인, 상담 이론 문서를 검색해 현재 회기 맥락에 맞는 답변을 제공합니다.
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
        st.session_state.page = "챗봇"
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "상담기록 작성":
    render_record_write()

elif st.session_state.page == "분석 대시보드":
    render_dashboard()

elif st.session_state.page == "보고서·계획":
    render_report_plan()

elif st.session_state.page == "내담자 목록":
    render_client_list()

elif st.session_state.page == "회기 목록":
    render_session_list()

elif st.session_state.page == "챗봇":
    render_chatbot()
