# =========================================================
# Streamlit 화면 구현 + 모델 교체 가능 구조
# 프로젝트: CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼
# =========================================================

import json
import base64
import hashlib
import os
import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

try:
    from src.report import make_docx_report_bytes, make_pdf_report_bytes
except Exception:
    def make_docx_report_bytes(*args, **kwargs):
        return None

    def make_pdf_report_bytes(*args, **kwargs):
        return None

from utils.hira_utils import (
    get_hira_context,
    format_hira_report_text,
    infer_hira_context_keys,
)

#============
@st.cache_data(ttl=1)
def load_hira_model_context():
    project_root = Path(__file__).resolve().parent
    hira_path = project_root / "data" / "processed" / "hira" / "hira_model_context.csv"
    return pd.read_csv(hira_path, encoding="utf-8-sig")
# =========================================================
# 1. Secrets / 환경 설정
# =========================================================
load_dotenv()


def get_secret(key: str, default=None):
    """
    우선순위: st.secrets → os.environ (.env 포함) → default
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


APP_NAME = get_secret("APP_NAME", "CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼")
APP_ENV = get_secret("APP_ENV", "demo")
DEMO_MODE = str(get_secret("DEMO_MODE", "true")).lower() == "true"

# 모델 백엔드
# 지금은 mock으로 동작.
# 나중에 3단계 KlueBERT 모델 코드, 4단계 Koalpaca 모델 코드를 추가한 뒤
# MODEL_BACKEND 값을 aihub_local 등으로 바꾸면 됨.
MODEL_BACKEND = get_secret("MODEL_BACKEND", "mock")
FACTOR_BACKEND = get_secret("FACTOR_BACKEND", "mock")
CLASSIFIER_BACKEND = get_secret("CLASSIFIER_BACKEND", "mock")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")

CHROMA_DB_PATH = get_secret("CHROMA_DB_PATH", "./chroma_db")
RAG_EMBEDDING_MODEL = get_secret(
    "RAG_EMBEDDING_MODEL",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

KLUEBERT_MODEL_NAME = get_secret("KLUEBERT_MODEL_NAME", "AIHub-KlueBERT-demo")
KOALPACA_MODEL_NAME = get_secret("KOALPACA_MODEL_NAME", "Koalpaca-demo")

# =========================================================
# 공통 컬러 시스템
# 첨부한 레퍼런스처럼 부드럽지만 서로 구분되는 색상으로 재설계
# =========================================================

CONTEXT_THEME = {
    "depression": {
        "accent": "#4F6EF7",
        "accent_dark": "#2F4ED8",
        "soft_bg": "#EEF3FF",
        "border": "#A9BAFF",
    },
    "anxiety": {
        "accent": "#A176F2",
        "accent_dark": "#7B57D1",
        "soft_bg": "#F5EFFF",
        "border": "#D8C4FF",
    },
    "addiction": {
        "accent": "#36B9D6",
        "accent_dark": "#1989A3",
        "soft_bg": "#ECFBFF",
        "border": "#B6ECF5",
    },
    "sleep": {
        "accent": "#62C9BC",
        "accent_dark": "#329E92",
        "soft_bg": "#EDFCF8",
        "border": "#BCEEE5",
    },
    "fatigue": {
        "accent": "#7E91C7",
        "accent_dark": "#5E73A9",
        "soft_bg": "#F1F5FF",
        "border": "#CAD7F4",
    },
    "intervention": {
        "accent": "#FF9DB5",
        "accent_dark": "#E46E8E",
        "soft_bg": "#FFF1F5",
        "border": "#FFD2DE",
    },
    "other": {
        "accent": "#94A3B8",
        "accent_dark": "#64748B",
        "soft_bg": "#F8FAFC",
        "border": "#CBD5E1",
    },
}

SERIES_COLOR_MAP = {
    "우울": "#6B8EF7",
    "불안": "#B08AF5",
    "중독": "#5ECADF",
    "수면문제": "#7DD6CC",
    "피로감": "#9AA9D6",
    "상담사 개입": "#F6A6BE",
    "변화/기타": "#CBD5E1",
}

FACTOR_CATEGORY_COLOR_MAP = {
    "우울": "#6B8EF7",
    "불안": "#B08AF5",
    "중독": "#5ECADF",
    "상담사 개입": "#F6A6BE",
    "변화/기타": "#CBD5E1",
}

DONUT_PALETTE = [
    "#BFDBFE",  # soft blue
    "#DDD6FE",  # soft violet
    "#99F6E4",  # soft teal
    "#FBCFE8",  # soft pink
    "#FDE68A",  # soft amber
    "#BBF7D0",  # soft green
    "#BAE6FD",  # soft sky
    "#CBD5E1",  # soft slate
    "#F0ABFC",  # soft orchid
    "#FDA4AF",  # soft rose
]

# HELPER
def get_context_theme(context_key: str) -> Dict[str, str]:
    return CONTEXT_THEME.get(str(context_key or "").strip(), CONTEXT_THEME["other"])

# =========================================================
# 2. Streamlit 기본 화면 설정
# =========================================================
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. 데모 데이터 정의
# =========================================================
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

# =========================================================
# processed 라벨링데이터 연결
# =========================================================
PROCESSED_SESSIONS_PATH = Path("data/processed/sessions.jsonl")


def _script_to_dialogue_df(script: str) -> pd.DataFrame:
    """
    sessions.jsonl의 script 문자열을 Streamlit data_editor용 DataFrame으로 변환한다.
    """
    rows = []

    for line in str(script or "").split("\n"):
        line = line.strip()

        if not line:
            continue

        if ":" in line:
            speaker, utterance = line.split(":", 1)
            speaker = speaker.strip()
            utterance = utterance.strip()
        else:
            speaker = "내담자"
            utterance = line

        if speaker not in ["상담사", "내담자"]:
            speaker = "내담자"

        rows.append(
            {
                "화자": speaker,
                "발화": utterance,
            }
        )

    if not rows:
        return DEFAULT_DIALOGUE.copy()

    return pd.DataFrame(rows)


def _format_session_label(value: Any) -> str:
    """
    session 값을 화면용 회기명으로 변환한다.
    """
    text = str(value or "").strip()

    if not text or text == "unknown":
        return "회기미상"

    if text.endswith("회기"):
        return text

    return f"{text}회기"


def load_processed_sessions() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    data/processed/sessions.jsonl이 있으면 실제 라벨링데이터 기반으로
    CLIENTS, SESSIONS, SESSION_DIALOGUES를 생성한다.

    파일이 없거나 읽기 실패 시 기존 데모 데이터를 유지한다.
    """
    if not PROCESSED_SESSIONS_PATH.exists():
        return CLIENTS, SESSIONS, {}

    try:
        raw = pd.read_json(PROCESSED_SESSIONS_PATH, lines=True)

        if raw.empty:
            return CLIENTS, SESSIONS, {}

        raw["client_id"] = raw["client_id"].astype(str)
        raw["회기"] = raw["session"].apply(_format_session_label)

        # 내담자 목록 생성
        clients = (
            raw.sort_values(["client_id", "session"])
            .groupby("client_id", as_index=False)
            .agg(
                {
                    "gender": "first",
                    "age": "first",
                    "class": "first",
                    "split": "first",
                    "회기": "last",
                }
            )
        )

        def age_band(age: Any) -> str:
            try:
                age = int(age)
                return f"{age // 10 * 10}대"
            except Exception:
                return "미상"

        def counseling_type(row: pd.Series) -> str:
            cls = str(row.get("class", "")).upper()

            if "DEPRESSION" in cls:
                return "우울"
            if "ANXIETY" in cls:
                return "불안"
            if "ADDICTION" in cls:
                return "중독"
            if "NORMAL" in cls:
                return "일반군"

            return cls or "미상"

        clients["내담자 ID"] = clients["client_id"]
        clients["이름"] = clients["client_id"]
        clients["성별"] = clients["gender"].replace({"여": "여성", "남": "남성"})
        clients["연령대"] = clients["age"].apply(age_band)
        clients["지역"] = clients["split"]
        clients["상담 유형"] = clients.apply(counseling_type, axis=1)
        clients["최근 회기"] = clients["회기"]
        clients["상태"] = "전처리 데이터"

        clients = clients[
            [
                "내담자 ID",
                "이름",
                "성별",
                "연령대",
                "지역",
                "상담 유형",
                "최근 회기",
                "상태",
            ]
        ].reset_index(drop=True)

        def make_session_date(row: pd.Series) -> str:
            """
            원본 데이터에는 실제 상담일이 없고 split이 Training/Validation으로 들어오는 경우가 있어,
            화면 표시용 날짜를 생성한다.
            """
            for candidate_col in ["date", "created_at", "session_date", "상담일"]:
                value = str(row.get(candidate_col, "") or "").strip()
                if value and value.lower() not in ["nan", "none", "training", "validation"]:
                    try:
                        return pd.to_datetime(value).strftime("%Y-%m-%d")
                    except Exception:
                        return value

            try:
                session_no = int(float(row.get("session", 1) or 1))
            except Exception:
                session_no = 1

            # 데모용 기준 날짜. 실제 상담일 데이터가 없으므로 회기 번호 기준으로 7일 간격 표시.
            base_date = pd.Timestamp("2026-05-01")
            return (base_date + pd.Timedelta(days=(session_no - 1) * 7)).strftime("%Y-%m-%d")

        def make_session_date(row: pd.Series) -> str:
            """
            원본 데이터에는 실제 상담일이 없고 split이 Training/Validation으로 들어오는 경우가 있어,
            화면 표시용 날짜를 생성한다.
            """
            for candidate_col in ["date", "created_at", "session_date", "상담일"]:
                value = str(row.get(candidate_col, "") or "").strip()
                if value and value.lower() not in ["nan", "none", "training", "validation"]:
                    try:
                        return pd.to_datetime(value).strftime("%Y-%m-%d")
                    except Exception:
                        return value

            try:
                session_no = int(float(row.get("session", 1) or 1))
            except Exception:
                session_no = 1

            # 데모용 기준 날짜. 실제 상담일 데이터가 없으므로 회기 번호 기준으로 7일 간격 표시.
            base_date = pd.Timestamp("2026-05-01")
            return (base_date + pd.Timedelta(days=(session_no - 1) * 7)).strftime("%Y-%m-%d")

        # 회기 목록 생성
        sessions = raw.copy()
        sessions["내담자 ID"] = sessions["client_id"]
        sessions["상담일"] = sessions.apply(make_session_date, axis=1)
        sessions["상담 주제"] = sessions["class"].fillna("상담 회기")
        sessions["보고서 상태"] = "전처리 완료"

        sessions = sessions[
            [
                "내담자 ID",
                "회기",
                "상담일",
                "상담 주제",
                "보고서 상태",
                "script",
                "summary",
                "filename",
                "split",
                "class",
                "depression",
                "anxiety",
                "addiction",
            ]
        ].reset_index(drop=True)

        # 회기별 상담 발화 DataFrame 생성
        session_dialogues: Dict[Tuple[str, str], pd.DataFrame] = {}

        for _, row in sessions.iterrows():
            key = (row["내담자 ID"], row["회기"])
            session_dialogues[key] = _script_to_dialogue_df(row.get("script", ""))

        return clients, sessions, session_dialogues

    except Exception as error:
        st.warning(f"processed sessions 데이터를 읽지 못해 데모 데이터를 사용합니다: {error}")
        return CLIENTS, SESSIONS, {}
    
CLIENTS, SESSIONS, SESSION_DIALOGUES = load_processed_sessions()

DEFAULT_CLIENT_ID = CLIENTS.iloc[0]["내담자 ID"] if not CLIENTS.empty else "C-001"

_DEFAULT_CLIENT_SESSIONS = SESSIONS[SESSIONS["내담자 ID"] == DEFAULT_CLIENT_ID]
if not _DEFAULT_CLIENT_SESSIONS.empty:
    DEFAULT_SESSION_NAME = _DEFAULT_CLIENT_SESSIONS.iloc[0]["회기"]
else:
    DEFAULT_SESSION_NAME = "새 상담"

DEFAULT_SESSION_DIALOGUE = SESSION_DIALOGUES.get(
    (DEFAULT_CLIENT_ID, DEFAULT_SESSION_NAME),
    DEFAULT_DIALOGUE.copy(),
)


FACTOR_LABELS = {
    "depressive_mood": "우울한 기분",
    "worthlessness": "무가치감",
    "guilt": "죄책감",
    "impaired_cognition": "사고력/집중력 저하",
    "suicidal": "자살 관련 사고",
    "anhedonia": "흥미 감소",
    "psychomotor_changes": "정신운동 변화",
    "weight_appetite": "체중/식욕 변화",
    "sleep_disturbance": "수면 문제",
    "fatigue": "피로감",
    "anxiety": "불안감",
    "loss_of_control": "통제감 상실",
    "social_avoidance": "사회적 회피",
    "physical_symptom": "신체 증상",
    "craving": "갈망",
    "withdrawal": "금단",
    "tolerance": "내성",
    "social_problem": "사회적 문제",
    "sympathy_support": "공감 및 지지",
    "clarification_reflection": "명료화 및 반영",
    "cognitive_restructuring": "인지 재구성",
    "information_provision": "정보 제공",
    "goal_setting": "목표 설정",
    "task_assignment": "과제 부여",
    "behavioral_intervention": "행동 개입",
    "coping_skill_training": "대처기술 훈련",
    "structuring": "구조화",
    "motivation_for_change": "변화 동기",
}


# =========================================================
# 4. 모델 교체용 클래스
# =========================================================
class MockKlueBERTClassifier:
    """
    임시 분류 모델.
    나중에 3단계에서 실제 AI Hub KlueBERT 모델 코드로 교체한다.

    입력:
        상담사/내담자 발화 구분이 포함된 상담 스크립트

    출력:
        {
            "depression": 0 또는 1,
            "anxiety": 0 또는 1,
            "addiction": 0 또는 1
        }
    """

    def predict(self, script: str) -> Dict[str, int]:
        text = script.lower()

        depression_terms = ["우울", "무기력", "피곤", "잠", "수면", "아무것도 하기 싫"]
        anxiety_terms = ["불안", "걱정", "긴장", "초조", "가슴이 답답", "공황"]
        addiction_terms = ["중독", "술", "음주", "갈망", "끊기", "금단"]

        return {
            "depression": int(any(term in text for term in depression_terms)),
            "anxiety": int(any(term in text for term in anxiety_terms)),
            "addiction": int(any(term in text for term in addiction_terms)),
        }
        
class KlueBertAPIClassifier:
    """
    외부 KlueBERT API를 호출해 우울/불안/중독 여부를 판별하는 클래스.

    최종 반환값:
    {
        "depression": 0/1,
        "anxiety": 0/1,
        "addiction": 0/1
    }
    """

    def __init__(self):
        self.last_result = {
            "ok": False,
            "status": "not_run",
            "message": "KlueBERT 분류가 아직 실행되지 않았습니다.",
            "backend": "kluebert_api",
        }

    def predict(self, script: str) -> Dict[str, int]:
        try:
            from src.classifier import classify_text

            result = classify_text(script)
            self.last_result = result

            return result.get(
                "classification",
                {
                    "depression": 0,
                    "anxiety": 0,
                    "addiction": 0,
                },
            )

        except Exception as error:
            self.last_result = {
                "ok": False,
                "status": "error",
                "message": f"KlueBERT 분류 모듈 실행 중 오류가 발생했습니다: {error}",
                "backend": "kluebert_api",
            }

            return {
                "depression": 0,
                "anxiety": 0,
                "addiction": 0,
            }

class MockFactorExtractor:
    """
    임시 28요인 추출 모델.
    나중에 Gemini few-shot 또는 별도 모델/규칙으로 교체한다.
    """

    def extract(self, script: str, classification: Dict[str, int]) -> Dict[str, int]:
        text = script.lower()

        factors = {
            "depressive_mood": 2 if "우울" in text or "아무것도 하기 싫" in text else 0,
            "worthlessness": 2 if "일을 잘 못" in text or "내가 문제" in text else 0,
            "guilt": 0,
            "impaired_cognition": 2 if "집중" in text else 0,
            "suicidal": 1 if "죽고" in text or "자살" in text or "사라지고" in text else 0,
            "anhedonia": 2 if "아무것도 하기 싫" in text or "흥미" in text else 0,
            "psychomotor_changes": 0,
            "weight_appetite": 0,
            "sleep_disturbance": 3 if "잠" in text or "수면" in text else 0,
            "fatigue": 3 if "피곤" in text or "힘들" in text else 0,
            "anxiety": 3 if "불안" in text or "가슴이 답답" in text else 0,
            "loss_of_control": 1 if "통제" in text else 0,
            "social_avoidance": 2 if "피하게" in text or "사람들을 만나는" in text else 0,
            "physical_symptom": 1 if "가슴이 답답" in text else 0,
            "craving": 0,
            "withdrawal": 0,
            "tolerance": 0,
            "social_problem": 1 if "회사" in text or "업무" in text else 0,
            "sympathy_support": 1,
            "clarification_reflection": 1,
            "cognitive_restructuring": 1,
            "information_provision": 0,
            "goal_setting": 1,
            "task_assignment": 1,
            "behavioral_intervention": 0,
            "coping_skill_training": 1,
            "structuring": 1,
            "motivation_for_change": 1,
        }

        return factors

class GeminiAPIFactorExtractor:
    """
    Gemini API 기반 28요인 추출 연결 클래스.

    현재 역할:
        - src/factor_extractor.py의 extract_factors() 함수를 호출한다.
        - GEMINI_API_KEY가 없거나 API 호출에 실패해도 앱이 깨지지 않게 한다.
        - 최종 반환값은 기존 MockFactorExtractor와 동일하게 factors dict만 반환한다.
    """

    def __init__(self):
        self.last_result = {
            "ok": False,
            "status": "not_run",
            "message": "Gemini 28요인 추출이 아직 실행되지 않았습니다.",
            "backend": "gemini_api",
        }

    def extract(self, script: str, classification: Dict[str, int]) -> Dict[str, int]:
        try:
            from src.factor_extractor import extract_factors

            result = extract_factors(
                script=script,
                classification=classification,
                backend="gemini_api",
            )

            self.last_result = result

            return result.get("factors", {})

        except Exception as error:
            self.last_result = {
                "ok": False,
                "status": "error",
                "message": f"Gemini 28요인 추출 모듈 실행 중 오류가 발생했습니다: {error}",
                "backend": "gemini_api",
            }

            try:
                from src.factor_extractor import FACTOR_KEYS

                return {key: 0 for key in FACTOR_KEYS}
            except Exception:
                return {}

class KoalpacaAPISummarizer:
    def __init__(self) -> None:
        self.gemini_sections: list = []

    def _call_gemini_section(self, transcript: str, section_title: str, description: str) -> str:
        """Gemini로 단일 섹션을 생성한다. 오류 시 빈 문자열 반환."""
        try:
            api_key = ""
            model = "gemini-2.5-flash"
            try:
                api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
                model = str(st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash"
            except Exception:
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

            if not api_key:
                return ""

            prompt = (
                f"아래 심리상담 전사 기록을 바탕으로 '{section_title}' 항목을 1~2문단으로 작성해주세요.\n"
                f"설명: {description}\n"
                "주의: 마크다운 문법(**, *, #, -, ` 등)을 절대 사용하지 마세요. 임상 진단 단정 표현 금지.\n"
                '반드시 JSON 형식으로만 답하세요: {"content": "<내용>"}\n\n'
                f"상담 전사 기록:\n{transcript[:9000]}"
            )

            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(raw).get("content", "").strip()
        except Exception:
            return ""

    def summarize(
        self,
        script: str,
        classification: Dict[str, int],
        factors: Dict[str, int],
    ) -> str:
        self.gemini_sections = []

        try:
            from src.summarizer import summarize as koalpaca_summarize
            result = koalpaca_summarize(script)
        except Exception as error:
            return (
                f"[KoAlpaca API 연결 오류]\n\n"
                f"KoAlpaca API 호출 모듈을 실행하는 중 오류가 발생했습니다.\n\n오류 내용:\n{error}"
            )

        koalpaca_ok = result.get("ok", False)
        koalpaca_sections = result.get("sections", {}) if koalpaca_ok else {}

        SECTION_MAP = [
            ("symptoms",             "1. 주요 증상",        "내담자가 호소하는 주요 심리·신체 증상"),
            ("risk_factors",         "2. 위험 요인",        "악화 가능성 또는 추가 확인이 필요한 위험 요인"),
            ("improvement_factors",  "3. 개선 요인",        "상담 개입의 발판이 될 수 있는 강점과 자원"),
            ("intervention_factors", "4. 상담사 개입 요인", "상담사가 다음 회기에서 초점을 맞출 개입 포인트"),
        ]
        SECTION_KEY_MAP = {
            "symptoms":             "report_section_1",
            "risk_factors":         "report_section_2",
            "improvement_factors":  "report_section_3",
            "intervention_factors": "report_section_4",
        }

        parts = []
        for key, numbered_heading, description in SECTION_MAP:
            content = koalpaca_sections.get(key, "").strip()
            body = ""
            if content:
                lines = content.split("\n", 1)
                body = lines[1].strip() if len(lines) > 1 else ""

            if not body:
                body = self._call_gemini_section(script, numbered_heading.split(". ", 1)[1], description)
                if body:
                    self.gemini_sections.append(SECTION_KEY_MAP[key])

            if body:
                parts.append(f"{numbered_heading}\n{body}")

        section5_body = self._call_gemini_section(
            script,
            "다음 회기 계획 추천",
            "다음 회기에서 다룰 주제와 상담사가 준비할 사항",
        )
        if section5_body:
            parts.append(f"5. 다음 회기 계획 추천\n{section5_body}")
            self.gemini_sections.append("report_section_5")

        if parts:
            return "\n\n".join(parts)

        if not koalpaca_ok:
            status = result.get("status", "unknown")
            message = result.get("message", "KoAlpaca API 호출 결과를 확인할 수 없습니다.")
            return (
                f"[KoAlpaca API 연결 상태: {status}]\n\n{message}\n\n"
                "KoAlpaca 호스팅이 완료되면 Streamlit Secrets에 KOALPACA_ENDPOINT_URL과 KOALPACA_API_KEY를 입력한 뒤 다시 실행하세요."
            )

        return result.get("text", "")

# =========================================================
# 5. 모델 로더
# =========================================================
def load_classifier_model():
    """
    우울/불안/중독 분류 모델 로더.

    CLASSIFIER_BACKEND 값에 따라 분류 백엔드를 선택한다.

    - mock: 기존 키워드 기반 mock 분류
    - kluebert_api: 외부 KlueBERT API 호출
    """
    if CLASSIFIER_BACKEND == "mock":
        return MockKlueBERTClassifier()

    if CLASSIFIER_BACKEND == "kluebert_api":
        return KlueBertAPIClassifier()

    return MockKlueBERTClassifier()


def load_factor_model():
    """
    28요인 추출 모델 로더.

    FACTOR_BACKEND 값에 따라 28요인 추출 백엔드를 선택한다.

    - mock: 기존 mock 28요인 추출 사용
    - gemini_api: src/factor_extractor.py를 통해 Gemini API 호출
    """
    if FACTOR_BACKEND == "mock":
        return MockFactorExtractor()

    if FACTOR_BACKEND == "gemini_api":
        return GeminiAPIFactorExtractor()

    if FACTOR_BACKEND == "aihub_local":
        # 향후 AI Hub 기반 28요인 모델을 직접 연결할 때 사용할 자리
        return MockFactorExtractor()

    return MockFactorExtractor()


def load_summarizer_model():
    return KoalpacaAPISummarizer()

# =========================================================
# 6. 분석 파이프라인
# =========================================================
def build_dialogue_text(dialogue_df: pd.DataFrame) -> str:
    lines = []

    for _, row in dialogue_df.iterrows():
        speaker = str(row.get("화자", "")).strip()
        utterance = str(row.get("발화", "")).strip()

        if speaker and utterance and utterance.lower() != "nan":
            lines.append(f"{speaker}: {utterance}")

    return "\n".join(lines)


def soften_diagnostic_expression(text: str) -> str:
    """
    진단 단정 표현 방지용 간단 후처리.
    실제 서비스에서는 금칙어 사전과 안전 문구를 별도 관리하는 것이 좋다.
    """
    replacements = {
        "우울증입니다": "우울 관련 호소가 확인됩니다",
        "불안장애입니다": "불안 관련 호소가 확인됩니다",
        "중독입니다": "중독 관련 호소가 확인됩니다",
        "진단됩니다": "가능성이 표시됩니다",
        "확진": "라벨상 표시",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def run_analysis(script: str) -> Dict[str, Any]:
    """
    전체 분석 파이프라인.

    현재:
        Mock 모델 기반

    추후:
        classifier만 KlueBERT로 교체
        summarizer만 Koalpaca로 교체
        factor_model만 Gemini/28요인 모델로 교체
    """
    classifier = load_classifier_model()
    factor_model = load_factor_model()
    summarizer = load_summarizer_model()

    classification = classifier.predict(script)
    factors = factor_model.extract(script, classification)
    summary = summarizer.summarize(script, classification, factors)
    summary = soften_diagnostic_expression(summary)

    return {
        "script": script,
        "classification": classification,
        "factors": factors,
        "summary": summary,
        "gemini_sections": getattr(summarizer, "gemini_sections", []),
        "model_info": {
            "backend": MODEL_BACKEND,
            "factor_backend": FACTOR_BACKEND,
            "classifier_backend": CLASSIFIER_BACKEND,
            "classifier": "KlueBERT API" if CLASSIFIER_BACKEND == "kluebert_api" else "MockKlueBERTClassifier",
            "classifier_status": getattr(classifier, "last_result", {}).get("status", "success" if CLASSIFIER_BACKEND == "mock" else "unknown"),
            "classifier_message": getattr(classifier, "last_result", {}).get("message", ""),
            "classifier_scores": getattr(classifier, "last_result", {}).get("scores", {}),
            "classifier_raw_scores": getattr(classifier, "last_result", {}).get("raw_scores", {}),
            "classifier": KLUEBERT_MODEL_NAME if MODEL_BACKEND != "mock" else "MockKlueBERTClassifier",
            "factor_extractor": "Gemini API" if FACTOR_BACKEND == "gemini_api" else "MockFactorExtractor",
            "summarizer": "KoAlpaca API",
            "factor_status": getattr(factor_model, "last_result", {}).get("status", "success" if FACTOR_BACKEND == "mock" else "unknown"),
            "factor_message": getattr(factor_model, "last_result", {}).get("message", ""),  
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_factor_dataframe(factors: Dict[str, int]) -> pd.DataFrame:
    rows = []

    for key, score in factors.items():
        label = FACTOR_LABELS.get(key, key)

        if key in [
            "depressive_mood",
            "worthlessness",
            "guilt",
            "impaired_cognition",
            "suicidal",
            "anhedonia",
            "psychomotor_changes",
            "weight_appetite",
            "sleep_disturbance",
            "fatigue",
        ]:
            category = "우울/증상"
        elif key in [
            "anxiety",
            "loss_of_control",
            "social_avoidance",
            "physical_symptom",
        ]:
            category = "불안"
        elif key in [
            "craving",
            "withdrawal",
            "tolerance",
            "social_problem",
        ]:
            category = "중독"
        elif key in [
            "sympathy_support",
            "clarification_reflection",
            "cognitive_restructuring",
            "information_provision",
            "goal_setting",
            "task_assignment",
            "behavioral_intervention",
            "coping_skill_training",
            "structuring",
        ]:
            category = "상담사 개입"
        else:
            category = "변화/기타"

        rows.append(
            {
                "요인코드": key,
                "요인": label,
                "카테고리": category,
                "점수": int(score),
            }
        )

    return pd.DataFrame(rows)


def build_report_text(analysis_result: Dict[str, Any]) -> str:
    classification = analysis_result["classification"]
    summary = analysis_result["summary"]

    return f"""[상담보고서 초안]

0. AI 판별 결과
- 우울 관련 라벨: {classification.get("depression", 0)}
- 불안 관련 라벨: {classification.get("anxiety", 0)}
- 중독 관련 라벨: {classification.get("addiction", 0)}

주의:
위 값은 모델 출력 기반 참고값이며, 임상 진단 또는 표준화 검사 점수로 단정하지 않는다.

{summary}
"""


def make_json_export(analysis_result: Dict[str, Any]) -> str:
    export_data = {
        "client": st.session_state.selected_client,
        "session": st.session_state.selected_session,
        "mode": st.session_state.record_mode,
        "dialogue": st.session_state.dialogue_rows.to_dict(orient="records"),
        "analysis_result": analysis_result,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "MVP 데모 데이터입니다. 실제 진단 목적으로 사용하지 않습니다.",
    }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


# =========================================================
# 7. Session State 초기화
# =========================================================
def init_session_state():
    global CLIENTS, SESSIONS, SESSION_DIALOGUES

    defaults = {
        "page": "내담자 홈",
        "selected_client": DEFAULT_CLIENT_ID,
        "selected_session": DEFAULT_SESSION_NAME,
        "client_search": "",
        "show_selected_client_label": False,
        "client_search_nonce": 0,
        "patient_home_tab": "내담자 정보",
        "record_mode": "existing",
        "dialogue_rows": DEFAULT_SESSION_DIALOGUE.copy(),
        "chat_history": [],
        "analysis_result": None,
        "session_notes": {},
        "registered_clients": [],
        "saved_reports": {},
        "saved_session_rows": [],
        "saved_session_dialogues": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.registered_clients:
        registered_df = pd.DataFrame(st.session_state.registered_clients)
        CLIENTS = pd.concat([CLIENTS, registered_df], ignore_index=True)
        CLIENTS = CLIENTS.drop_duplicates(subset=["내담자 ID"], keep="last")

    if st.session_state.saved_session_rows:
        saved_sessions_df = pd.DataFrame(st.session_state.saved_session_rows)
        SESSIONS = pd.concat([SESSIONS, saved_sessions_df], ignore_index=True)
        SESSIONS = SESSIONS.drop_duplicates(subset=["내담자 ID", "회기"], keep="last")

    for dialogue_key, rows in st.session_state.saved_session_dialogues.items():
        try:
            client_id, session_name = dialogue_key.split("||", 1)
        except ValueError:
            continue

        SESSION_DIALOGUES[(client_id, session_name)] = pd.DataFrame(rows)


# =========================================================
# 8. Helper 함수
# =========================================================
def go_page(page_name: str):
    st.session_state.page = page_name


def open_session_detail(session_name: str):
    select_session(session_name)
    go_page("회기 상세")


def back_to_patient_home():
    st.session_state.patient_home_tab = "상담관리"
    go_page("내담자 홈")


def select_session(session_name: str):
    st.session_state.selected_session = session_name
    st.session_state.record_mode = "existing"
    st.session_state.analysis_result = None

    key = (st.session_state.selected_client, session_name)

    if key in SESSION_DIALOGUES:
        st.session_state.dialogue_rows = SESSION_DIALOGUES[key].copy()
    else:
        st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()


NEW_SESSION_FORM_KEYS = [
    "new_session_name",
    "new_session_topic",
    "new_session_scope",
    "new_session_date",
    "new_session_input_mode",
    "new_session_input_mode_pills",
    "new_session_script_text",
    "new_session_dialogue_editor",
]


def get_empty_dialogue_rows() -> pd.DataFrame:
    return pd.DataFrame({"화자": ["상담사", "내담자"], "발화": ["", ""]})


def reset_new_session_form_state():
    for key in NEW_SESSION_FORM_KEYS:
        st.session_state.pop(key, None)

    st.session_state.dialogue_rows = get_empty_dialogue_rows()
    st.session_state.analysis_result = None


def reset_new_session_content_state():
    st.session_state.dialogue_rows = get_empty_dialogue_rows()
    st.session_state.new_session_script_text = ""
    st.session_state.pop("new_session_dialogue_editor", None)


def start_new_session():
    reset_new_session_form_state()
    st.session_state.record_mode = "new"
    st.session_state.selected_session = "새 상담"
    st.session_state.new_session_name = get_next_session_name()
    st.session_state.new_session_topic = ""
    st.session_state.new_session_scope = "복합"
    st.session_state.new_session_date = datetime.now().date()
    st.session_state.new_session_input_mode = "발화 단위 입력"
    st.session_state.analysis_result = None


def cancel_new_session():
    reset_new_session_form_state()
    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client]

    if client_sessions.empty:
        st.session_state.record_mode = "existing"
        go_page("상담내역 기록·추가")
        return

    select_session(client_sessions.iloc[0]["회기"])
    go_page("상담내역 기록·추가")


def get_next_session_name() -> str:
    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client]
    max_order = 0

    for value in client_sessions["회기"].tolist():
        match = re.search(r"(\d+)", str(value or ""))
        if match:
            max_order = max(max_order, int(match.group(1)))

    return f"{max_order + 1}회기"


def _build_new_session_payload(script: str) -> Tuple[str, pd.DataFrame]:
    input_mode = st.session_state.get("new_session_input_mode", "발화 단위 입력")

    if input_mode == "전사 텍스트 붙여넣기":
        clean_script = str(script or "").strip()
        dialogue_rows = _script_to_dialogue_df(clean_script) if clean_script else get_empty_dialogue_rows()
    else:
        dialogue_rows = st.session_state.dialogue_rows.copy()
        dialogue_rows["발화"] = dialogue_rows["발화"].fillna("").astype(str)
        dialogue_rows = dialogue_rows[dialogue_rows["발화"].str.strip() != ""].reset_index(drop=True)
        clean_script = build_dialogue_text(dialogue_rows).strip()

    return clean_script, dialogue_rows


def _persist_session_row(new_row: Dict[str, Any], dialogue_rows: pd.DataFrame):
    client_id = str(new_row["내담자 ID"])
    session_name = str(new_row["회기"])
    saved_rows = st.session_state.get("saved_session_rows", [])
    replaced = False

    for idx, row in enumerate(saved_rows):
        if str(row.get("내담자 ID")) == client_id and str(row.get("회기")) == session_name:
            saved_rows[idx] = new_row.copy()
            replaced = True
            break

    if not replaced:
        saved_rows.append(new_row.copy())

    st.session_state.saved_session_rows = saved_rows
    st.session_state.saved_session_dialogues[f"{client_id}||{session_name}"] = dialogue_rows.to_dict(orient="records")


def save_new_session(script: str, run_ai: bool = False) -> bool:
    global SESSIONS, SESSION_DIALOGUES

    session_name = str(st.session_state.get("new_session_name", get_next_session_name()) or get_next_session_name()).strip()
    session_date = st.session_state.get("new_session_date", datetime.now().date())
    session_scope = str(st.session_state.get("new_session_scope", "복합") or "복합").strip()
    session_topic = str(st.session_state.get("new_session_topic", "") or "").strip() or "(주제 미입력)"
    session_date_text = session_date.strftime("%Y-%m-%d") if hasattr(session_date, "strftime") else str(session_date)
    clean_script, dialogue_rows = _build_new_session_payload(script)

    if not clean_script:
        st.warning("상담 내용을 입력하세요.")
        return False

    new_row = {
        "내담자 ID": st.session_state.selected_client,
        "회기": session_name,
        "상담일": session_date_text,
        "상담 주제": session_topic,
        "상담 범위": session_scope,
        "분류 유형": session_scope,
        "입력 방식": st.session_state.get("new_session_input_mode", "발화 단위 입력"),
        "보고서 상태": "분석 완료" if run_ai else "임시 저장",
        "script": clean_script,
    }

    for column in SESSIONS.columns:
        if column not in new_row:
            new_row[column] = ""

    existing_mask = (
        (SESSIONS["내담자 ID"].astype(str) == str(st.session_state.selected_client))
        & (SESSIONS["회기"].astype(str) == session_name)
    )

    if existing_mask.any():
        SESSIONS.loc[existing_mask, list(new_row.keys())] = list(new_row.values())
    else:
        SESSIONS = pd.concat([SESSIONS, pd.DataFrame([new_row])], ignore_index=True)

    SESSION_DIALOGUES[(st.session_state.selected_client, session_name)] = dialogue_rows.copy()
    _persist_session_row(new_row, dialogue_rows)
    st.session_state.dialogue_rows = dialogue_rows.copy()
    st.session_state.selected_session = session_name

    if run_ai:
        st.session_state.record_mode = "existing"
        st.session_state.analysis_result = run_analysis(clean_script)
        go_page("분석 대시보드")
    else:
        st.session_state.record_mode = "existing"
        st.session_state.analysis_result = None
        st.session_state.new_session_save_success = True
        go_page("상담내역 기록·추가")

    return True


def open_draft_session_editor(session_name: str):
    client_id = st.session_state.selected_client
    session_rows = SESSIONS[
        (SESSIONS["내담자 ID"].astype(str) == str(client_id))
        & (SESSIONS["회기"].astype(str) == str(session_name))
    ]
    row = session_rows.iloc[0] if not session_rows.empty else pd.Series(dtype=object)
    dialogue_rows = SESSION_DIALOGUES.get((client_id, session_name), get_empty_dialogue_rows()).copy()
    script_text = str(row.get("script", "") or "").strip() or build_dialogue_text(dialogue_rows)
    date_text = str(row.get("상담일", datetime.now().strftime("%Y-%m-%d")) or "")

    try:
        session_date = pd.to_datetime(date_text).date()
    except Exception:
        session_date = datetime.now().date()

    st.session_state.selected_session = session_name
    st.session_state.new_session_name = session_name
    st.session_state.new_session_date = session_date
    st.session_state.new_session_scope = str(row.get("상담 범위", row.get("분류 유형", "복합")) or "복합")
    st.session_state.new_session_topic = str(row.get("상담 주제", "") or "")
    st.session_state.new_session_input_mode = str(row.get("입력 방식", "발화 단위 입력") or "발화 단위 입력")
    st.session_state.dialogue_rows = dialogue_rows
    st.session_state.new_session_script_text = script_text
    st.session_state.pop("new_session_input_mode_pills", None)
    st.session_state.pop("new_session_dialogue_editor", None)
    st.session_state.record_mode = "draft"
    st.session_state.analysis_result = None
    go_page("상담내역 기록·추가")


def register_new_client(name: str, gender: str, age: str, region: str, memo: str) -> bool:
    global CLIENTS

    clean_name = str(name or "").strip()
    if not clean_name:
        st.warning("이름/alias를 입력하세요.")
        return False

    client_id = hashlib.sha1(f"{clean_name}:{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:8]
    new_row = {
        "내담자 ID": client_id,
        "이름": clean_name,
        "성별": str(gender or "기타/미상").strip(),
        "연령대": str(age or "미상").strip() or "미상",
        "지역": str(region or "미상").strip() or "미상",
        "상담 유형": str(memo or "신규 등록").strip() or "신규 등록",
        "최근 회기": "0회기",
        "상태": "신규 등록",
        "메모": str(memo or "").strip(),
    }

    for column in CLIENTS.columns:
        if column not in new_row:
            new_row[column] = ""

    st.session_state.registered_clients.append(new_row.copy())
    CLIENTS = pd.concat([CLIENTS, pd.DataFrame([new_row])], ignore_index=True)
    st.session_state.selected_client = client_id
    st.session_state.selected_session = "새 상담"
    st.session_state.record_mode = "new"
    st.session_state.dialogue_rows = get_empty_dialogue_rows()
    st.session_state.analysis_result = None
    st.session_state.patient_home_tab = "내담자 정보"
    go_page("내담자 홈")
    st.success(f"{clean_name} 내담자를 등록했습니다.")
    return True


def get_client_row():
    row = CLIENTS[CLIENTS["내담자 ID"] == st.session_state.selected_client]
    if row.empty:
        return CLIENTS.iloc[0]
    return row.iloc[0]


def get_client_display_name(client_row: pd.Series | None = None) -> str:
    if client_row is None:
        client_row = get_client_row()
    return str(client_row.get("이름", client_row.get("내담자 ID", st.session_state.selected_client)))


def _short_gender(gender: Any) -> str:
    text = str(gender or "").strip()
    if text.startswith("여"):
        return "여"
    if text.startswith("남"):
        return "남"
    return text or "미상"


def _format_sidebar_client_label(client_id: str) -> str:
    row = CLIENTS[CLIENTS["내담자 ID"] == client_id]
    if row.empty:
        return str(client_id)
    item = row.iloc[0]
    name = get_client_display_name(item)
    gender = _short_gender(item.get("성별", ""))
    age = str(item.get("연령대", item.get("연령", "미상")) or "미상")
    region = str(item.get("지역", "미상") or "미상")
    return f"{name} · {gender}/{age}/{region}"


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


def clear_chat():
    st.session_state.chat_history = []

# =========================================================
# RAG / ChromaDB 검색 + Gemini 답변 생성
# =========================================================

RAG_COLLECTION_LABELS = {
    "counseling_cases": "유사 상담 발화",
    "session_summaries": "유사 회기 요약",
    "clinical_references": "임상 참고문서",
    "counseling_db": "상담 RAG DB",
}


@st.cache_resource(show_spinner=False)
def load_rag_embedding_model():
    """
    ChromaDB 검색용 query embedding 모델.
    주의: DB 구축 시 사용한 embedding 모델과 동일해야 한다.
    """
    return SentenceTransformer(RAG_EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_chroma_client():
    """
    현재 프로젝트 루트의 ./chroma_db를 로드한다.
    사진상 chroma_db/chroma.sqlite3가 있으므로 기본 경로는 현재 구조와 맞다.
    """
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def list_chroma_collection_names() -> List[str]:
    """
    현재 ChromaDB에 저장된 collection 이름 목록을 반환한다.
    ChromaDB 버전에 따라 list_collections() 반환 타입이 다를 수 있어 방어적으로 처리한다.
    """
    try:
        client = load_chroma_client()
        collections = client.list_collections()
    except Exception:
        return []

    names = []

    for item in collections:
        if hasattr(item, "name"):
            names.append(item.name)
        else:
            names.append(str(item))

    return names


def get_rag_target_collections() -> List[str]:
    """
    우선순위:
    1. counseling_cases / session_summaries / clinical_references
    2. counseling_db
    3. 실제 존재하는 모든 collection

    이렇게 하는 이유:
    - 네 문서 설계상 collection을 여러 개로 나누는 구조일 수도 있고,
    - LangChain 기본값처럼 하나의 collection만 있을 수도 있기 때문.
    """
    available = list_chroma_collection_names()

    if not available:
        return []

    preferred = [
        "counseling_cases",
        "session_summaries",
        "clinical_references",
        "counseling_db",
    ]

    matched = [name for name in preferred if name in available]

    if matched:
        return matched

    return available


def get_selected_session_script_for_rag() -> str:
    """
    현재 선택된 내담자/회기의 상담 스크립트를 RAG 답변 생성에 함께 넣는다.
    """
    try:
        return build_dialogue_text(st.session_state.dialogue_rows).strip()
    except Exception:
        return ""


def format_rag_source_title(collection_name: str, metadata: dict, rank: int) -> str:
    """
    검색 결과 metadata를 출처 표시용 제목으로 변환한다.
    """
    label = RAG_COLLECTION_LABELS.get(collection_name, collection_name)

    source = (
        metadata.get("source")
        or metadata.get("title")
        or metadata.get("filename")
        or metadata.get("client_id")
        or metadata.get("id")
        or f"검색 결과 {rank}"
    )

    session = metadata.get("session") or metadata.get("회기")
    class_label = metadata.get("class") or metadata.get("label")

    parts = [label, str(source)]

    if session not in [None, "", "nan"]:
        parts.append(f"{session}회기" if str(session).isdigit() else str(session))

    if class_label not in [None, "", "nan"]:
        parts.append(str(class_label))

    return " / ".join(parts)


def search_rag_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    ChromaDB에서 관련 문서를 검색한다.
    여러 collection이 있으면 collection별로 검색한 뒤 합친다.
    """
    clean_query = str(query or "").strip()

    if not clean_query:
        return []

    target_collections = get_rag_target_collections()

    if not target_collections:
        return []

    try:
        client = load_chroma_client()
        embedding_model = load_rag_embedding_model()
        query_embedding = embedding_model.encode(clean_query).tolist()
    except Exception as error:
        return [
            {
                "collection": "system",
                "title": "RAG 로딩 오류",
                "document": f"RAG 모델 또는 ChromaDB 로딩 중 오류가 발생했습니다: {error}",
                "metadata": {},
                "distance": None,
            }
        ]

    all_results = []

    for collection_name in target_collections:
        try:
            collection = client.get_collection(collection_name)

            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            for idx, document in enumerate(documents):
                metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                distance = distances[idx] if idx < len(distances) else None

                all_results.append(
                    {
                        "collection": collection_name,
                        "title": format_rag_source_title(collection_name, metadata, idx + 1),
                        "document": str(document or ""),
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

        except Exception as error:
            all_results.append(
                {
                    "collection": collection_name,
                    "title": f"{collection_name} 검색 오류",
                    "document": f"{collection_name} collection 검색 중 오류가 발생했습니다: {error}",
                    "metadata": {},
                    "distance": None,
                }
            )

    def distance_sort_key(item: Dict[str, Any]):
        distance = item.get("distance")
        return float(distance) if distance is not None else 999999

    all_results = sorted(all_results, key=distance_sort_key)

    return all_results[:top_k]


def build_rag_context_text(rag_results: List[Dict[str, Any]], max_chars_per_doc: int = 900) -> str:
    """
    Gemini prompt에 넣을 검색 근거 문자열 생성.
    """
    blocks = []

    for idx, item in enumerate(rag_results, start=1):
        title = item.get("title", f"검색 결과 {idx}")
        document = str(item.get("document", "") or "").strip()
        document = document[:max_chars_per_doc]

        blocks.append(
            f"[출처 {idx}] {title}\n{document}"
        )

    return "\n\n".join(blocks)

def clean_chatbot_answer(answer: str) -> str:
    """
    RAG/LLM 답변이 코드처럼 보이지 않도록 마크다운 잡음을 정리한다.
    - --- 제거
    - ### 제목 제거/정리
    - 과도한 공백 정리
    - '상담사님,' 같은 도입부는 유지
    """
    text = str(answer or "").strip()

    # 코드블록이 섞인 경우 제거
    text = text.replace("```markdown", "")
    text = text.replace("```", "")

    # 수평선 제거
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # ### 1. 제목 → 1. 제목 형태로 정리
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 굵게 표시용 **는 일부 남겨도 되지만, 너무 코드처럼 보이면 제거
    text = text.replace("**", "")

    # 특수 마크다운 리스트 기호 과다 정리
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)

    # 빈 줄 과다 제거
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def generate_rag_answer_with_gemini(
    question: str,
    rag_results: List[Dict[str, Any]],
) -> str:
    """
    Gemini API로 RAG 답변을 생성한다.
    GEMINI_API_KEY가 없으면 검색 결과 요약만 반환한다.
    """
    current_script = get_selected_session_script_for_rag()
    rag_context = build_rag_context_text(rag_results)

    if not rag_results:
        return (
            "ChromaDB에서 관련 검색 결과를 찾지 못했습니다.\n\n"
            "확인할 항목:\n"
            "1. chroma_db 경로가 ./chroma_db인지\n"
            "2. collection 이름이 실제로 존재하는지\n"
            "3. DB 구축 시 사용한 embedding 모델과 RAG_EMBEDDING_MODEL이 같은지"
        )

    if not GEMINI_API_KEY:
        source_titles = "\n".join(
            [f"- {item.get('title', '출처 없음')}" for item in rag_results]
        )

        return (
            "Gemini API 키가 없어 답변 생성은 실행하지 않고, ChromaDB 검색 결과만 표시합니다.\n\n"
            "검색된 근거:\n"
            f"{source_titles}\n\n"
            "Streamlit Secrets 또는 .env에 GEMINI_API_KEY를 설정하면 검색 결과 기반 답변 생성까지 실행됩니다."
        )

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )

        prompt = f"""
당신은 심리상담사를 보조하는 AI입니다.
아래의 [현재 선택 회기 상담 기록]과 [RAG 검색 근거]만 바탕으로 답변하세요.

중요 원칙:
- 진단 확정처럼 표현하지 마세요.
- "우울증입니다", "불안장애입니다"처럼 단정하지 마세요.
- 데이터 라벨, 유사 사례, 참고 문서는 상담사의 검토를 돕는 근거로만 표현하세요.
- 검색 근거에 없는 내용은 추측하지 마세요.
- 자해/자살 관련 표현이 있으면 별도 확인 필요 항목으로 분리하세요.
- 최종 판단은 상담사가 수행해야 한다고 명시하세요.
- 마지막에 참고한 출처 번호를 표시하세요.

[현재 선택 회기 상담 기록]
{current_script[:2500]}

[RAG 검색 근거]
{rag_context}

[상담사 질문]
{question}

[답변 형식]
아래 번호 형식만 사용하세요. 마크다운 제목 기호(###), 수평선(---), 코드블록(```), 표는 사용하지 마세요.

1. 핵심 요약
- 현재 회기에서 확인되는 내용을 3~5문장으로 요약

2. 근거 기반 유사 사례/참고 내용
- RAG 검색 결과와 연결되는 유사 패턴 정리

3. 다음 회기에서 확인할 내용
- 상담사가 다음 회기에서 확인할 질문 후보 정리

4. 개입 방향 초안
- 상담사가 참고할 수 있는 개입 방향 초안 정리

5. 주의사항
- 진단 확정 금지, 상담사 최종 판단 필요, 자해/자살 관련 확인 필요 여부 정리

6. 참고 출처
- 사용한 출처 번호만 간단히 표시
"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, json=payload, timeout=60)
        result = resp.json()
        answer = result["candidates"][0]["content"]["parts"][0]["text"]

        if not answer:
            return "Gemini 응답이 비어 있습니다. API 응답 상태를 확인하세요."

        return answer.strip()

    except Exception as error:
        return f"Gemini RAG 답변 생성 중 오류가 발생했습니다: {error}"

def add_mock_answer(user_prompt: str):
    """
    기존 mock 챗봇 응답 함수 이름은 유지한다.
    이유:
    - render_chatbot() 내부에서 이미 add_mock_answer(question)을 호출하고 있기 때문.
    - 함수 이름만 유지하고 내부 동작을 실제 RAG로 바꾸면 UI 코드를 거의 건드리지 않아도 된다.
    """
    question = str(user_prompt or "").strip()

    if not question:
        return

    now_label = datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
            "time": now_label,
        }
    )

    # 현재 선택 회기 상담 기록까지 query에 포함해 유사 사례 검색 품질을 높인다.
    current_script = get_selected_session_script_for_rag()

    if current_script:
        retrieval_query = (
            f"{question}\n\n"
            f"[현재 상담 기록]\n"
            f"{current_script[:1500]}"
        )
    else:
        retrieval_query = question

    rag_results = search_rag_documents(
        query=retrieval_query,
        top_k=5,
    )

    answer = generate_rag_answer_with_gemini(
        question=question,
        rag_results=rag_results,
    )
    answer = clean_chatbot_answer(answer)
    
    sources = []

    for item in rag_results:
        sources.append(
            {
                "title": item.get("title", "출처 없음"),
                "desc": str(item.get("document", ""))[:250],
            }
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": now_label,
        }
    )


# =========================================================
# 9. 전역 스타일
# =========================================================
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E40AF"
PRIMARY_LIGHT = "#EFF6FF"
PRIMARY_SOFT = "#BFDBFE"

# =========================================================
# 대시보드 공통 색상 팔레트
# 반복 지표는 모든 차트에서 같은 색을 사용한다.
# =========================================================

DASHBOARD_COLORS = {
    "우울": "#6E85B7",          # muted powder blue
    "불안": "#94B3FD",          # soft periwinkle
    "중독": "#7A6FC4",          # muted violet
    "수면문제": "#8CC0DE",      # iceberg blue
    "피로감": "#87AAAA",        # blue spruce
    "상담사 개입": "#96C7C1",   # muted teal
    "변화/기타": "#C4D7E0",     # pale blue gray
    "위험": "#D5D3DE",          # misty lavender gray
    "보호": "#B2C8DF",
}

CARD_BLUE = "#FFFFFF"
CARD_BLUE_BORDER = "#E2E8F0"
TEXT = "#0F172A"
SUBTEXT = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#F1F5F9"
CHATBOT_ICON_PATH = Path(__file__).resolve().parent / "assets" / "chatbot.png"
SEND_ICON_PATH = Path(__file__).resolve().parent / "assets" / "send_icon.png"
MD_ICON_PATH = Path(__file__).resolve().parent / "assets" / "md.png"
PDF_ICON_PATH = Path(__file__).resolve().parent / "assets" / "PDF Ribbon.png"
DOCX_ICON_PATH = Path(__file__).resolve().parent / "assets" / "docx.png"
SEARCH_ICON_PATH = Path(__file__).resolve().parent / "assets" / "search.png"
PROFILE_ICON_PATH = Path(__file__).resolve().parent / "assets" / "profile.png"


def get_svg_data_uri(path: Path) -> str:
    try:
        encoded_svg = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""

    return f"data:image/svg+xml;base64,{encoded_svg}"


def get_png_data_uri(path: Path) -> str:
    try:
        encoded_png = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""

    return f"data:image/png;base64,{encoded_png}"

def build_download_label(icon_path: Path, fallback_icon: str, text: str) -> str:
    icon_data_uri = get_png_data_uri(icon_path)

    if icon_data_uri:
        return f"![icon]({icon_data_uri}) {text}"

    return f"{fallback_icon} {text}"

def make_simple_pdf_report_bytes(report_text: str, title: str = "상담 요약 보고서") -> bytes | None:
    """
    WeasyPrint 기반 PDF 생성이 실패할 때 사용하는 ReportLab fallback.
    한글 출력을 위해 Windows의 Malgun Gothic 또는 Linux의 NotoSansCJK를 우선 탐색한다.
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return None

    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]

    font_name = "Helvetica"

    for font_path in font_candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", str(font_path)))
                font_name = "KoreanFont"
                break
            except Exception:
                continue

    try:
        buffer = BytesIO()
        page_width, page_height = A4
        c = canvas.Canvas(buffer, pagesize=A4)

        left = 18 * mm
        right = 18 * mm
        top = page_height - 20 * mm
        bottom = 18 * mm
        usable_width = page_width - left - right

        def draw_wrapped_text(text: str, x: float, y: float, font_size: int = 10, line_height: int = 15):
            c.setFont(font_name, font_size)

            for raw_line in str(text or "").splitlines():
                line = raw_line.strip()

                if not line:
                    y -= line_height
                    continue

                # 한글 기준 대략적 줄바꿈. ReportLab 기본 stringWidth 계산을 사용.
                current = ""
                for char in line:
                    test = current + char
                    if c.stringWidth(test, font_name, font_size) > usable_width:
                        c.drawString(x, y, current)
                        y -= line_height
                        current = char

                        if y < bottom:
                            c.showPage()
                            c.setFont(font_name, font_size)
                            y = top

                    else:
                        current = test

                if current:
                    c.drawString(x, y, current)
                    y -= line_height

                if y < bottom:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = top

            return y

        c.setFont(font_name, 16)
        c.drawString(left, top, title)

        y = top - 12 * mm
        y = draw_wrapped_text(report_text, left, y, font_size=10, line_height=15)

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    except Exception:
        return None

def build_download_label(icon_path: Path, fallback_icon: str, text: str) -> str:
    """
    assets 아이콘이 없으면 깨진 이미지 대신 이모지 fallback을 사용한다.
    """
    icon_data_uri = get_png_data_uri(icon_path)

    if icon_data_uri:
        return f"![icon]({icon_data_uri}) {text}"

    return f"{fallback_icon} {text}"

def apply_global_style():
    chatbot_icon_data_uri = get_svg_data_uri(CHATBOT_ICON_PATH)
    search_icon_data_uri = get_png_data_uri(SEARCH_ICON_PATH)
    profile_icon_data_uri = get_png_data_uri(PROFILE_ICON_PATH)

    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY};
            --primary-dark: {PRIMARY_DARK};
            --primary-soft: {PRIMARY_LIGHT};
            --text: {TEXT};
            --subtext: {SUBTEXT};
            --muted: #94A3B8;
            --border: {BORDER};
            --card: #FFFFFF;
            --sidebar-bg: {SIDEBAR_BG};
        }}

        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard", sans-serif;
            color: var(--text);
        }}

        .stApp {{
            background: #FFFFFF;
        }}

        .main .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0.5rem;
            max-width: 1180px;
            padding-left: 3rem;
            padding-right: 3rem;
            margin-left: auto;
            margin-right: auto;
            transform: translateY(-0.9rem);
        }}

        section[data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            box-shadow: 8px 0 24px rgba(15, 23, 42, 0.025);
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }}

        section[data-testid="stSidebar"]::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding: 1.75rem 1.35rem 1.4rem !important;
        }}

        section[data-testid="stSidebar"] h1 {{
            color: var(--primary-dark);
            font-size: 1.34rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.045em;
            line-height: 1.12;
            margin-bottom: 0.25rem;
        }}

        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {{
            color: var(--subtext);
            font-size: 0.77rem;
            line-height: 1.45;
        }}

        .sidebar-brand {{
            font-size: 1.34rem !important;
            font-weight: 720 !important;
            color: var(--primary-dark) !important;
            letter-spacing: -0.045em !important;
            line-height: 1.12 !important;
            margin-bottom: 0.25rem !important;
        }}

        .sidebar-subtitle {{
            font-size: 0.77rem !important;
            color: var(--subtext) !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            margin-bottom: 1rem !important;
        }}

        .sidebar-profile-text {{
            color: var(--text) !important;
            font-size: 0.8rem !important;
            line-height: 1.55 !important;
            margin-top: 0.2rem !important;
            margin-bottom: 1.08rem !important;
            font-weight: 400 !important;
        }}

        .sidebar-profile-text strong {{
            display: block;
            font-size: 0.9rem !important;
            color: var(--text) !important;
            font-weight: 620 !important;
            margin-bottom: 0.22rem !important;
        }}

        .sidebar-section-title {{
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: var(--text) !important;
            margin: 1.12rem 0 0.55rem !important;
            letter-spacing: -0.01em !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stAlert"] {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--subtext);
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.025);
        }}

        section[data-testid="stSidebar"] h3 {{
            color: var(--text);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin-top: 1.12rem;
        }}

        section[data-testid="stSidebar"] label {{
            color: var(--subtext) !important;
            font-size: 0.74rem !important;
            font-weight: 400 !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            border-radius: 11px !important;
            border-color: var(--border) !important;
            min-height: 2.42rem !important;
            background: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.025) !important;
            cursor: pointer !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
            cursor: pointer !important;
        }}

        section[data-testid="stSidebar"] hr {{
            margin: 1.25rem 0 !important;
            border-color: #CBD5E1 !important;
        }}

        .app-title {{
            font-size: 1.7rem;
            font-weight: 640;
            color: var(--text);
            letter-spacing: -0.045em;
            margin-bottom: 0.35rem;
        }}

        .section-title {{
            font-size: 1.65rem;
            font-weight: 660;
            color: var(--text);
            letter-spacing: -0.05em;
            margin-top: 0.4rem;
            margin-bottom: 0.45rem;
        }}

        .session-detail-section-title {{
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 560;
            letter-spacing: -0.04em;
            margin-top: 0.25rem;
            margin-bottom: 0.55rem;
        }}

        .page-desc {{
            color: var(--subtext);
            font-size: 0.9rem;
            margin-bottom: 1rem;
            line-height: 1.58;
        }}

        .tag {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: var(--primary-soft);
            color: var(--primary-dark);
            font-size: 0.72rem;
            font-weight: 500;
            margin-right: 0.34rem;
            margin-bottom: 0.2rem;
            border: 1px solid var(--primary-soft);
        }}

        .hero-card {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.05rem 1.1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.032);
        }}

        .hero-title {{
            font-size: 0.98rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.25rem;
            letter-spacing: -0.03em;
        }}

        .hero-desc {{
            color: var(--subtext);
            font-size: 0.88rem;
            line-height: 1.62;
        }}

        .summary-card {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 132px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.032);
        }}

        .summary-card-title {{
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--primary-dark);
            letter-spacing: -0.015em;
            margin-bottom: 0.72rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid var(--border);
        }}

        .summary-card-body {{
            font-size: 0.82rem;
            font-weight: 480;
            color: var(--text);
            line-height: 1.65;
        }}

        .patient-kicker {{
            color: var(--primary);
            font-size: 0.82rem;
            font-weight: 560;
            margin-bottom: 0.4rem;
            letter-spacing: -0.02em;
        }}

        .patient-title {{
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 580;
            letter-spacing: -0.055em;
            line-height: 1.18;
            margin-top: -1.2rem !important;
            margin-bottom: 0.2rem;
        }}

        .dashboard-title {{
            margin-top: -0.35rem !important;
        }}

        .patient-desc {{
            color: var(--subtext);
            font-size: 0.93rem;
            line-height: 1.6;
            margin-bottom: 1.15rem;
            font-weight: 400;
        }}

        .profile-card,
        .memo-card,
        .risk-card,
        .stat-card,
        .session-detail-header,
        .report-box {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.032);
        }}

        .profile-card {{
            padding: 1.15rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .profile-avatar {{
            width: 62px;
            height: 62px;
            border-radius: 999px;
            background: var(--primary);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 680;
            flex-shrink: 0;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
        }}

        .profile-summary {{
            font-size: 1rem;
            color: var(--text);
            font-weight: 580;
            margin-bottom: 0.28rem;
            letter-spacing: -0.025em;
        }}

        .profile-meta {{
            color: var(--subtext);
            font-size: 0.84rem;
            line-height: 1.55;
            font-weight: 400;
        }}

        .status-tag {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            background: var(--primary-soft);
            color: var(--primary-dark);
            border: 1px solid var(--primary-soft);
            padding: 0.14rem 0.48rem;
            font-size: 0.7rem;
            font-weight: 500;
            margin-left: 0.45rem;
            vertical-align: middle;
        }}

        .home-section-title {{
            color: var(--text);
            font-size: 0.96rem;
            font-weight: 540;
            margin: 0.9rem 0 0.65rem;
            letter-spacing: -0.025em;
        }}

        .home-card-subtle {{
            color: var(--subtext);
            font-size: 0.74rem;
            font-weight: 500;
            margin-bottom: 0.28rem;
        }}

        .home-card-title {{
            color: var(--text);
            font-size: 0.88rem;
            font-weight: 580;
            margin-bottom: 0.45rem;
        }}

        .memo-card,
        .risk-card {{
            padding: 0.95rem 1.05rem;
            min-height: 124px;
            color: var(--text);
            font-size: 0.88rem;
            line-height: 1.58;
        }}

        .risk-ok {{
            color: #059669;
        }}

        .risk-alert {{
            color: #DC2626;
        }}

        .stat-card {{
            padding: 0.85rem 1rem;
            min-height: 124px;
        }}

        .home-stats-caption {{
            margin-top: 0.65rem;
            color: var(--subtext);
            font-size: 0.82rem;
            line-height: 1.5;
        }}

        .stat-label {{
            color: var(--subtext);
            font-size: 0.78rem;
            font-weight: 500;
            margin-bottom: 0.32rem;
        }}

        .stat-value {{
            color: var(--text);
            font-size: 1.72rem;
            font-weight: 580;
            line-height: 1.1;
            letter-spacing: -0.045em;
            margin-top: 0.8rem;
        }}

        .session-detail-header,
        .report-box {{
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }}

        .session-detail-header {{
            margin-top: 1.05rem;
            margin-bottom: 1.55rem;
            padding: 1.25rem 1.25rem;
        }}

        .journal-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.25rem 1.35rem;
            margin-top: 0.2rem;
        }}

        .journal-label {{
            color: var(--subtext);
            font-size: 0.74rem;
            font-weight: 520;
            margin-bottom: 0.28rem;
        }}

        .journal-value {{
            color: var(--text);
            font-size: 0.88rem;
            font-weight: 450;
            line-height: 1.48;
            word-break: keep-all;
        }}

        .report-box {{
            min-height: 260px;
        }}

        .report-box-title {{
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 580;
            margin-bottom: 0.55rem;
            letter-spacing: -0.02em;
        }}

        .report-box-body {{
            color: var(--text);
            font-size: 0.88rem;
            line-height: 1.68;
            white-space: pre-wrap;
        }}

        .script-box {{
            background: #F8FAFC;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.05rem 1.1rem;
            min-height: 230px;
            color: var(--text);
            font-size: 0.9rem;
            line-height: 1.72;
            white-space: pre-wrap;
        }}

        .record-list-row {{
            display: grid;
            grid-template-columns: 0.12fr 0.18fr 0.42fr 0.18fr;
            gap: 1rem;
            align-items: center;
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 480;
        }}

        .record-list-meta {{
            color: var(--subtext);
            font-weight: 400;
        }}

        .new-session-title {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 580;
        }}

        .new-session-card-head {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.95rem;
        }}

        .new-session-card-icon {{
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 10px;
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #2563EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .new-session-card-title {{
            color: #0F172A;
            font-size: 1rem;
            font-weight: 620;
            letter-spacing: -0.025em;
            line-height: 1.25;
        }}

        .new-session-card-desc {{
            color: #64748B;
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 0.15rem;
        }}

        .new-session-mode-pills-marker {{
            display: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) div[data-testid="stButtonGroup"] {{
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 999px !important;
            padding: 0.18rem !important;
            display: inline-flex !important;
            width: fit-content !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) button[data-testid="stBaseButton-pills"],
        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) button[data-testid="stBaseButton-pillsActive"] {{
            border-radius: 999px !important;
            min-height: 2rem !important;
            height: 2rem !important;
            padding: 0 1.25rem !important;
            font-size: 0.82rem !important;
            font-weight: 560 !important;
            border: 0 !important;
            box-shadow: none !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) button[data-testid="stBaseButton-pills"] {{
            background: transparent !important;
            color: #334155 !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) button[data-testid="stBaseButton-pillsActive"] {{
            background: #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.18) !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.new-session-mode-pills-marker) button[data-testid="stBaseButton-pillsActive"] * {{
            color: #FFFFFF !important;
        }}

        .chart-panel-title {{
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 580;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }}

        .chart-panel-desc {{
            color: var(--subtext);
            font-size: 0.82rem;
            margin-bottom: 0.75rem;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: #E2E8F0 !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.025) !important;
        }}

        .dashboard-section-gap {{
            height: 1.05rem;
        }}

        .risk-metric-card {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.05rem 1rem;
            min-height: 8rem;
            box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.035);
        }}

        .risk-metric-head {{
            display: flex;
            align-items: center;
            gap: 0.62rem;
            margin-bottom: 0.62rem;
        }}

        .risk-metric-icon {{
            width: 1.55rem;
            height: 1.55rem;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.92rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .risk-metric-icon.blue {{
            background: #EFF6FF;
            color: #2563EB;
        }}

        .risk-metric-icon.sky {{
            background: #F0F7FF;
            color: #1D4ED8;
        }}

        .risk-metric-icon.orange {{
            background: #FFF7ED;
            color: #F97316;
        }}

        .risk-metric-icon.red {{
            background: #FEF2F2;
            color: #EF4444;
        }}

        .risk-metric-label {{
            color: var(--subtext);
            font-size: 0.85rem;
            font-weight: 500;
            line-height: 1.35;
        }}

        .risk-metric-value {{
            color: var(--text);
            font-size: 1.90rem;
            font-weight: 500;
            line-height: 1.5;
            letter-spacing: -0.055em;
            margin-left: 2rem;
            margin-top: 0.15rem;
        }}

        .risk-metric-status {{
            display: inline-flex;
            margin-top: 0.12rem;
            margin-left: 2rem;
            color: var(--primary-dark);
            background: var(--primary-soft);
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 540;
        }}

        .risk-pill-low,
        .risk-pill-stable {{
            background: #ECFDF3 !important;
            color: #159947 !important;
        }}

        .risk-pill-watch {{
            background: #EFF6FF !important;
            color: #2563EB !important;
        }}

        .risk-pill-caution {{
            background: #FFF7ED !important;
            color: #F97316 !important;
        }}

        .risk-pill-danger {{
            background: #FEF2F2 !important;
            color: #EF4444 !important;
        }}

        .ai-summary-section {{
            margin-top: 0.25rem;
        }}

        .ai-summary-title {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 620;
            margin-bottom: 0.75rem;
        }}

        .ai-summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
        }}

        .ai-summary-card {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.95rem;
            min-height: 190px;
        }}

        .ai-summary-head {{
            display: flex;
            gap: 0.55rem;
            align-items: center;
            margin-bottom: 0.65rem;
        }}

        .ai-summary-icon {{
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--primary-soft);
            color: var(--primary-dark);
            font-weight: 700;
        }}

        .ai-summary-card-title {{
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .ai-summary-list {{
            margin: 0;
            padding-left: 1.05rem;
            color: var(--text);
            font-size: 0.82rem;
            line-height: 1.65;
        }}

        .ai-summary-note {{
            color: var(--subtext);
            background: #F8FAFC;
            border-radius: 12px;
            padding: 0.62rem;
            margin-top: 0.7rem;
            font-size: 0.78rem;
            line-height: 1.5;
        }}

        .ai-summary-footnote {{
            display: flex;
            align-items: flex-start;
            gap: 0.48rem;
            color: #7A7F8C;
            font-size: 0.76rem;
            font-weight: 400;
            line-height: 1.48;
            margin-top: 0.75rem;
            word-break: keep-all;
        }}

        .dashboard-side-note {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            min-height: 285px;
        }}

        .hira-detail-title {{
            color: #0F172A;
            font-size: 1.50rem;
            font-weight: 520;
            letter-spacing: -0.045em;
            line-height: 2.05;
            margin: -0.4rem 0 0.28rem;
        }}

        .hira-detail-desc {{
            color: #64748B;
            font-size: 0.9rem;
            font-weight: 400;
            line-height: 1.65;
            margin-bottom: 1rem;
        }}

        .hira-donut-title {{
            color: #0F172A;
            font-size: 0.96rem;
            font-weight: 620;
            letter-spacing: -0.025em;
            margin-bottom: 0.2rem;
        }}

        .hira-donut-title span {{
            color: #64748B;
            font-size: 0.74rem;
            font-weight: 500;
        }}

        .hira-highlight-filter-marker {{
            display: none;
        }}

        div[data-testid="column"]:has(.hira-highlight-filter-marker) {{
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            transform: translateY(-1.5rem) !important;
        }}

        div[data-testid="column"]:has(.hira-highlight-filter-marker) label {{
            color: #0F172A !important;
            font-size: 0.78rem !important;
            font-weight: 540 !important;
            margin-bottom: 0.18rem !important;
        }}

        div[data-testid="column"]:has(.hira-highlight-filter-marker) div[data-baseweb="select"] > div {{
            min-height: 1.95rem !important;
            height: 1.95rem !important;
            border-radius: 9px !important;
            background: #F3F6FA !important;
            border-color: #E2E8F0 !important;
        }}

        div[data-testid="column"]:has(.hira-highlight-filter-marker) div[data-baseweb="select"] span {{
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            color: #0F172A !important;
        }}

        div[data-testid="column"]:has(.hira-highlight-filter-marker) div[data-baseweb="select"] svg {{
            width: 0.9rem !important;
            height: 0.9rem !important;
        }}

        .hira-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.72rem;
            margin-bottom: 1.15rem;
        }}

        .hira-kpi-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.82rem 0.9rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
            min-height: 5.2rem;
        }}

        .hira-kpi-label {{
            color: #475569;
            font-size: 0.78rem;
            font-weight: 500;
            line-height: 1.35;
            margin-bottom: 0.45rem;
        }}

        .hira-kpi-value {{
            color: #0F172A;
            font-size: 1.45rem;
            font-weight: 620;
            line-height: 1.2;
            letter-spacing: -0.045em;
            word-break: keep-all;
        }}

        .hira-kpi-note {{
            display: flex;
            align-items: flex-start;
            gap: 0.48rem;
            color: #7A7F8C;
            font-size: 0.76rem;
            font-weight: 400;
            line-height: 1.48;
            margin-top: 0.35rem;
            word-break: keep-all;
        }}

        .hira-kpi-note-icon {{
            width: 1.05rem;
            height: 1.05rem;
            min-width: 1.05rem;
            border-radius: 999px;
            background: #EFF6FF;
            color: #2563EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.68rem;
            font-weight: 650;
            line-height: 1;
            margin-top: 0.16rem;
        }}

        .dashboard-note-icon {{
            width: 1.05rem;
            height: 1.05rem;
            min-width: 1.05rem;
            border-radius: 999px;
            background: #EFF6FF;
            color: #2563EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.68rem;
            font-weight: 650;
            line-height: 1;
            margin-top: 0.12rem;
        }}

        .dashboard-note-line {{
            display: flex;
            align-items: flex-start;
            gap: 0.48rem;
            color: #7A7F8C;
            font-size: 0.76rem;
            font-weight: 400;
            line-height: 1.48;
            margin-top: 0.45rem;
            word-break: keep-all;
        }}

        .report-ai-note {{
            margin-top: 0.75rem !important;
            margin-bottom: 0.65rem !important;
        }}

        .hira-stat-warning-note {{
            margin-top: 0.55rem !important;
            margin-bottom: 0.85rem !important;
        }}

        .hira-interpret-title-row {{
            display: flex;
            align-items: center;
            gap: 0.48rem;
            margin-bottom: 0.65rem;
        }}

        .hira-interpret-title-icon {{
            width: 1.25rem;
            height: 1.25rem;
            min-width: 1.25rem;
            border-radius: 999px;
            background: #EFF6FF;
            color: #2563EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1;
        }}

        .hira-interpret-title {{
            color: #0F172A;
            font-size: 0.98rem;
            font-weight: 560;
            letter-spacing: -0.025em;
        }}

        .hira-interpret-card {{
            color: #334155;
            font-size: 0.88rem;
            font-weight: 400;
            line-height: 1.68;
            letter-spacing: -0.012em;
            word-break: keep-all;
        }}

        .hira-interpret-card p {{
            margin: 0 0 0.68rem;
        }}

        .hira-interpret-card p:last-child {{
            margin-bottom: 0;
        }}

        .hira-interpret-card strong {{
            color: #0F172A;
            font-weight: 620;
        }}

        .hira-interpret-lead {{
            color: #1E293B;
        }}

        .hira-interpret-final {{
            color: #475569;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.72rem 0.85rem;
            margin-top: 0.2rem !important;
            margin-bottom: 0.65rem !important;
        }}

        .dashboard-side-note-title {{
            color: var(--text);
            font-weight: 600;
            margin-bottom: 0.6rem;
        }}

        .dashboard-side-note-body,
        .dashboard-side-note-list {{
            color: var(--subtext);
            font-size: 0.84rem;
            line-height: 1.65;
        }}

        .factor-category-marker,
        .factor-category-is-selected,
        .factor-category-is-unselected {{
            display: none;
        }}

        div[data-testid="stMarkdownContainer"]:has(.factor-category-marker),
        div[data-testid="stMarkdownContainer"]:has(.factor-category-is-selected),
        div[data-testid="stMarkdownContainer"]:has(.factor-category-is-unselected) {{
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton {{
            width: fit-content !important;
            margin-right: 0.45rem !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child {{
            width: auto !important;
            min-width: auto !important;
            min-height: 2.15rem !important;
            height: 2.15rem !important;
            border-radius: 999px !important;
            padding: 0 0.92rem !important;
            font-size: 0.82rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            box-shadow: none !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child span {{
            font-size: 0.82rem !important;
            font-weight: 400 !important;
            line-height: 1 !important;
            margin: 0 !important;
            white-space: nowrap !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child {{
            background: #7DB3F7 !important;
            border: 1px solid #7DB3F7 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(125, 179, 247, 0.18) !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child span {{
            color: #FFFFFF !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child {{
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            color: #0F172A !important;
        }}


        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            color: #2563EB !important;
        }}

        /* AI 보고서 다운로드 버튼 디자인 */
        div[data-testid="stDownloadButton"] > button:first-child {{
            height: 2.65rem !important;
            min-height: 2.65rem !important;
            border-radius: 14px !important;
            border: 1px solid #D6E3F3 !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035) !important;
            font-size: 0.9rem !important;
            font-weight: 400 !important;
            letter-spacing: -0.025em !important;
            transition: all 0.15s ease !important;
        }}

        div[data-testid="stDownloadButton"] > button:first-child:hover {{
            background: #F8FBFF !important;
            border-color: #93C5FD !important;
            color: #1D4ED8 !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.08) !important;
        }}

        div[data-testid="stDownloadButton"] > button:first-child p {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.45rem !important;
            font-size: 0.9rem !important;
            font-weight: 400 !important;
            letter-spacing: -0.025em !important;
            margin: 0 !important;
            color: inherit !important;
            white-space: nowrap !important;
        }}

        div[data-testid="stDownloadButton"] > button:first-child img {{
            width: 1.08rem !important;
            height: 1.08rem !important;
            object-fit: contain !important;
            display: inline-block !important;
            vertical-align: middle !important;
        }}

        div[data-testid="stDownloadButton"] > button:first-child p {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.42rem !important;
            white-space: nowrap !important;
        }}

        .report-section-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.65rem;
        }}

        .report-section-title {{
            color: var(--text);
            font-size: 0.96rem;
            font-weight: 600;
        }}

        .report-block-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }}

        .report-block-title {{
            color: #0F172A;
            font-size: 0.92rem;
            font-weight: 650;
            letter-spacing: -0.025em;
        }}

        .report-edit-badge {{
            display: inline-flex;
            align-items: center;
            margin-left: 0.45rem;
            padding: 0.18rem 0.45rem;
            border-radius: 999px;
            background: #EFF6FF;
            color: #2563EB;
            font-size: 0.7rem;
            font-weight: 650;
            line-height: 1;
        }}

        .report-attached-chart-title {{
            color: #0F172A;
            font-size: 0.9rem;
            font-weight: 620;
            letter-spacing: -0.025em;
            margin-bottom: 0.25rem;
        }}

        .report-edit-badge {{
            display: inline-flex;
            margin-left: 0.45rem;
            padding: 0.14rem 0.42rem;
            border-radius: 999px;
            background: var(--primary-soft);
            color: var(--primary-dark);
            font-size: 0.68rem;
            font-weight: 560;
        }}

        .report-plan-panel,
        .report-chart-preview {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            margin-top: 1rem;
        }}

        .report-plan-title {{
            color: var(--text);
            font-weight: 620;
            margin-bottom: 0.75rem;
        }}

        .report-plan-grid,
        .chart-placeholder-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
        }}

        .report-plan-card,
        .chart-placeholder-card {{
            background: #F8FAFC;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.85rem;
            color: var(--subtext);
            font-size: 0.82rem;
            line-height: 1.55;
        }}

        .report-plan-card-title {{
            color: var(--text);
            font-weight: 600;
            margin-bottom: 0.45rem;
        }}

        .report-plan-card ul {{
            margin: 0;
            padding-left: 1rem;
        }}

        .report-footnote {{
            color: var(--subtext);
            font-size: 0.78rem;
            margin-top: 1rem;
        }}

        .report-preview-shell {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            padding: 2rem 2.25rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        .report-preview-title {{
            text-align: center;
            color: #0F172A;
            font-size: 1.9rem;
            font-weight: 720;
            letter-spacing: -0.055em;
            margin-bottom: 0.3rem;
        }}

        .report-preview-date {{
            text-align: right;
            color: #64748B;
            font-size: 0.78rem;
            margin-bottom: 0.8rem;
        }}


        .report-preview-chip-row {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-bottom: 1rem;
        }}

        .report-preview-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            border: 1px solid #BFDBFE;
            background: #F8FAFC;
            color: #1E40AF;
            border-radius: 999px;
            padding: 0.38rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 570;
        }}

        .report-preview-meta-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border: 1px solid #D8E1F0;
            border-radius: 10px;
            overflow: hidden;
            margin: 1.2rem 0 1.6rem;
        }}

        .report-preview-meta-table th,
        .report-preview-meta-table td {{
            border-right: 1px solid #D8E1F0;
            border-bottom: 1px solid #D8E1F0;
            padding: 0.75rem 0.9rem;
            text-align: left;
            font-size: 0.86rem;
        }}

        .report-preview-meta-table th {{
            background: #F8FAFC;
            color: #334155;
            font-weight: 700;
            width: 14%;
        }}

        .report-preview-meta-table td {{
            background: #FFFFFF;
            color: #0F172A;
            font-weight: 600;
            width: 19%;
        }}

        .report-preview-meta-table tr:last-child th,
        .report-preview-meta-table tr:last-child td {{
            border-bottom: none;
        }}

        .report-preview-meta-table th:last-child,
        .report-preview-meta-table td:last-child {{
            border-right: none;
        }}       

        .report-preview-summary {{
            background: #F1F7FF;
            border: 1px solid #D6E8FF;
            border-radius: 14px;
            padding: 1rem 1.05rem;
            margin: 1rem 0 1.4rem;
        }}

        .session-card-row {{
            display: grid;
            grid-template-columns: 0.13fr 0.18fr 0.35fr 0.2fr 0.14fr;
            gap: 1rem;
            align-items: center;
            width: 100%;
        }}

        .session-card-session-num {{
            color: var(--text);
            font-weight: 540;
            font-size: 0.9rem;
        }}

        .session-card-date {{
            color: var(--subtext);
            font-size: 0.85rem;
            font-weight: 450;
        }}

        .session-card-content {{
            display: flex;
            flex-direction: column;
            gap: 0.08rem;
            justify-content: center;
            transform: translateY(-0.30rem);
        }}

        .session-card-title {{
            color: var(--text);
            font-weight: 500;
            font-size: 0.97rem;
            line-height: 1.25;
            letter-spacing: -0.025em;
        }}

        .session-card-type {{
            color: var(--subtext);
            font-size: 0.72rem;
            font-weight: 400;
            line-height: 1.2;
            margin-top: 0;
        }}

        .session-card-status-wrapper {{
            display: grid;
            place-items: center;
            justify-content: center;
            height: 100%;
            min-height: 3.8rem;
            transform: translateY(-0.5rem);
        }}

        .session-card-status-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--primary-soft);
            color: var(--primary-dark);
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 540;
            white-space: nowrap;
            border: 0;
            box-shadow: none;
        }}



        /* 상담내역 카드 내부 구성 */
        .record-session-number-box {{
            width: 4.8rem;
            height: 3.9rem;
            border-radius: 13px;
            background: #EFF6FF;
            color: #1D4ED8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 550;
            letter-spacing: -0.055em;
        }}

        .record-session-title {{
            color: #0F172A;
            font-size: 1.02rem;
            font-weight: 500;
            letter-spacing: -0.045em;
            line-height: 1.25;
            margin-bottom: 0.26rem;
        }}

        .record-session-meta {{
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 430;
            line-height: 1.4;
        }}

        .record-session-card {{
            margin-bottom: 0.35rem !important;
        }}

        .record-session-status-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 3.35rem;
        }}

        .record-open-button-marker {{
            display: none;
        }}

        div[data-testid="element-container"]:has(.record-open-button-marker),
        div[data-testid="stElementContainer"]:has(.record-open-button-marker),
        div[data-testid="stMarkdownContainer"]:has(.record-open-button-marker) {{
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}

        div[data-testid="column"]:has(.record-open-button-marker) {{
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
        }}

        div[data-testid="column"]:has(.record-open-button-marker) div.stButton > button:first-child {{
            width: 100% !important;
            height: 2.45rem !important;
            min-height: 2.45rem !important;
            border-radius: 11px !important;
            border: 1px solid #2563EB !important;
            background: #FFFFFF !important;
            color: #1D4ED8 !important;
            font-size: 0.55rem !important;
            font-weight: 580 !important;
            box-shadow: none !important;
        }}

        div[data-testid="column"]:has(.record-open-button-marker) div.stButton > button:first-child:hover {{
            background: #EFF6FF !important;
            border-color: #1D4ED8 !important;
            color: #1D4ED8 !important;
        }}

        div[data-testid="column"]:has(.record-open-button-marker) div.stButton > button:first-child p {{
            color: #1D4ED8 !important;
            font-weight: 580 !important;
        }}

        /* 상담내역 카드 내부 요소 세로 위치 보정 */
        .record-session-number-box {{
            transform: translateY(-0.50rem);
        }}

        .record-session-title,
        .record-session-meta {{
            transform: translateY(-0.40rem);
        }}

        .record-session-status-wrap {{
            transform: translateY(-0.50rem);
        }}

        div[data-testid="column"]:has(.record-open-button-marker) div.stButton {{
            margin: 0 !important;
        }}

        div[data-testid="column"]:has(.record-open-button-marker) div.stButton > button {{
            transform: none !important;
        }}


        .session-card-status-complete {{
            background: var(--primary-soft);
            color: var(--primary-dark);
        }}

        .session-card-status-review {{
            background: #FFF7ED;
            color: #C2410C;
        }}

        .session-card-status-draft {{
            background: #F1F5F9;
            color: #475569;
        }}


        .report-preview-summary-title {{
            color: #1E40AF;
            font-size: 0.94rem;
            font-weight: 660;
            margin-bottom: 0.5rem;
        }}

        .report-preview-summary-body {{
            color: #0F172A;
            font-size: 0.9rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }}

        .report-preview-section {{
            margin-top: 1.15rem;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
        }}

        .report-preview-auto-badge {{
            display: inline-flex;
            align-items: center;
            margin-left: 0.45rem;
            padding: 0.16rem 0.45rem;
            border-radius: 999px;
            background: #EFF6FF;
            color: #2563EB;
            font-size: 0.68rem;
            font-weight: 600;
            vertical-align: middle;
        }}

        .report-preview-section-title {{
            color: #1E3A8A;
            font-size: 1rem;
            font-weight: 680;
            letter-spacing: -0.03em;
            padding-bottom: 0.45rem;
            border-bottom: 1px dashed #CBD5E1;
            margin-bottom: 0.6rem;
        }}

        .report-preview-section-body {{
            color: #0F172A;
            font-size: 0.9rem;
            line-height: 1.78;
            white-space: pre-wrap;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}

        .report-preview-chart-panel {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.9rem;
            margin-top: 1.4rem;
        }}

        .report-preview-chart-title {{
            color: #1E40AF;
            font-size: 0.92rem;
            font-weight: 660;
            margin-bottom: 0.65rem;
        }}

        .report-preview-chart-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }}

        .report-preview-mini-chart {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            min-height: 130px;
            padding: 0.8rem;
            color: #64748B;
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        .report-preview-footer-note {{
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #1E40AF;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 1rem;
        }}

        .chat-guide-banner {{
            background: var(--primary-soft);
            border: 1px solid #BFDBFE;
            color: var(--primary-dark);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            font-size: 0.84rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}

        .chat-page-card {{
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0.25rem 0.25rem 0.85rem 0.25rem !important;
            margin-bottom: 0.85rem !important;
            min-height: 58vh;
            box-sizing: border-box;
        }}

        .chat-row {{
            display: flex;
            gap: 0.65rem;
            margin-bottom: 0.85rem;
        }}

        .chat-row.user {{
            justify-content: flex-end;
        }}

        .chat-avatar {{
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: var(--primary-soft);
            color: var(--primary-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
        }}

        .chat-bubble-wrap {{
            max-width: 76%;
        }}

        .chat-bubble {{
            background: #F8FAFC;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.78rem 0.9rem;
            color: var(--text);
            font-size: 0.88rem;
            line-height: 1.65;
        }}

        .chat-row.user .chat-bubble {{
            background: var(--primary);
            color: #FFFFFF;
            border-color: var(--primary);
        }}

        .chat-time {{
            color: var(--muted);
            font-size: 0.7rem;
            margin-top: 0.25rem;
        }}

        .chat-row.user .chat-time {{
            text-align: right;
        }}

        .chat-source-row {{
            margin-top: 0.6rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }}

        .chat-source-chip {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.16rem 0.42rem;
            color: var(--subtext);
            font-size: 0.7rem;
        }}

        .quick-question-title {{
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 650;
            margin: 1rem 0 0.55rem;
        }}

        .quick-question-marker,
        .chat-input-marker,
        .chat-send-marker,
        .chat-composer-anchor {{
            display: none;
        }}

        .st-key-chat_composer_bar {{
            position: relative !important;
            width: 100% !important;
            z-index: 120 !important;
            background: rgba(255, 255, 255, 0.98) !important;
            border-top: 1px solid #EEF2F7 !important;
            padding: 0.2rem 0 0.35rem 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
            transform: translateY(-0.5rem) !important;
        }}

        .st-key-chat_composer_bar > div[data-testid="stVerticalBlock"] {{
            width: 100% !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }}

        .st-key-chat_composer_bar .quick-question-title {{
            margin-top: 0 !important;
        }}

        div[data-testid="column"]:has(.chat-input-marker) {{
            padding-right: 0 !important;
            position: relative !important;
            z-index: 1 !important;
        }}

        div[data-testid="column"]:has(.chat-input-marker) div[data-baseweb="input"] {{
            height: 2.65rem !important;
            min-height: 2.65rem !important;
            border-radius: 999px !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.07) !important;
        }}

        div[data-testid="column"]:has(.chat-input-marker) input {{
            height: 2.65rem !important;
            min-height: 2.65rem !important;
            color: var(--text) !important;
            font-size: 0.84rem !important;
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
        }}

        div[data-testid="column"]:has(.chat-input-marker) div[data-baseweb="input"]:hover {{
            background: #FFFFFF !important;
            border-color: #CBD5E1 !important;
        }}

        div[data-testid="column"]:has(.chat-send-marker) {{
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            position: relative !important;
            z-index: 3 !important;
            overflow: visible !important;
            margin-left: 0 !important; /* removed negative margin so button stays inside composer */
            min-width: 4rem !important;
            padding-top: 0 !important;
        }}

        div[data-testid="column"]:has(.chat-send-marker) div.stButton > button:first-child,
        .st-key-chatbot_send button {{
            width: 2.65rem !important;
            min-width: 2.65rem !important;
            max-width: 2.65rem !important;
            height: 2.65rem !important;
            min-height: 2.65rem !important;
            border-radius: 999px !important;
            border: 1px solid #4F63F6 !important;
            background: #4F63F6 !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 12px 24px rgba(79, 99, 246, 0.26) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.chat-send-marker) div.stButton > button:first-child:hover,
        .st-key-chatbot_send button:hover {{
            background: #3B4FE8 !important;
            border-color: #3B4FE8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 14px 28px rgba(79, 99, 246, 0.32) !important;
        }}

        div[data-testid="column"]:has(.chat-send-marker) div.stButton > button:first-child p,
        .st-key-chatbot_send button p {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            line-height: 1 !important;
            margin: 0 !important;
        }}

        div[data-testid="column"]:has(.chat-send-marker) div.stButton > button:first-child img,
        .st-key-chatbot_send button img {{
            width: 1.22rem !important;
            height: 1.22rem !important;
            object-fit: contain !important;
            display: block !important;
        }}

        div.stButton > button:first-child {{
            border-radius: 11px;
            min-height: 2.35rem;
            font-size: 0.84rem;
            line-height: 1.2;
            font-weight: 480;
            border: 1px solid #CBD5E1;
            color: var(--text);
            background: #FFFFFF;
            white-space: nowrap;
            box-shadow: none;
        }}

        div.stButton > button:hover {{
            border-color: var(--primary);
            color: var(--primary-dark);
            background-color: var(--primary-soft);
        }}

        div.stButton > button[kind="primary"] {{
            background: var(--primary) !important;
            border-color: var(--primary) !important;
            color: white !important;
            border-radius: 11px;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.14);
        }}

        div.stButton > button[kind="primary"] p {{
            color: #FFFFFF !important;
        }}

        div.stDownloadButton > button:first-child {{
            border-radius: 11px;
            min-height: 2.35rem;
            font-size: 0.84rem;
            font-weight: 500;
            border: 1px solid #CBD5E1;
        }}

        div.stDownloadButton > button:first-child p {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.38rem !important;
            white-space: nowrap !important;
        }}

        div.stDownloadButton > button:first-child img {{
            width: 1rem !important;
            height: 1rem !important;
            object-fit: contain !important;
            display: inline-block !important;
        }}

        section[data-testid="stSidebar"] input {{
            border-radius: 11px !important;
            font-size: 0.82rem !important;
            font-weight: 400 !important;
            background: #FFFFFF !important;
        }}

        .new-client-expander-marker {{
            display: none;
        }}

        /* 신규 등록 expander 바깥 기본 박스 제거 */
        /* 신규 등록 expander 전체 영역 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) {{
            border: none ! important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.025) ! important;
            background: rgba(255, 255, 255, 0.34) ! important;
            border-radius: 10px ! important;
            padding: 0.1rem !important;
            margin-bottom: 1rem ! important;
            overflow: hidden ! important;
        }}

        /* 신규 등록 버튼 본체 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary {{
            height: 2.85rem !important;
            min-height: 2.85rem !important;
            padding: 0 1rem !important;
            background: transparent !important;
            border: 1.6px solid #CBD5E1 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            color: #0F172A !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            font-size: 0.95rem !important;
            font-weight: 520 !important;
            letter-spacing: -0.025em !important;
        }}

        /* Streamlit expander 기본 테두리/배경 겹침 방지 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary > div {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}

        /* hover 상태 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:hover {{
            background: rgba(255, 255, 255, 0.38) !important;
            border-color: #94A3B8 !important;
            color: #0F172A !important;
        }}

        /* 신규 등록 텍스트 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary p,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary span {{
            color: #0F172A !important;
            font-size: 0.80rem !important;
            font-weight: 400 !important;
            letter-spacing: -0.025em !important;
        }}

        /* expander 화살표 위치 안정화 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary svg {{
            color: #0F172A !important;
        }}

        /* 신규 등록 텍스트 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary p,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary span {{
            color: #0F172A !important;
            font-size: 0.85rem !important;
            font-weight: 400 !important;
            letter-spacing: -0.025em !important;
        }}

        /* expander 화살표 위치 안정화 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary svg {{
            color: #0F172A !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:first-child {{
            display: flex !important;
            min-height: 2.35rem !important;
            height: 2.35rem !important;
            border-radius: 9px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: #0F172A !important;
            box-shadow: none !important;
            justify-content: flex-start !important;
            padding-left: 0.82rem !important;
            padding-right: 0.82rem !important;
            font-size: 0.9rem !important;
            font-weight: 430 !important;
            letter-spacing: -0.025em !important;
            margin-bottom: 0.25rem !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:first-child p {{
            font-size: 0.9rem !important;
            font-weight: 430 !important;
            color: inherit !important;
            width: 100% !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:first-child div[data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button,
        section[data-testid="stSidebar"] .st-key-side_nav_records button,
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button,
        section[data-testid="stSidebar"] .st-key-side_nav_report button,
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button,
        section[data-testid="stSidebar"] .st-key-side_nav_settings button {{
            justify-content: flex-start !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_records button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_report button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_settings button div[data-testid="stMarkdownContainer"] {{
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            width: 100% !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button p,
        section[data-testid="stSidebar"] .st-key-side_nav_records button p,
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button p,
        section[data-testid="stSidebar"] .st-key-side_nav_report button p,
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button p,
        section[data-testid="stSidebar"] .st-key-side_nav_settings button p {{
            width: 100% !important;
            text-align: left !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button *,
        section[data-testid="stSidebar"] .st-key-side_nav_records button *,
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button *,
        section[data-testid="stSidebar"] .st-key-side_nav_report button *,
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button *,
        section[data-testid="stSidebar"] .st-key-side_nav_settings button * {{
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:first-child:hover {{
            background: #EAF2FF !important;
            border-color: transparent !important;
            color: #1E40AF !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background: #EFF6FF !important;
            border-color: #BFDBFE !important;
            color: #2563EB !important;
            box-shadow: none !important;
            border-radius: 9px !important;
            min-height: 2.38rem !important;
            height: 2.38rem !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p {{
            color: #2563EB !important;
            font-weight: 560 !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:disabled {{
            background: transparent !important;
            border-color: transparent !important;
            color: #94A3B8 !important;
            opacity: 1 !important;
            font-weight: 400 !important;
        }}

        section[data-testid="stSidebar"] div.stButton > button:disabled p {{
            color: #94A3B8 !important;
            font-weight: 400 !important;
        }}

        .register-client-button-marker {{
            display: none;
        }}

        /* 신규 등록 버튼만 가운데 정렬 */
        section[data-testid="stSidebar"] .st-key-register_new_client button {{
            justify-content: center !important;
            text-align: center !important;
            padding-left: 0.82rem !important;
            padding-right: 0.82rem !important;
        }}


        section[data-testid="stSidebar"] .st-key-register_new_client button p,
        section[data-testid="stSidebar"] .st-key-register_new_client div[data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            text-align: center !important;
            justify-content: center !important;
        }}

        /* 사이드바 메뉴 버튼은 왼쪽 정렬 */
        section[data-testid="stSidebar"] .st-key-side_nav_home button,
        section[data-testid="stSidebar"] .st-key-side_nav_records button,
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button,
        section[data-testid="stSidebar"] .st-key-side_nav_report button,
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button,
        section[data-testid="stSidebar"] .st-key-side_nav_settings button {{
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_records button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_report button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button div[data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] .st-key-side_nav_settings button div[data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            display: block !important;
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] .st-key-side_nav_home button p,
        section[data-testid="stSidebar"] .st-key-side_nav_records button p,
        section[data-testid="stSidebar"] .st-key-side_nav_dashboard button p,
        section[data-testid="stSidebar"] .st-key-side_nav_report button p,
        section[data-testid="stSidebar"] .st-key-side_nav_chatbot button p,
        section[data-testid="stSidebar"] .st-key-side_nav_settings button p {{
            width: 100% !important;
            margin: 0 !important;
            text-align: left !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: #FFFFFF;
            padding: 0.85rem 0.9rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.03);
        }}

        div[data-testid="stMetricLabel"] {{
            color: var(--subtext) !important;
            font-weight: 500 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--text) !important;
            font-weight: 680 !important;
            letter-spacing: -0.045em !important;
        }}

        textarea:focus,
        input:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.18) !important;
            outline: none !important;
        }}

        div[data-baseweb="select"]:focus-within {{
            border-color: #93C5FD !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.12) !important;
            outline: none !important;
        }}

        @media (max-width: 1100px) {{
            .main .block-container {{
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }}


            .record-list-row {{
                grid-template-columns: 1fr;
                gap: 0.45rem;
            }}

            .ai-summary-grid {{
                grid-template-columns: 1fr;
            }}

            .report-plan-grid,
            .chart-placeholder-grid,
            .report-preview-chart-grid {{
                grid-template-columns: 1fr;
            }}

            .chat-bubble-wrap {{
                max-width: 92%;
            }}
        }}

        /* =========================================================
           Factor category pill buttons
        ========================================================= */
        .factor-category-marker,
        .factor-category-is-selected,
        .factor-category-is-unselected {{
            display: none;
        }}

        div[data-testid="stMarkdownContainer"]:has(.factor-category-marker),
        div[data-testid="stMarkdownContainer"]:has(.factor-category-is-selected),
        div[data-testid="stMarkdownContainer"]:has(.factor-category-is-unselected) {{
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) {{
            flex: 0 0 auto !important;
            width: auto !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton {{
            width: auto !important;
            display: inline-flex !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child {{
            width: auto !important;
            min-width: 4.2rem !important;
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            max-height: 2.25rem !important;
            border-radius: 9999px !important;
            padding: 0 1.05rem !important;
            font-size: 0.88rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            box-shadow: none !important;
            justify-content: center !important;
            transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child span {{
            font-size: 0.88rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            margin: 0 !important;
            white-space: nowrap !important;
            width: auto !important;
            text-align: center !important;
        }}
        
        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child {{
            background: #2563EB !important;
            border-color: #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.18) !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child span {{
            color: #FFFFFF !important;
            font-size: 0.88rem !important;
            font-weight: 520 !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child {{
            background: #FFFFFF !important;
            border-color: #E5E7EB !important;
            color: #0F172A !important;
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            max-height: 2.25rem !important;
            font-size: 0.88rem !important;
            font-weight: 520 !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            color: #2563EB !important;
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            max-height: 2.25rem !important;
            font-size: 0.88rem !important;
            font-weight: 520 !important;
        }}

        button[data-testid="stBaseButton-pills"],
        button[data-testid="stBaseButton-pillsActive"] {{
            border-radius: 9999px !important;
            min-height: 2.25rem !important;
            height: 2.25rem !important;
            max-height: 2.25rem !important;
            padding: 0 1.05rem !important;
            font-size: 0.88rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            box-shadow: none !important;
        }}

        button[data-testid="stBaseButton-pills"] {{
            background: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            color: #0F172A !important;
        }}

        button[data-testid="stBaseButton-pills"]:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            color: #2563EB !important;
        }}

        button[data-testid="stBaseButton-pillsActive"] {{
            background: #2563EB !important;
            border: 1px solid #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.20) !important;
        }}

        button[data-testid="stBaseButton-pillsActive"] p,
        button[data-testid="stBaseButton-pillsActive"] span,
        button[data-testid="stBaseButton-pillsActive"] div {{
            color: #FFFFFF !important;
        }}

        div[data-testid="column"]:has(.dashboard-ai-button-marker) {{
            display: flex !important;
            justify-content: flex-end !important;
        }}

        div[data-testid="column"]:has(.dashboard-ai-button-marker) div.stButton > button:first-child {{
            width: auto !important;
            min-width: 7.1rem !important;
            height: 2.55rem !important;
            min-height: 2.55rem !important;
            border-radius: 999px !important;
            border: 1px solid #BFDBFE !important;
            background: #FFFFFF !important;
            color: #1E40AF !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.45rem !important;
            padding: 0 0.9rem !important;
            font-size: 0.84rem !important;
            font-weight: 560 !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.dashboard-ai-button-marker) div.stButton > button:first-child:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            transform: translateY(-1px);
        }}

        div[data-testid="column"]:has(.dashboard-ai-button-marker) div.stButton > button:first-child p {{
            color: #1E40AF !important;
            font-size: 0.84rem !important;
            font-weight: 560 !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.42rem !important;
        }}

        div[data-testid="column"]:has(.dashboard-ai-button-marker) div.stButton > button:first-child img {{
            height: 1.35rem !important;
            width: 1.35rem !important;
            object-fit: contain !important;
            margin-right: 0.12rem !important;
            vertical-align: middle !important;
        }}

        .dashboard-ai-button-marker {{
            display: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.dashboard-session-select-marker) div[data-baseweb="select"] > div {{
            transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
            cursor: pointer !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.dashboard-session-select-marker) div[data-baseweb="select"] > div:hover {{
            background-color: #EFF6FF !important;
            border-color: #BFDBFE !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08) !important;
            cursor: pointer !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.dashboard-session-select-marker) div[data-baseweb="select"] * {{
            cursor: pointer !important;
        }}

        .dashboard-session-select-marker {{
            display: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.dashboard-session-select-marker) {{
            margin-top: -1.35rem !important;
        }}

        div[data-testid="column"]:has(.session-detail-list-button-marker),
        div[data-testid="column"]:has(.session-detail-dashboard-button-marker),
        div[data-testid="column"]:has(.session-detail-chat-button-marker) {{
            display: flex !important;
            justify-content: flex-end !important;
        }}

        div[data-testid="column"]:has(.session-detail-list-button-marker) div.stButton > button:first-child,
        div[data-testid="column"]:has(.session-detail-dashboard-button-marker) div.stButton > button:first-child {{
            height: 2.35rem !important;
            min-height: 2.35rem !important;
            border-radius: 11px !important;
            font-size: 0.84rem !important;
            font-weight: 520 !important;
            box-shadow: none !important;
            padding: 0 0.85rem !important;
        }}

        div[data-testid="column"]:has(.session-detail-chat-button-marker) div.stButton > button:first-child {{
            width: 2.35rem !important;
            min-width: 2.35rem !important;
            height: 2.35rem !important;
            min-height: 2.35rem !important;
            border-radius: 12px !important;
            border: 1px solid #C7D7EA !important;
            background: #F8FBFF !important;
            color: #2563EB !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.session-detail-chat-button-marker) div.stButton > button:first-child:hover {{
            background: #EAF3FF !important;
            border-color: #9FC4F4 !important;
            color: #1D4ED8 !important;
            transform: translateY(-1px) !important;
        }}

        div[data-testid="column"]:has(.session-detail-chat-button-marker) div.stButton > button:first-child p {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1 !important;
            width: 100% !important;
            text-align: center !important;
            color: inherit !important;
        }}

        div[data-testid="column"]:has(.session-detail-chat-button-marker) div.stButton > button:first-child img {{
            width: 1.32rem !important;
            height: 1.32rem !important;
            object-fit: contain !important;
            display: block !important;
        }}

        .session-detail-list-button-marker,
        .session-detail-dashboard-button-marker,
        .session-detail-chat-button-marker {{
            display: none;
        }}

        /* 챗봇 페이지는 전체 화면 스크롤을 막고, 메시지 영역만 스크롤 */
        .chatbot-page-marker {{
            display: none;
        }}

        div[data-testid="stMarkdownContainer"]:has(.chatbot-page-marker) {{
            display: none !important;
            height: 0 !important;
        }}

        .main .block-container:has(.chatbot-page-marker) {{
            min-height: auto !important;
            height: auto !important;
            overflow: visible !important;
            padding-bottom: 0 !important;
        }}

        .main .block-container:has(.chatbot-page-marker) .chat-page-card {{
            height: 50vh !important;
            min-height: 360px !important;
            overflow-y: auto !important;
            padding-bottom: 4.2rem !important;
            margin-bottom: 0.25rem !important;
            box-sizing: border-box !important;
        }}

        /* 챗봇 하단 composer 내부 세로 여백 강제 축소 */
        .st-key-chat_composer_bar div[data-testid="stVerticalBlock"] {{
            gap: 0.25rem !important;
        }}

        .st-key-chat_composer_bar div[data-testid="stHorizontalBlock"] {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}

        .st-key-chat_composer_bar div[data-testid="element-container"],
        .st-key-chat_composer_bar div[data-testid="stElementContainer"] {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}

        .st-key-chat_composer_bar div.stButton {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}

        .st-key-chat_composer_bar div[data-baseweb="input"] {{
            margin-top: 0 !important;
        }}

        .chat-answer-intro {{
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 1rem;
                padding: 0.9rem 1rem;
                margin-bottom: 0.85rem;
                color: #334155;
                font-size: 0.95rem;
                line-height: 1.75;
            }}

            .chat-answer-card {{
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 1rem;
                padding: 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }}

            .chat-answer-body {{
                color: #1E293B;
                font-size: 0.95rem;
                line-height: 1.8;
                word-break: keep-all;
                overflow-wrap: anywhere;
            }}

            .chat-answer-section {{
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                border: 1px solid #E2E8F0;
                border-radius: 1rem;
                padding: 1rem 1.05rem;
                margin: 0.85rem 0;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.035);
            }}

            .chat-answer-section-title {{
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: #0F172A;
                font-weight: 800;
                font-size: 1.02rem;
                margin-bottom: 0.65rem;
            }}

            .chat-answer-section-number {{
                display: inline-flex;
                width: 1.65rem;
                height: 1.65rem;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                background: #EEF3FF;
                color: #4F6EF7;
                font-size: 0.82rem;
                font-weight: 800;
                flex: 0 0 auto;
            }}

            .chat-answer-section-body {{
                color: #334155;
                font-size: 0.94rem;
                line-height: 1.78;
                word-break: keep-all;
                overflow-wrap: anywhere;
            }}


        /* 공통 챗봇 이동 버튼 */
        .chatbot-nav-button-marker {{
            display: none;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: flex-start !important;
            padding-right: 0 !important;
        }}

        .st-key-dashboard_chatbot_fab,
        .st-key-session_detail_chatbot_button {{
            width: 100% !important;
            display: flex !important;
            justify-content: flex-end !important;
        }}

        .st-key-dashboard_chatbot_fab {{
            transform: translateY(-0.65rem) !important;
        }}

        .st-key-session_detail_chatbot_button {{
            transform: translateY(0rem) !important;
        }}

        .st-key-dashboard_chatbot_fab div.stButton,
        .st-key-session_detail_chatbot_button div.stButton {{
            width: auto !important;
            display: flex !important;
            justify-content: flex-end !important;
        }}

        .st-key-dashboard_chatbot_fab div.stButton {{
            width: auto !important;
            display: flex !important;
            justify-content: flex-end !important;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) div.stButton > button:first-child,
        .st-key-session_detail_chatbot_button button,
        .st-key-dashboard_chatbot_fab button {{
            width: 2.65rem !important;
            min-width: 2.65rem !important;
            max-width: 2.65rem !important;
            height: 2.65rem !important;
            min-height: 2.65rem !important;
            max-height: 2.65rem !important;
            border-radius: 12px !important;
            border: 1px solid #2563EB !important;
            background: #2563EB !important;
            color: #FFFFFF !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.20) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) div.stButton > button:first-child:hover,
        .st-key-session_detail_chatbot_button button:hover,
        .st-key-dashboard_chatbot_fab button:hover {{
            background: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) div.stButton > button:first-child p,
        .st-key-session_detail_chatbot_button button p,
        .st-key-dashboard_chatbot_fab button p {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            color: #FFFFFF !important;
            line-height: 1 !important;
            margin: 0 !important;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) div.stButton > button:first-child img,
        .st-key-session_detail_chatbot_button button img,
        .st-key-dashboard_chatbot_fab button img {{
            width: 1.35rem !important;
            height: 1.35rem !important;
            object-fit: contain !important;
            display: block !important;
            filter: brightness(0) invert(1) !important;
        }}   
           
        /* =========================================================
           내담자 홈 리디자인: 알약 탭 + 상담 요약 카드
        ========================================================= */
        .patient-home-tab-marker,
        .patient-home-tab-active,
        .patient-home-tab-inactive {{
            display: none;
        }}

        .patient-home-tab-shell {{
            width: 36rem;
            max-width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 13px;
            padding: 0.18rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.025);
            margin: 0.95rem 0 1.45rem;
        }}

        div[data-testid="column"]:has(.patient-home-tab-marker) div.stButton > button:first-child {{
            height: 2.35rem !important;
            min-height: 2.35rem !important;
            border-radius: 10px !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 0.86rem !important;
            font-weight: 560 !important;
        }}

        div[data-testid="column"]:has(.patient-home-tab-active) div.stButton > button:first-child {{
            background: #EFF6FF !important;
            color: #1D4ED8 !important;
        }}

        div[data-testid="column"]:has(.patient-home-tab-active) div.stButton > button:first-child p {{
            color: #1D4ED8 !important;
            font-weight: 620 !important;
        }}

        div[data-testid="column"]:has(.patient-home-tab-inactive) div.stButton > button:first-child {{
            background: transparent !important;
            color: #0F172A !important;
        }}

        div[data-testid="column"]:has(.patient-home-tab-inactive) div.stButton > button:first-child:hover {{
            background: #F8FAFC !important;
            color: #1D4ED8 !important;
        }}

        .home-summary-title {{
            color: #0F172A;
            font-size: 1.05rem;
            font-weight: 580;
            letter-spacing: -0.035em;
            margin: 0.3rem 0 0.8rem;
        }}

        .home-summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.95rem;
            margin-top: 1.25rem;
            margin-bottom: 1.05rem;
        }}

        .home-summary-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.05rem 1rem;
            min-height: 8rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.035);
        }}

        .home-summary-head {{
            display: flex;
            align-items: center;
            gap: 0.62rem;
            margin-bottom: 0.62rem;
        }}

        .home-summary-icon {{
            width: 1.55rem;
            height: 1.55rem;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.92rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .home-summary-icon.blue {{
            background: #EFF6FF;
            color: #2563EB;
        }}

        .home-summary-icon.sky {{
            background: #F0F7FF;
            color: #1D4ED8;
        }}

        .home-summary-icon.orange {{
            background: #FFF7ED;
            color: #F97316;
        }}

        .home-summary-icon.red {{
            background: #FEF2F2;
            color: #EF4444;
        }}

        .home-summary-label {{
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 500;
            line-height: 1.35;
        }}

        .home-summary-value {{
            color: #0F172A;
            font-size: 1.90rem;
            font-weight: 500;
            letter-spacing: -0.055em;
            line-height: 1.5;
            margin-left: 2.0rem;
            margin-top: 0.15rem;
        }}

        .home-summary-value.warning {{
            color: #0F172A;
            font-size: 1.55rem;
            font-weight: 560;
            letter-spacing: -0.055em;
            line-height: 1.1;
            margin-left: 2.0rem;
            margin-top: 0.18rem;
        }}

        .home-summary-alert-text {{
            color: #EF4444;
            font-size: 0.8rem;
            line-height: 1.45;
            margin-top: 0.68rem;
            margin-left: 2.0rem;
            font-weight: 450;
            letter-spacing: -0.025em;
        }}

        .home-recent-summary-card {{
            display: flex;
            align-items: center;
            gap: 0.95rem;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 0.95rem 1.05rem;
            min-height: 4.5rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.032);
        }}

        .home-recent-summary-icon {{
            width: 2.45rem;
            height: 2.45rem;
            border-radius: 12px;
            background: #EFF6FF;
            color: #2563EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 720;
            flex-shrink: 0;
        }}

        .home-recent-summary-label {{
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 560;
            margin-bottom: 0.22rem;
        }}

        .home-recent-summary-text {{
            color: #0F172A;
            font-size: 0.88rem;
            font-weight: 520;
            line-height: 1.45;
        }}

        @media (max-width: 1100px) {{
            .home-summary-grid {{
                grid-template-columns: 1fr 1fr;
            }}

            .patient-home-tab-shell {{
                width: 100%;
            }}
        }} 
        /* 신규 등록 버튼 색상만 수정 */
        section[data-testid="stSidebar"] .st-key-register_new_client button {{
            background: #FFFFFF !important;
            border: 1px solid #BFDBFE !important;
            color: #2563EB !important;
            justify-content: center !important;
        }}

        section[data-testid="stSidebar"] .st-key-register_new_client button p {{
            color: #2563EB !important;
            width: 100% !important;
            text-align: center !important;
        }}

        section[data-testid="stSidebar"] .st-key-register_new_client button div[data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }}

        section[data-testid="stSidebar"] .st-key-register_new_client button div[data-testid="stMarkdownContainer"] p {{
            width: 100% !important;
            margin: 0 auto !important;
            text-align: center !important;
        }}

        section[data-testid="stSidebar"] .st-key-register_new_client button:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            color: #1D4ED8 !important;
        }}

        section[data-testid="stSidebar"] .st-key-register_new_client button:hover p {{
            color: #1D4ED8 !important;
        }}

        .new-client-gender-marker {{
            display: none;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.new-client-gender-marker) div[data-baseweb="select"] input,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.new-client-gender-marker) div[data-baseweb="select"] input:focus {{
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
            background: transparent !important;
            color: transparent !important;
            caret-color: transparent !important;
        }}

        section[data-testid="stSidebar"] .st-key-new_client_name {{
            margin-bottom: -0.9rem !important;
        }}

        /* 사이드바 내담자 검색창 - 검색 아이콘 */
        .sidebar-search-input-marker {{
            display: none;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.sidebar-search-input-marker) div[data-baseweb="input"] {{
            position: relative !important;
        }}

        /* 사이드바 내담자 선택 검색창에만 search.png 아이콘 적용 */
        .sidebar-search-input-marker {{
            display: none;
        }}

        section[data-testid="stSidebar"] .st-key-sidebar_client_search_box div[data-baseweb="input"] {{
            position: relative !important;
        }}

        section[data-testid="stSidebar"] .st-key-sidebar_client_search_box div[data-baseweb="input"]::before {{
            content: "";
            position: absolute;
            left: 0.82rem;
            top: 50%;
            transform: translateY(-50%);
            width: 1rem;
            height: 1rem;
            background-image: url("{search_icon_data_uri}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            z-index: 3;
            pointer-events: none;
            opacity: 0.55;
            transition: opacity 0.12s ease;
        }}

        section[data-testid="stSidebar"] .st-key-sidebar_client_search_box div[data-baseweb="input"]:focus-within::before {{
            opacity: 0;
        }}

        section[data-testid="stSidebar"] .st-key-sidebar_client_search_box input {{
            padding-left: 2.15rem !important;
        }}

        section[data-testid="stSidebar"] .st-key-sidebar_client_search_box div[data-baseweb="input"]:focus-within input {{
            padding-left: 0.82rem !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.sidebar-search-input-marker) div[data-baseweb="input"]:focus-within::before {{
            opacity: 0;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.sidebar-search-input-marker) input {{
            padding-left: 2.15rem !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.sidebar-search-input-marker) div[data-baseweb="input"]:focus-within input {{
            padding-left: 0.82rem !important;
        }}

        .sidebar-user-card {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0;
            border-radius: 0;
            background: transparent;
            border: 0;
            margin-top: 0.7rem;
        }}

        .sidebar-user-avatar {{
            width: 2.00rem;
            height: 2.00rem;
            border-radius: 999px;
            background: #EFF6FF;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            flex-shrink: 0;
        }}

        .sidebar-user-avatar img {{
            width: 2.00rem;
            height: 2.00rem;
            object-fit: contain;
            display: block;
        }}

        .sidebar-user-info {{
            min-width: 0;
        }}

        .sidebar-user-name {{
            color: #0F172A;
            font-size: 0.8rem;
            font-weight: 580;
            line-height: 1.25;
        }}

        .sidebar-user-role {{
            color: #64748B;
            font-size: 0.70rem;
            font-weight: 500;
            margin-top: 0.12rem;
        }}

        /* 신규 등록 expander 내부 text_input만 왼쪽 정렬 */
        section[data-testid="stSidebar"] .st-key-new_client_name div[data-baseweb="input"] input,
        section[data-testid="stSidebar"] .st-key-new_client_age div[data-baseweb="input"] input,
        section[data-testid="stSidebar"] .st-key-new_client_region div[data-baseweb="input"] input {{
            text-align: left !important;
        }}

        section[data-testid="stSidebar"] .st-key-new_client_name div[data-baseweb="input"] input::placeholder,
        section[data-testid="stSidebar"] .st-key-new_client_age div[data-baseweb="input"] input::placeholder,
        section[data-testid="stSidebar"] .st-key-new_client_region div[data-baseweb="input"] input::placeholder {{
            text-align: left !important;
        }}

        /* 신규 등록 expander 내부 입력창 왼쪽 정렬 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) div[data-baseweb="input"] input {{
            text-align: left !important;
            padding-left: 0.82rem !important;
        }}

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) div[data-baseweb="input"] input::placeholder {{
            text-align: left !important;
    }}

        /* Streamlit 버튼 hover/focus/active 주황색 제거 */
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            border-color: #CBD5E1 !important;
            color: #0F172A !important;
            box-shadow: none !important;
        }}

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: #93C5FD !important;
            color: #2563EB !important;
            background: #F8FBFF !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08) !important;
        }}

        div[data-testid="stButton"] > button:focus,
        div[data-testid="stButton"] > button:focus-visible,
        div[data-testid="stDownloadButton"] > button:focus,
        div[data-testid="stDownloadButton"] > button:focus-visible {{
            border-color: #3B82F6 !important;
            color: #2563EB !important;
            background: #F8FBFF !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.16) !important;
            outline: none !important;
        }}

        div[data-testid="stButton"] > button:active,
        div[data-testid="stDownloadButton"] > button:active {{
            border-color: #2563EB !important;
            color: #1D4ED8 !important;
            background: #EFF6FF !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
        }}

        div[data-testid="stButton"] > button:focus:not(:focus-visible),
        div[data-testid="stDownloadButton"] > button:focus:not(:focus-visible) {{
            outline: none !important;
        }}

        /* =====================================================
           Streamlit 기본 주황/빨강 focus 색상 제거 → 파란 계열 통일
        ===================================================== */

        /* 1) 검색 input focus 테두리 */
        section[data-testid="stSidebar"] div[data-baseweb="input"] {{
            border-color: #E2E8F0 !important;
            box-shadow: none !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {{
            border-color: #93C5FD !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.16) !important;
            outline: none !important;
        }}

        section[data-testid="stSidebar"] input:focus,
        section[data-testid="stSidebar"] input:focus-visible {{
            outline: none !important;
            box-shadow: none !important;
            border-color: transparent !important;
        }}


        /* 2) 신규 등록 expander hover/focus/active 색상 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:hover,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:focus,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:focus-visible,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:active {{
            background: #F8FBFF !important;
            border-color: #93C5FD !important;
            color: #0F172A !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.14) !important;
            outline: none !important;
        }}



        /* 3) 일반 버튼 / 다운로드 버튼 focus-visible 주황색 제거 */
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            border-color: #93C5FD !important;
            color: #2563EB !important;
            background: #F8FBFF !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08) !important;
            outline: none !important;
        }}

        div[data-testid="stButton"] > button:focus,
        div[data-testid="stButton"] > button:focus-visible,
        div[data-testid="stButton"] > button:active,
        div[data-testid="stDownloadButton"] > button:focus,
        div[data-testid="stDownloadButton"] > button:focus-visible,
        div[data-testid="stDownloadButton"] > button:active {{
            border-color: #3B82F6 !important;
            color: #2563EB !important;
            background: #F8FBFF !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.16) !important;
            outline: none !important;
        }}

        /* 4) Streamlit 기본 focus outline 사각형 제거 */
        button:focus,
        button:focus-visible,
        input:focus,
        input:focus-visible,
        summary:focus,
        summary:focus-visible,
        div[data-baseweb="select"]:focus,
        div[data-baseweb="select"]:focus-visible,
        div[data-baseweb="select"]:focus-within {{
            outline: none !important;
        }}

        /* selectbox 내부 input 때문에 생기는 파란 사각형 제거 */
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] input:focus,
        div[data-baseweb="select"] input:focus-visible {{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }}

        /* 신규 등록 expander 클릭 시 화살표 주변 focus 사각형 제거 */
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:focus,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:focus-visible,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:active {{
            outline: none !important;
        }}



        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary [data-testid],
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary [data-testid]:focus,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary [data-testid]:focus-visible,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary [data-testid]:active {{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
            background: transparent !important;
        }}

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary button,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary button:hover,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary button:focus,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary button:focus-visible,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary button:active {{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
            background: transparent !important;
            color: #0F172A !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 10. Sidebar
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">CounsHelper</div>
            <div class="sidebar-subtitle">심리상담사 AI 보조 플랫폼</div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown('<div class="sidebar-section-title">내담자 선택</div>', unsafe_allow_html=True)

        if st.session_state.pop("clear_client_search_input", False):
            st.session_state.clear_client_search_input = True
            st.session_state.client_search = ""

        search_input_key = f"client_search_input_{st.session_state.get('client_search_nonce', 0)}"

        with st.container(key="sidebar_client_search_box"):
            st.markdown('<span class="sidebar-search-input-marker"></span>', unsafe_allow_html=True)

            search_query = st.text_input(
                "내담자 검색",
                value=st.session_state.get("client_search", ""),
                placeholder="이름 / 내담자ID / 성별 / 지역",
                label_visibility="collapsed",
                key=search_input_key,
            )

        st.session_state.client_search = search_query

        filtered_clients = CLIENTS.copy()

        if search_query.strip() and not filtered_clients.empty:
            query = search_query.strip().lower()

            def matches_client(row: pd.Series) -> bool:
                values = [
                    row.get("이름", ""),
                    row.get("내담자 ID", ""),
                    row.get("성별", ""),
                    row.get("지역", ""),
                    row.get("상담 유형", ""),
                    row.get("메모", ""),
                ]
                return any(query in str(value).lower() for value in values)

            filtered_clients = filtered_clients[filtered_clients.apply(matches_client, axis=1)]
        else:
            filtered_clients = filtered_clients.iloc[0:0]

        if st.session_state.get("show_selected_client_label", False):
            current_client_label = _format_sidebar_client_label(st.session_state.selected_client)

            st.markdown(
                f"""
                <div style="
                    color:#64748B;
                    font-size:0.76rem;
                    line-height:1.45;
                    margin:0.45rem 0 0.55rem;
                ">
                    현재 선택: <span style="color:#0F172A; font-weight:560;">{html_escape(current_client_label)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if search_query.strip():
            if filtered_clients.empty:
                st.caption("검색 결과가 없습니다.")
            else:
                st.markdown(
                    """
                    <div style="
                        color:#64748B;
                        font-size:0.74rem;
                        font-weight:520;
                        margin:0.35rem 0 0.35rem;
                    ">
                        검색 결과
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for _, client in filtered_clients.head(6).iterrows():
                    client_id = str(client["내담자 ID"])
                    client_label = _format_sidebar_client_label(client_id)

                    if st.button(
                        client_label,
                        key=f"sidebar_client_result_{client_id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_client = client_id
                        st.session_state.client_search = ""
                        st.session_state.client_search_nonce = st.session_state.get("client_search_nonce", 0) + 1
                        st.session_state.show_selected_client_label = True
                        st.session_state.patient_home_tab = "내담자 정보"

                        client_sessions = SESSIONS[SESSIONS["내담자 ID"] == client_id]

                        if not client_sessions.empty:
                            selected_session = client_sessions.iloc[0]["회기"]
                            st.session_state.selected_session = selected_session
                            st.session_state.record_mode = "existing"

                            dialogue_key = (client_id, selected_session)

                            if dialogue_key in SESSION_DIALOGUES:
                                st.session_state.dialogue_rows = SESSION_DIALOGUES[dialogue_key].copy()
                            else:
                                st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()
                        else:
                            st.session_state.selected_session = "새 상담"
                            st.session_state.record_mode = "new"
                            st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()

                        st.session_state.analysis_result = None
                        go_page("내담자 홈")
                        st.rerun()

        with st.expander("신규 등록", expanded=False):
            st.markdown('<span class="new-client-expander-marker"></span>', unsafe_allow_html=True)
            new_client_name = st.text_input("이름", placeholder="예: 안녕", key="new_client_name")
            st.markdown('<span class="new-client-gender-marker"></span>', unsafe_allow_html=True)
            new_client_gender = st.selectbox("성별", ["여성", "남성", "기타/미상"], key="new_client_gender")
            new_client_age = st.text_input("연령", placeholder="예: 30대 또는 32", key="new_client_age")
            new_client_region = st.text_input("지역", placeholder="예: 서울", key="new_client_region")

            st.markdown('<span class="register-client-button-marker"></span>', unsafe_allow_html=True)
            if st.button("등록", type="primary", use_container_width=True, key="register_new_client"):
                did_register = register_new_client(
                    name=new_client_name,
                    gender=new_client_gender,
                    age=new_client_age,
                    region=new_client_region,
                    memo="",
                )

                if did_register:
                    st.rerun()

        st.divider()

        nav_items = [
            ("내담자 홈", "내담자 홈", "side_nav_home"),
            ("상담내역 기록·추가", "상담내역 기록·추가", "side_nav_records"),
            ("분석 대시보드", "분석 대시보드", "side_nav_dashboard"),
            ("AI 보고서", "AI 보고서", "side_nav_report"),
            ("챗봇", "챗봇", "side_nav_chatbot"),
        ]


        for label, page_name, key in nav_items:
            active = st.session_state.page == page_name

            if st.button(
                label,
                key=key,
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                go_page(page_name)
                st.rerun()

        st.markdown("<div style='height: 4.0rem;'></div>", unsafe_allow_html=True)

        profile_icon_data_uri = get_png_data_uri(PROFILE_ICON_PATH)        
        st.markdown(
            f"""
            <div class="sidebar-user-card">
                <div class="sidebar-user-avatar">
                    <img src="{profile_icon_data_uri}" alt="profile" />
                </div>
                <div class="sidebar-user-info">
                    <div class="sidebar-user-name">오은영의 데이터 상담소</div>
                    <div class="sidebar-user-role">MVP Demo</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 11. Header / Navigation
# =========================================================
def render_header():
    client_row = get_client_row()

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


def render_main_nav():
    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# 12. 상담내역 기록·추가 화면
# =========================================================
def render_session_cards():
    header_col, add_col = st.columns([0.82, 0.18], vertical_alignment="top")

    with header_col:
        st.markdown('<div class="patient-title">상담내역 기록·추가</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="patient-desc">회기별 상담 기록을 확인하고 새 상담 내역을 추가합니다.</div>',
            unsafe_allow_html=True,
        )    


    with add_col:
        if st.button("+ 새 상담 추가", key="add_new_session_top", use_container_width=True, type="primary"):
            start_new_session()
            st.rerun()

    st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client].copy()

    def _session_order(value: Any) -> int:
        match = re.search(r"(\d+)", str(value or ""))
        return int(match.group(1)) if match else 999

    client_sessions["_session_order"] = client_sessions["회기"].apply(_session_order)
    client_sessions = client_sessions.sort_values("_session_order", ascending=True).drop(columns=["_session_order"])

    if client_sessions.empty:
        st.info("기존 상담 내역이 없습니다. 우측 상단의 새 상담 추가 버튼을 눌러 첫 상담 내역을 입력해 주세요.")
        return

    for _, row in client_sessions.iterrows():
        render_session_list_card(row, key_prefix="record")

    st.markdown(
        """
        <div style="color:#64748B; font-size:0.82rem; text-align:center; margin-top:0.6rem;">
            회기를 클릭하면 상세 내용을 확인할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_record_editor():
    st.markdown('<div class="section-title">상담 기록 입력</div>', unsafe_allow_html=True)

    st.session_state.dialogue_rows = st.data_editor(
        st.session_state.dialogue_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "화자": st.column_config.SelectboxColumn(
                "화자",
                options=["상담사", "내담자"],
                required=True,
            ),
            "발화": st.column_config.TextColumn("발화", width="large"),
        },
    )

    script = build_dialogue_text(st.session_state.dialogue_rows)

    st.markdown("#### 상담 스크립트 미리보기")
    st.text_area(
        "모델 입력 형태",
        value=script,
        height=220,
        help="이 텍스트가 KlueBERT, Koalpaca, 28요인 추출 모델의 입력으로 들어갑니다.",
    )

    c1, c2, c3 = st.columns([0.22, 0.22, 0.56])

    with c1:
        if st.button("AI 분석 실행", type="primary", use_container_width=True):
            if not script.strip():
                st.warning("상담 발화를 먼저 입력하세요.")
            else:
                with st.spinner("AI 분석을 실행하는 중입니다..."):
                    st.session_state.analysis_result = run_analysis(script)
                st.success("분석 완료")
                go_page("분석 대시보드")
                st.rerun()

    with c2:
        if st.button("JSON 내보내기", use_container_width=True):
            temp_result = st.session_state.analysis_result or {}
            export_json = make_json_export(temp_result)
            st.download_button(
                "다운로드",
                data=export_json.encode("utf-8"),
                file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}.json",
                mime="application/json",
            )


def render_record_page():
    if st.session_state.get("record_mode") in ["new", "draft"]:
        render_new_session_form()
        return

    render_session_cards()


def _session_order(value: Any) -> int:
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else 999


def get_client_sessions_sorted() -> pd.DataFrame:
    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client].copy()

    if client_sessions.empty:
        return client_sessions

    client_sessions["_session_order"] = client_sessions["회기"].apply(_session_order)
    return client_sessions.sort_values("_session_order", ascending=True).drop(columns=["_session_order"])


def analyze_current_session_if_needed():
    if st.session_state.analysis_result is not None:
        return

    script = build_dialogue_text(st.session_state.dialogue_rows)

    if not script.strip():
        return

    with st.spinner("선택 회기 분석 결과를 준비하는 중입니다..."):
        st.session_state.analysis_result = run_analysis(script)


def risk_status(score: float) -> str:
    if score >= 2:
        return "주의"
    if score >= 1:
        return "관찰"
    return "낮음"


def style_dashboard_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", size=12),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            title=None,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def get_session_classification_label(session_row: pd.Series) -> str:
    explicit_value = str(session_row.get("분류 유형", "") or "").strip()

    if explicit_value and explicit_value.lower() != "nan":
        return explicit_value

    class_value = str(session_row.get("class", "") or "").strip()

    if class_value and class_value.lower() != "nan":
        return class_value

    score_labels = [
        ("depression", "우울"),
        ("anxiety", "불안"),
        ("addiction", "중독"),
    ]
    active_labels = []

    for column, label in score_labels:
        try:
            if int(float(session_row.get(column, 0) or 0)) > 0:
                active_labels.append(label)
        except (TypeError, ValueError):
            continue

    return "/".join(active_labels) if active_labels else "미분류"


def get_session_status_class(status_text: str) -> str:
    text = str(status_text or "")

    if any(keyword in text for keyword in ["검토", "주의", "확인", "중"]):
        return "session-card-status-review"

    if any(keyword in text for keyword in ["완료", "작성 완료", "전처리 완료", "분석 완료"]):
        return "session-card-status-complete"

    return "session-card-status-draft"


def render_session_list_card(row: pd.Series, key_prefix: str):
    session_name = str(row.get("회기", "회기"))
    session_date = str(row.get("상담일", "날짜 미상"))
    counseling_title = str(row.get("상담 주제", "상담 주제 미입력"))
    counseling_type_display = get_session_classification_label(row)

    status_text = str(row.get("보고서 상태", "상태 미상"))
    status_class = get_session_status_class(status_text)

    with st.container(border=True):
        num_col, info_col, status_col, open_col = st.columns(
            [0.13, 0.59, 0.14, 0.14],
            vertical_alignment="center",
        )

        with num_col:
            st.markdown(
                f"""
                <div class="record-session-number-box">
                    {html_escape(session_name)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info_col:
            st.markdown(
                f"""
                <div>
                    <div class="record-session-title">{html_escape(counseling_title)}</div>
                    <div class="record-session-meta">
                        {html_escape(session_date)} · 분류 유형: {html_escape(counseling_type_display)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with status_col:
            st.markdown(
                f"""
                <div class="record-session-status-wrap">
                    <span class="session-card-status-badge {status_class}">
                        {html_escape(status_text)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with open_col:
            st.markdown('<span class="record-open-button-marker"></span>', unsafe_allow_html=True)

            if st.button(
                "열기",
                key=f"{key_prefix}_open_{row['내담자 ID']}_{row['회기']}",
                use_container_width=True,
            ):
                if "임시 저장" in status_text:
                    open_draft_session_editor(row["회기"])
                else:
                    open_session_detail(row["회기"])
                st.rerun()


def _get_client_risk_signal(client_sessions: pd.DataFrame) -> Tuple[str, str]:
    risk_sessions = []

    for _, row in client_sessions.iterrows():
        session_name = row.get("회기", "")
        dialogue = SESSION_DIALOGUES.get((st.session_state.selected_client, session_name))

        if dialogue is None:
            continue

        script = build_dialogue_text(dialogue)

        if any(keyword in script for keyword in ["자살", "죽고 싶", "죽고싶", "사라지고"]):
            risk_sessions.append(str(session_name))

    if risk_sessions:
        return "risk-alert", f"자살/자해 관련 신호 확인 필요 · {', '.join(risk_sessions)}"

    return "risk-ok", "자살/자해 관련 신호 없음"


def render_patient_home():
    client_row = get_client_row()
    client_sessions = get_client_sessions_sorted()

    name = get_client_display_name(client_row)
    client_id = str(client_row.get("내담자 ID", st.session_state.selected_client))
    gender = str(client_row.get("성별", "미상"))
    age = str(client_row.get("연령대", "미상"))
    region = str(client_row.get("지역", "미상"))
    counseling_type = str(client_row.get("상담 유형", "미상"))
    recent_session = str(client_row.get("최근 회기", st.session_state.selected_session))
    avatar = (name or client_id or "C")[0]

    latest_date = client_sessions.iloc[-1]["상담일"] if not client_sessions.empty else "기록 없음"
    registered_date = client_sessions.iloc[0]["상담일"] if not client_sessions.empty else str(client_row.get("등록일", "등록일 미상"))

    analyzed_count = (
        int(client_sessions["보고서 상태"].astype(str).str.contains("작성 완료|전처리 완료|분석 완료", regex=True).sum())
        if not client_sessions.empty
        else 0
    )
    attention_count = (
        int(client_sessions["보고서 상태"].astype(str).str.contains("검토|주의|확인", regex=True).sum())
        if not client_sessions.empty
        else 0
    )

    risk_class, risk_text = _get_client_risk_signal(client_sessions)

    st.markdown('<div class="patient-title">내담자 홈</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="patient-desc">내담자의 상담 요약과 최근 회기 흐름을 확인합니다.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-avatar">{html_escape(avatar)}</div>
            <div>
                <div class="profile-summary">{html_escape(name)}</div>
                <div class="profile-meta">
                    등록일: {html_escape(str(registered_date))} · 성별: {html_escape(gender)} · 연령: {html_escape(age)} · 지역: {html_escape(region)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 내담자 홈 탭: st.pills 기반 알약형 탭
    st.markdown('<span class="patient-home-pills-marker"></span>', unsafe_allow_html=True)

    active_tab = st.pills(
        "내담자 홈 탭",
        options=["내담자 정보", "상담관리"],
        selection_mode="single",
        default=st.session_state.get("patient_home_tab", "내담자 정보"),
        key="patient_home_tab_pills",
        label_visibility="collapsed",
    )

    if active_tab is None:
        active_tab = st.session_state.get("patient_home_tab", "내담자 정보")

    if active_tab != st.session_state.get("patient_home_tab"):
        st.session_state.patient_home_tab = active_tab
        st.rerun()

    st.session_state.patient_home_tab = active_tab

    if active_tab == "내담자 정보":
        risk_value = "확인 필요" if risk_class == "risk-alert" else "없음"


        st.markdown(
            f"""<div class="home-summary-grid">
<div class="home-summary-card">
    <div class="home-summary-head">
        <div class="home-summary-icon blue">↻</div>
        <div class="home-summary-label">총 회기</div>
    </div>
    <div class="home-summary-value">{len(client_sessions)}회</div>
</div>
<div class="home-summary-card">
    <div class="home-summary-head">
        <div class="home-summary-icon sky">✓</div>
        <div class="home-summary-label">분석 완료</div>
    </div>
    <div class="home-summary-value">{analyzed_count}회</div>
</div>
<div class="home-summary-card">
    <div class="home-summary-head">
        <div class="home-summary-icon orange">!</div>
        <div class="home-summary-label">주의 회기</div>
    </div>
    <div class="home-summary-value">{attention_count}회</div>
</div>
<div class="home-summary-card">
    <div class="home-summary-head">
        <div class="home-summary-icon red">◇</div>
        <div class="home-summary-label">위험 신호</div>
    </div>
    <div class="home-summary-value warning">{html_escape(risk_value)}</div>
    <div class="home-summary-alert-text">{html_escape(risk_text)}</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        recent_summary_text = (
            f"{recent_session} · {latest_date} · {counseling_type}"
            if not client_sessions.empty
            else "아직 등록된 상담 회기가 없습니다."
        )

        st.markdown(
            f"""<div class="home-recent-summary-card">
<div class="home-recent-summary-icon">▤</div>
<div>
    <div class="home-recent-summary-label">최근 회기 요약</div>
    <div class="home-recent-summary-text">{html_escape(str(recent_summary_text))}</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    else:
        st.markdown('<div class="home-summary-title">상담 회기 목록</div>', unsafe_allow_html=True)

        if client_sessions.empty:
            st.info("등록된 상담 내역이 없습니다. 상담내역 기록·추가에서 새 상담을 입력해 주세요.")
        else:
            for _, row in client_sessions.iterrows():
                render_session_list_card(row, key_prefix="home")


def _build_session_summary_text(session_row: pd.Series, script: str) -> str:
    existing_summary = str(session_row.get("summary", "") or "").strip()

    if existing_summary and existing_summary.lower() != "nan":
        return existing_summary

    if st.session_state.analysis_result is not None:
        return build_report_text(st.session_state.analysis_result)

    preview_lines = []

    for line in script.split("\n"):
        if line.strip():
            preview_lines.append(line.strip())
        if len(preview_lines) >= 4:
            break

    preview = "\n".join(preview_lines) if preview_lines else "상담 스크립트가 아직 입력되지 않았습니다."

    return (
        f"상담일: {session_row.get('상담일', '미상')}\n"
        f"상담 주제: {session_row.get('상담 주제', '미상')}\n"
        f"보고서 상태: {session_row.get('보고서 상태', '미상')}\n\n"
        f"{preview}"
    )


def _get_journal_id(client_id: str, session_name: str, session_row: pd.Series) -> str:
    filename = str(session_row.get("filename", "") or "").strip()

    if filename and filename.lower() != "nan":
        return Path(filename).stem[:8]

    seed = f"{client_id}:{session_name}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def render_session_detail():
    client_row = get_client_row()
    session_row = get_session_row()
    script = build_dialogue_text(st.session_state.dialogue_rows)

    client_id = str(client_row.get("내담자 ID", st.session_state.selected_client))
    name = get_client_display_name(client_row)
    session_name = str(session_row.get("회기", st.session_state.selected_session))
    session_date = str(session_row.get("상담일", "미상"))
    session_topic = str(session_row.get("상담 주제", "미상"))
    report_status = str(session_row.get("보고서 상태", "미상"))
    journal_id = _get_journal_id(client_id, session_name, session_row)
    counseling_type = str(client_row.get("상담 유형", "상담"))

    title_col, spacer_col, list_col, dashboard_col, chatbot_col = st.columns(
        [0.50, 0.14, 0.13, 0.17, 0.06],
        vertical_alignment="center",
    )

    with title_col:
        st.markdown('<div class="patient-title">상담일지 열람</div>', unsafe_allow_html=True)

    with list_col:
        st.markdown('<span class="session-detail-list-button-marker"></span>', unsafe_allow_html=True)
        if st.button("목록으로", use_container_width=True):
            back_to_patient_home()
            st.rerun()

    with dashboard_col:
        st.markdown('<span class="session-detail-dashboard-button-marker"></span>', unsafe_allow_html=True)
        if st.button("분석 대시보드", use_container_width=True):
            if st.session_state.analysis_result is None and script.strip():
                with st.spinner("선택 회기 분석 대시보드를 준비하는 중입니다..."):
                    st.session_state.analysis_result = run_analysis(script)
            go_page("분석 대시보드")
            st.rerun()

    with chatbot_col:
        st.markdown('<span class="chatbot-nav-button-marker"></span>', unsafe_allow_html=True)
        chatbot_icon_label = f"![chatbot]({get_png_data_uri(CHATBOT_ICON_PATH)})"

        if st.button(
            chatbot_icon_label,
            key="session_detail_chatbot_button",
            use_container_width=False,
            help="챗봇으로 이동",
        ):
            go_page("챗봇")
            st.rerun()
            
    st.markdown(
        f"""
        <div class="session-detail-header">
            <div class="journal-grid">
                <div><div class="journal-label">일지번호</div><div class="journal-value">{html_escape(journal_id)}</div></div>
                <div><div class="journal-label">상담일자</div><div class="journal-value">{html_escape(session_date)}</div></div>
                <div><div class="journal-label">회기</div><div class="journal-value">{html_escape(session_name)}</div></div>
                <div><div class="journal-label">상태</div><div class="journal-value">{html_escape(report_status)}</div></div>
                <div><div class="journal-label">상담제목</div><div class="journal-value">{html_escape(session_topic)}</div></div>
                <div><div class="journal-label">상담분류</div><div class="journal-value">{html_escape(counseling_type)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    report_col, memo_col = st.columns([0.62, 0.38], gap="large")

    with report_col:
        st.markdown('<div class="session-detail-section-title">요약 보고서</div>', unsafe_allow_html=True)
        summary_text = _build_session_summary_text(session_row, script)
        st.markdown(
            f"""
            <div class="report-box">
                <div class="report-box-title">{html_escape(session_name)} 요약</div>
                <div class="report-box-body">{html_escape(summary_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with memo_col:
        st.markdown('<div class="session-detail-section-title">상담사 메모</div>', unsafe_allow_html=True)
        note_key = f"{client_id}_{session_name}"
        current_note = st.session_state.session_notes.get(note_key, "")
        note_value = st.text_area(
            "메모",
            value=current_note,
            height=340,
            placeholder="다음 회기 확인사항, 위험 신호, 보호요인, 상담사 관찰 메모를 입력하세요.",
            label_visibility="collapsed",
            key=f"session_note_input_{note_key}",
        )
        st.session_state.session_notes[note_key] = note_value


def render_new_session_form():
    client_row = get_client_row()
    client_name = get_client_display_name(client_row)

    if "new_session_name" not in st.session_state:
        st.session_state.new_session_name = get_next_session_name()
    if "new_session_date" not in st.session_state:
        st.session_state.new_session_date = datetime.now().date()
    if "new_session_scope" not in st.session_state:
        st.session_state.new_session_scope = "복합"
    if "new_session_topic" not in st.session_state:
        st.session_state.new_session_topic = ""
    if "new_session_input_mode" not in st.session_state:
        st.session_state.new_session_input_mode = "발화 단위 입력"

    if st.session_state.pop("reset_new_session_content_requested", False):
        reset_new_session_content_state()

    title_col, list_col = st.columns([0.82, 0.18], vertical_alignment="top")

    with title_col:
        st.markdown('<div class="patient-title">상담내역 기록·추가</div>', unsafe_allow_html=True)
        

    with list_col:
        if st.button("목록으로", use_container_width=True):
            cancel_new_session()
            st.rerun()

    f1, f2, f3, f4 = st.columns(4)

    with st.container(border=True):
        st.markdown(
            """
            <div class="new-session-card-head">
                <div class="new-session-card-icon">▣</div>
                <div>
                    <div class="new-session-card-title">상담 기본 정보</div>
                    <div class="new-session-card-desc">분석에 사용할 회기 메타 정보를 입력합니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.text_input("회기 번호", key="new_session_name")

        with f2:
            st.date_input("회기 일시", key="new_session_date")

        with f3:
            st.selectbox(
                "상담 범위",
                options=["우울", "불안", "중독", "복합"],
                key="new_session_scope",
            )

        with f4:
            st.text_input(
                "상담 주제",
                placeholder="예: 직장 스트레스로 인한 불면",
                key="new_session_topic",
            )

    with st.container(border=True):
        st.markdown(
            """
            <div class="new-session-card-head">
                <div class="new-session-card-icon">💬</div>
                <div>
                    <div class="new-session-card-title">상담 내용 입력</div>
                    <div class="new-session-card-desc">상담사와 내담자의 발화를 입력하면 AI가 회기 내용을 분석합니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<span class="new-session-mode-pills-marker"></span>', unsafe_allow_html=True)

        input_mode = st.pills(
            "입력 방식",
            options=["발화 단위 입력", "전사 텍스트 붙여넣기"],
            selection_mode="single",
            default=st.session_state.get("new_session_input_mode", "발화 단위 입력"),
            key="new_session_input_mode_pills",
            label_visibility="collapsed",
        )

        if input_mode is None:
            input_mode = st.session_state.get("new_session_input_mode", "발화 단위 입력")

        st.session_state.new_session_input_mode = input_mode

        st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

        if input_mode == "전사 텍스트 붙여넣기":
            default_script = build_dialogue_text(st.session_state.dialogue_rows)
            pasted_script = st.text_area(
                "전사 텍스트",
                value=default_script,
                height=300,
                placeholder="상담 전체 전사 텍스트를 붙여넣어 주세요.",
                label_visibility="collapsed",
                key="new_session_script_text",
            )
        else:
            st.session_state.dialogue_rows = st.data_editor(
                st.session_state.dialogue_rows,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "화자": st.column_config.SelectboxColumn(
                        "화자",
                        options=["상담사", "내담자"],
                        required=True,
                    ),
                    "발화": st.column_config.TextColumn("발화", width="large"),
                },
                key="new_session_dialogue_editor",
            )
            pasted_script = build_dialogue_text(st.session_state.dialogue_rows)

        reset_col, spacer_col = st.columns([0.14, 0.86])

        with reset_col:
            if st.button("입력 초기화", key="new_session_reset_dialogue", use_container_width=True):
                st.session_state.reset_new_session_content_requested = True
                st.rerun()

    spacer, b_save, b_analyze = st.columns(
        [0.62, 0.16, 0.22],
        gap="small",
    )

    with b_save:
        if st.button("임시저장", use_container_width=True):
            save_new_session(pasted_script, run_ai=False)
            st.success("임시 저장했습니다.")
            st.rerun()

    with b_analyze:
        if st.button("AI 분석 실행", type="primary", use_container_width=True):
            if not pasted_script.strip():
                st.warning("상담 내용을 입력하세요.")
            else:
                save_new_session(pasted_script, run_ai=True)
                st.rerun()


def scroll_chat_to_bottom():
    nonce = st.session_state.get("chat_scroll_nonce", 0)

    components.html(
        f"""
        <script>
        const scrollNonce = {nonce};

        function scrollToChatBottom() {{
            const parentDoc = window.parent.document;
            const chatBox =
                parentDoc.getElementById("chat-page-card") ||
                parentDoc.querySelector('[data-chat-scroll-box="true"]');

            if (chatBox) {{
                chatBox.scrollTop = chatBox.scrollHeight;

                const anchor = chatBox.querySelector("#chat-bottom-anchor");

                if (anchor) {{
                    anchor.scrollIntoView({{
                        behavior: "auto",
                        block: "end"
                    }});
                }}

                chatBox.scrollTop = chatBox.scrollHeight;
            }}
        }}

        window.requestAnimationFrame(scrollToChatBottom);
        setTimeout(scrollToChatBottom, 0);
        setTimeout(scrollToChatBottom, 40);
        setTimeout(scrollToChatBottom, 160);
        setTimeout(scrollToChatBottom, 360);
        setTimeout(scrollToChatBottom, 700);
        setTimeout(scrollToChatBottom, 1100);
        </script>
        """,
        height=1,
    )


# =========================================================
# 13. 분석 대시보드
# =========================================================
#hira
def get_factor_groups(factors: Dict[str, int]) -> Dict[str, int]:
    """
    28요인 점수를 4범주로 묶어 합산한다.
    """

    groups = {
        "증상 요인": 0,
        "위험 요인": 0,
        "개선 요인": 0,
        "개입 요인": 0,
    }

    symptom_keys = [
        "depressive_mood",
        "worthlessness",
        "guilt",
        "impaired_cognition",
        "suicidal",
        "anhedonia",
        "psychomotor_changes",
        "weight_appetite",
        "sleep_disturbance",
        "fatigue",
        "anxiety",
        "physical_symptom",
    ]

    risk_keys = [
        "loss_of_control",
        "social_avoidance",
        "craving",
        "withdrawal",
        "tolerance",
        "social_problem",
    ]

    intervention_keys = [
        "sympathy_support",
        "clarification_reflection",
        "cognitive_restructuring",
        "information_provision",
        "goal_setting",
        "task_assignment",
        "behavioral_intervention",
        "coping_skill_training",
        "structuring",
    ]

    improvement_keys = [
        "motivation_for_change",
    ]

    for key, value in factors.items():
        score = int(value or 0)

        if key in symptom_keys:
            groups["증상 요인"] += score
        elif key in risk_keys:
            groups["위험 요인"] += score
        elif key in improvement_keys:
            groups["개선 요인"] += score
        elif key in intervention_keys:
            groups["개입 요인"] += score

    return groups

# =========================================================
# HIRA 대시보드 보조 함수
# =========================================================
HIRA_DISEASE_GROUPS = {
    "depression": {
        "label": "우울",
        "keywords": ["우울"],
    },
    "anxiety": {
        "label": "불안",
        "keywords": ["불안", "공황"],
    },
    "addiction": {
        "label": "중독",
        "keywords": ["중독", "알코올", "물질", "약물", "도박", "의존"],
    },
}


def _get_positive_context_keys(classification: Dict[str, int]) -> List[str]:
    """
    KlueBERT 예측 결과에서 양성으로 나온 우울/불안/중독 key만 반환한다.
    """
    return [
        key
        for key in ["depression", "anxiety", "addiction"]
        if int(classification.get(key, 0) or 0) == 1
    ]


def _get_display_context_keys(
    classification: Dict[str, int],
    include_negative: bool = False,
) -> List[str]:
    """
    기본: KlueBERT 양성 항목만 표시
    include_negative=True: 음성 항목까지 함께 표시
    """
    positive_keys = _get_positive_context_keys(classification)

    if include_negative:
        return ["depression", "anxiety", "addiction"]

    return positive_keys


def _normalize_hira_dataframe(hira_df: pd.DataFrame) -> pd.DataFrame:
    """
    HIRA 데이터 컬럼 타입과 성별 표기를 화면 처리용으로 정리한다.
    """
    df = hira_df.copy()

    for col in ["disease", "gender", "age_group", "sido", "sigungu"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace(
            {
                "남자": "남",
                "남성": "남",
                "여자": "여",
                "여성": "여",
            }
        )

    for col in ["year", "patients", "visit_days", "visit_days_per_patient", "cost_per_patient"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "context_key" not in df.columns:
        df["context_key"] = df["disease"].apply(_infer_hira_context_key_from_disease)
    else:
        df["context_key"] = df["context_key"].fillna("").astype(str).str.strip()
        missing_mask = df["context_key"].eq("") | df["context_key"].eq("nan")
        df.loc[missing_mask, "context_key"] = df.loc[missing_mask, "disease"].apply(
            _infer_hira_context_key_from_disease
        )

    return df


def _infer_hira_context_key_from_disease(disease: Any) -> str:
    """
    질환명 문자열에서 우울/불안/중독 분류를 추정한다.
    """
    disease_text = str(disease or "")

    for context_key, meta in HIRA_DISEASE_GROUPS.items():
        if any(keyword in disease_text for keyword in meta["keywords"]):
            return context_key

    return "etc"


def _get_client_hira_filter_values() -> Tuple[str, str]:
    """
    현재 선택된 내담자의 성별·연령대를 HIRA 데이터 형식으로 변환한다.
    """
    client_row = get_client_row()

    raw_gender = str(client_row.get("성별", "")).strip()
    gender = "여" if raw_gender.startswith("여") else "남"

    age_band_map = {
        "0대": "0~9세",
        "10대": "10~19세",
        "20대": "20~29세",
        "30대": "30~39세",
        "40대": "40~49세",
        "50대": "50~59세",
        "60대": "60~69세",
        "70대": "70~79세",
        "80대": "80~89세",
        "90대": "90~99세",
    }

    age_group = age_band_map.get(str(client_row.get("연령대", "")).strip(), "30~39세")

    return gender, age_group


def build_client_hira_dataframe(
    classification: Dict[str, int],
    include_negative: bool = False,
) -> pd.DataFrame:
    """
    건강보험심사평가원_시군구별 성별 연령별 주요 정신질환 통계 2024 기반으로
    현재 내담자의 성별·연령대에 맞는 우울/불안/중독 관련 데이터를 정리한다.

    기본값:
    - KlueBERT 양성 항목만 반환
    - include_negative=True이면 음성 항목도 함께 반환
    """
    hira_df = _normalize_hira_dataframe(load_hira_model_context())

    gender, age_group = _get_client_hira_filter_values()
    display_context_keys = _get_display_context_keys(classification, include_negative)

    if not display_context_keys:
        return pd.DataFrame()

    target = hira_df[
        (hira_df["gender"] == gender)
        & (hira_df["age_group"] == age_group)
        & (hira_df["context_key"].isin(display_context_keys))
    ].copy()

    if target.empty:
        return target

    # 2024 데이터 우선 사용. 2024가 없으면 가장 최신 연도 사용.
    if "year" in target.columns:
        if 2024 in target["year"].dropna().astype(int).tolist():
            target = target[target["year"] == 2024]
        else:
            latest_year = target["year"].dropna().max()
            target = target[target["year"] == latest_year]

    target["질환군"] = target["context_key"].map(
        {key: value["label"] for key, value in HIRA_DISEASE_GROUPS.items()}
    )

    target["판별상태"] = target["context_key"].apply(
        lambda key: "양성" if int(classification.get(key, 0) or 0) == 1 else "음성"
    )

    return target


def build_hira_summary_dataframe(
    classification: Dict[str, int],
    include_negative: bool = False,
) -> pd.DataFrame:
    """
    증상별 입내원 KPI 카드와 HIRA 진료 현황 차트에 사용할 요약 데이터 생성.
    """
    target = build_client_hira_dataframe(
        classification=classification,
        include_negative=include_negative,
    )

    if target.empty:
        return pd.DataFrame()

    agg_dict = {}

    if "patients" in target.columns:
        agg_dict["patients"] = "sum"

    if "visit_days" in target.columns:
        agg_dict["visit_days"] = "sum"

    if "cost_per_patient" in target.columns:
        agg_dict["cost_per_patient"] = "mean"

    if not agg_dict:
        return pd.DataFrame()

    summary_df = (
        target.groupby(["context_key", "질환군", "disease", "판별상태"], as_index=False)
        .agg(agg_dict)
        .sort_values(["판별상태", "patients"], ascending=[False, False])
    )

    if "patients" not in summary_df.columns:
        summary_df["patients"] = 0

    if "visit_days" not in summary_df.columns:
        summary_df["visit_days"] = 0

    summary_df["visit_days_per_patient_calc"] = summary_df.apply(
        lambda row: float(row["visit_days"]) / float(row["patients"])
        if float(row.get("patients", 0) or 0) > 0
        else None,
        axis=1,
    )

    return summary_df


def format_number(value: Any, suffix: str = "") -> str:
    try:
        return f"{float(value):,.0f}{suffix}"
    except Exception:
        return f"계산 불가{suffix}"


def format_float(value: Any, suffix: str = "") -> str:
    try:
        return f"{float(value):,.2f}{suffix}"
    except Exception:
        return f"계산 불가{suffix}"


def format_money(value: Any) -> str:
    try:
        return f"{float(value):,.0f}원"
    except Exception:
        return "계산 불가"
    
def render_top_risk_cards(classification: Dict[str, int], factors: Dict[str, int]):
    """
    KlueBERT 우울/불안/중독 예측값을 나리 UI 톤의 KPI 카드로 표시한다.
    """

    depression_label = int(classification.get("depression", 0) or 0)
    anxiety_label = int(classification.get("anxiety", 0) or 0)
    addiction_label = int(classification.get("addiction", 0) or 0)
    suicidal_score = int(factors.get("suicidal", 0) or 0)
    positive_count = depression_label + anxiety_label + addiction_label
    review_percent = min(100, positive_count * 25 + (25 if suicidal_score > 0 else 0))

    def card_value(label_value: int) -> Tuple[str, str]:
        if label_value == 1:
            return "1", "관찰"
        return "0", "낮음"

    if suicidal_score > 0:
        review_status = "확인 필요"
    elif positive_count > 0:
        review_status = "관찰"
    else:
        review_status = "안정"

    depression_value, depression_status = card_value(depression_label)
    anxiety_value, anxiety_status = card_value(anxiety_label)
    addiction_value, addiction_status = card_value(addiction_label)

    def pill_class(status: str) -> str:
        status_map = {
            "낮음": "risk-pill-low",
            "안정": "risk-pill-stable",
            "관찰": "risk-pill-watch",
            "주의": "risk-pill-caution",
            "확인 필요": "risk-pill-danger",
        }
        return status_map.get(status, "risk-pill-watch")

    st.markdown('<div class="chart-panel-title">KlueBERT 예측 결과</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-panel-desc">우울·불안·중독 0/1 판별 결과와 자해·자살 관련 발화 확인 여부를 표시합니다.</div>',
        unsafe_allow_html=True,
    )

    kpi_values = [
        {
            "title": "우울 위험도",
            "value": depression_value,
            "status": depression_status,
            "icon": "D",
            "icon_class": "blue",
        },
        {
            "title": "불안 위험도",
            "value": anxiety_value,
            "status": anxiety_status,
            "icon": "A",
            "icon_class": "sky",
        },
        {
            "title": "중독 위험도",
            "value": addiction_value,
            "status": addiction_status,
            "icon": "S",
            "icon_class": "orange",
        },
        {
            "title": "검토 필요도",
            "value": f"{review_percent}%",
            "status": review_status,
            "icon": "!",
            "icon_class": "red",
        },
    ]

    cards_html = ""

    for item in kpi_values:
        status_class = pill_class(str(item["status"]))
        cards_html += f"""
<div class="risk-metric-card">
    <div class="risk-metric-head">
        <div class="risk-metric-icon {html_escape(item["icon_class"])}">{html_escape(item["icon"])}</div>
        <div class="risk-metric-label">{html_escape(item["title"])}</div>
    </div>
    <div class="risk-metric-value">{html_escape(item["value"])}</div>
    <div class="risk-metric-status {html_escape(status_class)}">{html_escape(item["status"])}</div>
</div>
"""

    st.markdown(
        f"""<div class="home-summary-grid" style="margin-top:0.55rem;">
{cards_html}
</div>""",
        unsafe_allow_html=True,
    )

    if suicidal_score > 0:
        st.markdown(
            """
            <div style="
                background:#FEF2F2;
                border:1px solid #FECACA;
                border-radius:16px;
                color:#DC2626;
                padding:0.95rem 1.1rem;
                margin-top:0.95rem;
                font-size:0.9rem;
                font-weight:560;
                line-height:1.55;
            ">
                자해/자살 관련 표현이 감지되어 추가 확인이 필요합니다. 상담사가 별도 안전 평가를 수행해 주세요.
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_hira_reference_card(result: Dict[str, Any], key_prefix: str = "ref_hira"):
    """
    HIRA 인구통계 비교 카드.
    주의: 현재 HIRA 데이터에는 인구분모가 없으므로 '진료율'이 아니라 '환자수' 중심으로 표시한다.
    """

    st.markdown("#### HIRA 인구통계 비교")
    st.caption("현재 내담자 조건과 HIRA 데이터 기반 유사 진료 통계를 비교합니다.")

    hira_df = load_hira_model_context()
    auto_context_keys = infer_hira_context_keys(result)

    if not auto_context_keys:
        st.info("현재 분석 결과에서 HIRA와 연결할 우울·불안·수면 항목을 찾지 못했습니다.")
        return []

    client_row = get_client_row()

    raw_gender = str(client_row.get("성별", "")).strip()
    default_gender = "여" if raw_gender.startswith("여") else "남"

    raw_age_band = str(client_row.get("연령대", "")).strip()
    age_band_to_hira = {
        "0대": "0~9세",
        "10대": "10~19세",
        "20대": "20~29세",
        "30대": "30~39세",
        "40대": "40~49세",
        "50대": "50~59세",
        "60대": "60~69세",
        "70대": "70~79세",
        "80대": "80~89세",
        "90대": "90~99세",
    }
    default_age_group = age_band_to_hira.get(raw_age_band, "30~39세")

    age_group_options = [
        "0~9세",
        "10~19세",
        "20~29세",
        "30~39세",
        "40~49세",
        "50~59세",
        "60~69세",
        "70~79세",
        "80~89세",
        "90~99세",
        "100세이상",
    ]

    available_age_groups = [
        age_group
        for age_group in age_group_options
        if age_group in hira_df["age_group"].dropna().unique().tolist()
    ]

    sido_options = sorted(hira_df["sido"].dropna().unique().tolist())

    f1, f2 = st.columns(2)

    with f1:
        selected_age_group = st.selectbox(
            "연령대",
            options=available_age_groups,
            index=available_age_groups.index(default_age_group)
            if default_age_group in available_age_groups
            else 0,
            key=f"{key_prefix}_age_group",
        )

        selected_gender = st.selectbox(
            "성별",
            options=["여", "남"],
            index=0 if default_gender == "여" else 1,
            key=f"{key_prefix}_gender",
        )

    with f2:
        selected_sido = st.selectbox(
            "시도",
            options=sido_options,
            index=sido_options.index("서울") if "서울" in sido_options else 0,
            key=f"{key_prefix}_sido",
        )

        sigungu_options = sorted(
            hira_df[hira_df["sido"] == selected_sido]["sigungu"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_sigungu = st.selectbox(
            "시군구",
            options=sigungu_options,
            index=sigungu_options.index("강남구")
            if selected_sido == "서울" and "강남구" in sigungu_options
            else 0,
            key=f"{key_prefix}_sigungu",
        )

    hira_results = get_hira_context(
        gender=selected_gender,
        sido=selected_sido,
        sigungu=selected_sigungu,
        age_group=selected_age_group,
        context_keys=auto_context_keys,
    )

    rows = []

    for item in hira_results:
        if item.get("matched"):
            rows.append(
                {
                    "질환": item["disease"],
                    "환자수": item["patients"],
                    "입내원일수": item["visit_days"],
                    "1인당 입내원일수": round(float(item["visit_days_per_patient"]), 2)
                    if item.get("visit_days_per_patient") is not None
                    else 0,
                }
            )

    if not rows:
        st.warning("선택 조건에 맞는 HIRA 통계를 찾지 못했습니다.")
        return []

    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df,
        x="질환",
        y="환자수",
        text="환자수",
        height=310,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="환자수",
        xaxis_title="",
    )

    st.plotly_chart(fig, use_container_width=True, key="depression_symptom_evidence_chart")

    st.caption(
        "주의: 현재 HIRA 데이터에는 지역별 인구분모가 포함되어 있지 않으므로 "
        "진료율이 아니라 환자수 기준으로 표시합니다."
    )

    return hira_results


def render_hira_report_sentence_card(
    classification: Dict[str, int],
    include_negative: bool = False,
):
    """
    증상별 입내원정보를 텍스트 대신 카드형 KPI + 조건부 색상으로 표시한다.
    KlueBERT 양성 항목을 우선 표시하고, include_negative=True이면 음성 항목도 함께 표시한다.
    """

    summary_df = build_hira_summary_dataframe(
        classification=classification,
        include_negative=include_negative,
    )

    st.markdown(
        '<div class="chart-panel-title">증상별 입내원정보</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-panel-desc">건강보험심사평가원_시군구별 성별 연령별 주요 정신질환 통계 2024 기준입니다. 기본적으로 KlueBERT 양성 항목만 표시합니다.</div>',
        unsafe_allow_html=True,
    )

    if summary_df.empty:
        st.info("현재 조건에서 표시할 양성 HIRA 입내원정보가 없습니다. 음성 항목도 함께 보기를 선택하면 전체 항목을 확인할 수 있습니다.")
        return

    # 질환군별로 대표 질환 1개씩 우선 표시
    summary_df = (
        summary_df.sort_values(["판별상태", "patients"], ascending=[False, False])
        .groupby("context_key", as_index=False)
        .head(1)
    )

    cols = st.columns(min(len(summary_df), 3), gap="medium")

    max_patients = summary_df["patients"].max() if "patients" in summary_df.columns else 0

    for col, (_, row) in zip(cols, summary_df.iterrows()):
        patients = row.get("patients", 0)
        visit_days = row.get("visit_days", 0)
        visit_days_per_patient = row.get("visit_days_per_patient_calc", None)
        cost_per_patient = row.get("cost_per_patient", None)

        is_positive = row.get("판별상태") == "양성"
        is_high = float(patients or 0) >= float(max_patients or 0)

        is_positive = row.get("판별상태") == "양성"
        context_key = str(row.get("context_key", "")).strip()
        theme = get_context_theme(context_key)

        if is_positive:
            if context_key == "depression":
                bg = "linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 100%)"
                border = "#BFD0FF"
                title_color = "#2F4ED8"
                value_color = "#2F4ED8"
            elif context_key == "anxiety":
                bg = "linear-gradient(180deg, #FBF8FF 0%, #FFFFFF 100%)"
                border = "#E3D6FF"
                title_color = "#7B57D1"
                value_color = "#7B57D1"
            elif context_key == "addiction":
                bg = "linear-gradient(180deg, #F7FDFF 0%, #FFFFFF 100%)"
                border = "#C8F2F7"
                title_color = "#1989A3"
                value_color = "#1989A3"
            else:
                bg = "linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%)"
                border = "#D8E2EF"
                title_color = "#475569"
                value_color = "#334155"
        else:
            bg = "linear-gradient(180deg, #FAFBFD 0%, #FFFFFF 100%)"
            border = "#D8E2EF"
            title_color = "#475569"
            value_color = "#334155"

        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:1px solid {border};
                    border-radius:1.15rem;
                    padding:0.95rem 1rem;
                    min-height:245px;
                    box-shadow:0 8px 20px rgba(15,23,42,0.05);
                ">
                    <div style="font-size:0.78rem; font-weight:560; color:{title_color}; margin-bottom:0.35rem;">
                        {row.get("판별상태", "")} 항목
                    </div>
                    <div style="font-size:1.00rem; font-weight:650; color:#0F172A; margin-bottom:0.25rem;">
                        {row.get("disease", "질환명 없음")}
                    </div>
                    <div style="font-size:0.8rem; color:#64748B; margin-bottom:0.9rem;">
                        질환군: {row.get("질환군", "")}
                    </div>
                    <div style="border-top:1px solid {border}; padding-top:0.75rem;">
                        <div style="font-size:0.76rem; color:#64748B;">환자수</div>
                        <div style="font-size:1.3rem; font-weight:700; color:{value_color};">
                            {format_number(patients, "명")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">입내원일수</div>
                        <div style="font-size:1.05rem; font-weight:650; color:{value_color};">
                            {format_number(visit_days, "일")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">1인당 평균 입내원일수</div>
                        <div style="font-size:1.05rem; font-weight:650; color:{value_color};">
                            {format_float(visit_days_per_patient, "일")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">1인당 평균 요양급여비용</div>
                        <div style="font-size:1.05rem; font-weight:650; color:{value_color};">
                            {format_money(cost_per_patient)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="dashboard-note-line hira-stat-warning-note">
            <span class="dashboard-note-icon">i</span>
            <span>
                주의: 위 통계는 요양기관 소재지 기준의 공공 진료 통계이며, 개별 내담자의 진단, 중증도, 위험도 판단 근거로 사용하지 않습니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_same_group_disease_chart(
    classification: Dict[str, int] = None,
    include_negative: bool = False,
):
    """
    이전 버전의 '같은 성별·연령대 주요 정신질환 진료 현황' 막대차트 복원.
    현재 선택된 내담자와 같은 성별·연령대의 주요 정신질환 환자수를 비교한다.
    """

    st.markdown(
        '<div class="chart-panel-title">같은 성별·연령대 주요 정신질환 진료 현황</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chart-panel-desc">내담자와 같은 성별·연령대의 주요 정신질환 진료 환자수를 비교합니다.</div>',
        unsafe_allow_html=True,
    )

    hira_df = load_hira_model_context().copy()
    client_row = get_client_row()

    for col in ["disease", "gender", "age_group"]:
        if col in hira_df.columns:
            hira_df[col] = hira_df[col].astype(str).str.strip()

    hira_df["gender"] = hira_df["gender"].replace(
        {
            "남자": "남",
            "남성": "남",
            "여자": "여",
            "여성": "여",
        }
    )

    raw_gender = str(client_row.get("성별", "")).strip()
    gender = "여" if raw_gender.startswith("여") else "남"

    age_band_map = {
        "0대": "0~9세",
        "10대": "10~19세",
        "20대": "20~29세",
        "30대": "30~39세",
        "40대": "40~49세",
        "50대": "50~59세",
        "60대": "60~69세",
        "70대": "70~79세",
        "80대": "80~89세",
        "90대": "90~99세",
    }

    age_group = age_band_map.get(str(client_row.get("연령대", "")).strip(), "20~29세")

    target = hira_df[
        (hira_df["gender"] == gender)
        & (hira_df["age_group"] == age_group)
    ].copy()

    disease_order = ["우울증", "불안장애", "불면증", "ADHD", "조울증", "조현병"]
    target = target[target["disease"].isin(disease_order)]

    # 2024 우선. 없으면 가장 최신 연도.
    if "year" in target.columns:
        target["year"] = pd.to_numeric(target["year"], errors="coerce")

        if 2024 in target["year"].dropna().astype(int).tolist():
            target = target[target["year"] == 2024]
        elif not target["year"].dropna().empty:
            latest_year = target["year"].dropna().max()
            target = target[target["year"] == latest_year]

    chart_df = (
        target.groupby("disease", as_index=False)["patients"]
        .sum()
        .sort_values("patients", ascending=False)
    )

    if chart_df.empty:
        st.warning("같은 성별·연령대 진료 통계를 찾지 못했습니다.")
        with st.expander("HIRA 데이터 확인"):
            st.write("선택 성별:", gender)
            st.write("선택 연령대:", age_group)
            st.write("사용 가능한 성별:", sorted(hira_df["gender"].dropna().unique().tolist()))
            st.write("사용 가능한 연령대:", sorted(hira_df["age_group"].dropna().unique().tolist()))
            st.write("사용 가능한 질환:", sorted(hira_df["disease"].dropna().unique().tolist()))
        return

    fig = px.bar(
        chart_df,
        x="disease",
        y="patients",
        text="patients",
        height=370,
        color_discrete_sequence=["#6B8EF7"],
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
        width=0.58,
        textfont=dict(
            size=11,
            color="#64748B",
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard", sans-serif',
        ),
        cliponaxis=False,
    )

    fig.update_layout(
        margin=dict(l=18, r=18, t=24, b=28),
        xaxis_title="질환",
        yaxis_title="환자수",
        bargap=0.34,
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(
            color="#64748B",
            size=11,
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard", sans-serif',
        ),
        xaxis=dict(
            tickfont=dict(size=11, color="#64748B"),
            title_font=dict(size=12, color="#64748B"),
            showgrid=False,
            zeroline=False,
            linecolor="#E2E8F0",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#64748B"),
            title_font=dict(size=12, color="#64748B"),
            showgrid=True,
            gridcolor="#E5EAF2",
            zeroline=False,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="same_group_disease_chart",
    )

    st.caption(
        f"현재 비교 기준: {age_group} {gender}. "
        "환자수는 진료 통계 기준이며, 인구분모가 없으므로 진료율이 아닙니다."
    )

def _get_context_label(context_key: str) -> str:
    label_map = {
        "depression": "우울",
        "anxiety": "불안",
        "addiction": "중독",
    }
    return label_map.get(context_key, context_key)


def _get_context_keywords(context_key: str) -> List[str]:
    keyword_map = {
        "depression": ["우울"],
        "anxiety": ["불안", "공황"],
        "addiction": ["중독", "알코올", "물질", "약물", "도박", "의존"],
    }
    return keyword_map.get(context_key, [])


def _filter_hira_by_context(context_key: str) -> pd.DataFrame:
    """
    HIRA 데이터에서 우울/불안/중독 관련 질환만 필터링한다.
    context_key 컬럼이 있으면 우선 사용하고, 없으면 질환명 keyword로 필터링한다.
    """
    hira_df = load_hira_model_context().copy()

    for col in ["disease", "gender", "age_group", "sido", "sigungu"]:
        if col in hira_df.columns:
            hira_df[col] = hira_df[col].astype(str).str.strip()

    if "gender" in hira_df.columns:
        hira_df["gender"] = hira_df["gender"].replace(
            {
                "남자": "남",
                "남성": "남",
                "여자": "여",
                "여성": "여",
            }
        )

    for col in ["year", "patients", "visit_days", "cost_per_patient"]:
        if col in hira_df.columns:
            hira_df[col] = pd.to_numeric(hira_df[col], errors="coerce")

    # 2024 우선. 없으면 최신 연도.
    if "year" in hira_df.columns:
        if 2024 in hira_df["year"].dropna().astype(int).tolist():
            hira_df = hira_df[hira_df["year"] == 2024]
        elif not hira_df["year"].dropna().empty:
            latest_year = hira_df["year"].dropna().max()
            hira_df = hira_df[hira_df["year"] == latest_year]

    if "context_key" in hira_df.columns:
        context_df = hira_df[hira_df["context_key"].astype(str).str.strip() == context_key].copy()

        if not context_df.empty:
            return context_df

    keywords = _get_context_keywords(context_key)

    if not keywords:
        return pd.DataFrame()

    pattern = "|".join(keywords)
    return hira_df[hira_df["disease"].str.contains(pattern, na=False)].copy()


def format_compact_korean_number(value: Any, suffix: str = "") -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        numeric_value = 0

    numeric_value = float(numeric_value)

    if abs(numeric_value) >= 10000:
        unit = f"만{suffix}" if suffix else "만"
        return f"{numeric_value / 10000:.1f}{unit}"

    if suffix == "원":
        return f"{numeric_value:,.0f}원"

    if suffix in ["명", "일"]:
        if float(numeric_value).is_integer():
            return f"{numeric_value:,.0f}{suffix}"
        return f"{numeric_value:,.2f}{suffix}"

    if suffix:
        return f"{numeric_value:,.2f}{suffix}"

    return f"{numeric_value:,.0f}"


def render_hira_donut_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str,
    key: str,
    top_n: int | None = 10,
    highlight_value: str = "전체",
    show_top_note: bool = True,
):
    """
    지역/성별/연령대 분포 도넛차트 공통 함수.

    동작:
    - 전체 통계에서 범주별 환자수를 집계
    - top_n이 있으면 상위 N개만 표시
    - highlight_value가 있으면 해당 조각에 네온 테두리 강조
    - 성별처럼 top 표기가 불필요한 경우 show_top_note=False로 처리
    """
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        st.info(f"{title} 데이터를 찾지 못했습니다.")
        return

    clean_df = df.copy()
    clean_df[names_col] = clean_df[names_col].astype(str).str.strip()
    clean_df[values_col] = pd.to_numeric(clean_df[values_col], errors="coerce").fillna(0)

    invalid_values = ["", "nan", "None", "none", "undefined", "Undefined", "NaN"]
    clean_df = clean_df[~clean_df[names_col].isin(invalid_values)]

    full_chart_df = (
        clean_df.groupby(names_col, as_index=False)[values_col]
        .sum()
        .sort_values(values_col, ascending=False)
    )

    if top_n is not None:
        chart_df = full_chart_df.head(top_n).copy()
    else:
        chart_df = full_chart_df.copy()

    if chart_df.empty or chart_df[values_col].sum() == 0:
        st.info(f"{title} 데이터를 표시할 수 없습니다.")
        return

    labels = chart_df[names_col].astype(str).tolist()

    highlight_text = str(highlight_value or "전체").strip()

    pull_values = [
        0.055 if highlight_text != "전체" and str(label) == highlight_text else 0
        for label in labels
    ]

    line_widths = [
        3 if highlight_text != "전체" and str(label) == highlight_text else 1
        for label in labels
    ]

    # 선택 조각은 과하게 튀지 않도록 연한 반투명 민트 테두리로 강조
    line_colors = [
        "rgba(37, 99, 235, 0.38)" if highlight_text != "전체" and str(label) == highlight_text else "rgba(255, 255, 255, 0.95)"
        for label in labels
    ]

    fig = px.pie(
        chart_df,
        names=names_col,
        values=values_col,
        hole=0.66,
        height=300,
        color_discrete_sequence=DONUT_PALETTE,
    )

    fig.update_traces(
        name="",
        textposition="inside",
        texttemplate="%{percent}",
        textfont=dict(size=9, color="#334155"),
        pull=pull_values,
        marker=dict(
            line=dict(
                color=line_colors,
                width=line_widths,
            )
        ),
        hovertemplate="<b>%{label}</b><br>환자수: %{value:,}명<br>비중: %{percent}<extra></extra>",
    )

    fig.update_layout(
        title=None,
        title_text="",
        margin=dict(l=0, r=0, t=0, b=66),
        showlegend=True,
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#475569"),
            itemwidth=38,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    if show_top_note and top_n is not None:
        st.markdown(
            f"<div class='hira-donut-title'>{html_escape(title)} <span>(Top {min(top_n, len(chart_df))})</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='hira-donut-title'>{html_escape(title)}</div>", unsafe_allow_html=True)
        
    st.plotly_chart(fig, use_container_width=True, key=key)

def build_context_factor_treemap_df(
    context_key: str,
    factors: Dict[str, int],
) -> pd.DataFrame:
    """
    우울/불안/중독 상세 영역에서 사용할 28요인 기반 트리맵 데이터 생성.

    기존 HIRA 질환 구성 트리맵이 아니라,
    현재 상담 기록에서 추출된 28요인 점수를 기준으로 구성한다.
    """
    factor_df = build_factor_dataframe(factors).copy()

    context_factor_keys = {
        "depression": [
            "depressive_mood",
            "worthlessness",
            "guilt",
            "impaired_cognition",
            "suicidal",
            "anhedonia",
            "psychomotor_changes",
            "weight_appetite",
            "sleep_disturbance",
            "fatigue",
        ],
        "anxiety": [
            "anxiety",
            "loss_of_control",
            "social_avoidance",
            "physical_symptom",
        ],
        "addiction": [
            "craving",
            "withdrawal",
            "tolerance",
            "social_problem",
            "loss_of_control",
            "motivation_for_change",
        ],
    }

    selected_keys = context_factor_keys.get(context_key, [])

    tree_df = factor_df[factor_df["요인코드"].isin(selected_keys)].copy()
    tree_df = tree_df[tree_df["점수"] > 0]

    if tree_df.empty:
        return tree_df

    tree_df["분류"] = _get_context_label(context_key)
    tree_df["값"] = tree_df["점수"].astype(int)

    return tree_df

def render_hira_context_detail_section(
    context_key: str,
    classification: Dict[str, int],
    factors: Dict[str, int],
    expanded_default: bool = True,
):
    """
    우울/불안/중독 상세 영역.

    수정 사항:
    1. 지역/성별/연령대 도넛차트는 HIRA 2024 기준으로 유지
    2. 오른쪽 필터는 데이터를 필터링하는 기능이 아니라, 도넛차트 조각을 하이라이트하는 기능
    3. 트리맵은 HIRA 질환 구성이 아니라 현재 상담 기록의 28요인 점수 기반으로 표시
    """

    label = _get_context_label(context_key)
    is_positive = int(classification.get(context_key, 0) or 0) == 1

    if not is_positive:
        return

    context_df = _filter_hira_by_context(context_key)

    if context_df.empty:
        st.info(f"{label} 관련 HIRA 데이터를 찾지 못했습니다.")
        return
        
    # 지역은 환자수 기준 Top10 안에서 선택
    sido_top10 = (
        context_df.groupby("sido", as_index=False)["patients"]
        .sum()
        .sort_values("patients", ascending=False)
        .head(10)["sido"]
        .astype(str)
        .tolist()
    )

    sido_options = ["전체"] + sido_top10

    # 연령대는 환자수 기준 Top6 안에서 선택
    age_top6 = (
        context_df.groupby("age_group", as_index=False)["patients"]
        .sum()
        .sort_values("patients", ascending=False)
        .head(6)["age_group"]
        .astype(str)
        .tolist()
    )

    age_options = ["전체"] + age_top6

    title_col, filter_sido_col, filter_age_col = st.columns(
        [0.70, 0.15, 0.15],
        gap="small",
        vertical_alignment="bottom",
    )

    with title_col:
        st.markdown(
            f"<div class='hira-detail-title'>{html_escape(label)} 관련 공공통계 상세 정보</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='hira-detail-desc'>{html_escape(label)} 관련 질환의 HIRA 2024 분포와 현재 상담 기록의 28요인 구성을 함께 확인합니다.</div>",
            unsafe_allow_html=True,
        )

    with filter_sido_col:
        st.markdown('<span class="hira-highlight-filter-marker"></span>', unsafe_allow_html=True)
        selected_sido = st.selectbox(
            "지역",
            options=sido_options,
            key=f"{context_key}_detail_sido_highlight",
        )

    with filter_age_col:
        st.markdown('<span class="hira-highlight-filter-marker"></span>', unsafe_allow_html=True)
        selected_age = st.selectbox(
            "연령",
            options=age_options,
            key=f"{context_key}_detail_age_highlight",
        )

    client_gender, client_age_group = _get_client_hira_filter_values()

    st.markdown("<div style='height:0.15rem;'></div>", unsafe_allow_html=True)

    # =====================================================
    # 2행: 도넛 차트 3개를 전체 폭 기준으로 균등 배치
    # =====================================================
    d1, d2, d3 = st.columns(3, gap="medium")

    with d1:
        render_hira_donut_chart(
            context_df,
            names_col="sido",
            values_col="patients",
            title="지역",
            key=f"{context_key}_sido_donut",
            top_n=10,
            highlight_value=selected_sido,
            show_top_note=True,
        )

    with d2:
        render_hira_donut_chart(
            context_df,
            names_col="gender",
            values_col="patients",
            title="성별",
            key=f"{context_key}_gender_donut",
            top_n=None,
            highlight_value="전체",
            show_top_note=False,
        )

    with d3:
        render_hira_donut_chart(
            context_df,
            names_col="age_group",
            values_col="patients",
            title="연령대",
            key=f"{context_key}_age_donut",
            top_n=6,
            highlight_value=selected_age,
            show_top_note=True,
        )

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # =====================================================
    # 2행: 28요인 트리맵 + 공공통계
    # =====================================================
    tree_col, stat_col = st.columns([0.57, 0.43], gap="medium")

    with tree_col:
        with st.container(border=True):
            st.markdown(f"<div class='chart-panel-title'>{html_escape(label)} 관련 28요인 구성</div>", unsafe_allow_html=True)
            st.caption("현재 상담 기록에서 추출된 28요인 점수를 기준으로 표시합니다.")

            tree_df = build_context_factor_treemap_df(
                context_key=context_key,
                factors=factors,
            )

            if tree_df.empty:
                st.info(f"현재 상담 기록에서 0점보다 큰 {label} 관련 28요인이 없습니다.")
            else:
                theme = get_context_theme(context_key)
                treemap_palette = {
                    "depression": {
                        "root": "#EFF6FF",
                        "category": "#DBEAFE",
                        "leaf": {
                            1: "#EAF2FF",
                            2: "#BFDBFE",
                            3: "#93C5FD",
                        },
                    },
                    "anxiety": {
                        "root": "#F5F3FF",
                        "category": "#EDE9FE",
                        "leaf": {
                            1: "#F5F3FF",
                            2: "#DDD6FE",
                            3: "#C4B5FD",
                        },
                    },
                    "addiction": {
                        "root": "#ECFEFF",
                        "category": "#CCFBF1",
                        "leaf": {
                            1: "#ECFEFF",
                            2: "#99F6E4",
                            3: "#5EEAD4",
                        },
                    },
                }.get(context_key, {})

                root_label = _get_context_label(context_key)

                labels = [root_label]
                ids = [root_label]
                parents = [""]
                values = [int(tree_df["값"].sum())]
                colors = [treemap_palette.get("root", theme["soft_bg"])]

                category_color = treemap_palette.get("category", "#E2E8F0")

                leaf_color_map = treemap_palette.get(
                    "leaf",
                    {
                        1: "#F1F5F9",
                        2: "#CBD5E1",
                        3: "#94A3B8",
                    },
                )

                for category_name, category_df in tree_df.groupby("카테고리"):
                    category_id = f"{root_label}/{category_name}"

                    labels.append(category_name)
                    ids.append(category_id)
                    parents.append(root_label)
                    values.append(int(category_df["값"].sum()))
                    colors.append(category_color)

                    for _, factor_row in category_df.iterrows():
                        factor_label = str(factor_row["요인"])
                        factor_value = int(factor_row["값"])
                        factor_id = f"{category_id}/{factor_label}"

                        labels.append(factor_label)
                        ids.append(factor_id)
                        parents.append(category_id)
                        values.append(factor_value)
                        colors.append(leaf_color_map.get(factor_value, "#DCE6F6"))

                fig_tree = go.Figure(
                    go.Treemap(
                        labels=labels,
                        ids=ids,
                        parents=parents,
                        values=values,
                        branchvalues="total",
                        marker=dict(
                            colors=colors,
                            line=dict(color="rgba(255,255,255,0.96)", width=2),
                        ),
                        texttemplate="%{label}<br>%{value}점",
                        textfont=dict(size=13, color="#0F172A"),
                        hovertemplate="<b>%{label}</b><br>점수: %{value}점<extra></extra>",
                        tiling=dict(pad=4),
                        pathbar=dict(visible=False),
                    )
                )

                fig_tree.update_layout(
                    height=355,
                    margin=dict(l=6, r=6, t=8, b=8),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    uniformtext_minsize=11,
                    uniformtext_mode="hide",
                )

                st.plotly_chart(
                    fig_tree,
                    use_container_width=True,
                    key=f"{context_key}_factor_treemap",
                )

    with stat_col:
        with st.container(border=True):
            st.markdown("<div class='chart-panel-title'>공공통계</div>", unsafe_allow_html=True)

            total_patients = context_df["patients"].sum() if "patients" in context_df.columns else 0
            total_visit_days = context_df["visit_days"].sum() if "visit_days" in context_df.columns else 0

            if total_patients > 0:
                avg_visit_days = total_visit_days / total_patients
            else:
                avg_visit_days = 0

            if "cost_per_patient" in context_df.columns:
                avg_cost = context_df["cost_per_patient"].mean()
            else:
                avg_cost = 0

            kpi_items = [
                ("환자수", format_compact_korean_number(total_patients, "명")),
                ("입내원일수", format_compact_korean_number(total_visit_days, "일")),
                ("1인당 입내원일수", format_compact_korean_number(avg_visit_days, "일")),
                ("1인당 요양급여비용", format_compact_korean_number(avg_cost, "원")),
            ]

            kpi_html = "".join(
                (
                    '<div class="hira-kpi-card">'
                    f'<div class="hira-kpi-label">{html_escape(label_text)}</div>'
                    f'<div class="hira-kpi-value">{html_escape(value_text)}</div>'
                    "</div>"
                )
                for label_text, value_text in kpi_items
            )
            st.markdown(f"<div class='hira-kpi-grid'>{kpi_html}</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="hira-kpi-note">
                    <span class="hira-kpi-note-icon">i</span>
                    <span>
                        위 공공통계는 현재 선택된 내담자의 성별·연령대 조건을 반영한 HIRA 2024 기준 합계입니다.
                        지역·연령대 선택은 도넛 차트의 하이라이트 용도이며, 개별 내담자의 진단·중증도·위험도 판단 근거가 아닙니다.
                    </span>
                </div>
                <div style="height:0.65rem;"></div>
                """,
                unsafe_allow_html=True,
            )


    # =====================================================
    # 3행: 상담자 해석 도우미
    # =====================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="hira-interpret-title-row">
                <span class="hira-interpret-title-icon">i</span>
                <span class="hira-interpret-title">상담자 해석 도우미</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        highlight_parts = []

        if selected_sido != "전체":
            highlight_parts.append(f"{selected_sido} 지역")

        if selected_age != "전체":
            highlight_parts.append(f"{selected_age}")

        if highlight_parts:
            highlight_text = " · ".join(highlight_parts)
        else:
            highlight_text = "전체 분포"

        st.markdown(
            f"""<div class="hira-interpret-card">
<p class="hira-interpret-lead">
현재 KlueBERT 판별에서 <strong>{html_escape(label)} 관련 항목이 양성</strong>으로 표시되었습니다.
</p>

<p>
도넛차트에서는 <strong>{html_escape(highlight_text)}</strong>가 강조되어 있습니다.
이 강조 기능은 상담자가 HIRA 공공통계에서 특정 지역·연령대의 상대적 위치를 빠르게 확인하기 위한 시각적 보조 기능입니다.
</p>

<p>
트리맵은 HIRA 질환 구성이 아니라,
<strong>현재 상담 기록에서 추출된 {html_escape(label)} 관련 28요인 점수 구성</strong>입니다.
따라서 공공통계는 인구통계적 참고자료로, 28요인 트리맵은 현재 상담 발화의 내용 기반 참고자료로 분리해서 해석해야 합니다.
</p>

<p class="hira-interpret-final">
최종 판단은 상담자가 실제 발화 맥락, 증상 지속 기간, 기능 저하 정도, 보호요인, 위험요인을 함께 검토해 수행해야 합니다.
</p>
</div>""",
            unsafe_allow_html=True,
        )

def render_depression_dashboard_section(classification, factors, key_prefix="depression"):
    if int(classification.get("depression", 0)) != 1:
        return

    st.markdown("### 우울 판정 설명 대시보드")
    st.caption("우울 관련 판정이 나온 경우에만 표시되는 영역입니다.")

    factor_df = build_factor_dataframe(factors)

    depression_keys = [
        "depressive_mood",
        "sleep_disturbance",
        "fatigue",
        "anhedonia",
        "worthlessness",
        "guilt",
        "impaired_cognition",
        "weight_appetite",
    ]

    symptom_df = factor_df[factor_df["요인코드"].isin(depression_keys)].copy()
    symptom_df = symptom_df.sort_values("점수", ascending=True)

    # 1행: 판정 근거 + 국가 KPI
    c1, c2 = st.columns([0.58, 0.42], gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("#### 우울 관련 핵심 증상 근거")
            fig = px.bar(
                symptom_df,
                x="점수",
                y="요인",
                orientation="h",
                range_x=[0, 3],
                height=360,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="depression_symptom_profile_chart",
            )

    with c2:
        with st.container(border=True):
            st.markdown("#### 우울 관련 공공통계 요약")
            k1, k2 = st.columns(2)
            k1.metric("우울감 경험률", "11.6%")
            k2.metric("우울장애 평생유병률", "7.7%")
            st.caption(
                "공공 정신건강 통계 기반 참고 지표입니다. "
                "개별 내담자의 진단 근거가 아니라 설명용 배경지표입니다."
            )

    # 2행: 현재 회기 요인 비교
    with st.container(border=True):
        st.markdown("#### 현재 회기 우울 요인 프로파일")
        profile_df = symptom_df.sort_values("점수", ascending=False)

        fig_profile = px.bar(
            profile_df,
            x="요인",
            y="점수",
            text="점수",
            range_y=[0, 3],
            height=340,
        )
        fig_profile.update_traces(textposition="outside")
        fig_profile.update_layout(
            xaxis_title="우울 관련 요인",
            yaxis_title="점수",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(
            fig_profile,
            use_container_width=True,
            key="depression_factor_profile_chart",
        )

    # 4행: 상담자 설명 문장
    with st.container(border=True):
        st.markdown("#### 상담자 해석 가이드")
        top_symptoms = symptom_df[symptom_df["점수"] > 0].sort_values("점수", ascending=False)

        if top_symptoms.empty:
            st.info("현재 0보다 큰 우울 관련 요인이 없습니다.")
        else:
            top_text = ", ".join(
                [
                    f"{row['요인']}({int(row['점수'])}점)"
                    for _, row in top_symptoms.head(4).iterrows()
                ]
            )

            st.markdown(
                f"""
                현재 상담 기록에서는 **{top_text}**이 주요하게 표시되었습니다.

                이 결과는 우울 관련 호소가 어떤 발화 요인에서 나타났는지 설명하기 위한 참고 정보입니다.
                상담자는 같은 성별·연령대·지역의 우울증 진료 통계와 함께 보면서, 다음 회기에서 증상의 지속 기간,
                기능 저하 정도, 수면·피로 문제의 변화 여부를 추가 확인할 수 있습니다.
                """
            )


def render_anxiety_dashboard_section(classification, factors, key_prefix="anxiety"):
    if int(classification.get("anxiety", 0)) != 1:
        return

    st.markdown("### 불안 판정 설명 대시보드")
    st.caption("불안 관련 판정이 나온 경우에만 표시되는 영역입니다.")

    factor_df = build_factor_dataframe(factors)

    anxiety_keys = [
        "anxiety",
        "loss_of_control",
        "social_avoidance",
        "physical_symptom",
    ]

    anxiety_df = factor_df[factor_df["요인코드"].isin(anxiety_keys)].copy()
    anxiety_df = anxiety_df.sort_values("점수", ascending=True)

    c1, c2 = st.columns([0.5, 0.5], gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("#### 불안 관련 핵심 위험요인")
            fig = px.bar(
                anxiety_df,
                x="점수",
                y="요인",
                orientation="h",
                range_x=[0, 3],
                height=340,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{key_prefix}_risk_factor_chart",
            )

    with c2:
        with st.container(border=True):
            st.markdown("#### 상담자 해석 가이드")
            st.markdown(
                """
                현재 상담 기록에서 불안 관련 라벨이 양성으로 표시되었습니다.

                이 결과는 불안 관련 호소가 상담 발화 안에서 탐지되었음을 의미하며,
                임상 진단을 의미하지 않습니다. 상담자는 다음 회기에서 아래 항목을 우선 확인할 수 있습니다.

                - 불안이 강해지는 구체적 상황
                - 신체 반응 여부
                - 회피 행동 여부
                - 통제감 상실 경험
                - 일상 기능 저하 정도
                """
            )

def render_addiction_dashboard_section(classification, factors, key_prefix="addiction"):
    if int(classification.get("addiction", 0)) != 1:
        return

    st.markdown("### 중독 판정 설명 대시보드")
    st.caption("중독 관련 판정이 나온 경우에만 표시되는 영역입니다.")

    factor_df = build_factor_dataframe(factors)

    addiction_keys = [
        "loss_of_control",
        "craving",
        "withdrawal",
        "tolerance",
        "social_problem",
        "motivation_for_change",
    ]

    addiction_df = factor_df[factor_df["요인코드"].isin(addiction_keys)].copy()
    addiction_df = addiction_df.sort_values("점수", ascending=True)

    c1, c2 = st.columns([0.5, 0.5], gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("#### 중독 관련 취약요인")
            fig = px.bar(
                addiction_df,
                x="점수",
                y="요인",
                orientation="h",
                range_x=[0, 3],
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_vulnerability_chart")

    with c2:
        with st.container(border=True):
            st.markdown("#### 중독 공공통계 요약")
            st.metric("알코올 사용장애 평생유병률", "11.6%")
            st.metric("알코올 사용장애 1년 유병률", "2.6%")
            st.caption("중독 주요 지표 모음집 기반 참고 통계입니다.")

    with st.container(border=True):
        st.markdown("#### 다음 회기 확인 포인트")
        st.markdown(
            """
            - 사용 빈도와 사용량
            - 조절 실패 경험
            - 갈망 또는 금단 경험
            - 일상 기능 손상 여부
            - 가족·사회적 문제 여부
            - 변화 동기와 서비스 연결 가능성
            """
        )

def _score_from_session_row(row: pd.Series, key: str, default: int = 0) -> int:
    """
    SESSIONS 행에서 depression/anxiety/addiction 같은 점수 컬럼을 안전하게 읽는다.
    데이터셋 원본 라벨은 0~2 또는 0~3일 수 있으므로 0~3 범위로 제한한다.
    """
    try:
        value = int(float(row.get(key, default) or 0))
        return max(0, min(3, value))
    except Exception:
        return default


def _keyword_factor_score(script: str, keywords: List[str], positive_score: int = 3) -> int:
    """
    session_scores.csv가 없을 때 회기별 추이용 보조 점수.
    실제 진단 점수가 아니라 차트 표시용 보조값이다.
    """
    text = str(script or "").lower()
    return positive_score if any(keyword in text for keyword in keywords) else 0


def build_session_trend_dataframe(
    selected_classification: Dict[str, int],
    selected_factors: Dict[str, int],
) -> pd.DataFrame:
    """
    현재 선택된 내담자의 전체 회기 목록을 기준으로 회기별 추이 데이터를 만든다.
    - 선택 회기는 현재 분석 결과를 우선 사용
    - 나머지 회기는 SESSIONS의 라벨 컬럼과 상담 스크립트 키워드 기반 보조 점수를 사용
    """
    client_sessions = get_client_sessions_sorted()

    rows = []

    for _, row in client_sessions.iterrows():
        session_name = str(row.get("회기", "회기미상"))
        is_selected = str(session_name) == str(st.session_state.selected_session)

        dialogue = SESSION_DIALOGUES.get(
            (st.session_state.selected_client, session_name),
            pd.DataFrame(),
        )
        script = build_dialogue_text(dialogue) if not dialogue.empty else str(row.get("script", "") or "")

        if is_selected:
            depression_score = int(selected_classification.get("depression", 0) or 0) * 3
            anxiety_score = int(selected_classification.get("anxiety", 0) or 0) * 3
            addiction_score = int(selected_classification.get("addiction", 0) or 0) * 3
            sleep_score = int(selected_factors.get("sleep_disturbance", 0) or 0)
            fatigue_score = int(selected_factors.get("fatigue", 0) or 0)
        else:
            depression_score = _score_from_session_row(row, "depression", 0)
            anxiety_score = _score_from_session_row(row, "anxiety", 0)
            addiction_score = _score_from_session_row(row, "addiction", 0)

            sleep_score = _keyword_factor_score(
                script,
                ["잠", "수면", "불면", "자주 깨", "못 자"],
                positive_score=3,
            )
            fatigue_score = _keyword_factor_score(
                script,
                ["피곤", "피로", "무기력", "힘들", "기운"],
                positive_score=3,
            )

        rows.append(
            {
                "회기": session_name,
                "상담일": row.get("상담일", ""),
                "우울": depression_score,
                "불안": anxiety_score,
                "중독": addiction_score,
                "수면문제": sleep_score,
                "피로감": fatigue_score,
                "선택회기": is_selected,
            }
        )

    return pd.DataFrame(rows)


def render_session_area_trend(classification: Dict[str, int], factors: Dict[str, int]):
    """
    선택 내담자의 전체 회기별 추이 변화 라인 차트.
    선택된 회기는 세로 음영과 마커로 강조한다.
    """
    trend = build_session_trend_dataframe(classification, factors)

    if trend.empty:
        st.info("회기별 추이를 표시할 상담 회기가 없습니다.")
        return

    color_map = {
        "우울": SERIES_COLOR_MAP["우울"],
        "불안": SERIES_COLOR_MAP["불안"],
        "중독": SERIES_COLOR_MAP["중독"],
        "수면문제": SERIES_COLOR_MAP["수면문제"],
        "피로감": SERIES_COLOR_MAP["피로감"],
    }

    fig = go.Figure()

    x_values = trend["회기"].astype(str).tolist()
    series_names = ["우울", "불안", "중독", "수면문제", "피로감"]

    for series in series_names:
        y_values = trend[series].astype(float).tolist()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=series,
                line=dict(
                    color=color_map[series],
                    width=3,
                    shape="linear",
                ),
                marker=dict(
                    size=8,
                    color=color_map[series],
                    opacity=0.78,
                    line=dict(width=2, color="white"),
                ),
                opacity=0.72,
                fill="tozeroy" if series in ["우울", "불안"] else None,
                fillcolor="rgba(37, 99, 235, 0.08)" if series == "우울" else (
                    "rgba(96, 165, 250, 0.08)" if series == "불안" else None
                ),
                hovertemplate=f"<b>{series}</b><br>회기: %{{x}}<br>점수: %{{y:.1f}}<extra></extra>",
            )
        )

    selected_session = str(st.session_state.selected_session)

    if selected_session in x_values:
        selected_index = x_values.index(selected_session)

        fig.add_vrect(
            x0=selected_index - 0.35,
            x1=selected_index + 0.35,
            fillcolor="rgba(37, 99, 235, 0.10)",
            line_width=0,
            layer="below",
        )

        fig.add_annotation(
            x=selected_session,
            y=3.15,
            text="선택 회기",
            showarrow=False,
            font=dict(size=12, color="#1D4ED8"),
            bgcolor="rgba(239, 246, 255, 0.95)",
            bordercolor="#93C5FD",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        height=390,
        margin=dict(l=20, r=140, t=20, b=20),
        xaxis_title="",
        yaxis_title="점수 (0-3)",
        yaxis=dict(range=[0, 3.2], showgrid=True, gridcolor="#E5EAF2", zeroline=False),
        xaxis=dict(showgrid=True, gridcolor="#E5EAF2", zeroline=False, type="category"),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            title=None,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=12),
    )

    st.plotly_chart(fig, use_container_width=True, key="session_line_trend_chart")

    st.caption(
        f"현재 선택 회기: {selected_session}. "
        "선택 회기는 차트에서 파란 음영으로 강조됩니다."
    )

def render_factor_detail_table(factor_df: pd.DataFrame):
    """
    사진 배치의 '28요인 세부내용' 표.
    기본은 0점 요인을 숨기고, 체크 시 전체 요인을 표시한다.
    """

    show_all = st.checkbox("0점 요인 포함", value=False, key="show_all_factor_detail")

    table_df = factor_df.copy()

    if not show_all:
        table_df = table_df[table_df["점수"] > 0]

    table_df = table_df.sort_values("점수", ascending=False)

    display_df = table_df.rename(
        columns={
            "카테고리": "내용",
            "요인": "요인",
            "점수": "점수",
        }
    )[["내용", "요인", "점수"]]

    if display_df.empty:
        st.info("현재 0점보다 큰 28요인 항목이 없습니다.")
    else:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

def render_factor_frequency_card(factors: Dict[str, int]):
    """
    4범주 빈도 차트.
    증상, 위험, 개선, 개입 영역의 점수 합계를 도넛차트로 보여준다.
    """

    st.markdown("#### 4범주 빈도 차트")
    st.caption("증상, 위험, 개선, 개입 영역의 점수 합계를 구성비로 보여줍니다.")

    groups = get_factor_groups(factors)

    group_df = pd.DataFrame(
        {
            "범주": list(groups.keys()),
            "빈도": list(groups.values()),
        }
    ).sort_values("빈도", ascending=False)

    if group_df["빈도"].sum() == 0:
        st.info("현재 표시할 4범주 점수가 없습니다.")
        return

    fig = px.pie(
        group_df,
        names="범주",
        values="빈도",
        hole=0.55,
        height=330,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key="factor_frequency_chart")


def render_ai_summary_cards(factors: Dict[str, int]):
    """
    민정 요약 추출 기준을 유지하면서 나리 UI 카드로 표시한다.
    """

    symptom_items = []
    risk_items = []
    improvement_items = []
    intervention_items = []

    if factors.get("sleep_disturbance", 0) > 0:
        symptom_items.append("수면 문제")
    if factors.get("fatigue", 0) > 0:
        symptom_items.append("피로감")
    if factors.get("depressive_mood", 0) > 0:
        symptom_items.append("우울감")
    if factors.get("anxiety", 0) > 0:
        symptom_items.append("불안감")

    if factors.get("social_problem", 0) > 0:
        risk_items.append("업무/사회적 스트레스")
    if factors.get("loss_of_control", 0) > 0:
        risk_items.append("통제감 저하")
    if factors.get("social_avoidance", 0) > 0:
        risk_items.append("사회적 회피")
    if factors.get("suicidal", 0) > 0:
        risk_items.append("자해·자살 관련 발화 확인 필요")

    if factors.get("motivation_for_change", 0) > 0:
        improvement_items.append("변화 동기")
    else:
        improvement_items.append("상담 참여 여부 확인 필요")

    if factors.get("sympathy_support", 0) > 0:
        intervention_items.append("공감 및 지지")
    if factors.get("cognitive_restructuring", 0) > 0:
        intervention_items.append("인지 재구성")
    if factors.get("goal_setting", 0) > 0:
        intervention_items.append("목표 설정")
    if factors.get("coping_skill_training", 0) > 0:
        intervention_items.append("대처기술 훈련")

    cards = [
        {
            "key": "symptom",
            "title": "주요 증상",
            "icon": "S",
            "items": symptom_items or ["뚜렷한 주요 증상 없음"],
            "note": "상담 기록에서 상대적으로 높게 표시된 증상 요인을 정리합니다.",
        },
        {
            "key": "risk",
            "title": "위험 요인",
            "icon": "!",
            "items": risk_items or ["추가 확인 필요"],
            "note": "추가 확인이 필요한 정서·행동 신호를 상담사가 검토합니다.",
        },
        {
            "key": "improve",
            "title": "개선 요인",
            "icon": "+",
            "items": improvement_items,
            "note": "내담자의 보호요인과 변화 자원을 함께 확인합니다.",
        },
        {
            "key": "intervention",
            "title": "개입 요인",
            "icon": "I",
            "items": intervention_items or ["개입 요인 확인 필요"],
            "note": "상담사 개입 방향을 다음 회기 계획과 연결합니다.",
        },
    ]

    cards_html = ""

    for card in cards:
        items_html = "".join(f"<li>{html_escape(str(item))}</li>" for item in card["items"][:4])
        cards_html += f"""
<div class="ai-summary-card {html_escape(card["key"])}">
    <div class="ai-summary-head">
        <div class="ai-summary-icon {html_escape(card["key"])}">{html_escape(card["icon"])}</div>
        <div class="ai-summary-card-title">{html_escape(card["title"])}</div>
    </div>
    <ul class="ai-summary-list">{items_html}</ul>
    <div class="ai-summary-note {html_escape(card["key"])}">{html_escape(card["note"])}</div>
</div>
"""

    st.markdown(
        f"""<div class="ai-summary-section">
<div class="ai-summary-title">AI 분석 요약</div>
<div class="ai-summary-grid">
{cards_html}
</div>
<div class="ai-summary-footnote">
    <span class="dashboard-note-icon">i</span>
    <span>
        위 요약은 AI 분석 결과의 보조 참고 자료이며, 최종 판단과 상담 방향은 상담사의 전문적 판단을 우선합니다.
    </span>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def normalize_factor_category(category: Any) -> str:
    category_text = str(category or "")

    if category_text.startswith("우울"):
        return "우울"

    return category_text


def factor_category_filter(factor_df: pd.DataFrame, selected_categories: List[str]) -> pd.DataFrame:
    if not selected_categories:
        return factor_df.iloc[0:0]

    chart_df = factor_df.copy()
    chart_df["표시 카테고리"] = chart_df["카테고리"].apply(normalize_factor_category)
    return chart_df[chart_df["표시 카테고리"].isin(selected_categories)]


def render_factor_top10_chart(factor_df: pd.DataFrame):
    chart_df = factor_df.copy()
    chart_df["표시 카테고리"] = chart_df["카테고리"].apply(normalize_factor_category)

    category_options = ["우울", "불안", "중독", "상담사 개입", "변화/기타"]

    if "factor_selected_categories" not in st.session_state:
        st.session_state.factor_selected_categories = ["우울"]

    st.markdown('<span class="factor-category-marker"></span>', unsafe_allow_html=True)
    selected_categories = st.pills(
        "요인 카테고리",
        options=category_options,
        selection_mode="multi",
        default=st.session_state.factor_selected_categories,
        key="factor_category_pills_integrated",
        label_visibility="collapsed",
    )

    st.session_state.factor_selected_categories = selected_categories or []

    filtered_factor_df = factor_category_filter(chart_df, st.session_state.factor_selected_categories)
    filtered_factor_df = filtered_factor_df[filtered_factor_df["점수"] > 0]

    if filtered_factor_df.empty:
        st.info("선택한 카테고리에 표시할 0점 초과 요인이 없습니다.")
    else:
        selected_factor_df = (
            filtered_factor_df.sort_values(["점수", "요인"], ascending=[False, True])
            .head(10)
            .sort_values("점수", ascending=True)
        )

        fig_factor = px.bar(
            selected_factor_df,
            x="점수",
            y="요인",
            color="표시 카테고리",
            orientation="h",
            range_x=[0, 3],
            color_discrete_map=FACTOR_CATEGORY_COLOR_MAP,
            height=440,
        )
        fig_factor.update_layout(
            margin=dict(l=10, r=140, t=20, b=20),
            xaxis_title="점수",
            yaxis_title="",
            bargap=0.58,
            legend_title_text="카테고리",
        )
        fig_factor.update_traces(marker_line_width=0)
        style_dashboard_chart(fig_factor)
        st.plotly_chart(fig_factor, use_container_width=True, key="factor_top10_category_chart")

def render_all_factor_expander(factor_df: pd.DataFrame):
    chart_df = factor_df.copy()
    chart_df["표시 카테고리"] = chart_df["카테고리"].apply(normalize_factor_category)

    with st.expander("전체 28요인 보기", expanded=False):
        all_factor_chart_df = chart_df.sort_values(["점수", "요인"], ascending=[False, True])
        all_factor_chart_df = all_factor_chart_df.sort_values("점수", ascending=True)

        fig_all_factor = px.bar(
            all_factor_chart_df,
            x="점수",
            y="요인",
            color="표시 카테고리",
            orientation="h",
            range_x=[0, 3],
            color_discrete_map=FACTOR_CATEGORY_COLOR_MAP,
            height=760,
        )

        fig_all_factor.update_layout(
            margin=dict(l=10, r=140, t=20, b=20),
            xaxis_title="점수",
            yaxis_title="",
            bargap=0.42,
            showlegend=True,
            legend_title_text="카테고리",
        )
        fig_all_factor.update_traces(marker_line_width=0)
        style_dashboard_chart(fig_all_factor)
        st.plotly_chart(fig_all_factor, use_container_width=True, key="all_factor_28_chart")

def render_dashboard():
    title_col, chat_col = st.columns([0.94, 0.06], vertical_alignment="top")

    with title_col:
        st.markdown('<div class="patient-title dashboard-title">분석 대시보드</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="patient-desc">AI 모델 출력 결과와 HIRA 공공 진료 통계를 기반으로 상담 기록의 주요 요인과 참고 통계를 시각화합니다.</div>',
            unsafe_allow_html=True,
        )

    with chat_col:
        st.markdown('<span class="chatbot-nav-button-marker"></span>', unsafe_allow_html=True)
        chatbot_button_label = f"![chatbot]({get_png_data_uri(CHATBOT_ICON_PATH)})"

        if st.button(
            chatbot_button_label,
            key="dashboard_chatbot_fab",
            use_container_width=False,
            help="챗봇으로 이동",
        ):
            go_page("챗봇")
            st.rerun()

    client_sessions = get_client_sessions_sorted()

    if client_sessions.empty:
        st.info("분석할 상담 회기가 없습니다. 먼저 상담내역 기록·추가에서 회기를 추가해 주세요.")
        return

    session_options = client_sessions["회기"].astype(str).tolist()

    if str(st.session_state.selected_session) not in session_options:
        select_session(session_options[0])

    st.markdown('<div style="height:0.15rem;"></div>', unsafe_allow_html=True)
    st.markdown('<span class="dashboard-session-select-marker"></span>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "회기 선택",
        options=session_options,
        index=session_options.index(str(st.session_state.selected_session)),
        format_func=lambda session_name: (
            f"{session_name} · "
            f"{client_sessions[client_sessions['회기'].astype(str) == str(session_name)].iloc[0].get('상담일', '날짜 미상')}"
        ),
        key="dashboard_session_selector",
    )

    if selected_label != st.session_state.selected_session:
        select_session(selected_label)
        analyze_current_session_if_needed()
        st.rerun()

    analyze_current_session_if_needed()
    result = st.session_state.analysis_result

    if result is None:
        st.info("선택한 회기의 상담 내용이 없어 분석 결과를 만들 수 없습니다.")
        return

    classification = result["classification"]
    factors = result["factors"]
    factor_df = build_factor_dataframe(factors)

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)
    render_top_risk_cards(classification, factors)

    st.markdown('<div style="height:0.35rem;"></div>', unsafe_allow_html=True)
    render_ai_summary_cards(factors)

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="chart-panel-title">회기별 추이 차트</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-panel-desc">상담 회기별 주요 지표 변화를 추적합니다.</div>',
            unsafe_allow_html=True,
        )
        render_session_area_trend(classification, factors)
        st.caption("실제 상담 기록을 기반으로 회기별 점수(0~3)가 표시됩니다.")

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="chart-panel-title">세부 요인 막대 그래프</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-panel-desc">28개 요인을 카테고리별로 탐색합니다. 선택한 카테고리에서 점수가 높은 상위 10개 요인을 표시합니다.</div>',
            unsafe_allow_html=True,
        )
        render_factor_top10_chart(factor_df)
        render_all_factor_expander(factor_df)

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)
    hira_left_col, hira_right_col = st.columns([0.44, 0.56], gap="medium")

    with hira_left_col:
        with st.container(border=True):
            try:
                render_same_group_disease_chart(classification=classification)
            except FileNotFoundError:
                st.markdown("#### 같은 성별·연령대 주요 정신질환 진료 현황")
                st.caption("내담자와 같은 성별·연령대의 주요 정신질환 진료 환자수를 비교합니다.")
                st.warning("HIRA 통계 CSV 파일을 찾지 못했습니다. data/processed/hira/hira_model_context.csv를 추가하면 차트가 표시됩니다.")

    with hira_right_col:
        with st.container(border=True):
            try:
                render_hira_report_sentence_card(
                    classification=classification,
                    include_negative=False,
                )
            except FileNotFoundError:
                st.markdown("### 증상별 입내원정보")
                st.caption("건강보험심사평가원_시군구별 성별 연령별 주요 정신질환 통계 2024 기준입니다.")
                st.warning("HIRA 통계 CSV 파일을 찾지 못했습니다. data/processed/hira/hira_model_context.csv를 추가하면 증상별 입내원정보가 표시됩니다.")

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)
    with st.expander("우울/불안/중독 상세 공공통계 대시보드 보기", expanded=False):
        positive_detail_count = 0

        for context_key in ["depression", "anxiety", "addiction"]:
            if int(classification.get(context_key, 0) or 0) == 1:
                positive_detail_count += 1
                try:
                    render_hira_context_detail_section(
                        context_key=context_key,
                        classification=classification,
                        factors=factors,
                    )
                except FileNotFoundError:
                    st.warning("HIRA 통계 CSV 파일을 찾지 못했습니다. data/processed/hira/hira_model_context.csv를 추가하면 상세 공공통계가 표시됩니다.")
                st.divider()

        if positive_detail_count == 0:
            st.info("현재 KlueBERT 양성 항목이 없어 우울/불안/중독 상세 공공통계 대시보드를 표시하지 않습니다.")

            
# =========================================================
# 14. AI 보고서
# =========================================================
def render_report():
    st.markdown('<div class="patient-title">AI 보고서</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI가 생성한 상담 보고서를 확인하고, 필요한 내용을 수정한 뒤 저장·다운로드할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.analysis_result

    if "report_preview_mode" not in st.session_state:
        st.session_state.report_preview_mode = False

    if result is None:
        st.info("아직 분석 결과가 없습니다. 먼저 상담내역 기록·추가 화면에서 AI 분석을 실행하세요.")
        return

    report_client_id = str(st.session_state.selected_client)
    report_session_label = str(st.session_state.selected_session)
    report_key = (report_client_id, report_session_label)
    current_report_context = f"{report_client_id}_{report_session_label}_{result.get('created_at', '')}"

    if st.session_state.get("current_report_context") != current_report_context:
        for section_key in [
            "report_section_1",
            "report_section_2",
            "report_section_3",
            "report_section_4",
            "report_section_5",
        ]:
            st.session_state.pop(section_key, None)

        st.session_state["current_report_context"] = current_report_context
    
    saved_report = st.session_state.get("saved_reports", {}).get(report_key, {})
    saved_sections = saved_report.get("edited_sections", {})

    report_text = build_report_text(result)
    client = get_client_row()
    session = get_session_row()
    gender = str(client.get("성별", ""))
    age = str(client.get("연령대", client.get("연령", "")))
    region = str(client.get("지역", ""))
    concern = str(session.get("상담 주제", "")) or str(client.get("상담 유형", "")) or "상담 분류 미상"
    created_date = str(session.get("상담일", datetime.now().strftime("%Y-%m-%d")))
    demographic = " · ".join([part for part in [f"{age} {gender}".strip(), region] if part])
    report_meta = {
        "내담자 ID": report_client_id,
        "회기": report_session_label,
        "성별/연령대/지역": demographic or "정보 없음",
        "상담 분류": concern,
        "작성일": created_date,
    }
    report_title = "상담 요약 보고서"
    chart_placeholders = [
        "위험도 요약 카드 영역",
        "요인 분석 바 차트 영역",
        "HIRA 비교 차트 영역",
    ]
    def build_report_factor_chart(factors: Dict[str, int]):
        factor_chart_df = build_factor_dataframe(factors).copy()
        factor_chart_df = factor_chart_df[factor_chart_df["점수"] > 0]
        factor_chart_df = factor_chart_df.sort_values("점수", ascending=False).head(8)
        factor_chart_df = factor_chart_df.sort_values("점수", ascending=True)

        if factor_chart_df.empty:
            return None

        fig = px.bar(
            factor_chart_df,
            x="점수",
            y="요인",
            color="카테고리",
            orientation="h",
            range_x=[0, 3],
            height=260,
            color_discrete_map={
                "우울/증상": "#7E9FE6",
                "불안": "#A991E6",
                "중독": "#8ECAD8",
                "상담사 개입": "#F3A8BE",
                "변화/기타": "#CBD5E1",
            },
        )

        fig.update_layout(
            margin=dict(l=6, r=6, t=8, b=10),
            xaxis_title="점수",
            yaxis_title="",
            legend_title_text="",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#475569", size=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        return fig


    def build_report_risk_chart(classification: Dict[str, int]):
        risk_df = pd.DataFrame(
            [
                {"항목": "우울", "값": int(classification.get("depression", 0) or 0)},
                {"항목": "불안", "값": int(classification.get("anxiety", 0) or 0)},
                {"항목": "중독", "값": int(classification.get("addiction", 0) or 0)},
            ]
        )

        fig = px.bar(
            risk_df,
            x="항목",
            y="값",
            text="값",
            range_y=[0, 1.1],
            height=230,
            color="항목",
            color_discrete_map={
                "우울": "#7E9FE6",
                "불안": "#A991E6",
                "중독": "#8ECAD8",
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            margin=dict(l=6, r=6, t=8, b=10),
            xaxis_title="",
            yaxis_title="KlueBERT 판별값",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#475569", size=10),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
        return fig


    def build_report_hira_chart(classification: Dict[str, int]):
        hira_summary_df = build_hira_summary_dataframe(
            classification=classification,
            include_negative=False,
        )

        if hira_summary_df.empty:
            return None

        chart_df = (
            hira_summary_df.sort_values("patients", ascending=False)
            .head(5)
            .sort_values("patients", ascending=True)
        )

        fig = px.bar(
            chart_df,
            x="patients",
            y="disease",
            orientation="h",
            text="patients",
            height=230,
            color="질환군",
            color_discrete_map={
                "우울": "#7E9FE6",
                "불안": "#A991E6",
                "중독": "#8ECAD8",
            },
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            margin=dict(l=6, r=20, t=8, b=10),
            xaxis_title="환자수",
            yaxis_title="",
            legend_title_text="",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#475569", size=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
        fig.update_yaxes(showgrid=False)
        return fig

    def make_report_chart_images(classification: Dict[str, int], factors: Dict[str, int]) -> List[Dict[str, Any]]:
        chart_images = []

        chart_specs = [
            ("AI 판별 요약", build_report_risk_chart(classification)),
            ("HIRA 입내원정보", build_report_hira_chart(classification)),
            ("주요 28요인", build_report_factor_chart(factors)),
        ]

        for chart_title, fig in chart_specs:
            if fig is None:
                continue

            try:
                image_bytes = fig.to_image(
                    format="png",
                    scale=2,
                    width=900,
                    height=420,
                )
            except Exception:
                continue

            chart_images.append(
                {
                    "title": chart_title,
                    "image_bytes": image_bytes,
                }
            )

        return chart_images


    def render_report_attached_charts(
        classification: Dict[str, int], 
        factors: Dict[str, int], 
        key_prefix: str,
        mode: str="edit",
    ): 
        risk_fig = build_report_risk_chart(classification)
        factor_fig = build_report_factor_chart(factors)
        hira_fig = build_report_hira_chart(classification)

        with st.container(border=True):
            if mode == "preview":
                st.markdown(
                    """
                    <div class="report-preview-section-title">
                        6. 첨부 차트
                        <span class="report-preview-auto-badge">자동 생성</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="report-block-header report-chart-block-header">
                        <div>
                            <span class="report-block-title">6. 첨부 차트</span>
                            <span class="report-edit-badge">자동 생성</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            top_left_col, top_right_col = st.columns([0.42, 0.58], gap="medium")

            with top_left_col:
                with st.container(border=True):
                    st.markdown(
                        '<div class="report-attached-chart-title">AI 판별 요약</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        risk_fig,
                        use_container_width=True,
                        key=f"{key_prefix}_risk_chart",
                        config={"displayModeBar": False},
                    )

            with top_right_col:
                with st.container(border=True):
                    st.markdown(
                        '<div class="report-attached-chart-title">HIRA 입내원정보</div>',
                        unsafe_allow_html=True,
                    )
                    if hira_fig is None:
                        st.info("표시할 HIRA 통계가 없습니다.")
                    else:
                        st.plotly_chart(
                            hira_fig,
                            use_container_width=True,
                            key=f"{key_prefix}_hira_chart",
                            config={"displayModeBar": False},
                        )

            st.markdown('<div style="height:0.65rem;"></div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown(
                    '<div class="report-attached-chart-title">주요 28요인</div>',
                    unsafe_allow_html=True,
                )
                if factor_fig is None:
                    st.info("표시할 28요인 점수가 없습니다.")
                else:
                    st.plotly_chart(
                        factor_fig,
                        use_container_width=True,
                        key=f"{key_prefix}_factor_chart",
                        config={"displayModeBar": False},
                    )

        st.markdown(
            """
            <div class="dashboard-note-line report-ai-note">
                <span class="dashboard-note-icon">i</span>
                <span>
                    AI가 생성한 보고서입니다. 상담사의 전문적 판단에 따라 내용을 검토·수정하여 사용하시기 바랍니다.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    report_caution = "AI가 생성한 보고서입니다. 상담사의 전문적 판단에 따라 내용을 검토·수정하여 사용하시기 바랍니다."

    def extract_section(text: str, number: int, fallback: str = "") -> str:
        pattern = rf"{number}\.\s*[^\n]*\n(.*?)(?=\n\d+\.\s|\Z)"
        match = re.search(pattern, text, flags=re.S)

        if match:
            return match.group(1).strip()

        return fallback

    def clean_report_section_text(text: str) -> str:
        """
        KoAlpaca/Gemini 요약 결과에 원문 대화, 특수기호, 타임스탬프, 깨진 토큰이 섞이는 경우를 제거한다.
        """
        clean_text = str(text or "").strip()

        # 깨진 특수문자/타임스탬프/마크다운 잡음 제거
        clean_text = re.sub(r"[*／＾＾¶；]+", " ", clean_text)
        clean_text = re.sub(r"\b\d+\s*[：:]\s*\d+\b", " ", clean_text)
        clean_text = re.sub(r"[#]{2,}.*", " ", clean_text)

        lines = []
        for line in clean_text.splitlines():
            line = line.strip()

            if not line:
                continue

            # 원문 대화처럼 들어온 줄 제거
            if re.match(r"^(상담자|상담사|내담자|면접자|상담소)\s*[:\t ]", line):
                continue

            # 숫자/기호만 길게 반복되는 줄 제거
            if re.fullmatch(r"[\d\s;；:：.,\-]+", line):
                continue

            # 디버그성/원문 재정리성 문구 제거
            if "상담 내용을 재정리" in line:
                continue
            if "상담 가능한가요" in line:
                continue
            if "상담료" in line:
                continue

            lines.append(line)

        clean_text = "\n".join(lines)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        return clean_text


    def is_noisy_report_section(text: str) -> bool:
        """
        보고서 섹션으로 쓰기 부적절한 오염 텍스트인지 판단한다.
        """
        clean_text = str(text or "").strip()

        if not clean_text:
            return True

        noise_patterns = [
            r"상담자\s*[:\t ]",
            r"상담사\s*[:\t ]",
            r"내담자\s*[:\t ]",
            r"면접자\s*[:\t ]",
            r"[*／＾＾¶；]{2,}",
            r"\d+；\d+；\d+",
            r"상담 가능한가요",
            r"상담료",
        ]

        if any(re.search(pattern, clean_text) for pattern in noise_patterns):
            return True

        # 주요 증상 섹션이 너무 길면 원문 대화가 섞였을 가능성이 높다.
        if len(clean_text) > 900:
            return True

        return False


    def build_clean_symptom_section(classification: Dict[str, int], factors: Dict[str, int]) -> str:
        """
        오염된 LLM 요약 대신 사용할 구조화된 주요 증상 문장.
        """
        factor_df = build_factor_dataframe(factors).copy()
        positive_df = (
            factor_df[factor_df["점수"] > 0]
            .sort_values(["점수", "요인"], ascending=[False, True])
            .head(5)
        )

        label_items = []

        if int(classification.get("depression", 0) or 0) == 1:
            label_items.append("우울 관련")
        if int(classification.get("anxiety", 0) or 0) == 1:
            label_items.append("불안 관련")
        if int(classification.get("addiction", 0) or 0) == 1:
            label_items.append("중독 관련")

        label_text = ", ".join(label_items) if label_items else "뚜렷한 양성 판별 항목은 제한적"

        if positive_df.empty:
            factor_text = "현재 0점 초과로 추출된 세부 요인은 제한적입니다."
        else:
            factor_text = ", ".join(
                [f"{row['요인']}({int(row['점수'])}점)" for _, row in positive_df.iterrows()]
            )

        return (
            f"현재 상담 기록 기준으로 {label_text} 호소가 확인됩니다.\n"
            f"주요 세부 요인은 {factor_text}입니다.\n"
            "위 내용은 상담 기록과 모델 출력에 기반한 참고 요약이며, 임상 진단 또는 표준화 검사 점수로 단정하지 않습니다."
        )

    classification = result.get("classification", {})
    factor_df = build_factor_dataframe(result.get("factors", {})).sort_values("점수", ascending=False).head(5)
    top_factor_text = ", ".join(
        [f"{row['요인']}({row['점수']})" for _, row in factor_df.iterrows() if int(row["점수"]) > 0]
    ) or "뚜렷하게 상승한 세부 요인은 제한적입니다."

    gemini_sections_result = result.get("gemini_sections", [])

    raw_section_1 = extract_section(report_text, 1, "")
    section_1 = clean_report_section_text(raw_section_1)

    if is_noisy_report_section(section_1) and "report_section_1" not in gemini_sections_result:
        section_1 = build_clean_symptom_section(
            classification=classification,
            factors=result.get("factors", {}),
        )
        
    section_2 = clean_report_section_text(
        extract_section(
            report_text,
            2,
            "수면, 피로, 회피 행동, 자기비하적 사고, 일상 기능 저하 가능성을 중심으로 추가 확인이 필요합니다.",
        )
    )
        
    section_3 = clean_report_section_text(
        extract_section(
            report_text,
            3,
            "내담자가 문제 상황을 언어화하고 상담 장면에 참여하고 있다는 점은 개입의 기반이 될 수 있습니다.",
        )
    )

    section_4 = clean_report_section_text(
        extract_section(
            report_text,
            4,
            "상담사는 수면 양상 확인, 감정 명명, 자동사고 탐색, 공감 및 지지, 다음 회기 과제 설정을 중심으로 개입할 수 있습니다.",
        )
    )

    section_5 = clean_report_section_text(
        extract_section(
            report_text,
            5,
            "다음 회기에서는 수면 양상, 출근 전 불안 상황, 회피 행동, 자기비하적 사고, 현재 대처 방식을 구체적으로 확인합니다.",
        )
    )

    base_sections = {
        "report_section_1": section_1,
        "report_section_2": section_2,
        "report_section_3": section_3,
        "report_section_4": section_4,
        "report_section_5": section_5,
    }

    editor_default_sections = {
        key: saved_sections.get(key, value)
        for key, value in base_sections.items()
    }

    edited_sections = {
        key: st.session_state.get(key, value)
        for key, value in editor_default_sections.items()
    }

    def compose_report(sections: Dict[str, str]) -> str:
        meta_lines = "\n".join([f"- {key}: {value}" for key, value in report_meta.items()])
        chart_lines = "\n".join([f"- {item}" for item in chart_placeholders])
        return f"""# {report_title}

## 기본 메타 정보
{meta_lines}

## 주요 증상
{sections["report_section_1"]}

## 위험 요인
{sections["report_section_2"]}

## 개선 요인
{sections["report_section_3"]}

## 상담사 개입 요인
{sections["report_section_4"]}

## 다음 회기 계획 추천
{sections["report_section_5"]}

## 첨부 차트 영역
{chart_lines}

## 주의 문구
{report_caution}
"""

    edited_report = compose_report(edited_sections)
    default_report = compose_report(base_sections)

    preview_saved_report = st.session_state.get("saved_reports", {}).get(report_key, {})

    if preview_saved_report:
        preview_sections = preview_saved_report.get("edited_sections", base_sections)
        preview_meta = preview_saved_report.get("meta", report_meta)
        preview_report = compose_report(preview_sections)
    else:
        preview_report = default_report
        preview_sections = base_sections
        preview_meta = report_meta

    chart_images = make_report_chart_images(
        classification=classification,
        factors=result.get("factors", {}),
    )

    report_download_kwargs = {
        "client_id": report_client_id,
        "session_label": report_session_label,
        "metadata": preview_meta,
        "sections": preview_sections,
        "chart_placeholders": chart_placeholders,
        "chart_images": chart_images,
        "caution_text": report_caution,
        "title": report_title,
    }
    pdf_bytes = make_pdf_report_bytes(preview_report, **report_download_kwargs)

    if pdf_bytes is None:
        pdf_bytes = make_simple_pdf_report_bytes(
            preview_report,
            title=report_title,
        )

    docx_bytes = make_docx_report_bytes(preview_report, **report_download_kwargs)
    md_download_label = build_download_label(MD_ICON_PATH, "📝", "MD 다운로드")
    pdf_download_label = build_download_label(PDF_ICON_PATH, "📄", "PDF 다운로드")
    docx_download_label = build_download_label(DOCX_ICON_PATH, "📘", "DOCX 다운로드")

    if st.session_state.report_preview_mode:
        close_col, spacer, md_col, pdf_col, docx_col = st.columns([0.06, 0.46, 0.16, 0.16, 0.16], gap="small")

        with close_col:
            if st.button("<", key="report_preview_close", use_container_width=True, help="편집 화면으로 돌아가기"):
                st.session_state.report_preview_mode = False
                st.rerun()

        with md_col:
            st.download_button(
                md_download_label,
                data=preview_report.encode("utf-8"),
                file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with pdf_col:
            st.download_button(
                pdf_download_label,
                data=pdf_bytes if pdf_bytes is not None else b"",
                file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=pdf_bytes is None,
            )

        with docx_col:
            st.download_button(
                docx_download_label,
                data=docx_bytes if docx_bytes is not None else b"",
                file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                disabled=docx_bytes is None,
            )
    else:
        spacer, b1, b2 = st.columns([0.72, 0.13, 0.15], gap="small")

        with b1:
            save_clicked = st.button("저장", type="primary", use_container_width=True)

        with b2:
            preview_clicked = st.button("미리보기", use_container_width=True)


        if save_clicked:
            st.session_state.saved_reports[report_key] = {
                "edited_report": edited_report,
                "edited_sections": edited_sections.copy(),
                "meta": report_meta.copy(),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            preview_report = edited_report
            preview_sections = edited_sections.copy()
            preview_meta = report_meta.copy()
            report_download_kwargs = {
                "client_id": report_client_id,
                "session_label": report_session_label,
                "metadata": preview_meta,
                "sections": preview_sections,
                "chart_placeholders": chart_placeholders,
                "chart_images": chart_images,
                "caution_text": report_caution,
                "title": report_title,
            }
            pdf_bytes = make_pdf_report_bytes(preview_report, **report_download_kwargs)

            if pdf_bytes is None:
                pdf_bytes = make_simple_pdf_report_bytes(
                    preview_report,
                    title=report_title,
                )
                
            docx_bytes = make_docx_report_bytes(preview_report, **report_download_kwargs)
            st.success("보고서 수정 내용을 현재 앱 세션에 저장했습니다.")

        if preview_clicked:
            st.session_state.report_preview_mode = True
            st.rerun()

    if pdf_bytes is None:
        st.caption("PDF 다운로드는 WeasyPrint 설치 후 활성화됩니다.")

    if docx_bytes is None:
        st.caption("DOCX 다운로드는 python-docx 설치 후 활성화됩니다.")

    if st.session_state.report_preview_mode:
        preview_html = f"""<div class="report-preview-shell">
<div class="report-preview-date">작성일: {html_escape(str(preview_meta.get("작성일", created_date)))}</div>
<div class="report-preview-title">{html_escape(report_title)}</div>
<div class="report-preview-divider"></div>

<table class="report-preview-meta-table">
    <tr>
        <th>내담자 ID</th>
        <td>{html_escape(str(preview_meta.get("내담자 ID", report_client_id)))}</td>
        <th>회기</th>
        <td>{html_escape(str(preview_meta.get("회기", report_session_label)))}</td>
        <th>성별·연령대</th>
        <td>{html_escape(str(preview_meta.get("성별/연령대/지역", demographic)).replace(" · " + str(preview_meta.get("지역", "")), ""))}</td>
    </tr>
    <tr>
        <th>지역</th>
        <td>{html_escape(str(preview_meta.get("지역", region)))}</td>
        <th>상담 분류</th>
        <td>{html_escape(str(preview_meta.get("상담 분류", concern)))}</td>
        <th>작성일</th>
        <td>{html_escape(str(preview_meta.get("작성일", created_date)))}</td>
    </tr>
</table>

<div class="report-preview-section">
    <div class="report-preview-section-title">1. 주요 증상</div>
    <div class="report-preview-section-body">{html_escape(preview_sections.get("report_section_1", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">2. 위험 요인</div>
    <div class="report-preview-section-body">{html_escape(preview_sections.get("report_section_2", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">3. 개선 요인</div>
    <div class="report-preview-section-body">{html_escape(preview_sections.get("report_section_3", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">4. 상담사 개입 요인</div>
    <div class="report-preview-section-body">{html_escape(preview_sections.get("report_section_4", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">5.다음 회기 계획 추천</div>
    <div class="report-preview-section-body">{html_escape(preview_sections.get("report_section_5", ""))}</div>
</div>
"""

        st.markdown(preview_html, unsafe_allow_html=True)

        render_report_attached_charts(
            classification=classification,
            factors=result.get("factors", {}),
            key_prefix="preview_report",
            mode="preview",
        )


        return

    report_sections = [
        ("1. 주요 증상", "report_section_1", editor_default_sections["report_section_1"]),
        ("2. 위험 요인", "report_section_2", editor_default_sections["report_section_2"]),
        ("3. 개선 요인", "report_section_3", editor_default_sections["report_section_3"]),
        ("4. 상담사 개입 요인", "report_section_4", editor_default_sections["report_section_4"]),
        ("5. 다음 회기 계획 추천", "report_section_5", editor_default_sections["report_section_5"]),
    ]

    def get_text_area_height(text: str) -> int:
        """
        텍스트 길이와 줄 수에 따라 보고서 편집 박스 높이를 유동적으로 계산한다.
        Streamlit text_area는 완전 자동 높이 조절은 없으므로, 렌더링 시 height를 계산한다.
        """
        clean_text = str(text or "")
        line_count = clean_text.count("\n") + 1
        estimated_wrapped_lines = max(1, len(clean_text) // 48)
        total_lines = max(line_count, estimated_wrapped_lines)

        return min(420, max(118, total_lines * 24 + 52))
    
    st.markdown(
        f"""
        <div class="report-preview-chip-row" style="justify-content:flex-start; margin-top:1.15rem; margin-bottom:1rem;">
            <div class="report-preview-chip">{html_escape(report_meta["내담자 ID"])}</div>
            <div class="report-preview-chip">{html_escape(report_meta["회기"])}</div>
            <div class="report-preview-chip">{html_escape(report_meta["성별/연령대/지역"])}</div>
            <div class="report-preview-chip">{html_escape(report_meta["상담 분류"])}</div>
            <div class="report-preview-chip">작성일: {html_escape(report_meta["작성일"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gemini_used = result.get("gemini_sections", [])

    for title, key, value in report_sections:
        with st.container(border=True):
            gemini_badge = (
                '<span style="font-size:0.72rem;color:#aaa;margin-left:0.4rem;">(gemini로 생성됨)</span>'
                if key in gemini_used else ""
            )
            st.markdown(
                f"""
                <div class="report-section-head">
                    <div class="report-section-title">
                        {title}{gemini_badge}
                        <span class="report-edit-badge">편집 가능</span>
                    </div>
                    <div class="report-edit-icon">edit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            current_value = st.session_state.get(key, value)

            st.text_area(
                title,
                value=current_value,
                height=get_text_area_height(current_value),
                label_visibility="collapsed",
                key=key,
            )
            
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    render_report_attached_charts(
        classification=classification,
        factors=result.get("factors", {}),
        key_prefix="edit_report",
    )


    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)



    download_spacer, md_col, pdf_col, docx_col = st.columns(
        [0.52, 0.16, 0.16, 0.16],
        gap="small",
    )

    with md_col:
        st.download_button(
            md_download_label,
            data=preview_report.encode("utf-8"),
            file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with pdf_col:
        st.download_button(
            pdf_download_label,
            data=pdf_bytes if pdf_bytes is not None else b"",
            file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=pdf_bytes is None,
        )

    with docx_col:
        st.download_button(
            docx_download_label,
            data=docx_bytes if docx_bytes is not None else b"",
            file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            disabled=docx_bytes is None,
        )


# =========================================================
# 15. RAG 챗봇
# =========================================================
def render_chat_messages():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                with st.chat_message("assistant"):
                    render_chatbot_answer_card(message.get("content", ""))

                    sources = message.get("sources", [])
                    if sources:
                        with st.expander("참고한 검색 결과", expanded=False):
                            for idx, source in enumerate(sources, start=1):
                                st.markdown(f"**출처 {idx}. {source.get('title', '출처 없음')}**")
                                st.caption(source.get("desc", ""))


def render_quick_question_buttons():
    st.markdown("#### 빠른 질문")

    q1, q2, q3 = st.columns(3)

    examples = [
        "현재 회기에서 다음 회기에 확인해야 할 내용은?",
        "수면 문제와 불안을 중심으로 상담 계획을 정리해줘.",
        "상담사가 기록할 때 주의해야 할 표현은?",
    ]

    for col, question in zip([q1, q2, q3], examples):
        with col:
            if st.button(question, use_container_width=True):
                add_mock_answer(question)
                st.rerun()

def render_chatbot_answer_card(content: str):
    """
    챗봇 답변을 코드처럼 보이지 않게 카드형 HTML로 렌더링한다.
    번호 섹션을 감지해서 제목/본문을 분리한다.
    """
    text = clean_chatbot_answer(content)

    # "1. 핵심 요약" 같은 번호 제목 기준으로 분리
    section_pattern = r"(?m)^\s*(\d+)\.\s+([^\n]+)"
    matches = list(re.finditer(section_pattern, text))

    if not matches:
        st.markdown(
            f"""
            <div class="chat-answer-card">
                <div class="chat-answer-body">{html_escape(text).replace(chr(10), "<br>")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    intro = text[:matches[0].start()].strip()

    if intro:
        st.markdown(
            f"""
            <div class="chat-answer-intro">
                {html_escape(intro).replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, match in enumerate(matches):
        number = match.group(1)
        title = match.group(2).strip()

        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        # bullet을 HTML 줄바꿈으로 자연스럽게 표시
        body_html = html_escape(body)
        body_html = body_html.replace("\n• ", "<br>• ")
        body_html = body_html.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="chat-answer-section">
                <div class="chat-answer-section-title">
                    <span class="chat-answer-section-number">{number}</span>
                    {html_escape(title)}
                </div>
                <div class="chat-answer-section-body">
                    {body_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_chatbot():
    st.markdown('<span class="chatbot-page-marker"></span>', unsafe_allow_html=True)
    header_text_col, clear_col = st.columns([0.84, 0.16], vertical_alignment="top")

    with header_text_col:
        st.markdown(
            '<div class="patient-title">챗봇</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-desc">상담 기록, 유사 사례, 임상 참고자료를 바탕으로 상담사가 다음 회기를 준비할 수 있도록 돕는 AI 보조 화면입니다.</div>',
            unsafe_allow_html=True,
        )

    with clear_col:
        st.markdown('<span class="chat-clear-button-marker"></span>', unsafe_allow_html=True)
        if st.button("대화 초기화", use_container_width=True):
            clear_chat()
            st.rerun()

    if not st.session_state.chat_history:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요. CounsHelper AI 도우미입니다.\n"
                    "상담 기록, 유사 사례, 임상 참고자료를 바탕으로 다음 회기 준비에 필요한 질문을 도와드릴게요."
                ),
                "sources": [],
                "time": datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후"),
            }
        ]

    if st.session_state.pop("clear_chatbot_input", False):
        st.session_state.chatbot_input = ""

    quick_questions = [
        ("유사사례 검색", "내담자의 주요 위험 요인과 비슷한 사례를 찾아줘."),
        ("임상 가이드라인", "수면 문제와 불안을 중심으로 참고할 임상 가이드라인을 정리해줘."),
        ("상담 이론", "현재 상담 기록에 적용할 수 있는 상담 이론을 알려줘."),
        ("다음 회기 개입 방법", "다음 회기에서 사용할 수 있는 개입 방법을 제안해줘."),
    ]

    def now_time_label():
        return datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")

    def mock_rag_answer(question: str) -> dict:
        q = question.strip()

        if "유사" in q or "사례" in q:
            content = (
                "선택한 내담자의 주요 위험 요인과 비슷한 사례를 기준으로 정리하면 다음과 같습니다.\n\n"
                "• 핵심 위험 요인: 지속적인 스트레스, 수면의 질 저하, 완벽주의 성향, 휴식 부족\n"
                "• 유사 사례: 비슷한 요인을 가진 직장인 상담 사례를 우선 확인할 수 있습니다.\n"
                "• 다음 회기 제안: 불안 완화 전략, 수면 습관 점검, 인지 재구성을 함께 다루는 것이 좋습니다."
            )
            sources = ["AI-Hub 유사 상담 사례", "상담 기록 요인 분석 결과"]

        elif "가이드라인" in q or "임상" in q:
            content = (
                "수면 문제와 불안을 함께 호소하는 경우에는 다음 항목을 우선 확인하는 것이 좋습니다.\n\n"
                "• 수면 양상: 입면 지연, 중간 각성, 기상 후 피로감\n"
                "• 불안 촉발 상황: 업무, 대인관계, 평가 상황, 미래 걱정\n"
                "• 안전 확인: 자해·자살 관련 발화 여부\n"
                "• 개입 방향: 수면위생 교육, 호흡 훈련, 걱정 기록지, 자동사고 탐색"
            )
            sources = ["임상 가이드라인 참고자료", "CBT 기반 상담 참고자료"]

        elif "다음 회기" in q or "개입" in q:
            content = (
                "다음 회기에서 활용할 수 있는 개입 방법을 제안드립니다.\n\n"
                "• 호흡 훈련: 복식호흡 1분 실습\n"
                "• 수면 위생 점검: 카페인, 취침 전 스마트폰, 수면 루틴 확인\n"
                "• 인지 재구성: 반복되는 자기비난 사고를 기록하고 대안적 사고 탐색\n"
                "• 행동 활성화: 부담이 낮은 활동을 작은 단위로 계획"
            )
            sources = ["CBT 참고자료", "상담사 개입 요인 분석"]

        elif "불안" in q:
            content = (
                "불안 완화를 위해 즉시 활용 가능한 간단한 실습은 다음과 같습니다.\n\n"
                "1. 복식 호흡 1분: 들숨과 날숨을 천천히 관찰하기\n"
                "2. grounding: 보이는 것, 들리는 것, 느껴지는 것을 차례로 말하기\n"
                "3. 감정 이름 붙이기: 지금 느끼는 감정을 한 단어로 표현하고 강도 0~10 체크"
            )
            sources = ["불안 완화 상담기법 참고자료", "CBT 실습 자료"]

        else:
            content = (
                "현재 질문을 바탕으로 상담 기록에서 확인할 수 있는 내용을 요약하면 다음과 같습니다.\n\n"
                "• 내담자의 핵심 호소와 위험 요인을 먼저 확인합니다.\n"
                "• 다음 회기에서는 수면, 피로, 불안, 회피 행동, 자기비하적 사고를 구체적으로 탐색할 수 있습니다.\n"
                "• 필요한 경우 유사 사례와 임상 참고자료를 함께 검토하는 방향이 적절합니다."
            )
            sources = ["현재 상담 기록", "RAG mock reference"]

        return {"content": content, "sources": sources}

    def submit_chat_question(question: str) -> bool:
        question = question.strip()

        if not question:
            return False

        add_mock_answer(question)
        st.session_state.chat_scroll_nonce = st.session_state.get("chat_scroll_nonce", 0) + 1

        return True

    def submit_chatbot_input():
        if submit_chat_question(st.session_state.get("chatbot_input", "")):
            st.session_state.chatbot_input = ""

    def build_message_html(message: dict) -> str:
        role = message.get("role", "assistant")
        content = html_escape(str(message.get("content", ""))).replace("\n", "<br>")
        sources = message.get("sources", [])
        time_label = html_escape(str(message.get("time", now_time_label())))

        if role == "user":
            return f"""<div class="chat-row user">
<div class="chat-bubble-wrap">
<div class="chat-bubble">{content}</div>
<div class="chat-time">{time_label} ✓✓</div>
</div>
</div>"""

        source_html = ""

        if sources:
            chips = "".join(
                f'<span class="chat-source-chip">출처: {html_escape(str(src.get("title", src)) if isinstance(src, dict) else str(src))}</span>'
                for src in sources
            )
            source_html = f'<div class="chat-source-row">{chips}</div>'

        return f"""<div class="chat-row assistant">
<div class="chat-avatar">AI</div>
<div class="chat-bubble-wrap">
<div class="chat-bubble">{content}{source_html}</div>
<div class="chat-time">{time_label}</div>
</div>
</div>"""

    messages_html = ""

    for message in st.session_state.chat_history:
        messages_html += build_message_html(message)

    st.markdown(
        f"""
        <div id="chat-page-card" class="chat-page-card" data-chat-scroll-box="true" style="height:50vh; min-height:360px; overflow-y:auto; padding-bottom:4.2rem; box-sizing:border-box;">
            {messages_html}
            <div id="chat-bottom-anchor"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="chat_composer_bar"):
        st.markdown('<span class="chat-composer-anchor"></span>', unsafe_allow_html=True)

        if "chat_quick_pill_nonce" not in st.session_state:
            st.session_state.chat_quick_pill_nonce = 0

        quick_question_map = dict(quick_questions)

        st.markdown('<span class="quick-question-marker"></span>', unsafe_allow_html=True)

        selected_quick_question = st.pills(
            "추천 질문",
            options=list(quick_question_map.keys()),
            selection_mode="single",
            default=None,
            key=f"chat_quick_question_pills_{st.session_state.chat_quick_pill_nonce}",
            label_visibility="collapsed",
        )

        if selected_quick_question:
            submit_chat_question(quick_question_map[selected_quick_question])
            st.session_state.chat_quick_pill_nonce += 1
            st.rerun()

        input_col, send_col = st.columns([0.94, 0.06], gap="small", vertical_alignment="center")

        with input_col:
            st.markdown('<span class="chat-input-marker"></span>', unsafe_allow_html=True)
            user_question = st.text_input(
                "상담 내용 입력",
                placeholder="상담 내용을 입력하거나 질문을 작성하세요",
                label_visibility="collapsed",
                key="chatbot_input",
                on_change=submit_chatbot_input,
            )

        with send_col:
            st.markdown('<span class="chat-send-marker"></span>', unsafe_allow_html=True)
            send_icon_label = f"![send]({get_png_data_uri(SEND_ICON_PATH)})"
            send_clicked = st.button(send_icon_label, key="chatbot_send", use_container_width=True, help="전송")

    if send_clicked and submit_chat_question(user_question):
        st.rerun()

    scroll_chat_to_bottom()


# =========================================================
# 16. Main
# =========================================================
def main():
    init_session_state()
    apply_global_style()
    render_sidebar()

    if st.session_state.page == "내담자 홈":
        render_patient_home()
    elif st.session_state.page == "회기 상세":
        render_session_detail()
    elif st.session_state.page == "상담내역 기록·추가":
        render_record_page()
    elif st.session_state.page == "분석 대시보드":
        render_dashboard()
    elif st.session_state.page == "AI 보고서":
        render_report()
    elif st.session_state.page == "챗봇":
        render_chatbot()


if __name__ == "__main__":
    main()
