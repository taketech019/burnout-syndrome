# =========================================================
# Streamlit 화면 구현 + 모델 교체 가능 구조
# 프로젝트: CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼
# =========================================================

import json
import base64
import hashlib
import re
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.analysis_pipeline import (
    build_dialogue_text as pipeline_build_dialogue_text,
    run_analysis as pipeline_run_analysis,
)
from src.data_loader import load_app_data
from src.report import make_docx_report_bytes, make_pdf_report_bytes


# =========================================================
# 1. Secrets / 환경 설정
# =========================================================
load_dotenv()


def get_secret(key: str, default=None):
    """
    Streamlit Community Cloud에서는 st.secrets에서 설정값을 읽고,
    Colab/로컬 테스트 중 secrets가 없으면 default 값을 사용한다.
    """
    try:
        return st.secrets[key]
    except Exception:
        return default


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
RAG_EMBEDDING_MODEL = get_secret("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")

KLUEBERT_MODEL_NAME = get_secret("KLUEBERT_MODEL_NAME", "AIHub-KlueBERT-demo")
KOALPACA_MODEL_NAME = get_secret("KOALPACA_MODEL_NAME", "Koalpaca-demo")


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

        # 회기 목록 생성
        sessions = raw.copy()
        sessions["내담자 ID"] = sessions["client_id"]
        sessions["상담일"] = sessions["split"]
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
    
CLIENTS, SESSIONS, SESSION_DIALOGUES, DEFAULT_DIALOGUE = load_app_data()

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

class MockKoalpacaSummarizer:
    """
    임시 요약 모델.
    나중에 4단계에서 Koalpaca 4bit 요약 모델 코드로 교체한다.
    """

    def summarize(
        self,
        script: str,
        classification: Dict[str, int],
        factors: Dict[str, int],
    ) -> str:
        symptom_items = []

        if factors.get("sleep_disturbance", 0) > 0:
            symptom_items.append("수면 문제")
        if factors.get("fatigue", 0) > 0:
            symptom_items.append("피로감")
        if factors.get("anxiety", 0) > 0:
            symptom_items.append("불안감")
        if factors.get("depressive_mood", 0) > 0:
            symptom_items.append("우울 관련 호소")
        if factors.get("social_avoidance", 0) > 0:
            symptom_items.append("사회적 회피 경향")

        symptom_text = ", ".join(symptom_items) if symptom_items else "뚜렷한 주요 증상 라벨 없음"

        return f"""1. 주요 증상
내담자는 제공된 상담 내용 기준으로 {symptom_text}을/를 호소하는 것으로 정리된다.

2. 위험 요인
업무 스트레스, 수면 부족, 피로 누적, 자기비하적 사고, 사회적 회피 가능성을 추가 확인할 필요가 있다.
자해·자살 관련 발화가 확인되는 경우 상담사가 별도 안전 평가를 수행해야 한다.

3. 개선 요인
내담자는 자신의 상태를 언어화하고 상담 장면에 참여하고 있으며, 상담 목표와 과제를 설정할 수 있는 가능성이 있다.

4. 상담사 개입 요인
상담사는 수면 양상 확인, 감정 명명, 자동사고 탐색, 공감 및 지지, 다음 회기 과제 설정을 중심으로 개입할 수 있다.

5. 다음 회기 계획
다음 회기에서는 수면 양상, 출근 전 불안 상황, 회피 행동, 자기비하적 사고, 현재 대처 방식을 구체적으로 확인한다.
"""
        
class KoalpacaAPISummarizer:
    """
    KoAlpaca API 요약 모델 연결 클래스.

    현재 역할:
        - src/summarizer.py의 summarize() 함수를 호출한다.
        - API가 아직 설정되지 않았거나 실패하면 앱이 깨지지 않도록 안내 문구를 반환한다.
    """

    def summarize(
        self,
        script: str,
        classification: Dict[str, int],
        factors: Dict[str, int],
    ) -> str:
        try:
            from src.summarizer import summarize as koalpaca_summarize

            result = koalpaca_summarize(script)

            if result.get("ok"):
                return result.get("text", "").strip()

            status = result.get("status", "unknown")
            message = result.get("message", "KoAlpaca API 호출 결과를 확인할 수 없습니다.")

            return f"""[KoAlpaca API 연결 상태: {status}]

{message}

현재 보고서 생성 기능은 KoAlpaca API 연결 자리만 준비된 상태입니다.
KoAlpaca 호스팅이 완료되면 Streamlit Secrets에 KOALPACA_ENDPOINT_URL과 KOALPACA_API_KEY를 입력한 뒤 다시 실행하세요.
"""

        except Exception as error:
            return f"""[KoAlpaca API 연결 오류]

KoAlpaca API 호출 모듈을 실행하는 중 오류가 발생했습니다.

오류 내용:
{error}
"""

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
    """
    요약보고서 생성 모델 로더.

    MODEL_BACKEND 값에 따라 요약 백엔드를 선택한다.

    - mock: 기존 mock 요약 사용
    - koalpaca_api: src/summarizer.py를 통해 KoAlpaca API 호출
    """
    if MODEL_BACKEND == "mock":
        return MockKoalpacaSummarizer()

    if MODEL_BACKEND == "koalpaca_api":
        return KoalpacaAPISummarizer()

    if MODEL_BACKEND == "aihub_local":
        # 향후 로컬 KoAlpaca 또는 AI Hub 모델을 직접 로딩할 때 사용할 자리
        return MockKoalpacaSummarizer()

    return MockKoalpacaSummarizer()

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
            "summarizer": "KoAlpaca API" if MODEL_BACKEND == "koalpaca_api" else "MockKoalpacaSummarizer",
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
            category = "우울"
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


# 친구 코드의 화면 흐름은 유지하고, 실제 데이터/분석 구현은 공통 src 모듈을 사용한다.
build_dialogue_text = pipeline_build_dialogue_text
run_analysis = pipeline_run_analysis


# =========================================================
# 7. Session State 초기화
# =========================================================
def init_session_state():
    global CLIENTS

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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.registered_clients:
        registered_df = pd.DataFrame(st.session_state.registered_clients)
        CLIENTS = pd.concat([CLIENTS, registered_df], ignore_index=True)
        CLIENTS = CLIENTS.drop_duplicates(subset=["내담자 ID"], keep="last")


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
    "new_session_script_text",
    "new_session_dialogue_editor",
]


def get_empty_dialogue_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "화자": ["상담사", "내담자"],
            "발화": ["", ""],
        }
    )


def reset_new_session_form_state():
    for key in NEW_SESSION_FORM_KEYS:
        st.session_state.pop(key, None)

    st.session_state.dialogue_rows = get_empty_dialogue_rows()
    st.session_state.analysis_result = None


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


def save_new_session(script: str, run_ai: bool = False):
    global SESSIONS, SESSION_DIALOGUES

    session_name = st.session_state.get("new_session_name", get_next_session_name())
    session_date = st.session_state.get("new_session_date", datetime.now().date())
    session_topic = str(st.session_state.get("new_session_topic", "") or "").strip() or "(주제 미입력)"
    session_date_text = session_date.strftime("%Y-%m-%d") if hasattr(session_date, "strftime") else str(session_date)

    new_row = {
        "내담자 ID": st.session_state.selected_client,
        "회기": session_name,
        "상담일": session_date_text,
        "상담 주제": session_topic,
        "보고서 상태": "작성 완료" if run_ai else "임시 저장",
    }

    for column in SESSIONS.columns:
        if column not in new_row:
            new_row[column] = ""

    existing_mask = (
        (SESSIONS["내담자 ID"] == st.session_state.selected_client)
        & (SESSIONS["회기"] == session_name)
    )

    if existing_mask.any():
        SESSIONS.loc[existing_mask, list(new_row.keys())] = list(new_row.values())
    else:
        SESSIONS = pd.concat([SESSIONS, pd.DataFrame([new_row])], ignore_index=True)

    st.session_state.dialogue_rows = _script_to_dialogue_df(script)
    SESSION_DIALOGUES[(st.session_state.selected_client, session_name)] = st.session_state.dialogue_rows.copy()
    st.session_state.selected_session = session_name
    st.session_state.record_mode = "existing"

    if run_ai and script.strip():
        st.session_state.analysis_result = run_analysis(script)
        go_page("분석 대시보드")
    else:
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
    st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()
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


def _short_gender(value: Any) -> str:
    text = str(value or "미상")

    if text.startswith("여"):
        return "여"
    if text.startswith("남"):
        return "남"

    return text[:1] if text else "?"


def _format_sidebar_client_label(client_id: str) -> str:
    row = CLIENTS[CLIENTS["내담자 ID"] == client_id]

    if row.empty:
        return str(client_id)

    client = row.iloc[0]
    name = str(client.get("이름", client_id))
    gender = _short_gender(client.get("성별", "미상"))
    age = str(client.get("연령대", "미상"))
    region = str(client.get("지역", "미상"))

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


def add_mock_answer(user_prompt: str):
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
                "현재는 RAG 연결 전 목업 응답입니다. 실제 구현 시 ChromaDB에서 유사 상담 사례, "
                "회기 요약, 임상 참고문서를 검색한 뒤 답변을 생성합니다.\n\n"
                "현재 상담 내용 기준으로는 수면 양상, 피로 지속 기간, 출근 전 불안 상황, "
                "회피 행동, 자기비하적 사고, 보호 요인을 다음 회기에서 확인할 수 있습니다. "
                "최종 판단은 상담사가 수행해야 합니다."
            ),
            "sources": [
                {
                    "title": "Mock 유사 상담 사례 #CASE-014",
                    "desc": "수면 문제·출근 전 불안·직무 스트레스가 함께 나타난 유사 상담 사례",
                },
                {
                    "title": "Mock 임상 reference",
                    "desc": "수면 양상, 불안 유발 상황, 일상 기능 저하 확인 항목",
                },
            ],
        }
    )


# =========================================================
# 9. 전역 스타일
# =========================================================
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E40AF"
PRIMARY_LIGHT = "#EFF6FF"
PRIMARY_SOFT = "#BFDBFE"
CARD_BLUE = "#FFFFFF"
CARD_BLUE_BORDER = "#E2E8F0"
TEXT = "#0F172A"
SUBTEXT = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#F1F5F9"
CHATBOT_ICON_PATH = Path(__file__).resolve().parent / "assets" / "chatbot.png"
SEND_ICON_PATH = Path(__file__).resolve().parent / "assets" / "send_icon.png"
MD_ICON_PATH = Path(__file__).resolve().parent / "assets" / "md.png"
PDF_ICON_PATH = Path(__file__).resolve().parent / "assets" / "pdf Ribbon.png"
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
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding: 1.75rem 1.35rem 1.4rem !important;
        }}

        section[data-testid="stSidebar"] h1 {{
            color: var(--primary-dark);
            font-size: 1.34rem !important;
            font-weight: 720 !important;
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
            font-weight: 500 !important;
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

        .dashboard-section-gap {{
            height: 1.35rem;
        }}

        .risk-metric-card {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.035);
        }}

        .risk-metric-label {{
            color: var(--subtext);
            font-size: 0.78rem;
            font-weight: 520;
        }}

        .risk-metric-value {{
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.15;
            margin-top: 0.28rem;
        }}

        .risk-metric-status {{
            display: inline-flex;
            margin-top: 0.55rem;
            color: var(--primary-dark);
            background: var(--primary-soft);
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 540;
        }}

        .ai-summary-section {{
            margin-top: 1.15rem;
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
            color: var(--subtext);
            font-size: 0.76rem;
            margin-top: 0.65rem;
        }}

        .dashboard-side-note {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            min-height: 285px;
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
            background: #60A5FA !important;
            border: 1px solid #60A5FA !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(96, 165, 250, 0.18) !important;
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
            width: 1.05rem !important;
            height: 1.05rem !important;
            object-fit: contain !important;
            display: inline-block !important;
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

        .report-preview-divider {{
            height: 1px;
            background: #E2E8F0;
            margin: 0.7rem 0 1rem;
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
            font-size: 0.88rem;
            line-height: 1.75;
            white-space: pre-wrap;
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

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) {{
            border-radius: 11px;
        }}

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary {{
            background: transparent !important;
            border: 1px solid #CBD5E1 !important;
            color: #0F172A !important;
            display: flex !important;
            align-items: center !important;
        }}

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary:hover {{
            background: rgba(255, 255, 255, 0.34) !important;
            border-color: #CBD5E1 !important;
            color: #0F172A !important;
        }}

        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary p,
        section[data-testid="stSidebar"] details:has(.new-client-expander-marker) summary span {{
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
        input:focus,
        div[data-baseweb="select"]:focus-within {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px var(--primary) !important;
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
            border-radius: 9999px !important;
            padding: 0 1.05rem !important;
            font-size: 0.92rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            box-shadow: none !important;
            justify-content: center !important;
            transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        }}

        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-marker) div.stButton > button:first-child span {{
            font-size: 0.92rem !important;
            font-weight: 520 !important;
            line-height: 1 !important;
            margin: 0 !important;
            white-space: nowrap !important;
            width: auto !important;
            text-align: center !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child {{
            background: #60A5FA !important;
            border-color: #60A5FA !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(96, 165, 250, 0.18) !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child p,
        div[data-testid="column"]:has(.factor-category-is-selected) div.stButton > button:first-child span {{
            color: #FFFFFF !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child {{
            background: #FFFFFF !important;
            border-color: #E5E7EB !important;
            color: #0F172A !important;
        }}

        div[data-testid="column"]:has(.factor-category-is-unselected) div.stButton > button:first-child:hover {{
            background: #EFF6FF !important;
            border-color: #93C5FD !important;
            color: #2563EB !important;
        }}

        button[data-testid="stBaseButton-pills"],
        button[data-testid="stBaseButton-pillsActive"] {{
            border-radius: 9999px !important;
            min-height: 2.35rem !important;
            padding: 0 1.12rem !important;
            font-size: 0.95rem !important;
            font-weight: 520 !important;
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
            background: #3B82F6 !important;
            border: 1px solid #3B82F6 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 14px rgba(59, 130, 246, 0.22) !important;
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

        /* 공통 챗봇 이동 버튼 */
        .chatbot-nav-button-marker {{
            display: none;
        }}

        div[data-testid="column"]:has(.chatbot-nav-button-marker) {{
            display: flex !important;
            justify-content: flex-end !important;
        }}

        .st-key-dashboard_chatbot_fab {{
            transform: translateY(-0.95rem) !important;
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
            <div class="sidebar-subtitle">상담 분석·보고서 자동화</div>
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

        st.markdown("<div style='height: 3.8rem;'></div>", unsafe_allow_html=True)

        profile_icon_data_uri = get_png_data_uri(PROFILE_ICON_PATH)        
        st.markdown(
            f"""
            <div class="sidebar-user-card">
                <div class="sidebar-user-avatar">
                    <img src="{profile_icon_data_uri}" alt="profile" />
                </div>
                <div class="sidebar-user-info">
                    <div class="sidebar-user-name">상담심리사</div>
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
# 12. 내담자 홈 / 회기 상세
# =========================================================
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


def build_hira_comparison_dataframe(review_score: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "그룹": ["현재 내담자", "동일 연령 평균", "전체 평균"],
            "비율": [review_score, 38, 31],
        }
    )


def build_trend_dataframe(
    client_sessions: pd.DataFrame,
    selected_session: str,
    depression_score: float,
    anxiety_score: float,
    factors: Dict[str, int],
) -> pd.DataFrame:
    rows = []

    for _, row in client_sessions.iterrows():
        session_name = str(row.get("회기", ""))
        order = _session_order(session_name)

        if session_name == selected_session:
            rows.append(
                {
                    "회기": session_name,
                    "우울": depression_score,
                    "불안": anxiety_score,
                    "수면문제": factors.get("sleep_disturbance", 0),
                    "피로감": factors.get("fatigue", 0),
                }
            )
        else:
            rows.append(
                {
                    "회기": session_name,
                    "우울": min(3, 1 + order * 0.35),
                    "불안": min(3, 0.8 + order * 0.32),
                    "수면문제": min(3, 1 + order * 0.28),
                    "피로감": min(3, 0.7 + order * 0.36),
                }
            )

    return pd.DataFrame(rows)


def factor_category_filter(factor_df: pd.DataFrame, selected_categories: List[str]) -> pd.DataFrame:
    if not selected_categories:
        return factor_df.iloc[0:0]

    category_map = {
        "우울": ["우울"],
        "불안": ["불안"],
        "중독": ["중독"],
    }
    allowed_categories = []

    for category in selected_categories:
        allowed_categories.extend(category_map.get(category, [category]))

    return factor_df[factor_df["카테고리"].isin(allowed_categories)]


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
        width="content",
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

    title_col, list_col, dashboard_col, chatbot_col = st.columns(
        [0.58, 0.13, 0.17, 0.06],
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
            use_container_width=True,
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


# =========================================================
# 13. 상담내역 기록·추가 화면
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
            width="content",
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
                st.session_state.dialogue_rows = get_empty_dialogue_rows()
                st.session_state.pop("new_session_dialogue_editor", None)
                st.session_state.pop("new_session_script_text", None)
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
    if st.session_state.get("record_mode") == "new":
        render_new_session_form()
        return

    render_session_cards()


# =========================================================
# 13. 분석 대시보드
# =========================================================
def render_dashboard():
    client_row = get_client_row()
    client_name = get_client_display_name(client_row)
    client_sessions = get_client_sessions_sorted()

    title_col, chat_col = st.columns([0.88, 0.12], vertical_alignment="top")

    with title_col:
        st.markdown('<div class="patient-title">분석 대시보드</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="patient-desc">상담 회기의 위험도, 주요 요인, 변화 추이를 한눈에 확인합니다.</div>',
            unsafe_allow_html=True,
        )

    with chat_col:
        st.markdown('<span class="chatbot-nav-button-marker"></span>', unsafe_allow_html=True)
        chatbot_button_label = f"![chatbot]({get_png_data_uri(CHATBOT_ICON_PATH)})"

        if st.button(
            chatbot_button_label,
            key="dashboard_chatbot_fab",
            use_container_width=True,
            help="챗봇으로 이동",
        ):
            go_page("챗봇")
            st.rerun()

    if client_sessions.empty:
        st.info("분석할 상담 회기가 없습니다. 먼저 상담내역 기록·추가에서 회기를 추가해 주세요.")
        return

    session_options = client_sessions["회기"].tolist()

    if st.session_state.selected_session not in session_options:
        select_session(session_options[0])

    st.markdown('<span class="dashboard-session-select-marker"></span>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "회기 선택",
        options=session_options,
        index=session_options.index(st.session_state.selected_session),
        format_func=lambda session_name: (
            f"{session_name} · "
            f"{client_sessions[client_sessions['회기'] == session_name].iloc[0].get('상담일', '날짜 미상')}"
        ),
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

    current_session = get_session_row()
    classification = result["classification"]
    factors = result["factors"]
    factor_df = build_factor_dataframe(factors)

    depression_score = float(classification.get("depression", 0))
    anxiety_score = float(classification.get("anxiety", 0))
    addiction_score = float(classification.get("addiction", 0))
    review_score = int(round(((depression_score + anxiety_score + addiction_score) / 9) * 100))

    metric_items = [
        ("우울 위험도", f"{depression_score:.0f} / 3", risk_status(depression_score)),
        ("불안 위험도", f"{anxiety_score:.0f} / 3", risk_status(anxiety_score)),
        ("중독 위험도", f"{addiction_score:.0f} / 3", risk_status(addiction_score)),
        ("검토 필요도", f"{review_score}%", "상담사 확인 필요" if review_score >= 45 else "안정"),
    ]

    metric_cols = st.columns(4)

    for col, (label, value, status) in zip(metric_cols, metric_items):
        with col:
            st.markdown(
                f"""
                <div class="risk-metric-card">
                    <div class="risk-metric-label">{html_escape(str(label))}</div>
                    <div class="risk-metric-value">{html_escape(str(value))}</div>
                    <span class="risk-metric-status">{html_escape(str(status))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    top_factor_df = factor_df.sort_values(["점수", "요인"], ascending=[False, True]).head(8)
    top_items = [str(row["요인"]) for _, row in top_factor_df.iterrows() if int(row["점수"]) > 0]
    symptom_items = top_items[:3] or ["상담 내용 추가 확인"]
    risk_items = top_items[3:6] or ["위험 요인 추가 확인"]

    ai_summary = {
        "symptom": {
            "title": "주요 증상",
            "icon": "S",
            "items": symptom_items,
            "note": "상담 기록에서 상대적으로 높게 표시된 증상 요인을 정리합니다.",
        },
        "risk": {
            "title": "위험 요인",
            "icon": "!",
            "items": risk_items,
            "note": "추가 확인이 필요한 정서·행동 신호를 상담사가 검토합니다.",
        },
        "improve": {
            "title": "개선 요인",
            "icon": "+",
            "items": ["상담 참여", "상태 언어화", "변화 가능성"],
            "note": "내담자의 보호요인과 변화 자원을 함께 확인합니다.",
        },
        "intervention": {
            "title": "개입 요인",
            "icon": "I",
            "items": ["감정 명명", "자동사고 탐색", "다음 회기 과제"],
            "note": "상담사 개입 방향을 다음 회기 계획과 연결합니다.",
        },
    }

    cards_html = ""

    for key, card in ai_summary.items():
        items_html = "".join([f"<li>{html_escape(str(item))}</li>" for item in card["items"]])
        cards_html += f"""
<div class="ai-summary-card {key}">
    <div class="ai-summary-head">
        <div class="ai-summary-icon {key}">{html_escape(str(card["icon"]))}</div>
        <div class="ai-summary-card-title">{html_escape(str(card["title"]))}</div>
    </div>
    <ul class="ai-summary-list">{items_html}</ul>
    <div class="ai-summary-note {key}">{html_escape(str(card["note"]))}</div>
</div>
"""

    st.markdown(
        f"""<div class="ai-summary-section">
<div class="ai-summary-title">AI 분석 요약</div>
<div class="ai-summary-grid">
{cards_html}
</div>
<div class="ai-summary-footnote">
    위 요약은 AI 분석 결과의 보조 참고 자료이며, 최종 판단과 상담 방향은 상담사의 전문적 판단을 우선합니다.
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)

    hira_col, insight_col = st.columns([0.52, 0.48], gap="large")

    with hira_col:
        st.markdown('<div class="chart-panel-title">HIRA 인구통계 비교</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-panel-desc">현재 내담자와 예시 인구통계 지표를 비교합니다.</div>',
            unsafe_allow_html=True,
        )
        comparison_df = build_hira_comparison_dataframe(review_score)
        fig_compare = px.bar(
            comparison_df,
            x="그룹",
            y="비율",
            text="비율",
            color="그룹",
            color_discrete_sequence=["#2563EB", "#93C5FD", "#BFDBFE"],
            height=285,
        )
        fig_compare.update_layout(
            showlegend=False,
            margin=dict(l=8, r=8, t=18, b=18),
            yaxis_title="비율(%)",
            xaxis_title="",
            bargap=0.55,
        )
        fig_compare.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        style_dashboard_chart(fig_compare)
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption("예시 데이터입니다. 실제 HIRA 통계 연동 후 연령·성별·지역별 비교 지표로 대체됩니다.")

    with insight_col:
        st.markdown(
            """<div class="dashboard-side-note">
<div class="dashboard-side-note-title">비교 해석 메모</div>
<div class="dashboard-side-note-body">
    현재 차트는 선택 내담자의 검토 필요도와 예시 인구통계 평균을 비교하는 자리입니다.
    실제 HIRA 통계가 연결되면 연령·성별·지역 기준의 상대적 수준을 함께 보여줄 수 있습니다.
</div>
<ul class="dashboard-side-note-list">
    <li>현재 내담자의 위험도 수준 확인</li>
    <li>동일 연령·성별·지역 평균과 비교</li>
    <li>상담 계획 수립 시 참고 지표로 활용</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)

    trend = build_trend_dataframe(
        client_sessions=client_sessions,
        selected_session=st.session_state.selected_session,
        depression_score=depression_score,
        anxiety_score=anxiety_score,
        factors=factors,
    )

    st.markdown('<div class="chart-panel-title">회기별 추이 차트</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-panel-desc">상담 회기별 주요 지표 변화를 추적합니다.</div>',
        unsafe_allow_html=True,
    )

    fig_trend = px.line(
        trend,
        x="회기",
        y=["우울", "불안", "수면문제", "피로감"],
        markers=True,
        range_y=[0, 3],
        color_discrete_sequence=["#2563EB", "#60A5FA", "#A78BFA", "#94A3B8"],
        height=390,
    )
    fig_trend.update_layout(margin=dict(l=10, r=140, t=20, b=20), yaxis_title="점수 (0-3)", xaxis_title="")
    fig_trend.update_traces(line=dict(width=2.2), marker=dict(size=7))
    style_dashboard_chart(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("실제 상담 기록을 기반으로 회기별 점수(0~3)가 표시됩니다.")

    st.markdown('<div class="dashboard-section-gap"></div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-panel-title">세부 요인 막대 그래프</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-panel-desc">28개 요인을 카테고리별로 그루핑합니다. 기본 화면에서는 상위 10개 요인을 먼저 보여줍니다.</div>',
        unsafe_allow_html=True,
    )

    if "factor_selected_categories" not in st.session_state:
        st.session_state.factor_selected_categories = ["우울", "불안", "중독"]

    category_options = ["우울", "불안", "중독"]
    selected_categories = st.pills(
        "요인 카테고리",
        options=category_options,
        selection_mode="multi",
        default=st.session_state.factor_selected_categories,
        key="factor_category_pills",
        label_visibility="collapsed",
    )
    st.session_state.factor_selected_categories = selected_categories

    filtered_factor_df = factor_category_filter(factor_df, st.session_state.factor_selected_categories)

    if filtered_factor_df.empty:
        st.info("선택한 카테고리에 표시할 요인이 없습니다.")
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
            color="카테고리",
            orientation="h",
            range_x=[0, 3],
            color_discrete_map={
                "우울": "#2563EB",
                "불안": "#60A5FA",
                "중독": "#93C5FD",
                "상담사 개입": "#CBD5E1",
                "변화/기타": "#E2E8F0",
            },
            height=440,
        )
        fig_factor.update_layout(margin=dict(l=10, r=140, t=20, b=20), xaxis_title="점수", yaxis_title="", bargap=0.58)
        fig_factor.update_traces(marker_line_width=0)
        style_dashboard_chart(fig_factor)
        st.plotly_chart(fig_factor, use_container_width=True)

        with st.expander("전체 28요인 보기", expanded=False):
            all_factor_chart_df = (
                factor_df.sort_values(["점수", "요인"], ascending=[False, True])
                .sort_values("점수", ascending=True)
            )

            fig_all_factor = px.bar(
                all_factor_chart_df,
                x="점수",
                y="요인",
                color="카테고리",
                orientation="h",
                range_x=[0, 3],
                color_discrete_map={
                    "우울": "#2563EB",
                    "불안": "#60A5FA",
                    "중독": "#93C5FD",
                    "상담사 개입": "#CBD5E1",
                    "변화/기타": "#E2E8F0",
                },
                height=760,
            )

            fig_all_factor.update_layout(margin=dict(l=10, r=140, t=20, b=20), xaxis_title="점수", yaxis_title="", bargap=0.42, showlegend=True,)

            fig_all_factor.update_traces(marker_line_width=0)
            style_dashboard_chart(fig_all_factor)

            st.plotly_chart(fig_all_factor, use_container_width=True)

    if factors.get("suicidal", 0) > 0:
        st.error("자해/자살 관련 발화 가능성이 표시되었습니다. 상담사가 별도 안전 평가를 수행해야 합니다.")

    st.caption(
        f"현재 선택 회기: {st.session_state.selected_client} / {st.session_state.selected_session} / "
        f"{current_session.get('상담일', '상담일 미상')} / {current_session.get('상담 주제', '주제 미상')}"
    )


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

    report_text = build_report_text(result)

    def extract_section(text: str, number: int, fallback: str = "") -> str:
        pattern = rf"{number}\.\s*[^\n]*\n(.*?)(?=\n\d+\.\s|\Z)"
        match = re.search(pattern, text, flags=re.S)

        if match:
            return match.group(1).strip()

        return fallback

    classification = result.get("classification", {})
    factor_df = build_factor_dataframe(result.get("factors", {})).sort_values("점수", ascending=False).head(5)
    top_factor_text = ", ".join(
        [f"{row['요인']}({row['점수']})" for _, row in factor_df.iterrows() if int(row["점수"]) > 0]
    ) or "뚜렷하게 상승한 세부 요인은 제한적입니다."

    section_1 = extract_section(
        report_text,
        1,
        (
            f"우울 {classification.get('depression', 0)}/3, 불안 {classification.get('anxiety', 0)}/3, "
            f"중독 {classification.get('addiction', 0)}/3 수준으로 분류되었습니다.\n"
            f"상위 세부 요인은 {top_factor_text}입니다.\n"
            f"{result.get('summary', '')}"
        ),
    )
    section_2 = extract_section(
        report_text,
        2,
        "수면, 피로, 회피 행동, 자기비하적 사고, 일상 기능 저하 가능성을 중심으로 추가 확인이 필요합니다.",
    )
    section_3 = extract_section(
        report_text,
        3,
        "내담자가 문제 상황을 언어화하고 상담 장면에 참여하고 있다는 점은 개입의 기반이 될 수 있습니다.",
    )
    section_4 = extract_section(
        report_text,
        4,
        "상담사는 수면 양상 확인, 감정 명명, 자동사고 탐색, 공감 및 지지, 다음 회기 과제 설정을 중심으로 개입할 수 있습니다.",
    )
    section_5 = extract_section(
        report_text,
        5,
        "다음 회기에서는 수면 양상, 출근 전 불안 상황, 회피 행동, 자기비하적 사고, 현재 대처 방식을 구체적으로 확인합니다.",
    )

    edited_sections = {
        "report_section_1": st.session_state.get("report_section_1", section_1),
        "report_section_2": st.session_state.get("report_section_2", section_2),
        "report_section_3": st.session_state.get("report_section_3", section_3),
        "report_section_4": st.session_state.get("report_section_4", section_4),
    }

    edited_report = f"""[상담보고서 초안]

0. AI 판별 결과
- 우울 관련 라벨: {classification.get("depression", 0)}
- 불안 관련 라벨: {classification.get("anxiety", 0)}
- 중독 관련 라벨: {classification.get("addiction", 0)}

주의:
위 값은 모델 출력 기반 참고값이며, 임상 진단 또는 표준화 검사 점수로 단정하지 않는다.

1. 주요 증상
{edited_sections["report_section_1"]}

2. 위험 요인
{edited_sections["report_section_2"]}

3. 개선 요인
{edited_sections["report_section_3"]}

4. 상담사 개입 요인
{edited_sections["report_section_4"]}

5. 다음 회기 계획
{section_5}
"""

    report_client_id = str(st.session_state.selected_client)
    report_session_label = str(st.session_state.selected_session)
    pdf_bytes = make_pdf_report_bytes(edited_report, client_id=report_client_id, session_label=report_session_label)
    docx_bytes = make_docx_report_bytes(edited_report, client_id=report_client_id, session_label=report_session_label)
    md_download_label = f"![md]({get_png_data_uri(MD_ICON_PATH)}) MD 다운로드"
    pdf_download_label = f"![pdf]({get_png_data_uri(PDF_ICON_PATH)}) PDF 다운로드"
    docx_download_label = f"![docx]({get_png_data_uri(DOCX_ICON_PATH)}) DOCX 다운로드"

    if st.session_state.report_preview_mode:
        close_col, spacer, md_col, pdf_col, docx_col = st.columns([0.06, 0.46, 0.16, 0.16, 0.16], gap="small")

        with close_col:
            if st.button("<", key="report_preview_close", use_container_width=True, help="편집 화면으로 돌아가기"):
                st.session_state.report_preview_mode = False
                st.rerun()

        with md_col:
            st.download_button(
                md_download_label,
                data=edited_report.encode("utf-8"),
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
            st.success("보고서 수정 내용이 현재 화면에 반영되었습니다. 실제 DB 저장은 추후 연동 예정입니다.")

        if preview_clicked:
            st.session_state.report_preview_mode = True
            st.rerun()

    if pdf_bytes is None:
        st.caption("PDF 다운로드는 WeasyPrint 설치 후 활성화됩니다.")

    if docx_bytes is None:
        st.caption("DOCX 다운로드는 python-docx 설치 후 활성화됩니다.")

    if st.session_state.report_preview_mode:
        client = get_client_row()
        session = get_session_row()
        gender = str(client.get("성별", ""))
        age = str(client.get("연령대", client.get("연령", "")))
        region = str(client.get("지역", ""))
        concern = str(client.get("상담 유형", ""))
        created_date = str(session.get("상담일", datetime.now().strftime("%Y.%m.%d")))
        one_line_summary = edited_sections.get("report_section_1", "").split("\n")[0]

        if not one_line_summary.strip():
            one_line_summary = "상담 기록을 바탕으로 주요 증상과 위험 요인을 요약한 보고서입니다."

        preview_html = f"""<div class="report-preview-shell">
<div class="report-preview-date">작성일: {html_escape(created_date)}</div>
<div class="report-preview-title">상담 요약 보고서</div>
<div class="report-preview-divider"></div>

<div class="report-preview-chip-row">
    <div class="report-preview-chip">👤 {html_escape(report_client_id)}</div>
    <div class="report-preview-chip">📅 {html_escape(report_session_label)}</div>
    <div class="report-preview-chip">📍 {html_escape(age)} {html_escape(gender)} · {html_escape(region)}</div>
    <div class="report-preview-chip">♡ {html_escape(concern)}</div>
</div>

<div class="report-preview-summary">
    <div class="report-preview-summary-title">한줄 요약</div>
    <div class="report-preview-summary-body">{html_escape(one_line_summary)}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">1. 주요 증상</div>
    <div class="report-preview-section-body">{html_escape(edited_sections.get("report_section_1", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">2. 위험 요인</div>
    <div class="report-preview-section-body">{html_escape(edited_sections.get("report_section_2", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">3. 개선 요인</div>
    <div class="report-preview-section-body">{html_escape(edited_sections.get("report_section_3", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">4. 상담사 개입 요인</div>
    <div class="report-preview-section-body">{html_escape(edited_sections.get("report_section_4", ""))}</div>
</div>

<div class="report-preview-section">
    <div class="report-preview-section-title">5. 다음 회기 계획</div>
    <div class="report-preview-section-body">{html_escape(section_5)}</div>
</div>

<div class="report-preview-chart-panel">
    <div class="report-preview-chart-title">첨부 차트 <span style="font-weight:400; color:#64748B;">(대시보드 차트 포함)</span></div>
    <div class="report-preview-chart-grid">
        <div class="report-preview-mini-chart">위험도 요약 카드</div>
        <div class="report-preview-mini-chart">요인 분석 바 차트</div>
        <div class="report-preview-mini-chart">HIRA 인구통계 비교</div>
    </div>
</div>
</div>

<div class="report-preview-footer-note">
    AI가 생성한 보고서입니다. 상담사의 전문적 판단에 따라 내용을 검토·수정하여 사용하시기 바랍니다.
</div>
"""

        st.markdown(preview_html, unsafe_allow_html=True)
        return

    report_sections = [
        ("1. 주요 증상", "report_section_1", section_1),
        ("2. 위험 요인", "report_section_2", section_2),
        ("3. 개선 요인", "report_section_3", section_3),
        ("4. 상담사 개입 요인", "report_section_4", section_4),
    ]

    for title, key, value in report_sections:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="report-section-head">
                    <div class="report-section-title">
                        {title}
                        <span class="report-edit-badge">편집 가능</span>
                    </div>
                    <div class="report-edit-icon">edit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(title, value=st.session_state.get(key, value), height=118, label_visibility="collapsed", key=key)

    st.markdown(
        """
        <div class="report-plan-panel">
            <div class="report-plan-title">다음 회기 계획 추천</div>
            <div class="report-plan-grid">
                <div class="report-plan-card">
                    <div class="report-plan-card-title">목표</div>
                    <ul>
                        <li>불안 및 걱정 수준을 관찰 가능한 지표로 기록한다.</li>
                        <li>수면과 피로 양상의 변화를 다음 회기까지 추적한다.</li>
                        <li>회피 행동을 줄이고 사회적 상호작용을 점진적으로 늘린다.</li>
                    </ul>
                </div>
                <div class="report-plan-card">
                    <div class="report-plan-card-title">기법</div>
                    <ul>
                        <li>인지 재구성 및 사고기록지 활용</li>
                        <li>호흡 명상 및 점진적 근이완 훈련</li>
                        <li>노출 계층표 작성 후 점진적 노출</li>
                    </ul>
                </div>
                <div class="report-plan-card">
                    <div class="report-plan-card-title">과제</div>
                    <ul>
                        <li>매일 10분 호흡 명상 실천 및 기록</li>
                        <li>수면 일지 작성 및 수면 위생 실천</li>
                        <li>주 3회 30분 이상 가벼운 운동 실천</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="report-chart-preview">
            <div class="report-section-title">포함 차트 미리보기</div>
            <div class="page-desc">보고서 다운로드 시 포함될 차트 영역입니다. 현재 화면에서는 자리만 표시합니다.</div>
            <div class="chart-placeholder-grid">
                <div class="chart-placeholder-card">위험도 요약 차트 영역</div>
                <div class="chart-placeholder-card">요인 분석 차트 영역</div>
                <div class="chart-placeholder-card">HIRA 비교 차트 영역</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-footnote">AI가 생성한 보고서입니다. 상담사의 전문적 판단에 따라 내용을 검토·수정하여 사용하시기 바랍니다.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)



    download_spacer, md_col, pdf_col, docx_col = st.columns(
        [0.52, 0.16, 0.16, 0.16],
        gap="small",
    )

    with md_col:
        st.download_button(
            md_download_label,
            data=edited_report.encode("utf-8"),
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

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("검색 근거 보기"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.caption(source["desc"])


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
            st.session_state.chat_messages = []
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요, CounsHelper AI 도우미입니다.\n"
                    "상담 관련 질문이나 도움이 필요한 내용을 자유롭게 입력해주세요."
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

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
                "sources": [],
                "time": now_time_label(),
            }
        )

        answer = mock_rag_answer(question)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer["content"],
                "sources": answer["sources"],
                "time": now_time_label(),
            }
        )
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
                f'<span class="chat-source-chip">출처: {html_escape(str(src))}</span>'
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

    for message in st.session_state.chat_messages:
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
            width="content",
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
