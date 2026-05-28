# =========================================================
# Streamlit 화면 구현 + 모델 교체 가능 구조
# 프로젝트: CounsHelper - 상담 기록 분석 & 보고서 자동화 플랫폼
# =========================================================

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

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
    defaults = {
        "page": "상담내역 기록·추가",
        "selected_client": DEFAULT_CLIENT_ID,
        "selected_session": DEFAULT_SESSION_NAME,
        "client_search": DEFAULT_CLIENT_ID,
        "record_mode": "existing",
        "dialogue_rows": DEFAULT_SESSION_DIALOGUE.copy(),
        "chat_history": [],
        "analysis_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# 8. Helper 함수
# =========================================================
def go_page(page_name: str):
    st.session_state.page = page_name


def select_session(session_name: str):
    st.session_state.selected_session = session_name
    st.session_state.record_mode = "existing"
    st.session_state.analysis_result = None

    key = (st.session_state.selected_client, session_name)

    if key in SESSION_DIALOGUES:
        st.session_state.dialogue_rows = SESSION_DIALOGUES[key].copy()
    else:
        st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()


def start_new_session():
    st.session_state.record_mode = "new"
    st.session_state.selected_session = "새 상담"
    st.session_state.dialogue_rows = pd.DataFrame(
        {
            "화자": ["상담사", "내담자"],
            "발화": ["", ""],
        }
    )
    st.session_state.analysis_result = None


def get_client_row():
    row = CLIENTS[CLIENTS["내담자 ID"] == st.session_state.selected_client]

    if row.empty:
        return CLIENTS.iloc[0]

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
PRIMARY_SOFT = "#DBEAFE"
CARD_BLUE = "#F3F8FF"
CARD_BLUE_BORDER = "#D9EAFE"
TEXT = "#0F172A"
SUBTEXT = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#F1F5F9"


def apply_global_style():
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
            font-weight: 720;
            color: {PRIMARY_DARK};
            margin-bottom: 0.25rem;
        }}

        .hero-desc {{
            color: {SUBTEXT};
            font-size: 0.9rem;
            line-height: 1.55;
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
        
        .dashboard-shell {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .clean-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 1.25rem;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
        }}

        .soft-card-blue {{
            background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%);
            border: 1px solid #DBEAFE;
            border-radius: 1.25rem;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.06);
        }}

        .soft-card-green {{
            background: linear-gradient(135deg, #ECFDF5 0%, #FFFFFF 100%);
            border: 1px solid #D1FAE5;
            border-radius: 1.25rem;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 8px 22px rgba(16, 185, 129, 0.06);
        }}

        .soft-card-orange {{
            background: linear-gradient(135deg, #FFF7ED 0%, #FFFFFF 100%);
            border: 1px solid #FED7AA;
            border-radius: 1.25rem;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 8px 22px rgba(249, 115, 22, 0.06);
        }}

        .soft-card-purple {{
            background: linear-gradient(135deg, #F5F3FF 0%, #FFFFFF 100%);
            border: 1px solid #DDD6FE;
            border-radius: 1.25rem;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 8px 22px rgba(124, 58, 237, 0.06);
        }}

        .card-title {{
            font-size: 0.92rem;
            font-weight: 720;
            color: #0F172A;
            margin-bottom: 0.45rem;
        }}

        .card-subtitle {{
            font-size: 0.78rem;
            color: #64748B;
            margin-bottom: 0.55rem;
            line-height: 1.45;
        }}

        .mini-label {{
            font-size: 0.75rem;
            color: #64748B;
            font-weight: 600;
        }}

        .mini-value {{
            font-size: 1.65rem;
            color: #0F172A;
            font-weight: 760;
            letter-spacing: -0.04em;
        }}

        .status-help {{
            display:inline-block;
            width:18px;
            height:18px;
            line-height:18px;
            text-align:center;
            border-radius:50%;
            background:#E2E8F0;
            color:#334155;
            font-size:12px;
            margin-left:6px;
            cursor:help;
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
        st.title("CounsHelper")
        st.caption("상담 기록 분석 & 보고서 자동화")

        st.info(
            f"보고서 백엔드: `{MODEL_BACKEND}`\n\n"
            f"분류 백엔드: `{CLASSIFIER_BACKEND}`\n\n"
            f"28요인 백엔드: `{FACTOR_BACKEND}`"
        )

        st.divider()

        st.sidebar.markdown("### 내담자 선택")

        client_options = CLIENTS["내담자 ID"].tolist()

        if not client_options:
            st.sidebar.warning("표시할 내담자 데이터가 없습니다.")
            selected_client_id = st.session_state.selected_client
        else:
            if st.session_state.selected_client in client_options:
                default_index = client_options.index(st.session_state.selected_client)
            else:
                default_index = 0

            selected_client_id = st.selectbox(
                "내담자",
                options=client_options,
                index=default_index,
            )
            
        if selected_client_id != st.session_state.selected_client:
            st.session_state.selected_client = selected_client_id
            client_sessions = SESSIONS[SESSIONS["내담자 ID"] == selected_client_id]

            if not client_sessions.empty:
                selected_session = client_sessions.iloc[0]["회기"]
                st.session_state.selected_session = selected_session
                st.session_state.record_mode = "existing"

                key = (selected_client_id, selected_session)

                if key in SESSION_DIALOGUES:
                    st.session_state.dialogue_rows = SESSION_DIALOGUES[key].copy()
                else:
                    st.session_state.dialogue_rows = DEFAULT_DIALOGUE.copy()

            st.session_state.analysis_result = None
            st.rerun()

        st.divider()

        st.caption("MVP Demo")
        st.caption("본 시스템은 상담사의 임상적 판단을 대체하지 않습니다.")


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
    st.markdown('<div class="section-title">상담 내역</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">선택한 내담자의 기존 상담 기록을 확인하거나 새 상담 내역을 추가합니다.</div>',
        unsafe_allow_html=True,
    )

    client_sessions = SESSIONS[SESSIONS["내담자 ID"] == st.session_state.selected_client].copy()

    def _session_order(value: Any) -> int:
        match = re.search(r"(\d+)", str(value or ""))
        return int(match.group(1)) if match else 999

    client_sessions["_session_order"] = client_sessions["회기"].apply(_session_order)
    client_sessions = client_sessions.sort_values("_session_order", ascending=True).drop(columns=["_session_order"])

    if client_sessions.empty:
        st.info("기존 상담 내역이 없습니다. 새 상담 내역을 추가해 주세요.")
    else:
        for _, row in client_sessions.iterrows():
            selected = (
                st.session_state.record_mode == "existing"
                and st.session_state.selected_session == row["회기"]
            )

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([0.14, 0.18, 0.34, 0.18, 0.16])

                c1.markdown(f"**{row['회기']}**")
                c3.write(row["상담 주제"])

                with c5:
                    button_label = "선택됨" if selected else "기록 보기"

                    if st.button(
                        button_label,
                        key=f"select_{row['내담자 ID']}_{row['회기']}",
                        use_container_width=True,
                        disabled=selected,
                    ):
                        select_session(row["회기"])
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
    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        render_session_cards()

    with right:
        render_record_editor()


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
    KlueBERT 우울/불안/중독 예측값 KPI 카드 복원.
    사진 배치의 상단 공통 영역 역할.
    """

    depression_label = int(classification.get("depression", 0) or 0)
    anxiety_label = int(classification.get("anxiety", 0) or 0)
    addiction_label = int(classification.get("addiction", 0) or 0)
    suicidal_label = 1 if int(factors.get("suicidal", 0) or 0) > 0 else 0

    review_required = (
        depression_label == 1
        or anxiety_label == 1
        or addiction_label == 1
        or suicidal_label == 1
    )

    st.markdown("### KlueBERT 예측 결과")
    st.caption("우울·불안·중독 0/1 판별 결과와 자해·자살 관련 발화 확인 여부를 표시합니다.")

    kpi_values = [
        {
            "title": "우울",
            "value": "양성" if depression_label == 1 else "음성",
            "delta": "모델 출력 1" if depression_label == 1 else "모델 출력 0",
            "class": "soft-card-blue",
        },
        {
            "title": "불안",
            "value": "양성" if anxiety_label == 1 else "음성",
            "delta": "모델 출력 1" if anxiety_label == 1 else "모델 출력 0",
            "class": "soft-card-purple",
        },
        {
            "title": "중독",
            "value": "양성" if addiction_label == 1 else "음성",
            "delta": "모델 출력 1" if addiction_label == 1 else "모델 출력 0",
            "class": "soft-card-orange",
        },
        {
            "title": "자해·자살",
            "value": "확인 필요" if suicidal_label == 1 else "미표시",
            "delta": "요인 점수 > 0" if suicidal_label == 1 else "요인 점수 0",
            "class": "soft-card-green",
        },
    ]

    kpi_cols = st.columns(4, gap="medium")

    for col, item in zip(kpi_cols, kpi_values):
        with col:
            st.markdown(
                f"""
                <div class="{item['class']}">
                    <div class="mini-label">{item['title']}</div>
                    <div class="mini-value">{item['value']}</div>
                    <div class="card-subtitle">{item['delta']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

    if review_required:
        review_text = "확인 필요"
        review_color = "#9A3412"
        review_bg = "#FFF7ED"
        review_border = "#FDBA74"
    else:
        review_text = "일반 확인"
        review_color = "#047857"
        review_bg = "#ECFDF5"
        review_border = "#A7F3D0"

    st.markdown(
        f"""
        <div style="
            background:{review_bg};
            border:1px solid {review_border};
            border-radius:1rem;
            padding:0.95rem 1.1rem;
            margin-bottom:0.6rem;
            box-shadow:0 6px 18px rgba(15,23,42,0.04);
        ">
            <div style="font-size:1.02rem; font-weight:760; color:{review_color};">
                상담사 검토: {review_text}
                <span class="status-help"
                    title="주의: 우울·불안·중독·자해·자살 표시는 모델과 라벨 기반 참고값이며, 임상 진단 또는 최종 위험도 판단을 의미하지 않습니다.">
                    ?
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("상세 사유 확인", expanded=False):
        reason_keys = [
            "depressive_mood",
            "sleep_disturbance",
            "fatigue",
            "anhedonia",
            "worthlessness",
            "impaired_cognition",
            "anxiety",
            "physical_symptom",
            "loss_of_control",
            "social_avoidance",
            "craving",
            "withdrawal",
            "tolerance",
            "social_problem",
            "suicidal",
        ]

        reason_rows = []

        for key in reason_keys:
            score = int(factors.get(key, 0) or 0)

            if score > 0:
                reason_rows.append(
                    {
                        "요인": FACTOR_LABELS.get(key, key),
                        "점수": score,
                    }
                )

        if not reason_rows:
            st.info("현재 0보다 큰 주요 요인 점수가 없습니다.")
        else:
            reason_df = pd.DataFrame(reason_rows).sort_values("점수", ascending=False)
            st.dataframe(reason_df, use_container_width=True, hide_index=True)

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

    st.markdown("### 증상별 입내원정보")
    st.caption(
        "건강보험심사평가원_시군구별 성별 연령별 주요 정신질환 통계 2024 기준입니다. "
        "기본적으로 KlueBERT 양성 항목만 표시합니다."
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

        if is_positive and is_high:
            bg = "#EFF6FF"
            border = "#93C5FD"
            title_color = "#1D4ED8"
        elif is_positive:
            bg = "#F5F3FF"
            border = "#C4B5FD"
            title_color = "#6D28D9"
        else:
            bg = "#F8FAFC"
            border = "#CBD5E1"
            title_color = "#475569"

        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:1px solid {border};
                    border-radius:1.15rem;
                    padding:1rem 1.05rem;
                    min-height:260px;
                    box-shadow:0 8px 20px rgba(15,23,42,0.05);
                ">
                    <div style="font-size:0.78rem; font-weight:700; color:{title_color}; margin-bottom:0.35rem;">
                        {row.get("판별상태", "")} 항목
                    </div>
                    <div style="font-size:1.08rem; font-weight:780; color:#0F172A; margin-bottom:0.25rem;">
                        {row.get("disease", "질환명 없음")}
                    </div>
                    <div style="font-size:0.8rem; color:#64748B; margin-bottom:0.9rem;">
                        질환군: {row.get("질환군", "")}
                    </div>
                    <div style="border-top:1px solid {border}; padding-top:0.75rem;">
                        <div style="font-size:0.76rem; color:#64748B;">환자수</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#0F172A;">
                            {format_number(patients, "명")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">입내원일수</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#334155;">
                            {format_number(visit_days, "일")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">1인당 평균 입내원일수</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#334155;">
                            {format_float(visit_days_per_patient, "일")}
                        </div>
                        <div style="height:0.65rem;"></div>
                        <div style="font-size:0.76rem; color:#64748B;">1인당 평균 요양급여비용</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#334155;">
                            {format_money(cost_per_patient)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "주의: 위 통계는 요양기관 소재지 기준의 공공 진료 통계이며, "
        "개별 내담자의 진단, 중증도, 위험도 판단 근거로 사용하지 않습니다."
    )

def render_same_group_disease_chart(
    classification: Dict[str, int] = None,
    include_negative: bool = False,
):
    """
    이전 버전의 '같은 성별·연령대 주요 정신질환 진료 현황' 막대차트 복원.
    현재 선택된 내담자와 같은 성별·연령대의 주요 정신질환 환자수를 비교한다.
    """

    st.markdown("#### 같은 성별·연령대 주요 정신질환 진료 현황")
    st.caption("내담자와 같은 성별·연령대의 주요 정신질환 진료 환자수를 비교합니다.")

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
        height=340,
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        xaxis_title="질환",
        yaxis_title="환자수",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, key="same_group_disease_chart_restored")

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


def render_hira_donut_chart(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str,
    key: str,
    top_n: int = 6,
    highlight_value: str = "전체",
):
    """
    지역/성별/연령대 분포 도넛차트 공통 함수.

    highlight_value가 '전체'가 아니면 해당 조각을 바깥으로 살짝 빼고,
    라인 두께를 키워 선택 상태를 강조한다.
    """
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        st.info(f"{title} 데이터를 찾지 못했습니다.")
        return

    chart_df = (
        df.groupby(names_col, as_index=False)[values_col]
        .sum()
        .sort_values(values_col, ascending=False)
        .head(top_n)
    )

    if chart_df.empty or chart_df[values_col].sum() == 0:
        st.info(f"{title} 데이터를 표시할 수 없습니다.")
        return

    labels = chart_df[names_col].astype(str).tolist()

    pull_values = [
        0.12 if highlight_value != "전체" and str(label) == str(highlight_value) else 0
        for label in labels
    ]

    line_widths = [
        4 if highlight_value != "전체" and str(label) == str(highlight_value) else 1
        for label in labels
    ]

    fig = px.pie(
        chart_df,
        names=names_col,
        values=values_col,
        hole=0.62,
        height=260,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        pull=pull_values,
        marker=dict(
            line=dict(
                color="white",
                width=line_widths,
            )
        ),
        hovertemplate="<b>%{label}</b><br>환자수: %{value:,}명<br>비중: %{percent}<extra></extra>",
    )

    fig.update_layout(
        title=None,
        margin=dict(l=5, r=5, t=10, b=5),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    st.markdown(f"**{title}**")

    if highlight_value != "전체":
        st.caption(f"선택 강조: {highlight_value}")

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

    st.markdown(f"### [{label}] 관련 공공통계 상세")
    st.caption(
        f"{label} 관련 질환의 HIRA 2024 분포와 현재 상담 기록의 28요인 구성을 함께 확인합니다."
    )

    # =====================================================
    # 1행: 필터는 '하이라이트 선택' 용도
    # =====================================================
    d1, d2, d3, filter_col = st.columns([0.24, 0.24, 0.24, 0.28], gap="medium")

    with filter_col:
        st.markdown("**하이라이트 필터**")
        st.caption("선택한 지역·연령대가 왼쪽 도넛차트에서 강조 표시됩니다.")

        sido_options = ["전체"] + sorted(context_df["sido"].dropna().unique().tolist())
        selected_sido = st.selectbox(
            "지역",
            options=sido_options,
            key=f"{context_key}_detail_sido_highlight",
        )

        age_options = ["전체"] + sorted(context_df["age_group"].dropna().unique().tolist())
        selected_age = st.selectbox(
            "연령",
            options=age_options,
            key=f"{context_key}_detail_age_highlight",
        )

    with d1:
        render_hira_donut_chart(
            context_df,
            names_col="sido",
            values_col="patients",
            title="지역",
            key=f"{context_key}_sido_donut",
            highlight_value=selected_sido,
        )

    with d2:
        render_hira_donut_chart(
            context_df,
            names_col="gender",
            values_col="patients",
            title="성별",
            key=f"{context_key}_gender_donut",
            highlight_value="전체",
        )

    with d3:
        render_hira_donut_chart(
            context_df,
            names_col="age_group",
            values_col="patients",
            title="연령대",
            key=f"{context_key}_age_donut",
            highlight_value=selected_age,
        )

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # =====================================================
    # 2행: 28요인 트리맵 + 공공통계
    # =====================================================
    tree_col, stat_col = st.columns([0.55, 0.45], gap="large")

    with tree_col:
        with st.container(border=True):
            st.markdown(f"#### {label} 관련 28요인 구성")
            st.caption("현재 상담 기록에서 추출된 28요인 점수를 기준으로 표시합니다.")

            tree_df = build_context_factor_treemap_df(
                context_key=context_key,
                factors=factors,
            )

            if tree_df.empty:
                st.info(f"현재 상담 기록에서 0점보다 큰 {label} 관련 28요인이 없습니다.")
            else:
                fig_tree = px.treemap(
                    tree_df,
                    path=["분류", "카테고리", "요인"],
                    values="값",
                    color="값",
                    height=330,
                )

                fig_tree.update_traces(
                    texttemplate="<b>%{label}</b><br>%{value}점",
                    hovertemplate="<b>%{label}</b><br>점수: %{value}점<extra></extra>",
                )

                fig_tree.update_layout(
                    margin=dict(l=5, r=5, t=10, b=5),
                )

                st.plotly_chart(
                    fig_tree,
                    use_container_width=True,
                    key=f"{context_key}_factor_treemap",
                )

    with stat_col:
        with st.container(border=True):
            st.markdown("#### 공공통계")

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

            c1, c2 = st.columns(2)
            c1.metric("환자수", f"{total_patients:,.0f}명")
            c2.metric("입내원일수", f"{total_visit_days:,.0f}일")

            c3, c4 = st.columns(2)
            c3.metric("1인당 입내원일수", f"{avg_visit_days:,.2f}일")
            c4.metric("1인당 요양급여비용", f"{avg_cost:,.0f}원")

            st.caption(
                "위 공공통계는 하이라이트 필터와 별개로 전체 HIRA 2024 기준 합계입니다. "
                "요양기관 소재지 기준 공공 진료 통계이며, 개별 내담자의 진단·중증도·위험도 판단 근거가 아닙니다."
            )

    # =====================================================
    # 3행: 상담자 해석 도우미
    # =====================================================
    with st.container(border=True):
        st.markdown("#### 상담자 해석 도우미")

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
            f"""
            현재 KlueBERT 판별에서 **{label} 관련 항목이 양성**으로 표시되었습니다.

            왼쪽 도넛차트에서는 **{highlight_text}**가 강조되어 있습니다.  
            이 강조 기능은 상담자가 HIRA 공공통계에서 특정 지역·연령대의 상대적 위치를 빠르게 확인하기 위한 시각적 보조 기능입니다.

            가운데 트리맵은 HIRA 질환 구성이 아니라, **현재 상담 기록에서 추출된 {label} 관련 28요인 점수 구성**입니다.  
            따라서 공공통계는 인구통계적 참고자료로, 28요인 트리맵은 현재 상담 발화의 내용 기반 참고자료로 분리해서 해석해야 합니다.

            최종 판단은 상담자가 실제 발화 맥락, 증상 지속 기간, 기능 저하 정도, 보호요인, 위험요인을 함께 검토해 수행해야 합니다.
            """
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

def render_session_area_trend(classification: Dict[str, int], factors: Dict[str, int]):
    """
    회기별 추이 변화.
    꺾은선 차트를 복원하되, 선 아래 영역은 여러 개의 반투명 fill layer를 겹쳐
    그라데이션처럼 보이게 만든다.
    """

    trend = pd.DataFrame(
        {
            "회기": ["1회기", "2회기", "3회기", "현재"],
            "우울": [2.8, 2.6, 2.4, int(classification.get("depression", 0) or 0) * 3],
            "불안": [2.7, 2.5, 2.4, int(classification.get("anxiety", 0) or 0) * 3],
            "중독": [1.2, 1.1, 1.0, int(classification.get("addiction", 0) or 0) * 3],
            "수면문제": [3.0, 3.0, 3.0, int(factors.get("sleep_disturbance", 0) or 0)],
            "피로감": [2.2, 2.6, 3.0, int(factors.get("fatigue", 0) or 0)],
        }
    )

    color_map = {
        "우울": "rgba(236, 72, 153, 1)",
        "불안": "rgba(124, 58, 237, 1)",
        "중독": "rgba(249, 115, 22, 1)",
        "수면문제": "rgba(34, 197, 94, 1)",
        "피로감": "rgba(59, 130, 246, 1)",
    }

    fill_color_map = {
        "우울": "rgba(236, 72, 153, {alpha})",
        "불안": "rgba(124, 58, 237, {alpha})",
        "중독": "rgba(249, 115, 22, {alpha})",
        "수면문제": "rgba(34, 197, 94, {alpha})",
        "피로감": "rgba(59, 130, 246, {alpha})",
    }

    fig = go.Figure()

    x_values = trend["회기"].tolist()
    series_names = ["우울", "불안", "중독", "수면문제", "피로감"]

    # 그라데이션 느낌을 위한 fill layer
    gradient_layers = [
        (0.35, 0.035),
        (0.55, 0.030),
        (0.75, 0.024),
        (1.00, 0.018),
    ]

    for series in series_names:
        y_values = trend[series].astype(float).tolist()

        for scale, alpha in gradient_layers:
            scaled_y = [value * scale for value in y_values]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=scaled_y,
                    mode="lines",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor=fill_color_map[series].format(alpha=alpha),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 실제 꺾은선
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=series,
                line=dict(
                    color=color_map[series],
                    width=3,
                    shape="spline",
                ),
                marker=dict(
                    size=8,
                    color=color_map[series],
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=f"<b>{series}</b><br>회기: %{{x}}<br>점수: %{{y:.1f}}<extra></extra>",
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="회기",
        yaxis_title="점수",
        yaxis=dict(range=[0, 3.2], gridcolor="rgba(148,163,184,0.18)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.10)"),
        legend_title_text="요인",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, key="session_gradient_line_trend_chart")


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
    레퍼런스 이미지의 AI 분석 요약 4개 카드.
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
        ("주요 증상", symptom_items or ["뚜렷한 주요 증상 없음"]),
        ("위험 요인", risk_items or ["추가 확인 필요"]),
        ("개선 요인", improvement_items),
        ("개입 요인", intervention_items or ["개입 요인 확인 필요"]),
    ]

    st.markdown("### AI 분석 요약")

    cols = st.columns(4)

    for col, (title, items) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                for item in items[:4]:
                    st.markdown(f"- {item}")

def render_dashboard():
    st.markdown('<div class="section-title">분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI 모델 출력 결과와 HIRA 공공 진료 통계를 기반으로 상담 기록의 주요 요인과 참고 통계를 시각화합니다.</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.analysis_result

    if result is None:
        st.info("아직 분석 결과가 없습니다. 먼저 상담내역 기록·추가 화면에서 AI 분석을 실행하세요.")
        return

    # 개발자용 모델 연결 상태: 기본 화면에서는 숨김
    with st.expander("모델 연결 상태 보기", expanded=False):
        backend = result.get("model_info", {}).get("backend", "unknown")
        classifier_backend = result.get("model_info", {}).get("classifier_backend", "mock")
        classifier_status = result.get("model_info", {}).get("classifier_status", "")
        classifier_message = result.get("model_info", {}).get("classifier_message", "")
        classifier_scores = result.get("model_info", {}).get("classifier_scores", {})
        classifier_raw_scores = result.get("model_info", {}).get("raw_scores", {})
        factor_backend = result.get("model_info", {}).get("factor_backend", "mock")
        factor_status = result.get("model_info", {}).get("factor_status", "")
        factor_message = result.get("model_info", {}).get("factor_message", "")

        status_rows = [
            {
                "구분": "보고서 생성",
                "백엔드": backend,
                "상태": "연결됨" if backend != "mock" else "mock",
                "비고": result.get("model_info", {}).get("summarizer", ""),
            },
            {
                "구분": "우울/불안/중독 판별",
                "백엔드": classifier_backend,
                "상태": classifier_status or ("mock" if classifier_backend == "mock" else "unknown"),
                "비고": classifier_message,
            },
            {
                "구분": "28요인 추출",
                "백엔드": factor_backend,
                "상태": factor_status or ("mock" if factor_backend == "mock" else "unknown"),
                "비고": factor_message,
            },
        ]

        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        if classifier_scores:
            st.caption(f"KlueBERT 0~3 예측 점수: {classifier_scores}")

        if classifier_raw_scores:
            st.caption(f"KlueBERT 회귀 원점수(raw score): {classifier_raw_scores}")

    selected_client_id = st.session_state.selected_client
    selected_session = st.session_state.selected_session

    current_session_rows = SESSIONS[
        (SESSIONS["내담자 ID"] == selected_client_id)
        & (SESSIONS["회기"] == selected_session)
    ]

    if not current_session_rows.empty:
        current_session = current_session_rows.iloc[0]
    else:
        current_session = None

    classification = result["classification"]
    factors = result["factors"]
    factor_df = build_factor_dataframe(factors)

    # 원본 라벨은 접어서 표시
    with st.expander("원본 데이터셋 라벨", expanded=False):
        if current_session is not None:
            label_cols = st.columns(4)
            label_cols[0].metric("우울 원본 라벨", int(current_session.get("depression", 0)))
            label_cols[1].metric("불안 원본 라벨", int(current_session.get("anxiety", 0)))
            label_cols[2].metric("중독 원본 라벨", int(current_session.get("addiction", 0)))
            label_cols[3].metric("증상", str(current_session.get("class", "미상")))
        else:
            st.info("현재 선택한 회기의 원본 라벨 정보를 찾지 못했습니다.")

    # =====================================================
    # 0. 상단 공통 영역: 상담사 확인 필요 배너
    # =====================================================
    render_top_risk_cards(classification, factors)

    # =====================================================
    # 1행: 회기별 추이 변화 + 28요인 세부내용
    # =====================================================
    row1_left, row1_right = st.columns([0.60, 0.40], gap="large")

    with row1_left:
        with st.container(border=True):
            st.markdown("### 회기별 추이 변화")
            st.caption("회기별 주요 요인 변화를 영역형 시계열 차트로 표시합니다.")
            render_session_area_trend(classification, factors)

    with row1_right:
        with st.container(border=True):
            st.markdown("### 28요인 세부내용")
            st.caption("점수가 높은 요인부터 표시합니다.")
            render_factor_detail_table(factor_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # 2행: 28요인 분석 + AI 분석 요약
    # =====================================================
    row2_left, row2_right = st.columns([0.55, 0.45], gap="large")

    with row2_left:
        with st.container(border=True):
            st.markdown("### 28요인 분석")
            st.caption("상담 기록에서 추출된 28요인을 카테고리별로 표시합니다.")

            fig_factor = px.bar(
                factor_df.sort_values("점수", ascending=True),
                x="점수",
                y="요인",
                color="카테고리",
                orientation="h",
                range_x=[0, 3],
                height=520,
            )

            fig_factor.update_layout(
                xaxis_title="점수",
                yaxis_title="요인",
                margin=dict(l=10, r=10, t=20, b=10),
                legend_title_text="카테고리",
            )

            st.plotly_chart(
                fig_factor,
                use_container_width=True,
                key="main_factor_horizontal_chart",
            )

    with row2_right:
        with st.container(border=True):
            render_ai_summary_cards(factors)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # 3행: 요인 도넛차트 + 내담자 조건 반영 정신질환 추세
    # =====================================================
    row3_left, row3_right = st.columns([0.35, 0.65], gap="large")

    with row3_left:
        with st.container(border=True):
            render_factor_frequency_card(factors)

    with row3_right:
        with st.container(border=True):
            include_negative_hira = st.checkbox(
                "증상별 입내원정보에서 음성 항목도 함께 보기",
                value=False,
                key="include_negative_hira_items",
                help="기본값은 KlueBERT 양성 항목만 증상별 입내원정보 카드에 표시합니다.",
            )

            render_same_group_disease_chart()

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # 4행: 증상별 입내원정보
    # =====================================================
    with st.container(border=True):
        render_hira_report_sentence_card(
            classification=classification,
            include_negative=include_negative_hira,
        )

    st.divider()

    # =====================================================
    # 하단 접힘 영역: 판정별 상세 대시보드
    # =====================================================
    with st.expander("우울/불안/중독 상세 공공통계 대시보드 보기", expanded=False):
        positive_detail_count = 0

        for context_key in ["depression", "anxiety", "addiction"]:
            if int(classification.get(context_key, 0) or 0) == 1:
                positive_detail_count += 1
                render_hira_context_detail_section(
                    context_key=context_key,
                    classification=classification,
                    factors=factors,
                )
                st.divider()

        if positive_detail_count == 0:
            st.info("현재 KlueBERT 양성 항목이 없어 우울/불안/중독 상세 공공통계 대시보드를 표시하지 않습니다.")
            
# =========================================================
# 14. AI 보고서
# =========================================================
def render_report():
    st.markdown('<div class="section-title">AI 보고서</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">KoAlpaca 요약 모델이 들어갈 위치입니다. 현재는 선택한 보고서 백엔드 결과를 표시합니다.</div>',
        unsafe_allow_html=True,
    )
    
    result = st.session_state.analysis_result

    if result is None:
        st.info("아직 분석 결과가 없습니다. 먼저 상담내역 기록·추가 화면에서 AI 분석을 실행하세요.")
        return

    backend = result.get("model_info", {}).get("backend", "unknown")
    summarizer_name = result.get("model_info", {}).get("summarizer", "unknown")

    if backend == "mock":
        st.warning("현재 보고서는 mock 요약 결과입니다. 실제 KoAlpaca API 결과가 아닙니다.")
    elif backend == "koalpaca_api":
        st.info(f"보고서 생성 백엔드: {summarizer_name}")
    else:
        st.info(f"보고서 생성 백엔드: {summarizer_name}")

    report_text = build_report_text(result)

    edited_report = st.text_area(
        "보고서 초안",
        value=report_text,
        height=620,
    )

    c1, c2 = st.columns([0.22, 0.78])

    with c1:
        st.download_button(
            "Markdown 다운로드",
            data=edited_report.encode("utf-8"),
            file_name=f"{st.session_state.selected_client}_{st.session_state.selected_session}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with c2:
        st.caption("PDF/DOCX 다운로드는 이후 단계에서 추가합니다.")


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


def render_chatbot():
    st.markdown('<div class="section-title">RAG 상담 보조 챗봇</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="page-desc">ChromaDB 기반 유사 상담 사례 검색과 임상 reference 검색이 들어갈 위치입니다.</div>',
        unsafe_allow_html=True,
    )
    st.warning("현재 RAG 챗봇은 mock 응답입니다. 아직 ChromaDB 검색과 실제 LLM 답변 생성이 연결되지 않았습니다.")
    
    c1, c2 = st.columns([0.22, 0.78])

    with c1:
        if st.button("대화 초기화", use_container_width=True):
            clear_chat()
            st.rerun()

        st.info(
            "현재 단계: mock RAG\n\n"
            "추후 단계에서 ChromaDB 검색 결과와 LLM 답변을 연결합니다."
        )

    with c2:
        render_quick_question_buttons()
        render_chat_messages()

        user_prompt = st.chat_input("상담 기록에 대해 질문하세요.")

        if user_prompt:
            add_mock_answer(user_prompt)
            st.rerun()


# =========================================================
# 16. Main
# =========================================================
def main():
    init_session_state()
    apply_global_style()
    render_sidebar()
    render_header()
    render_main_nav()

    if st.session_state.page == "상담내역 기록·추가":
        render_record_page()
    elif st.session_state.page == "분석 대시보드":
        render_dashboard()
    elif st.session_state.page == "AI 보고서":
        render_report()
    elif st.session_state.page == "챗봇":
        render_chatbot()


if __name__ == "__main__":
    main()


