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
import streamlit as st
from dotenv import load_dotenv


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
        
class KlueBertHFClassifier:
    """
    Hugging Face에 업로드된 KlueBERT 회귀 모델 3개를 사용해
    우울/불안/중독 여부를 판별하는 클래스.

    최종 반환값은 기존 MockKlueBERTClassifier와 동일하게
    {
        "depression": 0/1,
        "anxiety": 0/1,
        "addiction": 0/1
    }
    형식으로 맞춘다.
    """

    def __init__(self):
        self.last_result = {
            "ok": False,
            "status": "not_run",
            "message": "KlueBERT 분류가 아직 실행되지 않았습니다.",
            "backend": "kluebert_hf",
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
                "backend": "kluebert_hf",
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
    - kluebert_hf: Hugging Face에 업로드된 KlueBERT 회귀 모델 3개 사용
    """
    if CLASSIFIER_BACKEND == "mock":
        return MockKlueBERTClassifier()

    if CLASSIFIER_BACKEND == "kluebert_hf":
        return KlueBertHFClassifier()

    if CLASSIFIER_BACKEND == "aihub_local":
        # 향후 로컬 모델 직접 로딩 방식이 필요할 경우 사용할 자리
        return MockKlueBERTClassifier()

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
            "classifier": "KlueBERT HF" if CLASSIFIER_BACKEND == "kluebert_hf" else "MockKlueBERTClassifier",
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
def render_dashboard():
    st.markdown('<div class="section-title">분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">AI 모델 출력 결과를 기반으로 상담 내용의 주요 라벨과 요인을 시각화합니다.</div>',
        unsafe_allow_html=True,
    )
    
    backend = result.get("model_info", {}).get("backend", "unknown")
    
    if backend == "mock":
        st.warning("현재 결과는 mock 분석 결과입니다. 실제 KlueBERT, KoAlpaca, RAG 모델 결과가 아닙니다.")
    elif backend == "koalpaca_api":
        st.info("현재 보고서 생성 백엔드는 KoAlpaca API로 설정되어 있습니다.")
    else:
        st.info(f"현재 모델 백엔드: {backend}")
    
    result = st.session_state.analysis_result

    if result is None:
        st.info("아직 분석 결과가 없습니다. 먼저 상담내역 기록·추가 화면에서 AI 분석을 실행하세요.")
        return

    backend = result.get("model_info", {}).get("backend", "unknown")

    classifier_backend = result.get("model_info", {}).get("classifier_backend", "mock")
    classifier_status = result.get("model_info", {}).get("classifier_status", "")
    classifier_message = result.get("model_info", {}).get("classifier_message", "")
    classifier_scores = result.get("model_info", {}).get("classifier_scores", {})
    classifier_raw_scores = result.get("model_info", {}).get("classifier_raw_scores", {})
    
    if classifier_backend == "mock":
        st.warning("현재 우울/불안/중독 판별은 mock 분류 결과입니다. 실제 KlueBERT 모델 결과가 아닙니다.")
    elif classifier_backend == "kluebert_hf":
        if classifier_status == "success":
            st.success("우울/불안/중독 판별 백엔드: KlueBERT HF 연결 성공")
            if classifier_scores:
                st.caption(f"KlueBERT 0~3 예측 점수: {classifier_scores}")
            if classifier_raw_scores:
                st.caption(f"KlueBERT 회귀 원점수(raw score): {classifier_raw_scores}")
        else:
            st.info(f"우울/불안/중독 판별 백엔드: KlueBERT HF / 상태: {classifier_status}")
            if classifier_message:
                st.caption(classifier_message)

    if backend == "mock":
        st.warning("현재 보고서 생성 백엔드는 mock입니다. KoAlpaca API 보고서 결과가 아닙니다.")
    elif backend == "koalpaca_api":
        st.info("현재 보고서 생성 백엔드는 KoAlpaca API로 설정되어 있습니다.")
    else:
        st.info(f"현재 모델 백엔드: {backend}")

    factor_backend = result.get("model_info", {}).get("factor_backend", "mock")
    factor_status = result.get("model_info", {}).get("factor_status", "")
    factor_message = result.get("model_info", {}).get("factor_message", "")

    if factor_backend == "mock":
        st.warning("현재 28요인 점수는 mock 추출 결과입니다. 실제 Gemini API 결과가 아닙니다.")
    elif factor_backend == "gemini_api":
        if factor_status == "success":
            st.success("28요인 추출 백엔드: Gemini API 연결 성공")
        else:
            st.info(f"28요인 추출 백엔드: Gemini API / 상태: {factor_status}")
            if factor_message:
                st.caption(factor_message)
                
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("우울 0~3점", classification.get("depression", 0))
    c2.metric("불안 0~3점", classification.get("anxiety", 0))
    c3.metric("중독 0~3점", classification.get("addiction", 0))
    c4.metric("보고서 백엔드", result["model_info"]["backend"])
    
    st.markdown("### 데이터셋 원본 라벨")

    if current_session is not None:
        label_cols = st.columns(4)

        with label_cols[0]:
            st.metric("우울 원본 라벨", int(current_session.get("depression", 0)))

        with label_cols[1]:
            st.metric("불안 원본 라벨", int(current_session.get("anxiety", 0)))

        with label_cols[2]:
            st.metric("중독 원본 라벨", int(current_session.get("addiction", 0)))

        with label_cols[3]:
            st.metric("데이터 class", str(current_session.get("class", "미상")))

        st.caption(
            f"현재 선택 회기: {selected_client_id} / {selected_session} / "
            f"{current_session.get('split', 'split 미상')} / {current_session.get('filename', 'filename 미상')}"
        )
    else:
        st.info("현재 선택한 회기의 원본 라벨 정보를 찾지 못했습니다.")

    st.caption("주의: 위 값은 모델 출력 기반 참고값이며, 임상 진단 또는 표준화 검사 점수로 단정하지 않습니다.")

    st.divider()

    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.markdown("#### 세부 요인 점수")
        fig = px.bar(
            factor_df.sort_values("점수", ascending=True),
            x="점수",
            y="요인",
            color="카테고리",
            orientation="h",
            range_x=[0, 3],
            height=680,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### 주요 요인 표")
        st.dataframe(
            factor_df.sort_values(["점수", "카테고리"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 안전 확인")
        if factors.get("suicidal", 0) > 0:
            st.error("자해/자살 관련 발화 가능성이 표시되었습니다. 상담사가 별도 안전 평가를 수행해야 합니다.")
        else:
            st.success("현재 mock 분석 결과에서는 자해/자살 관련 라벨이 0입니다.")

    st.divider()

    trend = pd.DataFrame(
        {
            "회기": ["1회기", "2회기", "3회기", "현재"],
            "우울": [2.8, 2.6, 2.4, classification.get("depression", 0) * 3],
            "불안": [2.7, 2.5, 2.4, classification.get("anxiety", 0) * 3],
            "수면문제": [3.0, 3.0, 3.0, factors.get("sleep_disturbance", 0)],
            "피로감": [2.2, 2.6, 3.0, factors.get("fatigue", 0)],
        }
    )

    st.markdown("#### 회기별 추이 예시")
    fig_trend = px.line(
        trend,
        x="회기",
        y=["우울", "불안", "수면문제", "피로감"],
        markers=True,
        height=420,
    )
    st.plotly_chart(fig_trend, use_container_width=True)


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
