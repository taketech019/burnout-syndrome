# 우울증 음성 진단보조 시스템 개발 계획

## Context

PHQ-9 기반 이진 분류(정상/우울) 음성 진단보조 시스템을 구축한다.
입력 데이터는 `subject.id` 단위의 JSON + WAV 파일이며, `depression.PHQ-9` 총점을 기준으로 라벨을 생성한다.
기존 `svm.py`의 로깅 포맷(한국어 6섹션)과 데이터 포맷(.pkl 모델, .npz 테스트셋)을 그대로 유지하면서 전체 파이프라인을 신규 구축한다.

### JSON 스키마 (확인됨)

```json
{
  "subject": { "id": "000011", "sex": 1, "age": 48, "group": 1, "date_visited": "2021-08-05" },
  "voice":   { "category_id": 8, "file_name": "..\\..원천데이터\\음성\\8가을문단\\000011_가을문단all.wav" },
  "depression": { "category_name": "우울한 기분", "category_id": 1, "PHQ-9": 6 }
}
```

- `subject.id` → speaker_id (화자독립 분리 키)
- `subject.group` → 사전 이진 그룹 레이블 후보 (검증 필요)
- `depression.PHQ-9` → 이진 라벨 생성 기준 (< 10: 0, ≥ 10: 1)
- `voice.file_name` → Windows 상대경로 .wav (한국어 경로 포함)
- `voice.category_id` → 음성 과제 유형 (가을문단 낭독 등 다수 존재 가능)

---

## 프로젝트 디렉토리 구조

```
PR_BOAZ_2nd/
├── svm.py                  # 기존 파일 — 수정 없이 보존
├── config.py               # 경로·파라미터 중앙관리
├── data_loader.py          # JSON 파싱, 오디오 경로 수집
├── feature_extractor.py    # librosa 기반 음향 특징 추출
├── preprocess.py           # 화자독립 분리, .npz 저장
├── train.py                # GridSearchCV SVM 학습, .pkl 저장
├── evaluate.py             # 동적 혼동행렬 + 기존 로그 포맷 유지
├── pipeline.py             # 전체 파이프라인 진입점
├── artifacts/
│   ├── svm.pkl             # 학습된 Pipeline (StandardScaler + SVC)
│   ├── testSet.npz         # X_test, y_test, ids (svm.py 호환)
│   ├── trainSet.npz        # X_train, y_train, ids, groups
│   └── features.npz        # 특징 추출 캐시
├── data/
│   ├── audio/              # .wav 파일
│   └── labels/             # JSON 파일들
└── Logs.log
```

---

## 구현 계획 (파일별)

### 1. `config.py`

```python
PHQ9_THRESHOLD = 10        # 이진 분류 기준점
SR             = 16000     # 오디오 샘플링 레이트
N_MFCC         = 13
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
# 경로: os.getenv 우선, 기본값은 상대경로
MODEL_PATH     = "./artifacts/svm.pkl"
TEST_NPZ_PATH  = "./artifacts/testSet.npz"
TRAIN_NPZ_PATH = "./artifacts/trainSet.npz"
FEAT_CACHE     = "./artifacts/features.npz"
```

의존성: 없음

---

### 2. `data_loader.py`

**핵심 함수:**

```python
def load_all_labels(json_dir: str) -> pd.DataFrame:
    # JSON 디렉토리 전체 스캔
    # 각 파일에서 subject.id, subject.group, depression.PHQ-9,
    #   voice.file_name, voice.category_id 추출
    # label = 1 if PHQ-9 >= PHQ9_THRESHOLD else 0
    # 반환: DataFrame [audio_id, speaker_id, group, phq9, label, audio_path, category_id]

def resolve_audio_path(raw_path: str, audio_root: str) -> str:
    # Windows 백슬래시 정규화 (os.path.normpath)
    # 상대경로를 audio_root 기준 절대경로로 변환
    # 파일 존재 여부 확인 후 경로 반환 (없으면 None)
```

**주요 처리:**
- `file_name`의 `..\\..\\원천데이터\\...` Windows 경로를 실제 경로로 변환
- `subject.group`과 PHQ-9 기반 라벨 일치 여부 로그 출력 (불일치 시 경고)
- category_id별 샘플 수 로그 출력 (어떤 음성 과제가 몇 개인지)

의존성: `json`, `os`, `glob`, `pandas`, `logging`, `config`

---

### 3. `feature_extractor.py`

**음향 특징 세트 (~155차원):**

| 그룹 | 특징 | 차원 | 근거 |
|------|------|------|------|
| MFCCs | 13개 × 6 통계치 (mean/std/min/max/skew/kurtosis) | 78 | 우울증 연구 핵심 특징 |
| Delta MFCC | 13개 × (mean+std) | 26 | 시간적 변화 포착 |
| Delta-delta MFCC | 13개 × (mean+std) | 26 | 가속도 성분 |
| Pitch/F0 | pyin: mean/std/min/max/range + voiced_ratio | 6 | 우울 시 피치 범위 감소 |
| 에너지 | RMS: mean/std, Log energy: mean/std | 4 | 발화 강도 저하 |
| 스펙트럼 | ZCR/centroid/rolloff/bandwidth/onset 각 mean+std | 10 | 음성 질 변화 |
| 발화속도 대리 | voiced_ratio, pause_ratio | 2 | 느린 발화 패턴 |
| 음성질 | jitter/shimmer/HNR (parselmouth) | 3 | 성대 진동 불규칙성 |

**핵심 함수:**

```python
def extract_all_features(y: np.ndarray, sr: int) -> np.ndarray:
    # 각 그룹 함수 호출 → concatenate → 1D 벡터
    # NaN 감지 시 0으로 대체 + 경고 로그

def extract_dataset_features(
    df: pd.DataFrame, audio_root: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 반환: X, y, ids(audio_id), groups(speaker_id)
    # 10개 단위 진행률 로그, 실패 파일 스킵 후 로그
```

**parselmouth 선택적 사용:**
```python
try:
    import parselmouth
    # jitter/shimmer/HNR 계산
except ImportError:
    return np.zeros(3)  # 경고 로그만 출력
```

의존성: `librosa`, `numpy`, `scipy.stats`, `logging`; 선택: `parselmouth`

---

### 4. `preprocess.py`

**화자독립 분리 (핵심):**

```python
from sklearn.model_selection import GroupShuffleSplit

def speaker_independent_split(X, y, ids, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    # 화자 중복 없음 보장
    assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], ...
```

**svm.py 호환 저장:**
```python
def save_test_npz(X_test, y_test, ids_test, path):
    # 키 이름 고정: X_test, y_test, ids  ← svm.py 28~30줄과 정확히 일치
    np.savez(path, X_test=X_test, y_test=y_test, ids=ids_test)
```

분리 후 클래스 분포 + 화자 수 로그 출력.

의존성: `numpy`, `sklearn.model_selection`, `logging`, `config`

---

### 5. `train.py`

**Pipeline (StandardScaler 내장 → svm.py 무수정 호환):**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, class_weight='balanced'))
])

param_grid = {
    'svc__C':      [0.01, 0.1, 1, 10, 100],
    'svc__kernel': ['rbf', 'linear'],
    'svc__gamma':  ['scale', 0.001, 0.01, 0.1],
}
```

**CV 전략 (화자독립):**
```python
from sklearn.model_selection import GroupKFold, GridSearchCV

cv = GroupKFold(n_splits=5)
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=cv, scoring='balanced_accuracy',  # 클래스 불균형 대응
    n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train, groups=groups_train)
```

최적 파라미터 + CV 점수 로그 출력 후 `joblib.dump()` 저장.

**클래스 불균형 전략:**
- 1차: `class_weight='balanced'` (SVM 기본 대응)
- 불균형 4:1 초과 시: `imbalanced-learn`의 SMOTE를 파이프라인 앞단에 추가

의존성: `sklearn`, `joblib`, `numpy`, `logging`, `config`

---

### 6. `evaluate.py`

기존 `svm.py`의 6섹션 한국어 로그 포맷 유지, 혼동행렬 동적 계산:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

logger.debug("1) 시작 timestamp")
logger.debug(f"{t_now}\n")
logger.debug("\n2) 실행 명령어")
logger.debug("python svm.py")
logger.debug("\n3) 문항 개별 결과값")
logger.debug("데이터ID 모델예측값 GT값")
for i in range(len(X_test)):
    logger.debug(f"{testIds[i]} {pred[i]} {y_test[i]}")
logger.debug("\n4) 계산할 때 사용된 값")
logger.debug("클래스ID TP FP TN FN")
logger.debug(f"       0 {tn} {fp} {tp} {fn}")   # 동적 계산
logger.debug(f"       1 {tp} {fn} {tn} {fp}")   # 동적 계산
logger.debug("\n5) 최종 결과값\n")
logger.debug(f"{classification_report(y_test, pred, digits=4)}")
logger.debug("6) 종료 timestamp")
logger.debug(f"{t_now}")
```

argparse로 `--model-path`, `--test-npz`, `--log-path` 지원 (기본값은 config).

의존성: `numpy`, `joblib`, `sklearn.metrics`, `logging`, `datetime`, `argparse`, `config`

---

### 7. `pipeline.py`

전체 파이프라인 진입점. `--skip-extraction` / `--skip-training` 플래그로 캐시 활용:

```
data/labels/*.json
    │
    ▼ data_loader.load_all_labels()
DataFrame [audio_id, speaker_id, phq9, label, audio_path, category_id]
    │
    ▼ feature_extractor.extract_dataset_features()
X, y, ids, groups  ──▶  artifacts/features.npz (캐시)
    │
    ▼ preprocess.speaker_independent_split()
    ├──▶ artifacts/trainSet.npz
    └──▶ artifacts/testSet.npz
              │
              ▼ train.train_with_grid_search()
         artifacts/svm.pkl
              │
              ▼ evaluate.run_evaluation()
         Logs.log
```

의존성: 모든 모듈, `argparse`

---

## 구현 순서

| 단계 | 파일 | 검증 방법 |
|------|------|-----------|
| 1 | `config.py` | import 확인 |
| 2 | `data_loader.py` | 단일 JSON 파싱, 경로 해석 확인 |
| 3 | `feature_extractor.py` | 단일 WAV 파일 특징 벡터 shape 확인 |
| 4 | `preprocess.py` | 화자 중복 없음 assert 통과 |
| 5 | `train.py` | GridSearchCV 완료, .pkl 저장 |
| 6 | `evaluate.py` | Logs.log 6섹션 포맷 확인 |
| 7 | `pipeline.py` | end-to-end `python pipeline.py` 실행 |

---

## 검증 체크리스트

```bash
# 전체 파이프라인 실행
python pipeline.py --label-dir ./data/labels --audio-root ./data/audio

# 기존 svm.py 호환성 확인 (수정 없이 실행)
python svm.py
```

- [ ] `testSet.npz` 키: `X_test`, `y_test`, `ids` (svm.py 28~30줄 호환)
- [ ] `svm.pkl` Pipeline이 `predict(X_raw)` 호출 시 내부 스케일링 자동 적용
- [ ] train/test 화자 집합 교집합 = 0
- [ ] `Logs.log` 6섹션 한국어 포맷 정상 출력
- [ ] `classification_report` digits=4로 출력

---

## 필요 패키지

```
numpy>=1.21
pandas>=1.3
librosa>=0.10
scikit-learn>=1.2
joblib>=1.2
scipy>=1.7
praat-parselmouth>=0.4   # 선택: jitter/shimmer/HNR
imbalanced-learn>=0.11   # 선택: SMOTE (심한 불균형 시)
```
