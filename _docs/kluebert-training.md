# KlueBERT anxiety 모델 재학습 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KlueBERT anxiety 모델이 "어떤 입력이든 항상 1"을 반환하는 문제를 재학습으로 해결하고, 다양한 입력에 대해 변별력 있는 예측을 내도록 한다.

**Architecture:** 회귀(MSE→`closest_integer`) 구조를 4-class CrossEntropy 분류로 전환. 클래스 가중치, stratified split(이미 분리된 Validation 폴더 활용), validation 기반 best-model 선택, early stopping, 라벨 클리닝을 적용. 학습 전 **Phase 0 sanity check**로 로컬 가중치 자체가 망가졌는지 먼저 확인(통과 시 학습 진행, 실패 시 API 버그로 판단해 중단).

**Tech Stack:** Python 3.11, transformers, torch(CUDA), datasets, scikit-learn, pandas. 학습 환경은 **개인 데스크탑 PC, NVIDIA GTX 1070 Ti(VRAM 8GB, Pascal)**.

**환경 메모(GTX 1070 Ti):**
- 8GB VRAM이면 BERT-base + batch=16 + max_len=256 충분히 수용. 활성화/그라디언트/optimizer state 합산 ~4GB 수준.
- **FP16 비사용**: 소비자용 Pascal(GP104)은 FP16 처리량이 FP32의 1/64 — `fp16=True`는 메모리는 줄지만 속도는 오히려 더 느려진다. `fp16=False` 유지.
- 예상 학습 시간: ~520 train sample × 15 epoch(early stopping으로 보통 5~10에서 중단) ≈ **10~20분 내외**.

---

## Context

`src/classifier.py`는 외부 API(`KLUEBERT_ENDPOINT_URL`)로 추론을 호출하며, 응답을 0/1 이진값으로 변환한다. 현재 사용자가 보고한 문제는 "어떤 input을 넣어도 항상 1 반환". 사용자는 API 응답만 확인했으며 로컬 가중치(`ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_anxiety/`) 직접 추론은 미검증 상태다.

원본 학습 노트북(`kluebert_train.ipynb`)의 잠재 결함을 종합 분석한 결과:
1. **회귀(MSE) 구조** — 라벨 불균형(label 1이 anxiety+normal의 31.4%) 하에 예측이 평균값으로 수렴하는 고전적 collapse. `closest_integer`로 라운딩하면 평균 예측이 1로 고정되기 쉽다.
2. **`load_best_model_at_end` 부재** — 100 epochs를 돌고 **마지막** 체크포인트를 저장. 후반부 과적합/발산 상태가 그대로 배포됐을 가능성.
3. **Early stopping 부재**.
4. **클래스 가중치 부재**.
5. **라벨 anomaly 미정제** — 데이터에 label 4, 5 같은 정의 외 값 존재.

학습 데이터는 `data/references/심리상담데이터/Training/02.라벨링데이터/`(zip.part0)와 `Validation/02.라벨링데이터/`(zip.part0)에 분리되어 있다. 이미 별도 Validation 폴더가 존재하므로 그대로 held-out test set으로 사용한다.

**범위:** anxiety 모델만 재학습(검증용). 성공 시 동일 절차로 depression/addiction 별도 진행.

**Out of scope:**
- 학습된 가중치의 API 서버 업로드/재배포(사용자 수동 처리)
- `src/classifier.py`의 4-class→0/1 매핑 변경(API 응답 스키마 확정 후 별도)
- depression/addiction 모델

## Acceptance Gates

ML 학습이므로 일반 TDD 대신 수치적 합격 기준을 사용한다:

- **Gate A (Phase 0, gating):** 기존 `trained_model_kluebert_anxiety`를 로컬에서 로드해 5개 다양한 입력으로 추론. 모두 1을 반환하면 가중치 문제 확정 → 학습 진행. 다양한 예측이 나오면 API 버그이므로 학습 작업 중단하고 사용자에게 보고.
- **Gate B (no-collapse):** v2 모델이 의도적으로 다양한 입력(평온/불안/공포/일상/무관 텍스트 5개) 중 **최소 3개의 서로 다른 클래스**를 예측한다.
- **Gate C (macro F1):** Validation set 4-class **macro F1 ≥ 0.40** (random baseline 0.25 대비 의미 있는 개선).
- **Gate D (per-class recall):** Validation set에서 **각 클래스(0,1,2,3) recall ≥ 0.15** — 특정 클래스 완전 무시 방지.

## File Structure

```
ai-model/kluebert/1.모델소스코드/py 소스코드/
  kluebert_train.py                  # 기존 (보존)
  kluebert_run.py                    # 기존 (보존)
  kluebert_sanity_check.py           # 신규: Phase 0 로컬 추론 검증
  extract_data.py                    # 신규: zip.part0 → JSON 추출 일회성
  kluebert_train_v2.py               # 신규: 4-class 분류 학습 스크립트
  kluebert_eval_v2.py                # 신규: v2 모델 평가 + Gate B/C/D 체크

ai-model/kluebert/2.AI학습모델파일/   # gitignored
  trained_model_kluebert_anxiety/    # 기존 (보존, 비교용)
  trained_model_kluebert_anxiety_v2/ # 신규: 학습 산출물

data/raw/심리상담데이터/             # gitignored
  Training/anxiety/*.json
  Training/normal/*.json
  Validation/anxiety/*.json
  Validation/normal/*.json
```

`.gitignore` 확인 완료: `data/raw/`와 `ai-model/kluebert/2.AI학습모델파일/`, `**/trained_model_kluebert_*/` 모두 ignored. 모델 가중치는 HF Hub 배포 정책을 따른다.

---

## Task 1: Phase 0 — 로컬 가중치 sanity check (gating)

**Files:**
- Create: `ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_sanity_check.py`

기존 학습된 anxiety 모델을 외부 API 없이 직접 로드해 동작을 확인. 모든 후속 작업의 전제.

- [ ] **Step 1: sanity check 스크립트 작성**

`ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_sanity_check.py` 생성:

```python
"""Phase 0: 로컬 KlueBERT anxiety 가중치가 실제로 'always 1' 문제를 보이는지 확인.
실행: python ai-model/kluebert/1.모델소스코드/py\ 소스코드/kluebert_sanity_check.py
"""
from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

ROOT = Path(__file__).resolve().parents[3]  # repo root
MODEL_DIR = ROOT / "ai-model" / "kluebert" / "2.AI학습모델파일" / "trained_model_kluebert_anxiety"

class CustomBertForSequenceRegression(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.regressor = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs[0][:, 0, :]
        return self.regressor(pooled)

def closest_integer(p):
    return min(max(round(p), 0), 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR))
model = CustomBertForSequenceRegression.from_pretrained(str(MODEL_DIR)).to(device).eval()

probes = [
    "오늘 날씨가 참 좋고 기분이 상쾌합니다.",
    "심장이 두근거리고 숨이 막혀요. 너무 불안해요.",
    "그냥 그래요. 별로 할 말은 없어요.",
    "사람들 많은 곳이 너무 무서워서 외출을 못해요.",
    "안녕하세요 반갑습니다.",
]

print(f"device: {device}")
print(f"Loaded from: {MODEL_DIR}")
raws, ints = [], []
for p in probes:
    inputs = tokenizer(p, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    raw = out.squeeze().item()
    pred = closest_integer(raw)
    raws.append(raw); ints.append(pred)
    print(f"  raw={raw:+.4f}  pred={pred}  | {p[:40]}")

print(f"\nraw range: [{min(raws):.3f}, {max(raws):.3f}]  spread={max(raws)-min(raws):.3f}")
print(f"unique predictions: {sorted(set(ints))}")
if len(set(ints)) == 1 and ints[0] == 1:
    print("\n>>> Gate A PASSED: 로컬 가중치가 항상 1 반환. 재학습 진행.")
else:
    print("\n>>> Gate A FAILED: 로컬 모델은 다양한 예측. API 서버 측 버그일 가능성. 재학습 보류.")
```

- [ ] **Step 2: 실행 및 결과 기록**

Run: `python "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_sanity_check.py"`
Expected: 5개 입력 모두 `pred=1`이면 Gate A 통과 → Task 2로 진행. 다양한 예측이 나오면 사용자에게 즉시 보고하고 학습 작업 중단.

- [ ] **Step 3: 커밋**

```bash
git add "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_sanity_check.py"
git commit -m "chore: add kluebert local sanity check (Phase 0)"
```

---

## Task 2: 학습 데이터 압축 해제 + 라벨 분포 확인

**Files:**
- Create: `ai-model/kluebert/1.모델소스코드/py 소스코드/extract_data.py`
- Output: `data/raw/심리상담데이터/{Training,Validation}/{anxiety,normal}/*.json`

`*.zip.part0` 파일이 단일 zip 본체일 가능성이 높지만, multi-part일 수도 있다. 스크립트는 zipfile 시도 → 실패 시 사용자에게 수동 처리 요청.

- [ ] **Step 1: 추출 스크립트 작성**

`ai-model/kluebert/1.모델소스코드/py 소스코드/extract_data.py`:

```python
"""anxiety + normal zip.part0 파일을 data/raw/심리상담데이터/ 하위로 추출.
실행: python ai-model/kluebert/1.모델소스코드/py\ 소스코드/extract_data.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "references" / "심리상담데이터"
DST = ROOT / "data" / "raw" / "심리상담데이터"

# (zip 키워드, 출력 카테고리) 매핑. anxiety만 학습 대상.
MAP = {
    "Training": [("불안장애", "anxiety"), ("일반군", "normal")],
    "Validation": [("불안장애", "anxiety"), ("일반군", "normal")],
}

def extract():
    for split, items in MAP.items():
        for kw, cat in items:
            out = DST / split / cat
            out.mkdir(parents=True, exist_ok=True)
            src_dir = SRC / split / "02.라벨링데이터"
            zips = [p for p in src_dir.iterdir() if kw in p.name and (p.suffix == ".zip" or p.name.endswith(".zip.part0"))]
            for z in zips:
                try:
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(out)
                    print(f"OK  {z.name} -> {out}")
                except zipfile.BadZipFile:
                    print(f"FAIL {z.name} — multi-part 가능성. 수동 추출 필요.")

if __name__ == "__main__":
    extract()
    total = sum(1 for _ in DST.rglob("*.json"))
    print(f"\nTotal extracted JSON: {total}")
```

- [ ] **Step 2: 추출 실행**

Run: `python "ai-model/kluebert/1.모델소스코드/py 소스코드/extract_data.py"`
Expected: anxiety + normal JSON 약 649개 추출. FAIL 출력 시 사용자에게 수동 추출 요청.

- [ ] **Step 3: 라벨 분포 검증**

빠른 인라인 확인:

```python
# python 인터프리터에서 1회 실행
import json, collections
from pathlib import Path
ROOT = Path(".") / "data" / "raw" / "심리상담데이터"
counts = collections.Counter()
for split in ("Training", "Validation"):
    for cat in ("anxiety", "normal"):
        for p in (ROOT / split / cat).rglob("*.json"):
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            counts[(split, j.get("anxiety"))] += 1
print(counts)
```

Expected: Training label 0/1/2/3 모두 ≥10건. label 4,5 같은 anomaly가 소수 존재. 분포를 plan 실행 로그에 기록.

- [ ] **Step 4: 커밋**

```bash
git add "ai-model/kluebert/1.모델소스코드/py 소스코드/extract_data.py"
git commit -m "chore: add data extraction script for kluebert retrain"
```

---

## Task 3: 4-class 분류 학습 스크립트

**Files:**
- Create: `ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_train_v2.py`

기존 `kluebert_train.py`의 회귀 구조를 4-class CrossEntropy로 전환. 노트북 의존(Google Drive 마운트, 셀 분할) 제거하고 단일 스크립트로 작성.

- [ ] **Step 1: 학습 스크립트 작성**

`ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_train_v2.py`:

```python
"""KlueBERT anxiety v2: 4-class 분류 학습 스크립트 (GTX 1070 Ti, FP32).
실행: python ai-model/kluebert/1.모델소스코드/py\ 소스코드/kluebert_train_v2.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "raw" / "심리상담데이터"
OUT  = ROOT / "ai-model" / "kluebert" / "2.AI학습모델파일" / "trained_model_kluebert_anxiety_v2"
DISEASE = "anxiety"
NUM_LABELS = 4              # 0,1,2,3
MAX_LEN = 256
BATCH = 16                  # GTX 1070 Ti 8GB 충분히 수용
EPOCHS = 15                 # EarlyStopping이 더 일찍 끊을 것
LR = 2e-5
MODEL_NAME = "klue/bert-base"
SEED = 42

def load_split(split: str) -> pd.DataFrame:
    rows = []
    for cat in ("anxiety", "normal"):
        for p in (DATA / split / cat).rglob("*.json"):
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            label = j.get(DISEASE)
            if label is None or label not in (0, 1, 2, 3):
                continue  # anomaly(4,5) / 결측 필터
            text = "".join(
                f'{t.get("paragraph_speaker","")}: {t.get("paragraph_text","")}\n'
                for t in (j.get("paragraph") or [])
            )
            if not text.strip():
                continue
            rows.append({"input": text, "label": int(label)})
    return pd.DataFrame(rows)

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"CUDA available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_df = load_split("Training")
    val_df   = load_split("Validation")
    print(f"train={len(train_df)} val={len(val_df)}")
    print("train label dist:", train_df["label"].value_counts().sort_index().to_dict())
    print("val   label dist:", val_df["label"].value_counts().sort_index().to_dict())

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    def tok(ex): return tokenizer(ex["input"], padding="max_length", truncation=True, max_length=MAX_LEN)
    train_ds = Dataset.from_pandas(train_df).map(tok, batched=True).remove_columns(["input"])
    val_ds   = Dataset.from_pandas(val_df  ).map(tok, batched=True).remove_columns(["input"])
    train_ds.set_format("torch"); val_ds.set_format("torch")

    classes = np.array([0, 1, 2, 3])
    weights = compute_class_weight("balanced", classes=classes, y=train_df["label"].values)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print("class weights:", weights)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": (preds == labels).mean(),
        }

    args = TrainingArguments(
        output_dir=str(OUT / "_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=False,                # Pascal에서 FP16은 오히려 느림 (1/64 throughput)
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))

    # 최종 평가 + classification_report
    pred = trainer.predict(val_ds)
    y_true, y_pred = pred.label_ids, np.argmax(pred.predictions, axis=-1)
    print("\n=== Validation classification report ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 데이터 로드 dry run**

```bash
python -c "
import sys; sys.path.insert(0, 'ai-model/kluebert/1.모델소스코드/py 소스코드')
from kluebert_train_v2 import load_split
df = load_split('Training')
print('train rows:', len(df), 'label dist:', df['label'].value_counts().sort_index().to_dict())
"
```

Expected: 500~600 rows, 4개 라벨 모두 등장.

- [ ] **Step 3: 커밋**

```bash
git add "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_train_v2.py"
git commit -m "feat: add kluebert anxiety v2 training script (4-class, weighted CE)"
```

---

## Task 4: 학습 실행

**Files:** (실행만)

GTX 1070 Ti 환경에서는 빠른 학습이 가능하므로 곧바로 본 학습 진행.

- [ ] **Step 1: CUDA 인식 확인**

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Expected: `cuda: True NVIDIA GeForce GTX 1070 Ti`. False면 torch CUDA 빌드 설치 필요.

- [ ] **Step 2: 본 학습 실행**

Run: `python "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_train_v2.py" 2>&1 | tee ai-model/kluebert/train_v2.log`
Expected:
- 학습 ~10~20분 내 완료.
- EarlyStopping이 5~10 epoch 사이에서 멈춤 (val macro_f1 plateau).
- 최종 `=== Validation classification report ===` 블록이 로그에 남음.
- `trained_model_kluebert_anxiety_v2/` 디렉토리에 `model.safetensors`, `config.json`, `tokenizer_config.json`, `vocab.txt` 생성.

VRAM 부족(`CUDA out of memory`) 발생 시 batch=8로 낮춰 재시도.

- [ ] **Step 3: 학습 로그 보존**

`ai-model/*.log`는 `.gitignore`에 이미 등록되어 있어 자동 무시. 별도 커밋 불필요. 모델 가중치 디렉토리도 ignored.

---

## Task 5: 평가 + Gate B/C/D 자동 검증

**Files:**
- Create: `ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_eval_v2.py`

- [ ] **Step 1: 평가 스크립트 작성**

`ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_eval_v2.py`:

```python
"""v2 모델 평가 + Gate B/C/D 자동 체크."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, recall_score
from transformers import BertTokenizer, BertForSequenceClassification

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "ai-model" / "kluebert" / "2.AI학습모델파일" / "trained_model_kluebert_anxiety_v2"
DATA = ROOT / "data" / "raw" / "심리상담데이터"

def load_val():
    texts, labels = [], []
    for cat in ("anxiety", "normal"):
        for p in (DATA / "Validation" / cat).rglob("*.json"):
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            lbl = j.get("anxiety")
            if lbl not in (0, 1, 2, 3): continue
            text = "".join(f'{t.get("paragraph_speaker","")}: {t.get("paragraph_text","")}\n'
                           for t in (j.get("paragraph") or []))
            if text.strip():
                texts.append(text); labels.append(int(lbl))
    return texts, np.array(labels)

def predict_batch(texts, tokenizer, model, device):
    preds = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.append(int(torch.argmax(logits, dim=-1).item()))
    return np.array(preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device).eval()

    # Gate B: no-collapse probe
    probes = [
        "오늘 날씨가 참 좋고 기분이 상쾌합니다.",
        "심장이 두근거리고 숨이 막혀요. 너무 불안해요.",
        "그냥 그래요. 별로 할 말은 없어요.",
        "사람들 많은 곳이 너무 무서워서 외출을 못해요.",
        "안녕하세요 반갑습니다.",
    ]
    probe_preds = predict_batch(probes, tokenizer, model, device)
    unique = sorted(set(probe_preds.tolist()))
    print(f"[Gate B] probe preds: {probe_preds.tolist()}  unique={unique}")
    gate_b = len(unique) >= 3
    print(f"[Gate B] {'PASS' if gate_b else 'FAIL'} (need >=3 unique classes)")

    # Gate C/D: validation set
    texts, y = load_val()
    yhat = predict_batch(texts, tokenizer, model, device)
    macro_f1 = f1_score(y, yhat, average="macro", zero_division=0)
    per_class_recall = recall_score(y, yhat, labels=[0,1,2,3], average=None, zero_division=0)
    print(f"\n[Gate C] macro_f1 = {macro_f1:.3f} (need >= 0.40)")
    print(f"[Gate D] per-class recall = {dict(zip([0,1,2,3], per_class_recall.round(3)))} (each need >= 0.15)")
    gate_c = macro_f1 >= 0.40
    gate_d = all(r >= 0.15 for r in per_class_recall)
    print(f"[Gate C] {'PASS' if gate_c else 'FAIL'}")
    print(f"[Gate D] {'PASS' if gate_d else 'FAIL'}")

    print("\n=== Full classification report ===")
    print(classification_report(y, yhat, digits=3, zero_division=0))

    print("\n=== Summary ===")
    print(f"  Gate B (no-collapse): {gate_b}")
    print(f"  Gate C (macro F1>=0.40): {gate_c}")
    print(f"  Gate D (per-class recall>=0.15): {gate_d}")
    if gate_b and gate_c and gate_d:
        print(">>> ALL GATES PASS — 모델 배포 가능.")
    else:
        print(">>> 일부 Gate 실패 — 사용자에게 보고 후 하이퍼파라미터/데이터 재검토.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 평가 실행**

Run: `python "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_eval_v2.py"`
Expected: Gate B, C, D 모두 PASS. 어느 하나라도 FAIL이면 결과를 사용자에게 보고하고 다음 단계로 넘어가지 말 것.

- [ ] **Step 3: 커밋**

```bash
git add "ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_eval_v2.py"
git commit -m "feat: add kluebert anxiety v2 evaluation with gate checks"
```

---

## Task 6: 배포 인수인계 노트

학습 산출물(`trained_model_kluebert_anxiety_v2/`)을 외부 API 서버에 업로드하는 작업은 사용자가 수동 처리한다. 다음 사항을 사용자에게 명확히 전달한다.

- [ ] **Step 1: 인수인계 항목 정리 (대화에 출력)**

전달 사항:
1. **새 가중치 위치**: `ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_anxiety_v2/`
2. **출력 스키마 변경**: 1-차원 회귀(0~3 float) → **4-class logits/argmax (0,1,2,3 int)**
3. **API 서버 업데이트 항목**:
   - 모델 클래스: `CustomBertForSequenceRegression` → `BertForSequenceClassification(num_labels=4)`
   - 추론 후처리: `closest_integer(out)` → `argmax(logits)`
   - 응답에서 binary 0/1을 유지하려면 `argmax >= 1 → 1` 매핑(서버 측 또는 추후 `src/classifier.py` 측)
4. **검증 권장**: 서버 업로드 후 `kluebert_sanity_check.py`와 동일한 5개 probe로 외부 API 호출, 다양한 응답 확인.

- [ ] **Step 2: PR 본문에 위 1~4 그대로 적어 머지**

```bash
git push -u origin feature/yoon
gh pr create --title "feat: kluebert anxiety v2 retrain (4-class CE)" --body "(인수인계 1~4 그대로)"
```

---

## Verification (End-to-End)

전체 흐름이 한번에 동작하는지 최종 확인:

1. `python kluebert_sanity_check.py` → Gate A PASS (기존 모델 항상 1 재현)
2. `python extract_data.py` → JSON 파일 약 649개 추출
3. `python kluebert_train_v2.py` → 학습 완료(~15분), `trained_model_kluebert_anxiety_v2/` 생성
4. `python kluebert_eval_v2.py` → Gate B/C/D 모두 PASS

Gate A에서 다양한 예측이 나오면(=local 모델은 멀쩡) Task 2~6를 중단하고 사용자에게 "API 서버 측 버그로 추정됨"을 보고. 이 경우 본 계획은 폐기.

## 위험 요소

- **VRAM 부족**: 가능성 낮지만 batch=16에서 OOM 발생 시 batch=8로 낮춤. max_len을 192로 줄이는 것도 옵션.
- **클래스 0 데이터 부족**: anxiety 폴더에는 label 0이 거의 없고 "normal" 폴더가 label 0 공급원. 추출 시 normal이 누락되면 Gate D 실패 가능 — 추출 결과의 라벨 분포 반드시 확인.
- **Gate C 0.40 미달**: 데이터 품질 한계일 수 있음. 사용자에게 보고하고 임계치 조정 또는 hyperparameter 재시도 결정 (max_len 512, lr 5e-5, epoch 30+ 등).
- **Pascal에서 `fp16=True` 사용 금지**: Pascal(GP104)은 FP16 throughput이 FP32의 1/64 — 오히려 학습이 4~10배 느려진다.
