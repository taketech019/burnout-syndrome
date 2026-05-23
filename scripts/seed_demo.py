"""scripts/seed_demo.py — 데모 내담자 + 회기 1건 시드.

실행: `python scripts/seed_demo.py`
환자 데이터는 src/storage.py JSON 파일에 저장. 분석 결과는 비워둔 상태로 시드 —
회기 분석 페이지에서 "분석 실행"을 누르면 채워짐. (Gemma 호출이 필요해서 시드에 미포함)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import add_patient, add_session, list_patients


DEMO_TRANSCRIPT = """상담사: 안녕하세요. 오늘은 어떻게 지내셨어요?
내담자: 지난주 회기 이후로 잠을 거의 못 잤어요. 새벽 3-4시까지 깨어 있고, 일어나도 피곤해요.
내담자: 직장에서도 자꾸 멍해지고, 동료들이 물어봐도 대답을 못 하겠어요.
상담사: 그렇군요. 식사는 어떠셨어요?
내담자: 입맛이 없어서 거의 굶다가, 밤에 폭식하고 자책해요.
내담자: 가끔은 그냥 죽고 싶다는 생각도 들어요. 살아도 의미가 없어 보여서요.
상담사: 자살에 대한 구체적 계획이 있으신가요?
내담자: 아직 계획까지는 아닌데, 그런 생각이 자꾸 들어요. 혼자 있을 때 더 심해요.
상담사: 그 생각이 들 때 어떻게 대처하시나요?
내담자: 술을 마시면 잠시 잊을 수 있어요. 그래서 거의 매일 마셔요.
내담자: 친구들도 안 만난 지 한 달 넘었어요. 부모님과는 자주 다투고요.
상담사: 알겠습니다. 이번 회기에서 우리가 다룰 우선순위를 정해볼까요?
상담사: 우선 자살 사고에 대한 안전 계약을 함께 작성하고, 인지행동치료 기반의 사고 기록지를 시작해봅시다.
상담사: 매일 자기 전에 10분 동안 호흡 명상을 해보시고, 술 대신 자기 전 산책을 시도해보세요.
내담자: 네, 한 번 해볼게요. 솔직히 도움이 필요해요."""


def main() -> None:
    existing = {p["alias"] for p in list_patients()}
    if "데모A" not in existing:
        p = add_patient(
            alias="데모A",
            gender="여성",
            age=32,
            region="서울",
            note="MVP 시연용 — 우울 + 자살 사고 + 음주 동반",
        )
        s = add_session(
            patient_id=p["id"],
            session_date="2026-05-24",
            transcript=DEMO_TRANSCRIPT,
        )
        print(f"데모A 시드 완료. patient_id={p['id']}, session_id={s['id']}")
    else:
        print("데모A 이미 존재 — 시드 skip")

    if "데모B" not in existing:
        p = add_patient(
            alias="데모B",
            gender="남성",
            age=45,
            region="부산",
            note="MVP 시연용 — 정상군 (스트레스 호소 정도)",
        )
        print(f"데모B 시드 완료. patient_id={p['id']}")
    else:
        print("데모B 이미 존재 — 시드 skip")

    print("총 내담자:", len(list_patients()))


if __name__ == "__main__":
    main()
