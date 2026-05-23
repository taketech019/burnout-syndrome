"""F5 내담자 관리 페이지 — 목록, 신규 등록, JSON 내보내기, 삭제."""
import json

import streamlit as st

from src.storage import (
    add_patient, delete_patient, export_patient_json, list_patients, list_sessions
)


def render() -> None:
    st.title("내담자 관리")
    st.caption("MVP는 데모 데이터 전용. 실제 식별정보(실명·연락처)를 입력하지 마세요.")

    tab_list, tab_add = st.tabs(["목록", "신규 등록"])

    with tab_list:
        patients = list_patients()
        if not patients:
            st.info("등록된 내담자가 없습니다. '신규 등록' 탭에서 추가하세요.")
            return

        for p in patients:
            with st.expander(f"**{p['alias']}** — {p['gender']} / {p['age']}세 / {p['region']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"ID: `{p['id']}`")
                    st.write(f"등록일: {p['created_at']}")
                    if p.get("note"):
                        st.write(f"메모: {p['note']}")
                    n_sessions = len(list_sessions(p["id"]))
                    st.write(f"회기 수: **{n_sessions}회**")
                with col2:
                    export_data = export_patient_json(p["id"])
                    st.download_button(
                        "JSON 내보내기",
                        data=json.dumps(export_data, ensure_ascii=False, indent=2),
                        file_name=f"{p['alias']}_{p['id']}.json",
                        mime="application/json",
                        key=f"export_{p['id']}",
                    )
                with col3:
                    if st.button("삭제", key=f"del_{p['id']}", type="secondary"):
                        delete_patient(p["id"])
                        st.rerun()

    with tab_add:
        with st.form("add_patient_form"):
            alias = st.text_input("익명 식별자 (alias)", placeholder="예: 내담자A, P-001")
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("성별", ["여성", "남성", "기타"])
            with col2:
                age = st.number_input("연령", min_value=10, max_value=100, value=30)
            with col3:
                region = st.text_input("지역", placeholder="예: 서울")
            note = st.text_area("메모 (선택)", height=80)

            if st.form_submit_button("등록"):
                if not alias.strip():
                    st.error("alias는 필수입니다.")
                else:
                    p = add_patient(alias, gender, age, region, note)
                    st.success(f"등록 완료: {p['alias']} (ID: {p['id']})")
                    st.rerun()
