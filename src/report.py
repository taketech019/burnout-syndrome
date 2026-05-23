"""src/report.py — F3: 요약 보고서 출력 (.md / .docx / .pdf).

KoAlpaca 4섹션 요약 + 차트 이미지 포함. 다운로드 가능한 단일 파일 생성.
다음 회기 계획은 PRD §F3 추천 영역 — Gemini 또는 사용자가 직접 입력.
"""
import io
from datetime import datetime
from typing import Optional


SECTION_LABELS = {
    "symptoms": "주요 증상",
    "risk_factors": "위험 요인",
    "improvement_factors": "개선 요인",
    "intervention_factors": "상담사 개입 요인",
}


def _build_markdown(
    patient: dict,
    session: dict,
    summary: dict,
    next_plan: str = "",
) -> str:
    """4섹션 요약 + 회기 메타 → Markdown."""
    lines = [
        f"# 상담 회기 요약 보고서",
        "",
        f"- **내담자**: {patient.get('alias', '-')}",
        f"- **성별/연령/지역**: {patient.get('gender', '-')} / {patient.get('age', '-')}세 / {patient.get('region', '-')}",
        f"- **회기일**: {session.get('session_date', '-')}",
        f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for key, label in SECTION_LABELS.items():
        body = summary.get(key, "").strip() or "(추출 실패 또는 미작성)"
        lines.append(f"## {label}")
        lines.append("")
        lines.append(body)
        lines.append("")
    if next_plan.strip():
        lines.append("## 다음 회기 계획")
        lines.append("")
        lines.append(next_plan.strip())
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> *본 보고서는 AI 보조 도구로 생성되었으며 상담사의 임상적 판단을 대체하지 않습니다.*")
    return "\n".join(lines)


# ── 출력 포맷별 빌더 ──────────────────────────────────────────────────────────

def build_md(patient: dict, session: dict, summary: dict, next_plan: str = "") -> bytes:
    return _build_markdown(patient, session, summary, next_plan).encode("utf-8")


def build_docx(
    patient: dict,
    session: dict,
    summary: dict,
    next_plan: str = "",
    chart_pngs: Optional[list[bytes]] = None,
) -> bytes:
    """python-docx로 .docx 생성. 차트 PNG bytes 리스트가 있으면 본문 끝에 삽입."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading("상담 회기 요약 보고서", level=0)

    doc.add_paragraph(f"내담자: {patient.get('alias', '-')}")
    doc.add_paragraph(
        f"성별/연령/지역: {patient.get('gender', '-')} / "
        f"{patient.get('age', '-')}세 / {patient.get('region', '-')}"
    )
    doc.add_paragraph(f"회기일: {session.get('session_date', '-')}")
    doc.add_paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    for key, label in SECTION_LABELS.items():
        doc.add_heading(label, level=1)
        body = summary.get(key, "").strip() or "(추출 실패 또는 미작성)"
        doc.add_paragraph(body)

    if next_plan.strip():
        doc.add_heading("다음 회기 계획", level=1)
        doc.add_paragraph(next_plan.strip())

    if chart_pngs:
        doc.add_heading("회기 분석 차트", level=1)
        for png in chart_pngs:
            doc.add_picture(io.BytesIO(png), width=Inches(6))

    doc.add_paragraph(
        "본 보고서는 AI 보조 도구로 생성되었으며 상담사의 임상적 판단을 대체하지 않습니다."
    ).italic = True

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_pdf(
    patient: dict,
    session: dict,
    summary: dict,
    next_plan: str = "",
) -> Optional[bytes]:
    """weasyprint로 .pdf 생성. 실패 시 None 반환 — 호출자가 .md/.docx로 fallback.

    weasyprint는 GTK 의존성이 무거워 일부 환경에서 import 실패할 수 있음.
    """
    try:
        import markdown as md_lib
        from weasyprint import HTML
    except Exception:
        return None

    md_text = _build_markdown(patient, session, summary, next_plan)
    html_body = md_lib.markdown(md_text, extensions=["extra"])
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: 'Malgun Gothic', 'NanumGothic', sans-serif; line-height: 1.6; padding: 2cm; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ color: #2c5282; margin-top: 1.5em; }}
  blockquote {{ color: #666; font-style: italic; }}
</style>
</head><body>{html_body}</body></html>"""
    try:
        return HTML(string=html).write_pdf()
    except Exception:
        return None
