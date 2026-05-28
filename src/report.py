import os
import re
from datetime import datetime
from html import escape as html_escape
from io import BytesIO
from typing import Any, Dict


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


def make_docx_report_bytes(
    report_text: str,
    client_id: str = "",
    session_label: str = "",
    created_at: datetime | None = None,
) -> bytes | None:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except Exception:
        return None

    created_at = created_at or datetime.now()
    doc = Document()
    title = doc.add_heading("상담 요약 보고서", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f"작성일: {created_at.strftime('%Y.%m.%d')}")

    if client_id or session_label:
        doc.add_paragraph(f"내담자: {client_id or '-'} / 회기: {session_label or '-'}")

    doc.add_paragraph("")

    for line in str(report_text or "").split("\n"):
        stripped = line.strip()

        if not stripped:
            doc.add_paragraph("")
        elif stripped.startswith("[") and stripped.endswith("]"):
            doc.add_heading(stripped.strip("[]"), level=1)
        elif re.match(r"^\d+\.\s", stripped):
            doc.add_heading(stripped, level=2)
        elif stripped.startswith("## "):
            doc.add_heading(stripped.replace("## ", "", 1), level=1)
        elif stripped.startswith("# "):
            doc.add_heading(stripped.replace("# ", "", 1), level=0)
        else:
            doc.add_paragraph(stripped)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def make_pdf_report_bytes(
    report_text: str,
    client_id: str = "",
    session_label: str = "",
    created_at: datetime | None = None,
) -> bytes | None:
    try:
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
        from weasyprint import HTML
    except Exception:
        return None

    created_at = created_at or datetime.now()

    def render_line(line: str) -> str:
        stripped = line.strip()

        if not stripped:
            return "<div class='spacer'></div>"
        if stripped.startswith("[") and stripped.endswith("]"):
            return f"<h2>{html_escape(stripped.strip('[]'))}</h2>"
        if re.match(r"^\d+\.\s", stripped):
            return f"<h3>{html_escape(stripped)}</h3>"
        if stripped.startswith("# "):
            return f"<h1>{html_escape(stripped.replace('# ', '', 1))}</h1>"
        if stripped.startswith("## "):
            return f"<h2>{html_escape(stripped.replace('## ', '', 1))}</h2>"
        if stripped.startswith("- "):
            return f"<p class='bullet'>{html_escape(stripped)}</p>"
        return f"<p>{html_escape(stripped)}</p>"

    body_html = "\n".join(render_line(line) for line in str(report_text or "").split("\n"))
    meta_parts = [f"작성일: {created_at.strftime('%Y.%m.%d')}"]

    if client_id:
        meta_parts.append(client_id)

    if session_label:
        meta_parts.append(session_label)

    meta = html_escape(" / ".join(meta_parts))
    html = f"""
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 18mm;
        }}
        body {{
            font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", sans-serif;
            color: #111827;
            font-size: 11pt;
            line-height: 1.75;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .title {{
            color: #123160;
            font-size: 22pt;
            font-weight: 700;
            text-align: center;
            margin: 0 0 8mm;
        }}
        .meta {{
            color: #64748B;
            font-size: 9.5pt;
            text-align: center;
            border-bottom: 1px solid #D8E1F0;
            padding-bottom: 8mm;
            margin-bottom: 8mm;
        }}
        h1, h2, h3 {{
            color: #1E40AF;
            line-height: 1.35;
            page-break-after: avoid;
        }}
        h1 {{
            font-size: 18pt;
            text-align: center;
            margin: 8mm 0;
        }}
        h2 {{
            font-size: 14pt;
            margin: 7mm 0 3mm;
        }}
        h3 {{
            font-size: 12.5pt;
            margin: 6mm 0 2mm;
        }}
        p {{
            margin: 0 0 3mm;
            white-space: pre-wrap;
        }}
        .bullet {{
            padding-left: 4mm;
        }}
        .spacer {{
            height: 3mm;
        }}
    </style>
</head>
<body>
    <div class="title">상담 요약 보고서</div>
    <div class="meta">{meta}</div>
    {body_html}
</body>
</html>
"""
    return HTML(string=html).write_pdf()
