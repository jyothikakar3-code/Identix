
"""Export helpers for attendance and analytics reports."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert records to a DataFrame with a stable empty state."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Export rows as CSV bytes."""
    return rows_to_dataframe(rows).to_csv(index=False).encode("utf-8")


def to_excel_bytes(rows: list[dict[str, Any]], sheet_name: str = "Report") -> bytes:
    """Export rows as Excel bytes."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        rows_to_dataframe(rows).to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def to_simple_pdf_bytes(title: str, rows: list[dict[str, Any]]) -> bytes:
    """Create a lightweight text-based PDF without external dependencies."""
    lines = [title, ""]
    if not rows:
        lines.append("No records available.")
    else:
        columns = list(rows[0].keys())[:6]
        lines.append(" | ".join(columns))
        lines.append("-" * 90)
        for row in rows[:120]:
            lines.append(" | ".join(str(row.get(column, ""))[:22] for column in columns))

    content = "\n".join(lines).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    text_lines = content.split("\n")
    stream_lines = ["BT", "/F1 10 Tf", "50 790 Td"]
    for index, line in enumerate(text_lines[:70]):
        if index > 0:
            stream_lines.append("0 -14 Td")
        stream_lines.append(f"({line}) Tj")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines)
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj",
    ]
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("utf-8")))
        pdf += obj + "\n"
    xref_start = len(pdf.encode("utf-8"))
    pdf += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
    return pdf.encode("utf-8")
