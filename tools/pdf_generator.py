"""
Tool: PDF Generator

Tạo báo cáo PDF chuyên nghiệp từ list[RankedCandidate].

Cấu trúc PDF:
  Page 1 — Cover: tiêu đề, job title, ngày tạo, số ứng viên
  Page 2 — Executive Summary: shortlist top 5, tổng quan batch
  Page 3+ — Candidate Details: mỗi candidate 1 section
              └── Score breakdown table
              └── Strengths & Concerns
              └── Recommendation

Design:
  - Màu sắc encode tier: xanh (strong), cam (good), vàng (moderate), đỏ (weak)
  - Score bar chart ngang cho từng dimension
  - Không cần chart library — vẽ bằng reportlab primitives
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from schemas.score_schema import RankedCandidate, ScoreTier

# ── Màu sắc ──────────────────────────────────────────────────────────────────
COLOR_STRONG   = colors.HexColor("#16a34a")   # Xanh lá — strong (>=80)
COLOR_GOOD     = colors.HexColor("#2563eb")   # Xanh dương — good (60-79)
COLOR_MODERATE = colors.HexColor("#d97706")   # Cam — moderate (40-59)
COLOR_WEAK     = colors.HexColor("#dc2626")   # Đỏ — weak (<40)
COLOR_PRIMARY  = colors.HexColor("#1e293b")   # Text chính
COLOR_MUTED    = colors.HexColor("#64748b")   # Text phụ
COLOR_BG_LIGHT = colors.HexColor("#f8fafc")   # Background nhạt
COLOR_BORDER   = colors.HexColor("#e2e8f0")   # Border

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


def _tier_color(tier: str) -> colors.Color:
    return {
        ScoreTier.STRONG:   COLOR_STRONG,
        ScoreTier.GOOD:     COLOR_GOOD,
        ScoreTier.MODERATE: COLOR_MODERATE,
        ScoreTier.WEAK:     COLOR_WEAK,
    }.get(tier, COLOR_MUTED)


def _tier_label(tier: str) -> str:
    return {
        ScoreTier.STRONG:   "STRONG MATCH",
        ScoreTier.GOOD:     "GOOD MATCH",
        ScoreTier.MODERATE: "MODERATE",
        ScoreTier.WEAK:     "WEAK MATCH",
    }.get(tier, tier.upper())


# ── Styles ────────────────────────────────────────────────────────────────────
def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontSize=28, leading=34,
            textColor=COLOR_PRIMARY,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontSize=14, leading=20,
            textColor=COLOR_MUTED,
            fontName="Helvetica",
            alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontSize=13, leading=18,
            textColor=COLOR_PRIMARY,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10, leading=15,
            textColor=COLOR_PRIMARY,
            fontName="Helvetica",
        ),
        "body_muted": ParagraphStyle(
            "body_muted",
            fontSize=9, leading=13,
            textColor=COLOR_MUTED,
            fontName="Helvetica",
        ),
        "candidate_name": ParagraphStyle(
            "candidate_name",
            fontSize=14, leading=18,
            textColor=COLOR_PRIMARY,
            fontName="Helvetica-Bold",
        ),
        "label": ParagraphStyle(
            "label",
            fontSize=8, leading=11,
            textColor=COLOR_MUTED,
            fontName="Helvetica",
            spaceAfter=1,
        ),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    ranked:    list[RankedCandidate],
    job_title: str,
    job_id:    str,
    shortlist_count: int = 5,
) -> bytes:
    """
    Tạo PDF báo cáo từ danh sách ứng viên đã được xếp hạng.

    Args:
        ranked:          list[RankedCandidate] sắp xếp theo rank (rank 1 = tốt nhất)
        job_title:       Tên vị trí tuyển dụng
        job_id:          ID của JD
        shortlist_count: Số ứng viên trong shortlist (mặc định top 5)

    Returns:
        bytes: PDF binary
    """
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f"CV Screening Report — {job_title}",
        author="HR CV Scanner System",
    )

    styles   = _build_styles()
    elements = []

    # Page 1: Cover
    elements += _build_cover(job_title, job_id, ranked, styles)
    elements.append(PageBreak())

    # Page 2: Executive Summary + Ranking Table
    elements += _build_summary(ranked, shortlist_count, styles)
    elements.append(PageBreak())

    # Page 3+: Candidate detail cards
    for candidate in ranked:
        elements += _build_candidate_section(candidate, styles)
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(HRFlowable(width="100%", thickness=0.5,
                                   color=COLOR_BORDER, spaceAfter=8))

    doc.build(elements)
    return buffer.getvalue()


# ── Page builders ─────────────────────────────────────────────────────────────

def _build_cover(
    job_title: str,
    job_id:    str,
    ranked:    list[RankedCandidate],
    styles:    dict,
) -> list:
    """Cover page: title, stats, date."""
    strong_count   = sum(1 for r in ranked if r.score.tier == ScoreTier.STRONG)
    good_count     = sum(1 for r in ranked if r.score.tier == ScoreTier.GOOD)
    avg_score      = (
        sum(r.score.total_score for r in ranked) / len(ranked)
        if ranked else 0
    )

    elements = [
        Spacer(1, 3 * cm),
        Paragraph("CV Screening Report", styles["cover_title"]),
        Spacer(1, 0.5 * cm),
        Paragraph(job_title, ParagraphStyle(
            "jt", fontSize=18, leading=24, textColor=COLOR_GOOD,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Job ID: {job_id}", styles["cover_sub"]),
        Spacer(1, 2 * cm),
        HRFlowable(width="60%", thickness=1, color=COLOR_BORDER,
                   hAlign="CENTER", spaceAfter=16),
        Spacer(1, 0.5 * cm),
    ]

    # Stats table
    stats_data = [
        ["Total Candidates", "Strong Match", "Good Match", "Avg Score"],
        [
            str(len(ranked)),
            str(strong_count),
            str(good_count),
            f"{avg_score:.1f}/100",
        ],
    ]
    stats_table = Table(
        stats_data,
        colWidths=[(PAGE_W - 2 * MARGIN) / 4] * 4,
    )
    stats_table.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  COLOR_MUTED),
        ("FONTNAME",    (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, 1),  22),
        ("TEXTCOLOR",   (0, 1), (-1, 1),  COLOR_PRIMARY),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 2 * cm))
    elements.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}",
        styles["cover_sub"],
    ))
    return elements


def _build_summary(
    ranked:          list[RankedCandidate],
    shortlist_count: int,
    styles:          dict,
) -> list:
    """Executive Summary: ranking table với color-coded tiers."""
    elements = [
        Paragraph("Executive Summary", styles["section_header"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Showing all {len(ranked)} candidates ranked by total score. "
            f"Top {min(shortlist_count, len(ranked))} are recommended for interview.",
            styles["body_muted"],
        ),
        Spacer(1, 0.5 * cm),
    ]

    # Header row
    col_w = (PAGE_W - 2 * MARGIN)
    headers = ["Rank", "Candidate", "Score", "Tier", "Technical", "Experience", "Rec."]
    col_widths = [
        col_w * 0.07,  # Rank
        col_w * 0.26,  # Name
        col_w * 0.10,  # Score
        col_w * 0.13,  # Tier
        col_w * 0.13,  # Technical
        col_w * 0.13,  # Experience
        col_w * 0.18,  # Recommendation
    ]

    table_data = [headers]
    row_styles: list[tuple] = []

    for i, r in enumerate(ranked, start=1):
        sc  = r.score
        bd  = sc.breakdown
        rec = (sc.recommendation or "—")[:30]  # Truncate
        tech_score = f"{bd.technical_skills.raw_score:.0f}/{bd.technical_skills.max_score}" \
            if bd.technical_skills else "—"
        exp_score = f"{bd.experience.raw_score:.0f}/{bd.experience.max_score}" \
            if bd.experience else "—"

        row = [
            f"#{r.rank}",
            sc.candidate_name,
            f"{sc.total_score:.1f}",
            _tier_label(sc.tier),
            tech_score,
            exp_score,
            rec,
        ]
        table_data.append(row)

        # Color tier cell
        tier_col   = 3
        tier_color = _tier_color(sc.tier)
        row_i      = i          # 0-indexed header, data starts at 1
        row_styles.append(("TEXTCOLOR", (tier_col, row_i), (tier_col, row_i), tier_color))
        row_styles.append(("FONTNAME",  (tier_col, row_i), (tier_col, row_i), "Helvetica-Bold"))

        # Highlight shortlist rows with subtle background
        if r.rank <= shortlist_count:
            row_styles.append(("BACKGROUND", (0, row_i), (-1, row_i), COLOR_BG_LIGHT))

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    base_style = [
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (1, 0), (1, -1),  "LEFT"),
        ("ALIGN",         (6, 0), (6, -1),  "LEFT"),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  COLOR_MUTED),
        ("BACKGROUND",    (0, 0), (-1, 0),  COLOR_BG_LIGHT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, COLOR_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]
    table.setStyle(TableStyle(base_style + row_styles))
    elements.append(table)
    return elements


def _build_candidate_section(
    r:      RankedCandidate,
    styles: dict,
) -> list:
    """
    1 section cho 1 candidate:
      - Header: rank, name, score, tier badge
      - Score breakdown bar chart (vẽ bằng Table)
      - Strengths & Concerns
      - Recommendation
    """
    sc       = r.score
    elements = []

    # ── Header ──
    tier_color = _tier_color(sc.tier)
    header_data = [[
        Paragraph(f"#{r.rank}  {sc.candidate_name}", styles["candidate_name"]),
        Paragraph(
            f"<font color='#{_hex(tier_color)}'>{_tier_label(sc.tier)}</font>"
            f"   <b>{sc.total_score:.1f}/100</b>   "
            f"<font color='#64748b'>Top {100 - r.percentile + 1:.0f}%</font>",
            ParagraphStyle("hdr_right", fontSize=10, fontName="Helvetica",
                           alignment=TA_RIGHT, textColor=COLOR_PRIMARY),
        ),
    ]]
    header_table = Table(
        header_data,
        colWidths=[(PAGE_W - 2 * MARGIN) * 0.55, (PAGE_W - 2 * MARGIN) * 0.45],
    )
    header_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (0, 0), "LEFT"),
        ("ALIGN",         (1, 0), (1, 0), "RIGHT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.25 * cm))

    # ── Score breakdown bar chart ──
    elements.append(_build_score_bars(sc.breakdown, styles))
    elements.append(Spacer(1, 0.3 * cm))

    # ── Strengths & Concerns (2 columns) ──
    strengths_text = "<br/>".join(
        f"✓ {s}" for s in (sc.strengths or ["No strengths noted."])
    )
    concerns_text  = "<br/>".join(
        f"△ {c}" for c in (sc.concerns or ["No concerns noted."])
    )

    sc_data = [[
        [
            Paragraph("Strengths", styles["label"]),
            Paragraph(strengths_text, styles["body"]),
        ],
        [
            Paragraph("Concerns", styles["label"]),
            Paragraph(concerns_text, styles["body"]),
        ],
    ]]
    sc_table = Table(
        sc_data,
        colWidths=[(PAGE_W - 2 * MARGIN) * 0.5] * 2,
    )
    sc_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
    ]))
    elements.append(sc_table)
    elements.append(Spacer(1, 0.25 * cm))

    # ── Recommendation ──
    if sc.recommendation:
        elements.append(Paragraph(
            f"<b>Recommendation:</b> {sc.recommendation}",
            styles["body"],
        ))

    # ── Warnings ──
    if sc.low_confidence:
        elements.append(Spacer(1, 0.2 * cm))
        elements.append(Paragraph(
            "⚠ Low confidence parse — results should be manually verified.",
            ParagraphStyle("warn", fontSize=8, fontName="Helvetica",
                           textColor=COLOR_MODERATE),
        ))

    return elements


def _build_score_bars(breakdown, styles: dict) -> Table:
    """
    Horizontal bar chart cho 5 dimensions.
    Vẽ bằng reportlab Table — không cần matplotlib.
    """
    dims = breakdown.as_list()
    if not dims:
        return Table([[Paragraph("No score breakdown available.", styles["body_muted"])]])

    bar_max_w = (PAGE_W - 2 * MARGIN) * 0.55  # Bar kéo dài tối đa
    rows      = []

    for ds in dims:
        pct       = ds.percentage / 100
        bar_fill  = max(2, bar_max_w * pct)      # Tối thiểu 2pt để hiển thị
        bar_empty = bar_max_w - bar_fill
        bar_color = _tier_color(
            ScoreTier.from_score(ds.percentage)
        )

        label_cell = Paragraph(
            ds.dimension.replace("_", " ").title(),
            ParagraphStyle("bl", fontSize=8, fontName="Helvetica",
                           textColor=COLOR_PRIMARY),
        )
        score_cell = Paragraph(
            f"{ds.raw_score:.0f}/{ds.max_score}",
            ParagraphStyle("sc", fontSize=8, fontName="Helvetica-Bold",
                           textColor=COLOR_PRIMARY, alignment=TA_RIGHT),
        )

        # Bar: 2-cell Table (filled | empty)
        bar_table = Table(
            [[" ", " "]],
            colWidths=[bar_fill, max(bar_empty, 0.1)],
            rowHeights=[7],
        )
        bar_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), bar_color),
            ("BACKGROUND",    (1, 0), (1, 0), COLOR_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))

        rationale_cell = Paragraph(
            ds.rationale[:80] + ("…" if len(ds.rationale) > 80 else ""),
            styles["body_muted"],
        )

        rows.append([label_cell, score_cell, bar_table, rationale_cell])

    col_w = PAGE_W - 2 * MARGIN
    chart = Table(
        rows,
        colWidths=[col_w * 0.17, col_w * 0.09, col_w * 0.35, col_w * 0.39],
    )
    chart.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return chart


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex(color: colors.Color) -> str:
    """Convert reportlab Color → hex string cho Paragraph markup."""
    r = int(color.red   * 255)
    g = int(color.green * 255)
    b = int(color.blue  * 255)
    return f"{r:02x}{g:02x}{b:02x}"