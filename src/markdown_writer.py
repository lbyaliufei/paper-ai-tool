from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any

import markdown as md

from .models import Figure, ParsedPaper


class MarkdownWriter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write_translated(self, paper: ParsedPaper, structured: dict[str, Any], output_path: Path) -> None:
        lines: list[str] = []
        title = paper.paper_info.title_zh or paper.paper_info.title or "未命名论文"
        lines += [f"# {title}", ""]
        lines += ["## 论文基本信息"]
        if paper.paper_info.title:
            lines.append(f"- 原文标题：{paper.paper_info.title}")
        if paper.paper_info.authors:
            lines.append(f"- 作者：{', '.join(paper.paper_info.authors)}")
        if paper.paper_info.doi:
            lines.append(f"- DOI：{paper.paper_info.doi}")
        if paper.paper_info.journal:
            lines.append(f"- 期刊：{paper.paper_info.journal}")
        if paper.paper_info.volume:
            lines.append(f"- 卷：{paper.paper_info.volume}")
        if paper.paper_info.issue:
            lines.append(f"- 期：{paper.paper_info.issue}")
        if paper.paper_info.pages:
            lines.append(f"- 页码/文章编号：{paper.paper_info.pages}")
        if paper.paper_info.year:
            lines.append(f"- 年份：{paper.paper_info.year}")
        lines.append("")
        if paper.paper_info.abstract_zh or paper.paper_info.abstract:
            lines += ["## 摘要", paper.paper_info.abstract_zh or paper.paper_info.abstract, ""]

        inserted: set[str] = set()
        body_ref_ids = self._body_referenced_figure_ids(paper)
        for section in paper.sections:
            heading = section.section_title_zh or section.section_title
            if heading:
                lines += [f"## {heading}", ""]
            for para in section.paragraphs:
                if para.kind == "non_content":
                    continue
                if self._is_duplicate_abstract(para.text_original, paper.paper_info.abstract):
                    continue
                if para.kind == "caption":
                    matching = self._find_caption_figure(para.text_original, paper.figures)
                    if matching and self._fig_ref_key(matching.figure_id) not in body_ref_ids and self._fig_key(matching) not in inserted:
                        self._append_figure(lines, matching)
                        inserted.add(self._fig_key(matching))
                    continue
                paragraph_text = para.text_zh or para.text_original
                subheading, body = self._split_leading_subheading(section.section_title, para.text_original, paragraph_text)
                if subheading:
                    lines += [f"### {subheading}", ""]
                    if body:
                        lines += [body, ""]
                else:
                    lines += [paragraph_text, ""]
                for fig in self._figures_referenced_by_paragraph(para.text_original, paragraph_text, paper.figures, inserted):
                    self._append_figure(lines, fig)
                    inserted.add(self._fig_key(fig))

        for fig in paper.figures:
            if self._fig_key(fig) not in inserted:
                lines += ["", "<!-- 未能准确定位，按页面顺序补充的图 -->"]
                self._append_figure(lines, fig)

        lines += ["", "## 自动结构化总结", ""]
        for row in structured.get("device_performance", [])[:20]:
            lines.append(f"- 器件性能：PCE={row.get('pce_percent')}%，VOC={row.get('voc_v')} V，JSC={row.get('jsc_ma_cm2')}，FF={row.get('ff_percent')}；来源：{row.get('source_text','')[:160]}")
        for row in structured.get("stability_tests", [])[:20]:
            lines.append(f"- 稳定性：{row.get('test_type')}，保持率={row.get('retained_pce_operator','')}{row.get('retained_pce_percent')}%，时长={row.get('duration_h')} h；来源：{row.get('source_text','')[:160]}")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def write_summary(self, summary: str, output_path: Path) -> None:
        output_path.write_text(summary, encoding="utf-8")

    def markdown_to_html(self, markdown_path: Path, html_path: Path) -> None:
        body = md.markdown(markdown_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc"])
        html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(markdown_path.stem)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; line-height: 1.65; max-width: 920px; margin: 32px auto; padding: 0 24px; color: #202124; }}
h1, h2, h3 {{ line-height: 1.28; page-break-after: avoid; }}
h1 {{ font-size: 28px; border-bottom: 2px solid #ddd; padding-bottom: 12px; }}
h2 {{ margin-top: 30px; font-size: 22px; }}
p, li {{ font-size: 14px; }}
img {{ max-width: 100%; display: block; margin: 14px auto 8px; page-break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
td, th {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
code {{ background: #f5f5f5; padding: 1px 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
        html_path.write_text(html_doc, encoding="utf-8")

    def _append_figure(self, lines: list[str], fig: Figure) -> None:
        caption = fig.caption_zh or fig.caption_original
        if fig.image_base64:
            alt = caption.replace("\n", " ")[:120]
            lines.append(f"![{alt}](data:{fig.image_mime};base64,{fig.image_base64})")
        else:
            lines.append(f"> 图片裁切失败：{fig.warning or '未识别到图片区域'}")
        lines += [f"*{caption}*", ""]

    def _find_caption_figure(self, caption: str, figures: list[Figure]) -> Figure | None:
        m = re.match(r"^(Fig\.\s*\d+|Figure\s*\d+|Table\s*\d+|图\s*\d+)", caption, re.I)
        if not m:
            return None
        key = re.sub(r"^Figure", "Fig.", m.group(1), flags=re.I).lower().replace(" ", "")
        for fig in figures:
            if fig.figure_id.lower().replace(" ", "") == key:
                return fig
        return None

    def _figures_referenced_by_paragraph(self, original: str, rendered: str, figures: list[Figure], inserted: set[str]) -> list[Figure]:
        combined = f"{original}\n{rendered}"
        referenced_ids = self._referenced_figure_ids(combined)
        if not referenced_ids:
            return []
        by_id = {self._fig_ref_key(fig.figure_id): fig for fig in figures}
        out: list[Figure] = []
        for ref_id in referenced_ids:
            fig = by_id.get(ref_id)
            if fig and self._fig_key(fig) not in inserted:
                out.append(fig)
        return out

    def _body_referenced_figure_ids(self, paper: ParsedPaper) -> set[str]:
        refs: set[str] = set()
        for section in paper.sections:
            for para in section.paragraphs:
                if para.kind in {"caption", "non_content"}:
                    continue
                text = f"{para.text_original}\n{para.text_zh}"
                refs.update(self._referenced_figure_ids(text))
        return refs

    def _referenced_figure_ids(self, text: str) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        patterns = [
            r"(?<!Supplementary\s)(?<!Supplementary\sFig\.\s)(?<!Supplementary\sFigure\s)\bFig\.\s*(\d+)(?:\s*[A-Za-z]|\s*[a-z](?:[,–-]\s*[a-z])*)?",
            r"(?<!Supplementary\s)\bFigure\s*(\d+)(?:\s*[A-Za-z]|\s*[a-z](?:[,–-]\s*[a-z])*)?",
            r"(?<!补充)\b图\s*(\d+)(?:\s*[A-Za-z]|\s*[a-z](?:[,–-]\s*[a-z])*)?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                ref = f"fig.{match.group(1)}"
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
        return refs

    def _fig_ref_key(self, figure_id: str) -> str:
        match = re.search(r"(\d+)", figure_id)
        return f"fig.{match.group(1)}" if match else figure_id.lower().replace(" ", "")

    def _fig_key(self, fig: Figure) -> str:
        return f"{fig.figure_id.lower()}@{fig.page}"

    def _is_duplicate_abstract(self, text: str, abstract: str) -> bool:
        if not text or not abstract:
            return False
        a = re.sub(r"\s+", " ", abstract).strip()
        t = re.sub(r"\s+", " ", text).strip()
        return len(t) > 120 and (t in a or a.startswith(t[:120]) or t.startswith(a[:120]))

    def _split_leading_subheading(self, section_title: str, original: str, rendered: str) -> tuple[str, str]:
        section = section_title.lower()
        if section not in {"results", "methods", "正文"}:
            return "", rendered
        labels = [
            "Barrier energy quantification",
            "Scattering barrier preparation",
            "Drift barrier preparation",
            "Photovoltaic performance",
            "Inhibition effect for iodide ion migration",
            "Materials",
            "Perovskite solar cells fabrication",
            "Stability tests",
            "Relative dielectric constant",
            "Carrier concentration characterization",
            "Space charge limited current (SCLC) characterization",
            "SCAPS simulation",
            "Fitting of Fick’s second law of diffusion",
            "Fitting of Fick's second law of diffusion",
            "Characterization",
        ]
        for label in labels:
            if original.startswith(label + " "):
                body = rendered
                if body.startswith(label):
                    body = body[len(label) :].strip()
                return label, body
        return "", rendered
