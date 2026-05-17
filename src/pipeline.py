from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Callable, Any

from .config import get_settings
from .excel_writer import ExcelWriter
from .figure_extractor import FigureExtractor
from .llm_client import LLMClient
from .markdown_writer import MarkdownWriter
from .pdf_parser import PDFParser
from .structured_extractor import StructuredExtractor
from .summarizer import Summarizer
from .translator import Translator
from .utils import ensure_dir, safe_slug, setup_logger, write_json


Progress = Callable[[str, float], None]


def process_pdf(
    pdf_path: Path,
    output_root: Path,
    image_format: str = "png",
    compress_images: bool = True,
    progress: Progress | None = None,
    output_name: str | None = None,
    output_markdown: bool | None = None,
    output_html: bool | None = None,
    output_summary: bool | None = None,
    output_json: bool | None = None,
    output_excel: bool | None = None,
    run_structured_extraction: bool | None = None,
) -> dict[str, Any]:
    def report(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    settings = get_settings()
    should_write_markdown = settings.output_markdown if output_markdown is None else output_markdown
    should_write_html = settings.output_html if output_html is None else output_html
    should_write_summary = settings.output_summary if output_summary is None else output_summary
    should_write_json = settings.output_json if output_json is None else output_json
    should_write_excel = settings.output_excel if output_excel is None else output_excel
    should_extract_structured = settings.run_structured_extraction if run_structured_extraction is None else run_structured_extraction
    should_extract_structured = should_extract_structured or should_write_json or should_write_excel
    slug = safe_slug(output_name or pdf_path.name)
    out_dir = ensure_dir(output_root / slug)
    log_file = out_dir / f"{slug}_processing.log"
    logger = setup_logger(log_file)
    logger.info("Starting processing for %s", pdf_path)

    copied_pdf = out_dir / (output_name or pdf_path.name)
    input_pdf = pdf_path
    if settings.output_source_pdf and pdf_path.resolve() != copied_pdf.resolve():
        shutil.copy2(pdf_path, copied_pdf)
        input_pdf = copied_pdf

    paths = {
        "markdown": out_dir / f"{slug}_translated_embedded.md" if should_write_markdown else None,
        "html": out_dir / f"{slug}_translated.html" if should_write_html else None,
        "summary": out_dir / f"{slug}_summary.md" if should_write_summary else None,
        "json": out_dir / f"{slug}_structured.json" if should_write_json else None,
        "excel": out_dir / f"{slug}_tables.xlsx" if should_write_excel else None,
        "source_pdf": copied_pdf if settings.output_source_pdf else None,
        "log": log_file,
    }
    warnings: list[str] = []
    try:
        report("读取 PDF 并提取正文", 0.08)
        parser = PDFParser(logger)
        paper, blocks = parser.parse(input_pdf)

        report("识别图注并裁切图片", 0.22)
        figures = FigureExtractor(logger, zoom=settings.figure_zoom).extract(
            input_pdf,
            blocks,
            image_format=image_format,
            compress=compress_images,
            debug_dir=out_dir / "debug_figures" if settings.output_debug_figures else None,
        )
        paper.figures = figures
        _mark_figure_region_text_as_non_content(paper, figures)
        _merge_continuation_fragments_after_filter(paper)
        for fig in figures:
            if fig.warning:
                warnings.append(f"{fig.figure_id}: {fig.warning}")
        paper.warnings.extend(warnings)

        report("翻译正文和图注", 0.42)
        llm = LLMClient(settings, logger)
        if llm.available():
            report("测试 LLM 连通性", 0.44)
            if not llm.health_check():
                paper.warnings.append("LLM 连通性测试失败，本次处理自动使用本地降级方案。")
        paper = Translator(
            llm,
            logger,
            progress=report,
            batch_size=settings.translation_batch_size,
            batch_max_chars=settings.translation_batch_max_chars,
            max_workers=settings.translation_max_workers,
        ).translate_paper(paper)
        if not settings.openai_api_key:
            paper.warnings.append("未设置 OPENAI_API_KEY，翻译和总结使用本地降级方案。")
        elif not llm.available():
            paper.warnings.append("LLM 请求失败或超时，本次处理已自动使用本地降级方案。")

        if should_extract_structured:
            report("抽取结构化器件性能和稳定性数据", 0.62)
            structured = StructuredExtractor(llm, logger, settings.max_llm_chars).extract(paper)
        else:
            report("生成本地启发式结构化摘要", 0.62)
            structured = StructuredExtractor(llm, logger, settings.max_llm_chars).extract_heuristic(paper)
        structured["warnings"] = list({*structured.get("warnings", []), *paper.warnings})

        report("生成中文总结", 0.74)
        summary = Summarizer(llm, logger).summarize(paper, structured)

        report("写入输出文件", 0.84)
        writer = MarkdownWriter(logger)
        if should_write_markdown and paths["markdown"]:
            writer.write_translated(paper, structured, paths["markdown"])
        if should_write_html and paths["html"]:
            if paths["markdown"] and Path(paths["markdown"]).exists():
                writer.markdown_to_html(paths["markdown"], paths["html"])
            else:
                temp_markdown = out_dir / f"{slug}_translated_embedded.tmp.md"
                writer.write_translated(paper, structured, temp_markdown)
                writer.markdown_to_html(temp_markdown, paths["html"])
                temp_markdown.unlink(missing_ok=True)
        if should_write_summary and paths["summary"]:
            writer.write_summary(summary, paths["summary"])
        if should_write_json and paths["json"]:
            write_json(paths["json"], structured)
        if should_write_excel and paths["excel"]:
            ExcelWriter(logger).write(structured, paths["excel"])

        report("处理完成", 1.0)
        return {
            "ok": True,
            "output_dir": out_dir,
            "paths": paths,
            "warnings": structured.get("warnings", []),
            "llm_enabled": llm.available(),
            "outputs": {
                "markdown": should_write_markdown,
                "html": should_write_html,
                "summary": should_write_summary,
                "json": should_write_json,
                "excel": should_write_excel,
                "structured_extraction": should_extract_structured,
            },
        }
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        report("处理失败，已写入日志", 1.0)
        return {"ok": False, "output_dir": out_dir, "paths": paths, "warnings": warnings + [str(exc)], "llm_enabled": False}


def _mark_figure_region_text_as_non_content(paper, figures) -> None:
    for section in paper.sections:
        for para in section.paragraphs:
            if para.kind == "caption" or not para.bbox:
                continue
            for fig in figures:
                if para.source_page != fig.page:
                    continue
                in_figure = _bbox_overlap_ratio(para.bbox, fig.bbox) > 0.35 or _bbox_center_inside(para.bbox, fig.bbox, padding=8)
                in_caption = _bbox_overlap_ratio(para.bbox, fig.caption_bbox) > 0.45
                if in_figure or in_caption:
                    para.kind = "non_content"
                    para.text_zh = para.text_original
                    break


def _merge_continuation_fragments_after_filter(paper) -> None:
    for section in paper.sections:
        last_body = None
        for para in section.paragraphs:
            if para.kind in {"non_content", "caption"}:
                continue
            if last_body is not None and _should_merge_fragment_after_filter(last_body, para):
                last_body.text_original = f"{last_body.text_original.rstrip()} {para.text_original.lstrip()}"
                para.kind = "non_content"
                para.text_zh = para.text_original
                continue
            last_body = para


def _should_merge_fragment_after_filter(previous, current) -> bool:
    if previous.kind != "paragraph" or current.kind != "paragraph":
        return False
    if current.source_page < previous.source_page or current.source_page > previous.source_page + 1:
        return False
    prev = previous.text_original.rstrip()
    cur = current.text_original.lstrip()
    if not prev or not cur:
        return False
    if _starts_known_subheading(cur):
        return False
    if re.match(r"^[a-z0-9),;/%±~<>≥≤]", cur):
        return bool(re.search(r"[,;:]$", prev) or not re.search(r"[.!?。！？]$", prev))
    return False


def _starts_known_subheading(text: str) -> bool:
    """Whether text looks like a subheading based on structural patterns.

    Mirror of PDFParser._starts_known_subheading for cross-module use.
    """
    if not text or text[0].islower():
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 14:
        return False
    if re.search(r"[.!?。！？]$", text):
        return False
    if re.match(
        r"^(We|The|Our|This|These|A|An|In|On|For|At|By|To|Here|It|Its|They|Their|"
        r"However|Moreover|Furthermore|Therefore|Nevertheless|Additionally|Meanwhile)\b",
        text,
    ):
        return False
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text):
        return True
    if re.search(
        r"(characterization|fabrication|preparation|performance|analysis|evaluation|"
        r"measurement|simulation|calculation|estimation|quantification|assessment|"
        r"testing|modeling|study|investigation|measurements|tests|properties|"
        r"diffusion|migration|transport|filling|effect|behavior|mechanism|"
        r"stability|efficiency|optimization|degradation|passivation)\b",
        text, re.I,
    ):
        return True
    if len(words) <= 6 and re.match(r"^[A-Z]", words[0]) and re.match(r"^[A-Z]", words[-1]):
        meaningful = [w for w in words if len(w) > 2 and re.match(r"^[A-Za-z]", w)]
        if len(meaningful) >= 2:
            return True
    return False


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return inter / area


def _bbox_center_inside(a: list[float], b: list[float], padding: float = 0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    return (bx0 - padding) <= cx <= (bx1 + padding) and (by0 - padding) <= cy <= (by1 + padding)
