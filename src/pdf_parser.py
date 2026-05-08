from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from .models import PaperInfo, Paragraph, ParsedPaper, Section


CAPTION_RE = re.compile(r"^(Fig\.\s*\d+[A-Za-z]?|Figure\s*\d+[A-Za-z]?|Table\s*\d+[A-Za-z]?|图\s*\d+)", re.I)
SECTION_HINT_RE = re.compile(
    r"^(Abstract|Introduction|Results|Discussion|Conclusion|Conclusions|Methods|Materials and methods|Experimental|References|Acknowledg|Supplementary|Data availability)\b",
    re.I,
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NON_CONTENT_RE = re.compile(
    r"("
    r"©|copyright|all rights reserved|"
    r"received:|accepted:|published:|"
    r"correspondence|corresponding author|"
    r"publisher'?s note|"
    r"state key laboratory|e-mail:|these authors contributed equally|"
    r"www\.|https?://|"
    r"nature communications|nature energy|science|cell reports|joule|advanced materials|"
    r"supplementary information|"
    r"reporting summary|"
    r"author contributions|competing interests|conflict of interest|"
    r"data availability|code availability|"
    r"references\b"
    r")",
    re.I,
)


class PDFParser:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def parse(self, pdf_path: Path) -> tuple[ParsedPaper, list[dict]]:
        doc = fitz.open(pdf_path)
        raw_blocks: list[dict] = []
        paragraphs: list[Paragraph] = []
        page_texts: list[str] = []
        try:
            for page_index, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks")
                blocks = self._sort_page_blocks(blocks, page.rect)
                page_parts: list[str] = []
                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    clean = self._clean_text(text)
                    if not clean:
                        continue
                    if self._is_repeated_margin_block(clean, [x0, y0, x1, y1], page.rect):
                        kind = "non_content"
                    else:
                        kind = "caption" if CAPTION_RE.match(clean) else ("heading" if self._looks_like_heading(clean) else "paragraph")
                    item = {"page": page_index, "bbox": [x0, y0, x1, y1], "text": clean, "page_rect": list(page.rect)}
                    raw_blocks.append(item)
                    page_parts.append(clean)
                    if kind != "caption" and self._looks_like_non_content(clean):
                        kind = "non_content"
                    if page_index == 1 and y0 < page.rect.height * 0.35 and self._looks_like_author_line(clean):
                        kind = "non_content"
                    paragraphs.append(Paragraph(source_page=page_index, text_original=clean, bbox=[x0, y0, x1, y1], kind=kind))
                page_texts.append("\n".join(page_parts))
        finally:
            doc.close()

        full_text = "\n\n".join(page_texts)
        paper_info = self._extract_info(page_texts, paragraphs)
        self._mark_abstract_paragraph(paragraphs, paper_info.abstract)
        sections = self._build_sections(paragraphs)
        return ParsedPaper(paper_info=paper_info, sections=sections, figures=[], full_text=full_text), raw_blocks

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = text.replace("\x01", "-")
        text = text.translate(str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}))
        text = text.replace("ð", "(").replace("Þ", ")")
        text = re.sub(r"ffiffi(?:ffi)*\s*([A-Za-z0-9]+)\s*p", r"sqrt(\1)", text)
        text = re.sub(r"(?<=\d)(°C|°F|K|h|hours|min|s)", r" \1", text)
        text = re.sub(r"-\n(?=[a-z])", "", text)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _sort_page_blocks(self, blocks: list, page_rect: fitz.Rect) -> list:
        midpoint = page_rect.width / 2
        top_full_width: list = []
        bottom_full_width: list = []
        left: list = []
        right: list = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            width = x1 - x0
            if y0 < page_rect.height * 0.18 and width > page_rect.width * 0.72:
                top_full_width.append(block)
            elif width > page_rect.width * 0.72 and (y0 > page_rect.height * 0.86 or y1 < page_rect.height * 0.22):
                bottom_full_width.append(block)
            elif (x0 + x1) / 2 < midpoint:
                left.append(block)
            else:
                right.append(block)
        key = lambda b: (round(b[1], 1), round(b[0], 1))
        return sorted(top_full_width, key=key) + sorted(left, key=key) + sorted(right, key=key) + sorted(bottom_full_width, key=key)

    def _looks_like_heading(self, text: str) -> bool:
        if CAPTION_RE.match(text):
            return False
        if SECTION_HINT_RE.match(text):
            return True
        if re.match(r"^\d+\.?\s+(Introduction|Results|Discussion|Conclusion|Methods|Experimental)\b", text, re.I):
            return True
        return False

    def _looks_like_non_content(self, text: str) -> bool:
        if CAPTION_RE.match(text):
            return False
        if re.search(r"check for updates", text, re.I):
            return True
        if NON_CONTENT_RE.search(text):
            return True
        if DOI_RE.search(text) and len(text) < 220:
            return True
        if re.fullmatch(r"\d+", text.strip()):
            return True
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", text) and len(text.strip()) < 40:
            return True
        if len(text) < 18 and re.search(r"\b(article|volume|issue|page|doi)\b", text, re.I):
            return True
        return False

    def _is_repeated_margin_block(self, text: str, bbox: list[float], page_rect: fitz.Rect) -> bool:
        x0, y0, x1, y1 = bbox
        height = page_rect.height
        width = page_rect.width
        text_len = len(text.strip())
        in_header = y1 < height * 0.075
        in_footer = y0 > height * 0.93
        narrow_or_short = text_len < 120 or (x1 - x0) < width * 0.35
        if (in_header or in_footer) and narrow_or_short:
            return True
        return False

    def _extract_info(self, page_texts: list[str], paragraphs: list[Paragraph]) -> PaperInfo:
        first_page = page_texts[0] if page_texts else ""
        doi_match = DOI_RE.search(first_page)
        year_match = YEAR_RE.search(first_page)
        publication = self._extract_publication_info("\n".join(page_texts[:2]))
        title = ""
        for p in paragraphs[:12]:
            t = p.text_original
            doi = DOI_RE.search(t)
            if doi and len(t) > doi.end() + 20:
                candidate = t[doi.end() :].strip(" .;:-")
                if candidate:
                    title = candidate
                    break
            if self._looks_like_author_line(t):
                continue
            if 20 < len(t) < 250 and not DOI_RE.search(t) and not SECTION_HINT_RE.match(t) and not self._looks_like_non_content(t):
                title = t
                break
        abstract = ""
        abstract_match = re.search(r"\bAbstract\b\s*(.+?)(?:\bIntroduction\b|\bKeywords\b|1\.\s*Introduction)", "\n".join(page_texts[:3]), re.I | re.S)
        if abstract_match:
            abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip()
        if not abstract:
            abstract = self._guess_abstract_from_layout(paragraphs, title)
        authors: list[str] = []
        if title:
            for p in paragraphs[:12]:
                if p.text_original == title:
                    continue
                if 5 < len(p.text_original) < 300 and self._looks_like_author_line(p.text_original):
                    authors = [a.strip() for a in re.split(r",| and ", p.text_original) if a.strip()][:20]
                    break
        return PaperInfo(
            title=title,
            authors=authors,
            doi=doi_match.group(0) if doi_match else "",
            journal=publication.get("journal", ""),
            volume=publication.get("volume", ""),
            issue=publication.get("issue", ""),
            pages=publication.get("pages", ""),
            year=publication.get("year", "") or (year_match.group(0) if year_match else ""),
            abstract=abstract,
        )

    def _extract_publication_info(self, text: str) -> dict[str, str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        info = {"journal": "", "volume": "", "issue": "", "pages": "", "year": ""}
        if not normalized:
            return info

        pipe_match = re.search(
            r"(?P<journal>[A-Z][A-Za-z0-9 &+\-]+?)\s*\|\s*\(?(?P<year>(?:19|20)\d{2})\)?\s*\|\s*"
            r"(?P<volume>\d+[A-Za-z]?)\s*(?:\((?P<issue>[^)]+)\))?\s*[:;,]\s*(?P<pages>[A-Za-z]?\d+(?:[-–]\d+)?)",
            normalized,
        )
        if not pipe_match:
            pipe_match = re.search(
                r"(?P<journal>[A-Z][A-Za-z0-9 &+\-]+?)\s*\|\s*\(?(?P<year>(?:19|20)\d{2})\)?\s*"
                r"(?P<volume>\d+[A-Za-z]?)\s*[:;,]\s*(?P<pages>[A-Za-z]?\d+(?:[-–]\d+)?)",
                normalized,
            )
        if not pipe_match:
            pipe_match = re.search(
                r"(?P<journal>[A-Z][A-Za-z0-9 &+\-]{3,80}?)\s+\(?(?P<year>(?:19|20)\d{2})\)?\s+"
                r"(?P<volume>\d+[A-Za-z]?)\s*(?:\((?P<issue>[^)]+)\))?\s*[:;,]\s*(?P<pages>[A-Za-z]?\d+(?:[-–]\d+)?)",
                normalized,
            )
        if pipe_match:
            for key in info:
                value = pipe_match.groupdict().get(key) or ""
                info[key] = value.strip(" .,:;|")

        if not info["journal"]:
            journal_match = re.search(
                r"\b(Nature Communications|Nature Energy|Science|Joule|Advanced Materials|Energy & Environmental Science|"
                r"ACS Energy Letters|Nano Energy|Cell Reports Physical Science|Chemical Engineering Journal|Solar RRL)\b",
                normalized,
                re.I,
            )
            if journal_match:
                info["journal"] = journal_match.group(1)

        if not info["volume"]:
            volume_match = re.search(r"\b(?:vol(?:ume)?\.?\s*)?(\d+[A-Za-z]?)\s*\(([^)]+)\)\s*[:;,]\s*([A-Za-z]?\d+(?:[-–]\d+)?)", normalized, re.I)
            if volume_match:
                info["volume"] = volume_match.group(1)
                info["issue"] = volume_match.group(2)
                info["pages"] = volume_match.group(3)
        if not info["volume"]:
            volume_match = re.search(r"\b(?:vol(?:ume)?\.?\s*)(\d+[A-Za-z]?)\b", normalized, re.I)
            if volume_match:
                info["volume"] = volume_match.group(1)
        if not info["issue"]:
            issue_match = re.search(r"\bissue\s+([A-Za-z0-9.-]+)\b", normalized, re.I)
            if issue_match:
                info["issue"] = issue_match.group(1)
        if not info["pages"]:
            pages_match = re.search(r"\b(?:pp?\.?|pages?)\s*([A-Za-z]?\d+(?:[-–]\d+)?)\b", normalized, re.I)
            if pages_match:
                info["pages"] = pages_match.group(1)
        if not info["year"]:
            year_match = YEAR_RE.search(normalized)
            if year_match:
                info["year"] = year_match.group(0)
        return info

    def _looks_like_author_line(self, text: str) -> bool:
        if len(text) > 180 or "." in text:
            return False
        if not ("," in text or "&" in text):
            return False
        if DOI_RE.search(text) or SECTION_HINT_RE.match(text):
            return False
        if re.search(r"\d", text) and re.search(r"[A-Z][a-z]+", text):
            return True
        return False

    def _guess_abstract_from_layout(self, paragraphs: list[Paragraph], title: str) -> str:
        """Nature-style papers often omit the literal 'Abstract' label on page 1."""
        candidates: list[Paragraph] = []
        for para in paragraphs:
            if para.source_page != 1 or not para.bbox:
                continue
            text = para.text_original
            x0, y0, x1, y1 = para.bbox
            if text == title or para.kind == "non_content" or self._looks_like_author_line(text):
                continue
            if len(text) < 180 or y0 < 180 or y0 > 460:
                continue
            if x0 < 150:
                continue
            candidates.append(para)
        if not candidates:
            return ""
        candidates.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        return candidates[0].text_original

    def _mark_abstract_paragraph(self, paragraphs: list[Paragraph], abstract: str) -> None:
        if not abstract:
            return
        normalized_abstract = re.sub(r"\s+", " ", abstract).strip()
        for para in paragraphs:
            normalized_text = re.sub(r"\s+", " ", para.text_original).strip()
            if normalized_text == normalized_abstract:
                para.kind = "abstract"

    def _build_sections(self, paragraphs: list[Paragraph]) -> list[Section]:
        sections: list[Section] = []
        current = Section(section_title="正文")
        in_references = False
        for para in paragraphs:
            if para.kind == "abstract":
                continue
            if para.kind == "non_content":
                if re.search(r"^references\b|^bibliography\b", para.text_original, re.I):
                    in_references = True
                continue
            if para.kind == "heading":
                if current.paragraphs or current.section_title != "正文":
                    sections.append(current)
                title, remainder = self._split_inline_heading(para.text_original)
                current = Section(section_title=title)
                in_references = bool(re.search(r"^references\b|^bibliography\b", title, re.I))
                if remainder:
                    para.kind = "paragraph"
                    para.text_original = remainder
                    current.paragraphs.append(para)
            else:
                if in_references:
                    para.kind = "non_content"
                self._append_paragraph(current, para)
        if current.paragraphs or current.section_title:
            sections.append(current)
        return sections

    def _append_paragraph(self, section: Section, para: Paragraph) -> None:
        if (
            para.kind == "paragraph"
            and section.paragraphs
            and section.paragraphs[-1].kind == "paragraph"
            and self._should_merge_paragraphs(section.paragraphs[-1], para)
        ):
            section.paragraphs[-1].text_original = f"{section.paragraphs[-1].text_original.rstrip()} {para.text_original.lstrip()}"
            if (
                section.paragraphs[-1].source_page == para.source_page
                and section.paragraphs[-1].bbox
                and para.bbox
                and para.bbox[1] >= section.paragraphs[-1].bbox[1]
            ):
                section.paragraphs[-1].bbox = self._union_bbox(section.paragraphs[-1].bbox, para.bbox)
            return
        section.paragraphs.append(para)

    def _should_merge_paragraphs(self, previous: Paragraph, current: Paragraph) -> bool:
        if not previous.text_original or not current.text_original:
            return False
        if previous.bbox and current.bbox:
            if previous.source_page == current.source_page:
                prev_y0, prev_y1 = previous.bbox[1], previous.bbox[3]
                cur_y0, cur_y1 = current.bbox[1], current.bbox[3]
                column_wrap = previous.bbox[2] < current.bbox[0] and prev_y1 > 500 and cur_y0 < 140
                if cur_y0 < prev_y0 - 8 and not column_wrap:
                    return False
                if not column_wrap and cur_y0 - prev_y1 > 80:
                    return False
                if not column_wrap and max(prev_y1, cur_y1) - min(prev_y0, cur_y0) > 320:
                    return False
            elif current.source_page != previous.source_page + 1:
                return False
            elif current.bbox[1] > 180:
                return False
            elif len(current.text_original.strip()) < 50:
                return False
        if self._starts_known_subheading(current.text_original):
            return False
        prev = previous.text_original.rstrip()
        cur = current.text_original.lstrip()
        if re.search(r"[.!?。！？:]$", prev):
            return False
        if re.match(r"^[a-z0-9),;/%±~<>≥≤]", cur):
            return True
        if re.search(r"[,;]$", prev):
            return True
        return False

    def _starts_known_subheading(self, text: str) -> bool:
        return bool(
            re.match(
                r"^(Barrier energy quantification|Scattering barrier preparation|Drift barrier preparation|"
                r"Photovoltaic performance|Inhibition effect for iodide ion migration|Materials|"
                r"Perovskite solar cells fabrication|Stability tests|Relative dielectric constant|"
                r"Carrier concentration characterization|Space charge limited current|SCAPS simulation|"
                r"Fitting of Fick|Characterization)\b",
                text,
            )
        )

    def _union_bbox(self, a: list[float], b: list[float]) -> list[float]:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def _split_inline_heading(self, text: str) -> tuple[str, str]:
        match = re.match(
            r"^(Abstract|Introduction|Results|Discussion|Conclusion|Conclusions|Methods|Materials and methods|Experimental|References|Acknowledg(?:ements)?|Supplementary|Data availability)\b[:.\s-]*(.*)$",
            text,
            re.I,
        )
        if not match:
            return text, ""
        return match.group(1), match.group(2).strip()
