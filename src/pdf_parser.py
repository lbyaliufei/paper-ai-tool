from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from .models import PaperInfo, Paragraph, ParsedPaper, Section


CAPTION_RE = re.compile(r"^(Fig\.\s*\d+[A-Za-z]?|Figure\s+\d+[A-Za-z]?|Table\s+\d+[A-Za-z]?|图\s*\d+)", re.I)
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
                        kind = "caption" if self._looks_like_caption(clean) else ("heading" if self._looks_like_heading(clean) else "paragraph")
                    item = {"page": page_index, "bbox": [x0, y0, x1, y1], "text": clean, "page_rect": list(page.rect)}
                    raw_blocks.append(item)
                    page_parts.append(clean)
                    if kind != "caption" and self._looks_like_non_content(clean):
                        kind = "non_content"
                    if page_index == 1 and y0 < page.rect.height * 0.35 and self._looks_like_author_line(clean):
                        kind = "non_content"
                    paragraphs.append(Paragraph(source_page=page_index, text_original=clean, bbox=[x0, y0, x1, y1], kind=kind))
                raw_text = self._clean_text(page.get_text("text"))
                page_texts.append(raw_text or "\n".join(page_parts))
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
        if self._looks_like_caption(text):
            return False
        if not text or text[0].islower():
            return False
        if re.match(r"^Supplementary\s+(?:Fig\.?|Figure|Table|Note)\b", text, re.I):
            return False
        section_match = SECTION_HINT_RE.match(text)
        if section_match:
            keyword = section_match.group(1)
            if keyword.lower() == "supplementary":
                return False
            rest = text[section_match.end() :].strip()
            if rest and not re.match(r"^[:.\-–]", rest):
                inline_allowed = keyword.lower() in {
                    "results",
                    "discussion",
                    "conclusion",
                    "conclusions",
                    "methods",
                    "references",
                    "acknowledg",
                    "acknowledgements",
                    "data availability",
                }
                if not inline_allowed:
                    return False
            return True
        if re.match(r"^\d+\.?\s+(Introduction|Results|Discussion|Conclusion|Methods|Experimental)\b", text, re.I):
            return True
        if re.match(r"^(Package Description|Package Testing|Outdoor Testing|Experimental Methods|Device Fabrication|Data Availability)\.?\b", text, re.I):
            return True
        return False

    def _looks_like_caption(self, text: str) -> bool:
        text = re.sub(r"\s+", " ", text).strip()
        if not text or re.search(r"\bSupplementary\s+(?:Fig|Figure|Table)\b", text, re.I):
            return False
        match = CAPTION_RE.match(text)
        if not match:
            return False
        raw_id = match.group(1)
        number_match = re.search(r"(\d+)", raw_id)
        number = int(number_match.group(1)) if number_match else 0
        if number > 20:
            return False
        rest = text[match.end() :].lstrip()
        if re.match(r"^[),;]", rest):
            return False
        if re.match(r"^[a-z](?:[,)]|\b)", rest):
            return False
        if re.match(r"^[.:]\s*(These|This|The result|The results|after|before|reveals?|indicat(?:e|es)|suggests?)\b", rest, re.I):
            return False
        if re.match(r"^[\|:：]\s*\S+", rest):
            return True
        if re.match(r"^\.\s+(Overview|Characterization|Photovoltaic|Durability|Stability|Properties|Device|Figures of merit|Optoelectronic|Hole extraction|In-situ|The passivation|Performance)\b", rest, re.I):
            return True
        if re.match(r"^[-–]\s*\S+", rest):
            return True
        if re.match(r"^[A-Z][A-Za-z0-9,()/ -]{8,}", rest):
            return True
        return False

    def _looks_like_non_content(self, text: str) -> bool:
        if self._looks_like_caption(text):
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
        publication = self._extract_publication_info("\n".join(page_texts[:4]))
        title = ""
        for p in paragraphs[:12]:
            t = p.text_original
            doi = DOI_RE.search(t)
            if doi and len(t) > doi.end() + 20:
                candidate = self._clean_title_candidate(t[doi.end() :])
                if candidate:
                    title = candidate
                    break
            candidate = self._clean_title_candidate(t)
            if 20 < len(candidate) < 260 and not self._looks_like_title_noise(candidate) and not DOI_RE.search(candidate) and not SECTION_HINT_RE.match(candidate) and not self._looks_like_non_content(candidate):
                title = candidate
                break
            if self._looks_like_author_line(t):
                continue
        abstract = ""
        abstract = self._extract_labeled_abstract(page_texts)
        if not abstract:
            abstract = self._guess_abstract_from_layout(paragraphs, title)
        authors = self._extract_authors(paragraphs, title)
        return PaperInfo(
            title=title,
            authors=authors,
            doi=doi_match.group(0) if doi_match else "",
            journal=publication.get("journal", ""),
            volume=publication.get("volume", ""),
            issue=publication.get("issue", ""),
            pages=publication.get("pages", ""),
            year=publication.get("year", ""),
            abstract=abstract,
        )

    def _clean_title_candidate(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" .;:-")
        text = re.sub(r"^(Article|Research Article|Original Article|ARTICLE IN PRESS)\b[:.\s-]*", "", text, flags=re.I).strip()
        text = re.sub(r"^Nature Communications Article in Press\b[:.\s-]*", "", text, flags=re.I).strip()
        return text

    def _looks_like_title_noise(self, text: str) -> bool:
        return bool(
            re.search(
                r"^(Nature Communications Article in Press|ARTICLE IN PRESS|Received:|Accepted:|Published|Cite this|Cite This|"
                r"We are providing|If this paper|Open Access|ACCESS Metrics|Check for updates|Supporting Information)\b",
                text,
                re.I,
            )
        )

    def _extract_labeled_abstract(self, page_texts: list[str]) -> str:
        text = "\n".join(page_texts[:8])
        text = re.sub(r"\s+", " ", text).strip()
        candidates = []
        for match in re.finditer(r"\bAbstract\b\s*(.+?)(?=\bIntroduction\b|\bKeywords\b|\bResults\b|\bMethods\b|1\.\s*Introduction)", text, re.I | re.S):
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
            if len(candidate) >= 120 and not re.match(r"^(ARTICLE IN PRESS|maximum power point)", candidate, re.I):
                candidates.append(candidate)
        if candidates:
            return max(candidates, key=len)
        return ""

    def _extract_authors(self, paragraphs: list[Paragraph], title: str) -> list[str]:
        if not title:
            return []
        title_seen = False
        for p in paragraphs[:20]:
            text = p.text_original
            if p.source_page != 1 or (p.bbox and p.bbox[1] > 430):
                continue
            if text == title or self._clean_title_candidate(text) == title or (title in text and DOI_RE.search(text)):
                title_seen = True
                continue
            if not title_seen and p.bbox and p.bbox[1] > 120:
                title_seen = True
            if not title_seen:
                continue
            if self._looks_like_author_line(text):
                cleaned = re.sub(r"(?<=[A-Za-z])\d+(?:,\d+)*", "", text)
                cleaned = re.sub(r"\b\d+(?:,\d+)*\b", "", cleaned)
                cleaned = re.sub(r"\*|†|‡|§", "", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
                authors = [a.strip(" ,;") for a in re.split(r",|\s+&\s+|\s+and\s+", cleaned) if a.strip(" ,;")]
                return authors[:40]
        return []

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
                r"\b(Nature Communications|Nature Photonics|Nature Energy|Science|Joule|Advanced Materials|Energy & Environmental Science|"
                r"ACS Energy Letters|ACS Energy Lett\.|Nano Energy|Cell Reports Physical Science|Chemical Engineering Journal|Solar RRL|Nat Commun)\b",
                normalized,
                re.I,
            )
            if journal_match:
                info["journal"] = journal_match.group(1)
                if info["journal"].lower() == "nat commun":
                    info["journal"] = "Nature Communications"
                if info["journal"].lower() == "acs energy lett.":
                    info["journal"] = "ACS Energy Letters"

        acs_match = re.search(r"\bACS Energy Lett\.\s*((?:19|20)\d{2}),\s*(\d+),\s*([A-Za-z]?\d+(?:[-−–]\d+)?)", normalized, re.I)
        if acs_match:
            info["journal"] = "ACS Energy Letters"
            info["year"] = acs_match.group(1)
            info["volume"] = acs_match.group(2)
            info["pages"] = acs_match.group(3).replace("−", "-").replace("–", "-")

        nat_match = re.search(r"\bNat(?:ure)?\s+Commun(?:ications)?\s*\(((?:19|20)\d{2})\)", normalized, re.I)
        if nat_match:
            info["journal"] = "Nature Communications"
            info["year"] = nat_match.group(1)

        if not info["volume"]:
            volume_match = re.search(r"\b(?:vol(?:ume)?\.?\s*)?(\d+[A-Za-z]?)\s*\(([^)]+)\)\s*[:;,]\s*([A-Za-z]?\d+(?:[-−–]\d+)?)", normalized, re.I)
            if volume_match:
                info["volume"] = volume_match.group(1)
                issue = volume_match.group(2).strip()
                info["issue"] = "" if re.search(r"\b(has|ref|refs|figure|table)\b", issue, re.I) else issue
                info["pages"] = volume_match.group(3).replace("−", "-").replace("–", "-")
        if not info["volume"]:
            volume_match = re.search(r"\b(?:vol(?:ume)?\.?\s*)(\d+[A-Za-z]?)\b", normalized, re.I)
            if volume_match:
                info["volume"] = volume_match.group(1)
        if not info["issue"]:
            issue_match = re.search(r"\bissue\s+([A-Za-z0-9.-]+)\b", normalized, re.I)
            if issue_match:
                issue = issue_match.group(1)
                if not re.search(r"\b(has|for|with|and|the|of|limits?)\b", issue, re.I):
                    info["issue"] = issue
        if not info["pages"]:
            pages_match = re.search(r"\b(?:pp?\.?|pages?)\s*([A-Za-z]?\d+(?:[-−–]\d+)?)\b", normalized, re.I)
            if pages_match:
                info["pages"] = pages_match.group(1).replace("−", "-").replace("–", "-")
        if not info["pages"]:
            article_match = re.search(r"\|\s*\(((?:19|20)\d{2})\)\s*(\d+[A-Za-z]?)\s*[:;,]\s*([A-Za-z]?\d+(?:[-−–]\d+)?)", normalized)
            if article_match:
                info["year"] = info["year"] or article_match.group(1)
                info["volume"] = info["volume"] or article_match.group(2)
                info["pages"] = article_match.group(3).replace("−", "-").replace("–", "-")
        if not info["year"]:
            published = re.search(r"\b(?:Published(?: online)?|Copyright|Cite This:|Cite this article as:)[^.;]{0,160}\b((?:19|20)\d{2})\b", normalized, re.I)
            accepted = re.search(r"\bAccepted:[^.;]{0,80}\b((?:19|20)\d{2})\b", normalized, re.I)
            year_match = published or accepted or YEAR_RE.search(normalized)
            if year_match:
                info["year"] = year_match.group(1) if year_match.lastindex else year_match.group(0)
        return info

    def _looks_like_author_line(self, text: str) -> bool:
        if len(text) > 1200 or DOI_RE.search(text) or SECTION_HINT_RE.match(text):
            return False
        if re.search(r"\b(Abstract|Introduction|Received:|Accepted:|Published|Cite this|Open Access|ARTICLE IN PRESS|ACCESS Metrics|Supporting Information)\b", text, re.I):
            return False
        if "." in text and len(text) < 220:
            return False
        if not ("," in text or "&" in text):
            return False
        if re.search(r"[A-Z][a-z]+", text):
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
            if text[:1].islower() or self._split_embedded_heading(text):
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
                if remainder and not in_references:
                    para.kind = "paragraph"
                    para.text_original = remainder
                    current.paragraphs.append(para)
            else:
                if in_references:
                    para.kind = "non_content"
                embedded = self._split_embedded_heading(para.text_original)
                if embedded and not in_references:
                    before, embedded_title, after = embedded
                    if before:
                        before_para = Paragraph(
                            source_page=para.source_page,
                            text_original=before,
                            bbox=para.bbox,
                            kind=para.kind,
                        )
                        self._append_paragraph(current, before_para)
                    if current.paragraphs or current.section_title != "正文":
                        sections.append(current)
                    current = Section(section_title=embedded_title)
                    para.text_original = after
                    para.kind = "paragraph"
                    if after:
                        self._append_paragraph(current, para)
                    continue
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
        specific = re.match(r"^(Results and discussion)\b[:.\s-]*(.*)$", text, re.I)
        if specific:
            return specific.group(1), specific.group(2).strip()
        match = re.match(
            r"^(Abstract|Introduction|Results|Discussion|Conclusion|Conclusions|Methods|Materials and methods|Experimental|References|Acknowledg(?:ements)?|Supplementary|Data availability)\b[:.\s-]*(.*)$",
            text,
            re.I,
        )
        if match:
            if match.group(1).lower() == "supplementary":
                return text, ""
            return match.group(1), match.group(2).strip()
        extended = re.match(
            r"^(Package Description|Package Testing|Outdoor Testing|Experimental Methods|Device Fabrication|Fabrication of [^.]{5,80}|"
            r"Stability Tests?|Characterization|Data Availability|Code Availability)\b[.:]?\s*(.*)$",
            text,
            re.I,
        )
        if extended:
            return extended.group(1), extended.group(2).strip()
        return text, ""

    def _split_embedded_heading(self, text: str) -> tuple[str, str, str] | None:
        match = re.search(
            r"(?<=\.)\s+("
            r"Package Description|Package Testing|Outdoor Testing|Experimental Methods|Device Fabrication|"
            r"Results and Discussion|Conclusions?"
            r")\.\s+",
            text,
            re.I,
        )
        if not match:
            return None
        before = text[: match.start()].strip()
        title = match.group(1).strip()
        after = text[match.end() :].strip()
        if len(before) < 40 and len(after) < 40:
            return None
        return before, title, after
