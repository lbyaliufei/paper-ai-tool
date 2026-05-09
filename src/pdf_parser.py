from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from .models import PaperInfo, Paragraph, ParsedPaper, Section


CAPTION_RE = re.compile(
    r"^(Extended\s+Data\s+Fig\.?\s*\d+[A-Za-z]?|Fig\.?\s*\d+[A-Za-z]?|Figure\s+\d+[A-Za-z]?|"
    r"Table\s+(?:\d+[A-Za-z]?|[IVXLCDM]+)|Scheme\s+\d+[A-Za-z]?|Chart\s+\d+[A-Za-z]?|Box\s+\d+[A-Za-z]?|图\s*\d+)",
    re.I,
)
SECTION_HINT_RE = re.compile(
    r"^(Abstract|Introduction|Background|Results(?: and discussion)?|Discussion|Conclusion|Conclusions|"
    r"Methods?|Materials and methods|Methodology|Experimental(?: section| details)?|"
    r"References|Bibliography|Notes and references|Acknowledg(?:ement|ements)?|"
    r"Supplementary|Data availability|Code availability|Appendix|Author contributions|"
    r"Competing interests?|Conflict of interest|Conflicts of interest|Associated content|Supporting information)\b",
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
    r"supplementary information|supporting information|associated content|"
    r"science direct|sciencedirect|elsevier|article in press|available online|"
    r"highlights|graphical abstract|keywords|index terms|nomenclature|"
    r"open access|access metrics|article recommendations|"
    r"reporting summary|"
    r"author contributions|competing interests|conflict of interest|conflicts of interest|"
    r"data availability|code availability|"
    r"references\b|bibliography\b|notes and references\b"
    r")",
    re.I,
)


class PDFParser:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def parse(self, pdf_path: Path) -> tuple[ParsedPaper, list[dict]]:
        doc = fitz.open(pdf_path)
        metadata = dict(doc.metadata or {})
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
        paper_info = self._extract_info(page_texts, paragraphs, metadata)
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
        if re.match(r"^(Supplementary|Supporting)\s+(?:Fig\.?|Figure|Table|Note|Information)\b", text, re.I):
            return False
        section_match = SECTION_HINT_RE.match(text)
        if section_match:
            keyword = section_match.group(1)
            low_keyword = keyword.lower()
            if low_keyword in {
                "supplementary",
                "supporting information",
                "associated content",
                "author contributions",
                "competing interest",
                "competing interests",
                "conflict of interest",
                "conflicts of interest",
            }:
                return False
            rest = text[section_match.end() :].strip()
            if rest and not re.match(r"^[:.\-–]", rest):
                inline_allowed = low_keyword in {
                    "results",
                    "results and discussion",
                    "discussion",
                    "conclusion",
                    "conclusions",
                    "method",
                    "methods",
                    "methodology",
                    "references",
                    "bibliography",
                    "notes and references",
                    "acknowledg",
                    "acknowledgement",
                    "acknowledgements",
                    "data availability",
                    "code availability",
                }
                if not inline_allowed:
                    return False
            return True
        if re.match(r"^(?:\d+\.?|[IVXLCDM]+\.?)\s+(Introduction|Background|Results|Discussion|Conclusion|Methods|Methodology|Experimental)\b", text, re.I):
            return True
        if re.match(
            r"^(Package Description|Package Testing|Outdoor Testing|Experimental Methods|Experimental Details|"
            r"Device Fabrication|Data Availability|Sample Preparation|Materials Characterization|"
            r"Solar Cell Fabrication|Device Characterization)\.?\b",
            text,
            re.I,
        ):
            return True
        return False

    def _looks_like_caption(self, text: str) -> bool:
        text = re.sub(r"\s+", " ", text).strip()
        if not text or re.search(r"\b(Supplementary|Supporting)\s+(?:Fig|Figure|Table)\b", text, re.I):
            return False
        match = CAPTION_RE.match(text)
        if not match:
            return False
        raw_id = match.group(1)
        if re.match(r"^Table\s+[IVXLCDM]+$", raw_id, re.I):
            number = 1
        else:
            number_match = re.search(r"(\d+)", raw_id)
            number = int(number_match.group(1)) if number_match else 0
        if number > 80:
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
        if re.match(
            r"^\.\s+(Overview|Characterization|Photovoltaic|Durability|Stability|Properties|Device|"
            r"Figures of merit|Optoelectronic|Hole extraction|In-situ|The passivation|Performance|"
            r"Summary|Comparison|Schematic|Architecture|Mechanism|Experimental|Structural|Electrical|Optical)\b",
            rest,
            re.I,
        ):
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

    def _extract_info(self, page_texts: list[str], paragraphs: list[Paragraph], metadata: dict | None = None) -> PaperInfo:
        first_page = page_texts[0] if page_texts else ""
        doi_match = DOI_RE.search(first_page)
        publication = self._extract_publication_info("\n".join(page_texts[:4]))
        metadata_title = self._clean_title_candidate((metadata or {}).get("title", "") or "")
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
        if (not title or self._looks_like_title_noise(title)) and 20 < len(metadata_title) < 260:
            if not self._looks_like_title_noise(metadata_title) and not DOI_RE.search(metadata_title):
                title = metadata_title
        abstract = ""
        abstract = self._extract_labeled_abstract(page_texts)
        if not abstract:
            abstract = self._guess_abstract_from_layout(paragraphs, title)
        authors = self._extract_authors(paragraphs, title)
        if not authors and metadata:
            metadata_author = metadata.get("author", "") or metadata.get("Author", "")
            authors = self._parse_author_names(metadata_author)
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
        text = re.sub(r"^(Article|Research Article|Original Article|Review Article|Communication|Letter|ARTICLE IN PRESS|Open Access)\b[:.\s-]*", "", text, flags=re.I).strip()
        text = re.sub(r"^Nature Communications Article in Press\b[:.\s-]*", "", text, flags=re.I).strip()
        return text

    def _looks_like_title_noise(self, text: str) -> bool:
        return bool(
            re.search(
                r"^(Nature Communications Article in Press|ARTICLE IN PRESS|Received:|Accepted:|Published|Cite this|Cite This|"
                r"We are providing|If this paper|Open Access|ACCESS Metrics|Check for updates|Supporting Information|"
                r"ScienceDirect|Available online|Highlights|Graphical abstract|Keywords|Index Terms|Research Article|Review Article)\b",
                text,
                re.I,
            )
        )

    def _extract_labeled_abstract(self, page_texts: list[str]) -> str:
        text = "\n".join(page_texts[:8])
        text = re.sub(r"\s+", " ", text).strip()
        candidates = []
        stop = (
            r"\b(?:Introduction|Keywords?|Index Terms|Background|Results|Methods?|Experimental|"
            r"1\.\s*Introduction|I\.\s*Introduction|Highlights|Graphical Abstract)\b"
        )
        for match in re.finditer(rf"\bAbstract\b\s*(?:[-—:]\s*)?(.+?)(?={stop})", text, re.I | re.S):
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .:-—")
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
                return self._parse_author_names(cleaned)
        return []

    def _parse_author_names(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip(" ,;")
        if not cleaned:
            return []
        cleaned = re.sub(r"\b(?:and|&)\b", ",", cleaned)
        cleaned = re.sub(r";", ",", cleaned)
        authors = [a.strip(" ,;") for a in cleaned.split(",") if a.strip(" ,;")]
        return [a for a in authors if 1 < len(a) < 80][:40]

    def _extract_publication_info(self, text: str) -> dict[str, str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        info = {"journal": "", "volume": "", "issue": "", "pages": "", "year": ""}
        if not normalized:
            return info

        ieee_match = re.search(
            r"\b(?P<journal>IEEE\s+[A-Z][A-Z0-9 &/\-]+?)\s*,?\s+VOL\.?\s*(?P<volume>\d+[A-Za-z]?)"
            r"(?:\s*,?\s+NO\.?\s*(?P<issue>\d+[A-Za-z]?))?.{0,80}?\b(?P<year>(?:19|20)\d{2})\b",
            normalized,
            re.I,
        )
        if ieee_match:
            for key in info:
                value = ieee_match.groupdict().get(key) or ""
                info[key] = value.strip(" .,:;|")
            info["journal"] = self._clean_journal_name(info["journal"])

        citation_patterns = [
            # Elsevier and many Springer PDFs: Journal Name 123 (2026) 123456
            r"(?P<journal>[A-Z][A-Za-z0-9 &.,+\-/]{3,90}?)\s+(?P<volume>\d{1,4}[A-Za-z]?)\s*"
            r"\((?P<year>(?:19|20)\d{2})\)\s*(?P<pages>[A-Za-z]?\d+(?:[-−–]\d+)?)",
            # RSC/ACS/Wiley compact citations: Journal Name, 2026, 12, 1234-1245
            r"(?P<journal>[A-Z][A-Za-z0-9 &.,+\-/]{3,90}?),\s*(?P<year>(?:19|20)\d{2}),\s*"
            r"(?P<volume>\d{1,4}[A-Za-z]?)\s*,\s*(?P<pages>[A-Za-z]?\d+(?:[-−–]\d+)?|[A-Za-z]\d{4,})",
            # MDPI style: Journal 2026, 18, 1234
            r"(?P<journal>[A-Z][A-Za-z0-9 &.,+\-/]{3,90}?)\s+(?P<year>(?:19|20)\d{2}),\s*"
            r"(?P<volume>\d{1,4}[A-Za-z]?)\s*,\s*(?P<pages>[A-Za-z]?\d+(?:[-−–]\d+)?|[A-Za-z]\d{4,})",
            # Wiley style: Advanced Energy Materials 2026, 16, 2500000
            r"(?P<journal>Adv(?:anced)?\.?\s+[A-Za-z &]+|Angew(?:andte)?\.?\s+[A-Za-z &]+|Small|Solar RRL)"
            r"\s*,?\s*(?P<year>(?:19|20)\d{2})\s*,\s*(?P<volume>\d{1,4}[A-Za-z]?)"
            r"\s*,\s*(?P<pages>[A-Za-z]?\d+(?:[-−–]\d+)?)",
        ]
        for pattern in citation_patterns:
            if info["journal"] and info["year"]:
                break
            match = re.search(pattern, normalized, re.I)
            if match and self._valid_journal_candidate(match.group("journal")):
                for key in info:
                    value = match.groupdict().get(key) or ""
                    if value and not info[key]:
                        info[key] = value.strip(" .,:;|").replace("−", "-").replace("–", "-")
                info["journal"] = self._clean_journal_name(info["journal"])

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
            info["journal"] = self._clean_journal_name(info["journal"])

        if not info["journal"]:
            journal_match = re.search(
                r"\b(Nature Communications|Nature Photonics|Nature Energy|Science|Joule|Advanced Materials|Energy & Environmental Science|"
                r"ACS Energy Letters|ACS Energy Lett\.|Nano Energy|Cell Reports Physical Science|Chemical Engineering Journal|Solar RRL|Nat Commun|"
                r"Advanced Energy Materials|Advanced Functional Materials|Angewandte Chemie|Journal of Materials Chemistry A|"
                r"Energy & Environmental Materials|Matter|Chem|Device|Science Advances|PNAS|Small|Small Methods|"
                r"IEEE [A-Z][A-Z0-9 &/\-]+|Applied Physics Letters|Physical Review [A-Z])\b",
                normalized,
                re.I,
            )
            if journal_match:
                info["journal"] = self._clean_journal_name(journal_match.group(1))

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

    def _clean_journal_name(self, journal: str) -> str:
        cleaned = re.sub(r"\s+", " ", journal or "").strip(" .,:;|")
        cleaned = re.sub(r"^(Cite This:|Original Article|Research Article|Article)\s*", "", cleaned, flags=re.I).strip()
        aliases = {
            "nat commun": "Nature Communications",
            "acs energy lett": "ACS Energy Letters",
            "acs energy lett.": "ACS Energy Letters",
        }
        return aliases.get(cleaned.lower(), cleaned)

    def _valid_journal_candidate(self, journal: str) -> bool:
        if not journal:
            return False
        if len(journal) > 100:
            return False
        return not bool(
            re.search(
                r"\b(Abstract|Introduction|Results|Discussion|Methods|References|Figure|Table|Supplementary|"
                r"Received|Accepted|Published|Downloaded|Copyright|Open Access|Article in Press)\b",
                journal,
                re.I,
            )
        )

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
                if re.search(r"^(references|bibliography|notes and references)\b", para.text_original, re.I):
                    in_references = True
                continue
            if para.kind == "heading":
                if current.paragraphs or current.section_title != "正文":
                    sections.append(current)
                title, remainder = self._split_inline_heading(para.text_original)
                current = Section(section_title=title)
                in_references = bool(re.search(r"^(references|bibliography|notes and references)\b", title, re.I))
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
                r"Fitting of Fick|Characterization|Device fabrication|Sample preparation|Materials characterization|"
                r"Solar cell fabrication|Device characterization|Experimental details|Synthesis)\b",
                text,
                re.I,
            )
        )

    def _union_bbox(self, a: list[float], b: list[float]) -> list[float]:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def _split_inline_heading(self, text: str) -> tuple[str, str]:
        specific = re.match(r"^(Results and discussion)\b[:.\s-]*(.*)$", text, re.I)
        if specific:
            return specific.group(1), specific.group(2).strip()
        match = re.match(
            r"^(Abstract|Introduction|Background|Results(?: and discussion)?|Discussion|Conclusion|Conclusions|"
            r"Methods?|Materials and methods|Methodology|Experimental(?: section| details)?|References|Bibliography|"
            r"Notes and references|Acknowledg(?:ement|ements)?|Supplementary|Data availability|Code availability|Appendix)\b[:.\s-]*(.*)$",
            text,
            re.I,
        )
        if match:
            if match.group(1).lower() == "supplementary":
                return text, ""
            return match.group(1), match.group(2).strip()
        extended = re.match(
            r"^(Package Description|Package Testing|Outdoor Testing|Experimental Methods|Device Fabrication|Fabrication of [^.]{5,80}|"
            r"Stability Tests?|Characterization|Data Availability|Code Availability|Sample Preparation|Materials Characterization|"
            r"Solar Cell Fabrication|Device Characterization|Synthesis|Film Deposition|Module Fabrication)\b[.:]?\s*(.*)$",
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
            r"Results and Discussion|Experimental Details|Sample Preparation|Materials Characterization|"
            r"Solar Cell Fabrication|Device Characterization|Conclusions?"
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
