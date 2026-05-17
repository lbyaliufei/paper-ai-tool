from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Paragraph:
    source_page: int
    text_original: str
    text_zh: str = ""
    bbox: list[float] | None = None
    kind: str = "paragraph"
    font_size: float | None = None
    is_bold: bool = False
    heading_level: int = 0


@dataclass
class Section:
    section_title: str
    section_title_zh: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)
    heading_level: int = 1


@dataclass
class PaperInfo:
    title: str = ""
    title_zh: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    year: str = ""
    abstract: str = ""
    abstract_zh: str = ""


@dataclass
class Figure:
    figure_id: str
    page: int
    caption_original: str
    caption_zh: str = ""
    bbox: list[float] = field(default_factory=list)
    image_mime: str = "image/png"
    image_base64: str = ""
    caption_bbox: list[float] = field(default_factory=list)
    warning: str = ""
    image_source: str = "heuristic"


@dataclass
class ParsedPaper:
    paper_info: PaperInfo
    sections: list[Section]
    figures: list[Figure]
    full_text: str
    warnings: list[str] = field(default_factory=list)


def dataclass_to_dict(obj: Any) -> Any:
    return asdict(obj)


def empty_structured_data() -> dict[str, Any]:
    return {
        "paper_info": asdict(PaperInfo()),
        "sections": [],
        "figures": [],
        "device_performance": [],
        "stability_tests": [],
        "experimental_methods": [],
        "characterization_methods": [],
        "key_innovations": [],
        "new_knowledge": [],
        "limitations": [],
        "warnings": [],
    }
