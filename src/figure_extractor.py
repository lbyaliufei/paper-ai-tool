from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import fitz
from PIL import Image

from .models import Figure
from .utils import bytes_to_base64, ensure_dir


CAPTION_RE = re.compile(r"^(Fig\.\s*\d+[A-Za-z]?|Figure\s*\d+[A-Za-z]?|Table\s*\d+[A-Za-z]?|图\s*\d+)\s*[\|.:：-]?\s*(.*)", re.I)


class FigureExtractor:
    def __init__(self, logger: logging.Logger, zoom: float = 2.5):
        self.logger = logger
        self.zoom = zoom

    def extract(self, pdf_path: Path, blocks: list[dict], image_format: str = "png", compress: bool = True, debug_dir: Path | None = None) -> list[Figure]:
        figures: list[Figure] = []
        doc = fitz.open(pdf_path)
        if debug_dir:
            ensure_dir(debug_dir)
        try:
            for block in blocks:
                match = CAPTION_RE.match(block["text"])
                if not match:
                    continue
                page_num = block["page"]
                page = doc[page_num - 1]
                fig_id = self._normalize_id(match.group(1))
                caption_blocks = self._caption_blocks(block, blocks)
                caption = " ".join(b["text"] for b in caption_blocks)
                bbox = self._infer_figure_bbox(page, block, blocks)
                image_bytes, mime, warning = self._crop_page(page, bbox, image_format, compress)
                base64_data = bytes_to_base64(image_bytes) if image_bytes else ""
                if debug_dir and image_bytes:
                    suffix = "jpg" if mime == "image/jpeg" else "png"
                    (debug_dir / f"{fig_id.replace(' ', '_').replace('.', '')}.{suffix}").write_bytes(image_bytes)
                figures.append(
                    Figure(
                        figure_id=fig_id,
                        page=page_num,
                        caption_original=caption,
                        bbox=[round(v, 2) for v in bbox],
                        image_mime=mime,
                        image_base64=base64_data,
                        caption_bbox=[round(v, 2) for v in self._union_bbox([b["bbox"] for b in caption_blocks])],
                        warning=warning,
                    )
                )
        except Exception as exc:
            self.logger.exception("Figure extraction failed: %s", exc)
        finally:
            doc.close()
        return self._dedupe(figures)

    def _normalize_id(self, raw: str) -> str:
        raw = re.sub(r"\s+", " ", raw.strip())
        raw = re.sub(r"^Figure", "Fig.", raw, flags=re.I)
        return raw

    def _caption_blocks(self, caption_block: dict, blocks: list[dict]) -> list[dict]:
        cap_x0, cap_y0, cap_x1, cap_y1 = caption_block["bbox"]
        same_page = [b for b in blocks if b["page"] == caption_block["page"]]
        selected = []
        for block in same_page:
            if block is caption_block:
                continue
            x0, y0, x1, y1 = block["bbox"]
            text = block["text"]
            if CAPTION_RE.match(text):
                continue
            same_row = abs(y0 - cap_y0) < 8 and x0 > cap_x0
            wrapped_below = cap_y1 <= y0 <= cap_y1 + 16 and x0 <= cap_x1 + 20 and len(text) > 40
            if same_row or wrapped_below:
                selected.append(block)
        return [caption_block] + sorted(selected, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))

    def _union_bbox(self, bboxes: list[list[float]]) -> list[float]:
        return [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]

    def _infer_figure_bbox(self, page: fitz.Page, caption_block: dict, blocks: list[dict]) -> list[float]:
        page_rect = page.rect
        cap_x0, cap_y0, cap_x1, cap_y1 = caption_block["bbox"]
        same_page = [b for b in blocks if b["page"] == caption_block["page"] and b is not caption_block]
        above = [b for b in same_page if b["bbox"][3] < cap_y0 - 8]
        page_margin = 8.0
        top_crop_min = 40.0
        horizontal_pad = 34.0
        top_pad = 18.0
        bottom_pad = 4.0
        previous_y = top_crop_min
        if cap_y0 > page_rect.height * 0.65:
            previous_y = top_crop_min
        elif above:
            candidates = [b for b in above if b["bbox"][3] > page_rect.height * 0.12]
            if candidates:
                previous_y = max(b["bbox"][3] for b in candidates)
        x0 = max(page_margin, min(cap_x0 - horizontal_pad, page_rect.width * 0.015))
        x1 = min(page_rect.width - page_margin, max(cap_x1 + horizontal_pad, page_rect.width * 0.985))
        y0 = max(top_crop_min, previous_y - top_pad)
        y1 = max(y0 + 80.0, cap_y0 - bottom_pad)
        if y1 - y0 < 100:
            y0 = max(top_crop_min, cap_y0 - page_rect.height * 0.52)
            y1 = max(y0 + 80.0, cap_y0 - bottom_pad)
        return [x0, y0, x1, min(y1, page_rect.height - page_margin)]

    def _crop_page(self, page: fitz.Page, bbox: list[float], image_format: str, compress: bool) -> tuple[bytes, str, str]:
        try:
            clip = fitz.Rect(*bbox) & page.rect
            if clip.is_empty or clip.height < 20 or clip.width < 20:
                return b"", "image/png", "Inferred figure bbox is too small."
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), clip=clip, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            out = io.BytesIO()
            fmt = image_format.lower()
            if fmt in {"jpg", "jpeg"}:
                img.save(out, format="JPEG", quality=82 if compress else 95, optimize=True)
                return out.getvalue(), "image/jpeg", ""
            img.save(out, format="PNG", optimize=compress)
            return out.getvalue(), "image/png", ""
        except Exception as exc:
            self.logger.exception("Could not crop figure: %s", exc)
            return b"", "image/png", str(exc)

    def _dedupe(self, figures: list[Figure]) -> list[Figure]:
        seen: set[tuple[str, int]] = set()
        unique: list[Figure] = []
        for fig in figures:
            key = (fig.figure_id.lower(), fig.page)
            if key in seen:
                continue
            seen.add(key)
            unique.append(fig)
        return unique
