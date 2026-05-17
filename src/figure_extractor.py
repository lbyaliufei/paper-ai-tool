from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import fitz
from PIL import Image

from .models import Figure
from .utils import bytes_to_base64, ensure_dir


CAPTION_RE = re.compile(
    r"^(Extended\s+Data\s+Fig\.?\s*\d+[A-Za-z]?|Fig\.?\s*\d+[A-Za-z]?|Figure\s+\d+[A-Za-z]?|"
    r"Table\s+(?:\d+[A-Za-z]?|[IVXLCDM]+)|Scheme\s+\d+[A-Za-z]?|Chart\s+\d+[A-Za-z]?|Box\s+\d+[A-Za-z]?|图\s*\d+)",
    re.I,
)


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
                if not self._looks_like_caption(block["text"]):
                    continue
                match = CAPTION_RE.match(block["text"])
                if not match:
                    continue
                page_num = block["page"]
                page = doc[page_num - 1]
                fig_id = self._normalize_id(match.group(1))
                caption_blocks = self._caption_blocks(block, blocks)
                caption = " ".join(b["text"] for b in caption_blocks)
                if self._caption_kind(fig_id) == "table":
                    bbox = self._infer_table_bbox(page, block, blocks)
                    source = "heuristic"
                else:
                    content_bbox, source = self._find_figure_from_content(page, block["bbox"])
                    if content_bbox is not None:
                        bbox = content_bbox
                    else:
                        bbox = self._infer_figure_bbox(page, block, blocks)
                        source = "heuristic"
                target_page = page
                target_page_num = page_num
                if self._should_try_next_page(page, block) and page_num < len(doc):
                    next_page = doc[page_num]
                    next_bbox = self._infer_next_page_figure_bbox(next_page, blocks, page_num + 1)
                    if self._figure_region_score(next_page, next_bbox) > self._figure_region_score(page, bbox) * 1.35:
                        target_page = next_page
                        target_page_num = page_num + 1
                        bbox = next_bbox
                if self._is_legend_only_caption(block, blocks):
                    image_bytes, mime, warning = b"", "image/png", "Caption appears in a Figure Legends section; no nearby figure image was cropped."
                else:
                    image_bytes, mime, warning = self._crop_page(target_page, bbox, image_format, compress)
                base64_data = bytes_to_base64(image_bytes) if image_bytes else ""
                if debug_dir and image_bytes:
                    suffix = "jpg" if mime == "image/jpeg" else "png"
                    (debug_dir / f"{fig_id.replace(' ', '_').replace('.', '')}.{suffix}").write_bytes(image_bytes)
                figures.append(
                    Figure(
                        figure_id=fig_id,
                        page=target_page_num,
                        caption_original=caption,
                        bbox=[round(v, 2) for v in bbox],
                        image_mime=mime,
                        image_base64=base64_data,
                        caption_bbox=[round(v, 2) for v in self._union_bbox([b["bbox"] for b in caption_blocks])],
                        warning=warning,
                        image_source=source,
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
        raw = re.sub(r"^Fig\s+", "Fig. ", raw, flags=re.I)
        raw = re.sub(r"^Extended Data Fig\s+", "Extended Data Fig. ", raw, flags=re.I)
        return raw

    def _caption_kind(self, figure_id: str) -> str:
        if re.match(r"^Table\b", figure_id, re.I):
            return "table"
        if re.match(r"^(Scheme|Chart|Box)\b", figure_id, re.I):
            return "graphic"
        return "figure"

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

    def _is_legend_only_caption(self, caption_block: dict, blocks: list[dict]) -> bool:
        page = caption_block["page"]
        cap_y0 = caption_block["bbox"][1]
        same_page = [b for b in blocks if b["page"] == page]
        has_legend_heading = any(
            re.fullmatch(r"Figure Legends?", b["text"].strip(), re.I) and b["bbox"][1] < cap_y0
            for b in same_page
        )
        if has_legend_heading:
            return True
        legend_pages = [
            b["page"]
            for b in blocks
            if re.fullmatch(r"Figure Legends?", b["text"].strip(), re.I)
        ]
        if legend_pages and min(legend_pages) <= page <= min(legend_pages) + 3:
            return True
        caption_count = sum(1 for b in same_page if self._looks_like_caption(b["text"]))
        return caption_count >= 3 and cap_y0 > 250

    def _caption_blocks(self, caption_block: dict, blocks: list[dict]) -> list[dict]:
        cap_x0, cap_y0, cap_x1, cap_y1 = caption_block["bbox"]
        same_page = [b for b in blocks if b["page"] == caption_block["page"]]
        selected = []
        for block in same_page:
            if block is caption_block:
                continue
            x0, y0, x1, y1 = block["bbox"]
            text = block["text"]
            if self._looks_like_caption(text):
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

    def _infer_table_bbox(self, page: fitz.Page, caption_block: dict, blocks: list[dict]) -> list[float]:
        page_rect = page.rect
        cap_x0, cap_y0, cap_x1, cap_y1 = caption_block["bbox"]
        same_page = [b for b in blocks if b["page"] == caption_block["page"] and b is not caption_block]
        below = [b for b in same_page if b["bbox"][1] > cap_y1 + 6]
        page_margin = 8.0
        horizontal_pad = 34.0
        y1 = page_rect.height - 44.0
        next_caption_y = [
            b["bbox"][1]
            for b in below
            if self._looks_like_caption(b["text"]) and b["bbox"][1] > cap_y1 + 30
        ]
        if next_caption_y:
            y1 = min(next_caption_y) - 8.0
        else:
            dense_text = [b for b in below if b["bbox"][1] < cap_y1 + page_rect.height * 0.45]
            if dense_text:
                y1 = max(b["bbox"][3] for b in dense_text) + 8.0
        x0 = max(page_margin, min(cap_x0 - horizontal_pad, page_rect.width * 0.015))
        x1 = min(page_rect.width - page_margin, max(cap_x1 + horizontal_pad, page_rect.width * 0.985))
        y0 = min(page_rect.height - 120.0, cap_y1 + 3.0)
        return [x0, max(40.0, y0), x1, min(max(y0 + 80.0, y1), page_rect.height - page_margin)]

    def _should_try_next_page(self, page: fitz.Page, caption_block: dict) -> bool:
        cap_y0 = caption_block["bbox"][1]
        return cap_y0 > page.rect.height * 0.72

    def _infer_next_page_figure_bbox(self, page: fitz.Page, blocks: list[dict], page_num: int) -> list[float]:
        page_rect = page.rect
        page_margin = 8.0
        top = 40.0
        bottom = page_rect.height - 44.0
        next_captions = [b for b in blocks if b["page"] == page_num and self._looks_like_caption(b["text"])]
        if next_captions:
            first_caption_y = min(b["bbox"][1] for b in next_captions)
            if first_caption_y > page_rect.height * 0.35:
                bottom = first_caption_y - 4.0
        return [page_margin, top, page_rect.width - page_margin, max(top + 100.0, bottom)]

    def _figure_region_score(self, page: fitz.Page, bbox: list[float]) -> float:
        try:
            clip = fitz.Rect(*bbox) & page.rect
            if clip.is_empty or clip.height < 50 or clip.width < 50:
                return 0.0
            pix = page.get_pixmap(matrix=fitz.Matrix(0.55, 0.55), clip=clip, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pixels = list(img.getdata())
            if not pixels:
                return 0.0
            non_white = 0
            saturated = 0
            dark = 0
            for r, g, b in pixels[:: max(1, len(pixels) // 120000)]:
                mx, mn = max(r, g, b), min(r, g, b)
                if mx < 245:
                    non_white += 1
                if mx - mn > 25:
                    saturated += 1
                if mx < 120:
                    dark += 1
            total = max(1, len(pixels[:: max(1, len(pixels) // 120000)]))
            return (non_white / total) + 1.8 * (saturated / total) + 0.25 * (dark / total)
        except Exception:
            return 0.0

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

    def _find_embedded_images(self, page: fitz.Page, min_size: float = 120.0) -> list[dict]:
        images: list[dict] = []
        page_rect = page.rect
        try:
            for img_info in page.get_images(full=True):
                try:
                    bbox = page.get_image_bbox(img_info)
                except Exception:
                    continue
                w, h = bbox.width, bbox.height
                if bbox.is_empty or w < min_size or h < min_size:
                    continue
                # Sanity check: bbox must be within page bounds and not occupy > 80% of page
                bbox_pct = (w * h) / (page_rect.width * page_rect.height)
                if bbox_pct > 0.85:
                    self.logger.debug("Skipping image that occupies %.0f%% of page", bbox_pct * 100)
                    continue
                images.append({
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "width": img_info[2],
                    "height": img_info[3],
                    "xref": img_info[0],
                })
        except Exception:
            pass
        return images

    def _find_drawings(self, page: fitz.Page, min_size: float = 80.0) -> list[list[float]]:
        page_rect = page.rect
        page_w, page_h = page_rect.width, page_rect.height
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        bboxes: list[list[float]] = []
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect is None:
                continue
            clip_x0 = max(0.0, rect.x0)
            clip_y0 = max(0.0, rect.y0)
            clip_x1 = min(page_w, rect.x1)
            clip_y1 = min(page_h, rect.y1)
            w = clip_x1 - clip_x0
            h = clip_y1 - clip_y0
            if w > min_size and h > min_size and w < page_w * 0.95 and h < page_h * 0.9:
                bboxes.append([clip_x0, clip_y0, clip_x1, clip_y1])
        if len(bboxes) <= 1:
            return bboxes
        merged: list[list[float]] = []
        sorted_boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        current = list(sorted_boxes[0])
        for bbox in sorted_boxes[1:]:
            if bbox[1] <= current[3] + 30 and abs(bbox[0] - current[0]) < 80:
                current[2] = max(current[2], bbox[2])
                current[3] = max(current[3], bbox[3])
            else:
                merged.append(current)
                current = list(bbox)
        merged.append(current)
        return merged

    def _find_figure_from_content(self, page: fitz.Page, caption_bbox: list[float]) -> tuple[list[float] | None, str]:
        cap_y0 = caption_bbox[1]
        page_height = page.rect.height
        page_width = page.rect.width
        page_area = page_width * page_height
        min_figure_area = page_area * 0.015
        candidates: list[tuple[list[float], float, str]] = []
        images = self._find_embedded_images(page, min_size=150.0)
        for img in images:
            img_bbox = img["bbox"]
            if img_bbox[3] < cap_y0 + 10 and img_bbox[1] > page_height * 0.03:
                area = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
                if area > min_figure_area:
                    candidates.append((img_bbox, area, "embedded"))
        drawings = self._find_drawings(page, min_size=150.0)
        for d_bbox in drawings:
            if d_bbox[3] < cap_y0 + 10 and d_bbox[1] > page_height * 0.03:
                area = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
                if area > min_figure_area:
                    candidates.append((d_bbox, area, "drawing"))
        if not candidates:
            if images:
                candidates = [(img["bbox"], (img["bbox"][2] - img["bbox"][0]) * (img["bbox"][3] - img["bbox"][1]), "embedded") for img in images if img["bbox"][3] < cap_y0 + 10]
            if not candidates and drawings:
                candidates = [(d, (d[2] - d[0]) * (d[3] - d[1]), "drawing") for d in drawings if d[3] < cap_y0 + 10]
        if not candidates:
            return (None, "heuristic")
        candidates.sort(key=lambda c: c[1], reverse=True)
        best_bbox, _area, source = candidates[0]
        # Validate the candidate: ensure it's not too far from the caption
        best_y1 = best_bbox[3]
        if best_y1 > cap_y0 + page_height * 0.4:
            self.logger.debug(
                "Content-figure candidate is too far from caption (y1=%.1f vs cap_y0=%.1f); falling back to heuristic",
                best_y1, cap_y0,
            )
            return (None, "heuristic")
        padded = self._pad_bbox(best_bbox, page.rect)
        return (padded, source)

    def _pad_bbox(self, bbox: list[float], page_rect: fitz.Rect, pad: float = 8.0) -> list[float]:
        return [
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(page_rect.width, bbox[2] + pad),
            min(page_rect.height, bbox[3] + pad),
        ]

    def _dedupe(self, figures: list[Figure]) -> list[Figure]:
        by_id: dict[str, Figure] = {}
        for fig in figures:
            key = fig.figure_id.lower().replace(" ", "")
            existing = by_id.get(key)
            if existing is None:
                by_id[key] = fig
                continue
            # Prefer the entry with image data; if both have image data, keep the one
            # with a larger base64 payload (more detail).
            existing_has = bool(existing.image_base64)
            fig_has = bool(fig.image_base64)
            if fig_has and not existing_has:
                by_id[key] = fig
            elif fig_has and existing_has and len(fig.image_base64) > len(existing.image_base64) * 1.1:
                # New one has significantly more image data – use it instead
                by_id[key] = fig
            # Keep existing otherwise (first seen wins in a tie)
        return list(by_id.values())
