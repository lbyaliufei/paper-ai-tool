from __future__ import annotations

import base64
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


def safe_slug(name: str) -> str:
    stem = Path(name).stem or "paper"
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip()
    return stem or "paper"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"paper_ai_tool.{log_file}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            return None
    return None


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in re.split(r"\n\s*\n", text):
        if size + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def parse_operator_number(raw: str) -> tuple[str, float | None]:
    match = re.search(r"([>~<≈≥≤])?\s*(\d+(?:\.\d+)?)", raw)
    if not match:
        return "", None
    op = match.group(1) or ""
    if op == "≈":
        op = "~"
    return op, float(match.group(2))
