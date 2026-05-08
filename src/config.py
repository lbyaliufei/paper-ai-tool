from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_output_dir() -> Path:
    configured = os.getenv("OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    desktop = Path.home() / "Desktop"
    if desktop.exists():
        return desktop

    localized_desktop = Path.home() / "桌面"
    if localized_desktop.exists():
        return localized_desktop

    return Path("outputs")


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "").strip())
    openai_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")))
    openai_max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    openai_stream: bool = field(default_factory=lambda: os.getenv("OPENAI_STREAM", "true").lower() in {"1", "true", "yes", "on"})
    openai_max_output_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "8192")))
    figure_zoom: float = 2.5
    default_image_format: str = "png"
    outputs_dir: Path = field(default_factory=default_output_dir)
    max_llm_chars: int = 70000
    translation_batch_size: int = field(default_factory=lambda: int(os.getenv("TRANSLATION_BATCH_SIZE", "6")))
    translation_batch_max_chars: int = field(default_factory=lambda: int(os.getenv("TRANSLATION_BATCH_MAX_CHARS", "6000")))
    translation_max_workers: int = field(default_factory=lambda: int(os.getenv("TRANSLATION_MAX_WORKERS", "2")))
    output_markdown: bool = field(default_factory=lambda: env_bool("OUTPUT_MARKDOWN", True))
    output_html: bool = field(default_factory=lambda: env_bool("OUTPUT_HTML", False))
    output_summary: bool = field(default_factory=lambda: env_bool("OUTPUT_SUMMARY", True))
    output_json: bool = field(default_factory=lambda: env_bool("OUTPUT_JSON", False))
    output_excel: bool = field(default_factory=lambda: env_bool("OUTPUT_EXCEL", False))
    output_source_pdf: bool = field(default_factory=lambda: env_bool("OUTPUT_SOURCE_PDF", False))
    output_debug_figures: bool = field(default_factory=lambda: env_bool("OUTPUT_DEBUG_FIGURES", False))
    run_structured_extraction: bool = field(default_factory=lambda: env_bool("RUN_STRUCTURED_EXTRACTION", False))


TERMINOLOGY = {
    "perovskite solar cells": "钙钛矿太阳能电池",
    "power conversion efficiency": "光电转换效率",
    "PCE": "PCE",
    "open-circuit voltage": "开路电压",
    "short-circuit current density": "短路电流密度",
    "fill factor": "填充因子",
    "hole transport layer": "空穴传输层",
    "electron transport layer": "电子传输层",
    "maximum power point tracking": "最大功率点跟踪",
    "damp-heat stability": "湿热稳定性",
    "light/dark cycling": "光暗循环",
    "retained initial efficiency": "初始效率保持率",
    "iodide migration": "碘离子迁移",
    "time-of-flight secondary ion mass spectrometry": "飞行时间二次离子质谱",
    "X-ray photoelectron spectroscopy": "X 射线光电子能谱",
    "Kelvin probe force microscopy": "开尔文探针力显微镜",
    "time-resolved photoluminescence": "时间分辨光致发光",
}


def get_settings() -> Settings:
    return Settings()
