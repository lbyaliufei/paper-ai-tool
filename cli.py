from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.config import get_settings
from src.pipeline import process_pdf
from src.utils import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="本地处理英文科研论文 PDF，生成中文 Markdown/总结，并按 .env 开关输出 HTML/JSON/Excel。"
    )
    parser.add_argument("pdf", nargs="?", help="输入 PDF 文件路径；不传时会提示拖入 PDF 文件")
    default_output = get_settings().outputs_dir
    parser.add_argument("-o", "--output-dir", default=str(default_output), help=f"输出根目录，默认 {default_output}")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png", help="裁切图片格式，默认 png")
    parser.add_argument("--no-compress-images", action="store_true", help="不压缩裁切图片")
    parser.add_argument("--output-name", default="", help="自定义输出文件名前缀，默认使用 PDF 文件名")
    args = parser.parse_args()

    pdf_arg = args.pdf or prompt_for_pdf_path()
    pdf_path = normalize_dragged_path(pdf_arg).expanduser().resolve()
    if not pdf_path.exists():
        print(f"输入 PDF 不存在：{pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"输入文件不是 PDF：{pdf_path}", file=sys.stderr)
        return 2

    settings = get_settings()
    output_root = ensure_dir(Path(args.output_dir).expanduser().resolve())

    print("输出开关：")
    print(f"  Markdown: {int(settings.output_markdown)}")
    print(f"  HTML: {int(settings.output_html)}")
    print(f"  Summary: {int(settings.output_summary)}")
    print(f"  JSON: {int(settings.output_json)}")
    print(f"  Excel: {int(settings.output_excel)}")
    print(f"  Source PDF copy: {int(settings.output_source_pdf)}")
    print(f"  Debug figures: {int(settings.output_debug_figures)}")
    print(f"  Structured extraction: {int(settings.run_structured_extraction or settings.output_json or settings.output_excel)}")
    print("")

    last_message = ""

    def progress(message: str, value: float) -> None:
        nonlocal last_message
        if message == last_message:
            return
        last_message = message
        print(f"[{value * 100:5.1f}%] {message}", flush=True)

    result = process_pdf(
        pdf_path=pdf_path,
        output_root=output_root,
        image_format=args.image_format,
        compress_images=not args.no_compress_images,
        progress=progress,
        output_name=args.output_name or pdf_path.name,
    )

    print("")
    if not result.get("ok"):
        print("处理失败，详情见日志。", file=sys.stderr)
        for warning in result.get("warnings", []):
            print(f"- {warning}", file=sys.stderr)
        return 1

    print("处理完成")
    print(f"输出目录：{result.get('output_dir')}")
    for label, path in (result.get("paths") or {}).items():
        if path:
            print(f"{label}: {path}")
    warnings = result.get("warnings") or []
    if warnings:
        print("\n警告：")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def prompt_for_pdf_path() -> str:
    print("")
    print("请把 PDF 文件从访达或 Windows 文件管理器拖到这个窗口，然后按回车。")
    print("也可以直接粘贴 PDF 完整路径。")
    while True:
        raw = input("PDF 路径> ").strip()
        if raw:
            return raw
        print("路径为空，请重新拖入或粘贴。")


def normalize_dragged_path(raw: str) -> Path:
    value = raw.strip()
    if not value:
        return Path("")

    if value.startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
        if parsed.netloc and not value.startswith(f"//{parsed.netloc}"):
            value = f"//{parsed.netloc}{value}"
        return Path(value)

    try:
        parts = shlex.split(value)
        if parts:
            value = parts[0]
    except ValueError:
        value = value.strip("'\"")

    return Path(value.strip("'\""))


if __name__ == "__main__":
    raise SystemExit(main())
