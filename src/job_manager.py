from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .pipeline import process_pdf
from .utils import ensure_dir, safe_slug


def jobs_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / ".jobs")


def uploads_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / ".uploads")


def status_path(output_root: Path, job_id: str) -> Path:
    return jobs_dir(output_root) / f"{job_id}.json"


def read_job_status(output_root: Path, job_id: str) -> dict[str, Any] | None:
    path = status_path(output_root, job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_job_status(output_root: Path, job_id: str, data: dict[str, Any]) -> None:
    path = status_path(output_root, job_id)
    tmp = path.with_suffix(".json.tmp")
    data["updated_at"] = time.time()
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def start_processing_job(
    uploaded_name: str,
    uploaded_bytes: bytes,
    output_root: Path,
    image_format: str,
    compress_images: bool,
) -> str:
    ensure_dir(output_root)
    job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    safe_name = safe_slug(uploaded_name) + ".pdf"
    input_path = uploads_dir(output_root) / f"{job_id}_{safe_name}"
    input_path.write_bytes(uploaded_bytes)

    initial = {
        "job_id": job_id,
        "status": "running",
        "message": "任务已创建",
        "progress": 0.0,
        "uploaded_name": uploaded_name,
        "input_path": str(input_path),
        "output_root": str(output_root),
        "result": None,
        "error": "",
        "created_at": time.time(),
    }
    write_job_status(output_root, job_id, initial)

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "input_path": input_path,
            "uploaded_name": uploaded_name,
            "output_root": output_root,
            "image_format": image_format,
            "compress_images": compress_images,
        },
        daemon=True,
    )
    thread.start()
    return job_id


def _run_job(
    job_id: str,
    input_path: Path,
    uploaded_name: str,
    output_root: Path,
    image_format: str,
    compress_images: bool,
) -> None:
    def progress(message: str, value: float) -> None:
        current = read_job_status(output_root, job_id) or {}
        current.update({"status": "running", "message": message, "progress": max(0.0, min(1.0, value))})
        write_job_status(output_root, job_id, current)

    try:
        result = process_pdf(
            pdf_path=input_path,
            output_root=output_root,
            image_format=image_format,
            compress_images=compress_images,
            progress=progress,
            output_name=uploaded_name,
        )
        status = read_job_status(output_root, job_id) or {}
        status.update(
            {
                "status": "completed" if result.get("ok") else "failed",
                "message": "处理完成" if result.get("ok") else "处理失败或部分失败",
                "progress": 1.0,
                "result": _serialize_result(result),
                "error": "" if result.get("ok") else str(result.get("warnings") or ""),
            }
        )
        write_job_status(output_root, job_id, status)
    except Exception as exc:
        status = read_job_status(output_root, job_id) or {}
        status.update({"status": "failed", "message": "处理失败", "progress": 1.0, "error": str(exc)})
        write_job_status(output_root, job_id, status)


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    if "output_dir" in out:
        out["output_dir"] = str(out["output_dir"])
    if "paths" in out and isinstance(out["paths"], dict):
        out["paths"] = {key: (str(value) if value else "") for key, value in out["paths"].items()}
    return out
