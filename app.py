from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.job_manager import read_job_status, start_processing_job
from src.utils import ensure_dir


st.set_page_config(page_title="AI 论文处理工具", layout="wide")


def main() -> None:
    st.title("AI 论文处理工具")
    st.caption("本地处理英文科研论文 PDF，默认生成中文 Markdown 和结构化总结；HTML/JSON/Excel 可通过 .env 开启。")

    settings = get_settings()
    if not settings.openai_api_key:
        st.warning("未检测到 OPENAI_API_KEY。程序仍可运行，但翻译、总结和结构化抽取会使用降级启发式方案。")

    with st.sidebar:
        st.header("处理设置")
        output_dir = st.text_input("输出目录", value=str(settings.outputs_dir.expanduser().resolve()))
        image_format = st.selectbox("图片格式", options=["png", "jpg"], index=0)
        compress_images = st.checkbox("压缩图片", value=True)
        st.divider()
        st.write(f"模型：`{settings.openai_model}`")
        st.write("输出：")
        st.write(f"- Markdown：`{int(settings.output_markdown)}`")
        st.write(f"- HTML：`{int(settings.output_html)}`")
        st.write(f"- 总结：`{int(settings.output_summary)}`")
        st.write(f"- JSON：`{int(settings.output_json)}`")
        st.write(f"- Excel：`{int(settings.output_excel)}`")
        st.write(f"- 原 PDF 副本：`{int(settings.output_source_pdf)}`")
        st.write(f"- 调试图片：`{int(settings.output_debug_figures)}`")

    output_root = ensure_dir(Path(output_dir).expanduser().resolve())
    job_id = _current_job_id()
    if job_id:
        _render_job_status(output_root, job_id)
        st.divider()

    uploaded = st.file_uploader("上传英文科研论文 PDF", type=["pdf"])
    start = st.button("开始处理", type="primary", disabled=uploaded is None)

    if uploaded and start:
        new_job_id = start_processing_job(
            uploaded_name=uploaded.name,
            uploaded_bytes=uploaded.getbuffer().tobytes(),
            output_root=output_root,
            image_format=image_format,
            compress_images=compress_images,
        )
        st.session_state["active_job_id"] = new_job_id
        st.query_params["job_id"] = new_job_id
        st.rerun()


def _current_job_id() -> str:
    query_job = st.query_params.get("job_id", "")
    if isinstance(query_job, list):
        query_job = query_job[0] if query_job else ""
    if query_job:
        st.session_state["active_job_id"] = query_job
        return query_job
    return st.session_state.get("active_job_id", "")


def _render_job_status(output_root: Path, job_id: str) -> None:
    status = read_job_status(output_root, job_id)
    if not status:
        st.warning(f"未找到任务状态：{job_id}")
        return

    progress = float(status.get("progress") or 0.0)
    state = status.get("status", "running")
    message = status.get("message", "")
    st.subheader("当前任务")
    st.caption(f"任务 ID：{job_id}")
    st.progress(min(max(progress, 0.0), 1.0))

    if state == "running":
        st.info(message or "正在处理")
        st.caption("页面可刷新，进度会从磁盘状态文件恢复。")
        time.sleep(2)
        st.rerun()
    elif state == "completed":
        st.success(message or "处理完成")
        result = status.get("result") or {}
        _render_result(result)
    else:
        st.error(message or "处理失败")
        if status.get("error"):
            st.code(str(status["error"]))
        result = status.get("result") or {}
        if result:
            _render_result(result)


def _render_result(result: dict) -> None:
    output_dir = result.get("output_dir", "")
    if output_dir:
        st.subheader("输出目录")
        st.code(str(output_dir))

    warnings = result.get("warnings") or []
    if warnings:
        with st.expander("警告和降级信息", expanded=True):
            for warning in warnings:
                st.write(f"- {warning}")

    st.subheader("输出文件")
    cols = st.columns(3)
    paths = result.get("paths") or {}
    display = [
        ("中文 Markdown", paths.get("markdown"), "text/markdown"),
        ("HTML", paths.get("html"), "text/html"),
        ("总结 Markdown", paths.get("summary"), "text/markdown"),
        ("结构化 JSON", paths.get("json"), "application/json"),
        ("Excel", paths.get("excel"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("原 PDF 副本", paths.get("source_pdf"), "application/pdf"),
        ("处理日志", paths.get("log"), "text/plain"),
    ]
    for idx, (label, path_value, mime) in enumerate(display):
        with cols[idx % 3]:
            if not path_value:
                st.write(label)
                st.caption("未生成")
                continue
            path = Path(path_value)
            if path.exists():
                st.write(label)
                st.caption(str(path))
                st.download_button(
                    label=f"下载 {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=str(path),
                )
            else:
                st.write(label)
                st.caption("未生成")


if __name__ == "__main__":
    main()
