from __future__ import annotations

import json
import logging
from typing import Any

from .llm_client import LLMClient
from .models import ParsedPaper


SUMMARY_SYSTEM_PROMPT = """你是材料、器件、光伏、钙钛矿太阳能电池论文总结专家。
输出中文研究型论文总结，重点总结实验方法、器件结构、效率、稳定性、表征手段、创新点、新知识。
必须特别列出不同材料、不同结构、不同尺寸器件的效率和稳定性。
如果某一类信息没有找到，要明确写“未在正文中明确找到”，不要编造。"""


class Summarizer:
    def __init__(self, llm: LLMClient, logger: logging.Logger, max_chars: int = 60000):
        self.llm = llm
        self.logger = logger
        self.max_chars = max_chars

    def summarize(self, paper: ParsedPaper, structured: dict[str, Any]) -> str:
        if self.llm.available():
            prompt = self._build_prompt(paper, structured)
            result = self.llm.chat(SUMMARY_SYSTEM_PROMPT, prompt)
            if result.strip():
                return result.strip()
        return self._fallback_summary(paper, structured)

    def _build_prompt(self, paper: ParsedPaper, structured: dict[str, Any]) -> str:
        text = self._filtered_text_for_prompt(paper)[: self.max_chars]
        compact_structured = self._compact_structured_for_prompt(structured)
        return f"""论文标题：{paper.paper_info.title}
摘要：{paper.paper_info.abstract}

已抽取结构化数据：
{json.dumps(compact_structured, ensure_ascii=False)}

全文片段：
{text}

请按以下标题输出：
- 论文基本信息
- 研究背景与核心问题
- 核心创新点
- 本研究发现的新知识
- 实验方法
- 器件结构
- 不同材料 / 不同结构 / 不同尺寸电池的效率
- 稳定性测试条件和保持率
- 核心表征手段及其作用
- 关键图表解读
- 局限性与后续研究方向"""

    def _filtered_text_for_prompt(self, paper: ParsedPaper) -> str:
        parts = [paper.paper_info.title, paper.paper_info.abstract]
        for section in paper.sections:
            parts.append(f"\n## {section.section_title}")
            for para in section.paragraphs:
                if para.kind == "non_content":
                    continue
                if para.kind == "caption":
                    continue
                text = para.text_original.strip()
                if text:
                    parts.append(f"[page {para.source_page}] {text}")
        for fig in paper.figures:
            if fig.caption_original:
                parts.append(f"[page {fig.page}] {fig.figure_id}: {fig.caption_original}")
        return "\n".join(part for part in parts if part)

    def _compact_structured_for_prompt(self, structured: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key in [
            "paper_info",
            "device_performance",
            "stability_tests",
            "experimental_methods",
            "characterization_methods",
            "key_innovations",
            "new_knowledge",
            "limitations",
            "warnings",
        ]:
            compact[key] = structured.get(key, [] if key != "paper_info" else {})
        compact["figures"] = [
            {
                "figure_id": fig.get("figure_id", ""),
                "page": fig.get("page"),
                "caption_original": fig.get("caption_original", ""),
                "caption_zh": fig.get("caption_zh", ""),
            }
            for fig in structured.get("figures", [])
            if isinstance(fig, dict)
        ]
        return compact

    def _fallback_summary(self, paper: ParsedPaper, structured: dict[str, Any]) -> str:
        info = paper.paper_info
        lines = [
            "# 论文结构化总结",
            "",
            "## 论文基本信息",
            f"- 标题：{info.title or '未在正文中明确找到'}",
            f"- DOI：{info.doi or '未在正文中明确找到'}",
            f"- 期刊：{info.journal or '未在正文中明确找到'}",
            f"- 卷：{info.volume or '未在正文中明确找到'}",
            f"- 期：{info.issue or '未在正文中明确找到'}",
            f"- 页码/文章编号：{info.pages or '未在正文中明确找到'}",
            f"- 年份：{info.year or '未在正文中明确找到'}",
            "",
            "## 研究背景与核心问题",
            info.abstract_zh or info.abstract or "未在正文中明确找到",
            "",
            "## 核心创新点",
        ]
        innovations = structured.get("key_innovations") or []
        lines.extend([f"- {x.get('innovation') or x.get('explanation')}" for x in innovations] or ["- 未在正文中明确找到"])
        lines += ["", "## 不同材料 / 不同结构 / 不同尺寸电池的效率"]
        perf = structured.get("device_performance") or []
        lines.extend([f"- PCE={r.get('pce_percent')}%，VOC={r.get('voc_v')} V，JSC={r.get('jsc_ma_cm2')} mA cm-2，FF={r.get('ff_percent')}%，证据：{r.get('source_text','')[:180]}" for r in perf] or ["- 未在正文中明确找到"])
        lines += ["", "## 稳定性测试条件和保持率"]
        stab = structured.get("stability_tests") or []
        lines.extend([f"- {r.get('test_type')}，{r.get('duration_h') or r.get('cycles')}，保持率{r.get('retained_pce_operator','')}{r.get('retained_pce_percent')}%，证据：{r.get('source_text','')[:180]}" for r in stab] or ["- 未在正文中明确找到"])
        lines += ["", "## 核心表征手段及其作用"]
        chars = structured.get("characterization_methods") or []
        lines.extend([f"- {r.get('method')}：{r.get('purpose') or r.get('key_finding') or '作用未在正文中明确找到'}" for r in chars] or ["- 未在正文中明确找到"])
        lines += ["", "## 局限性与后续研究方向", "- 未在正文中明确找到"]
        return "\n".join(lines)
