from __future__ import annotations

import logging
import re
from typing import Any

from .llm_client import LLMClient
from .models import ParsedPaper, empty_structured_data
from .utils import extract_json_object, parse_operator_number


EXTRACTION_SYSTEM_PROMPT = """你是材料与光伏论文结构化抽取专家。从论文文本中抽取结构化数据。
严格输出 JSON，不要输出 Markdown，不要输出任何解释文字。只输出 JSON 对象。
如果某字段没有找到，填 null 或空字符串。不要编造数据。"""

SCHEMA_PERFORMANCE = """输出 JSON:
{
  "device_performance": [{"device_label":"","sample_role":"target/control/unknown","pce_percent":null,"certified_pce_percent":null,"steady_state_pce_percent":null,"voc_v":null,"jsc_ma_cm2":null,"ff_percent":null,"active_area_cm2":null,"device_structure":"","perovskite_composition":"","htl":"","source_text":"","source_page":null,"source_figure":""}],
  "stability_tests": [{"device_label":"","sample_role":"target/control/unknown","test_type":"MPP/thermal/damp_heat/reverse_bias/other","duration_h":null,"temperature_c":null,"retained_pce_percent":null,"t80_h":null,"t95_h":null,"qualitative_result":"","source_text":"","source_page":null,"source_figure":""}]
}"""

SCHEMA_METHODS = """输出 JSON:
{
  "experimental_methods": [{"method_name":"","description":"","source_page":null}],
  "characterization_methods": [{"method":"","purpose":"","key_finding":"","source_text":"","source_page":null}]
}"""

SCHEMA_INSIGHTS = """输出 JSON:
{
  "key_innovations": [{"innovation":"","explanation":"","source_text":"","source_page":null}],
  "new_knowledge": [{"finding":"","evidence":"","source_text":"","source_page":null}],
  "limitations": []
}"""


class StructuredExtractor:
    def __init__(self, llm: LLMClient, logger: logging.Logger, max_chars: int = 70000):
        self.llm = llm
        self.logger = logger
        self.max_chars = max_chars

    def extract(self, paper: ParsedPaper) -> dict[str, Any]:
        data = self._base_data(paper)
        if not self.llm.available():
            return self.extract_heuristic(paper)
        perf_text = self._make_performance_text(paper)[: self.max_chars]
        method_text = self._make_method_text(paper)[: self.max_chars // 2]
        insight_text = self._make_insight_text(paper)[: self.max_chars]
        calls = [
            ("device_performance+stability", SCHEMA_PERFORMANCE, perf_text,
             ["device_performance", "stability_tests"]),
            ("methods", SCHEMA_METHODS, method_text,
             ["experimental_methods", "characterization_methods"]),
            ("insights", SCHEMA_INSIGHTS, insight_text,
             ["key_innovations", "new_knowledge", "limitations"]),
        ]
        for name, schema, text, keys in calls:
            if not text.strip():
                self.logger.warning("LLM structured extraction skipped %s: empty text", name)
                continue
            prompt = f"{schema}\n\n论文文本：\n{text}"
            raw = self.llm.chat(EXTRACTION_SYSTEM_PROMPT, prompt, temperature=0)
            parsed = extract_json_object(raw)
            if parsed:
                for key in keys:
                    if key in parsed and isinstance(parsed[key], list):
                        data[key] = parsed[key]
            else:
                self.logger.warning(
                    "LLM structured extraction JSON parse failed for %s. Raw (first 500): %s",
                    name, raw[:500] if raw else "(empty)"
                )
        heuristic = self._heuristic_extract(paper)
        for key in ["device_performance", "stability_tests"]:
            if not data.get(key):
                data[key] = heuristic.get(key, [])
        for key in ["key_innovations", "new_knowledge"]:
            if not data.get(key):
                data[key] = heuristic.get(key, [])
        for key in ["experimental_methods", "characterization_methods"]:
            h = heuristic.get(key, [])
            data[key] = h if h else data.get(key, [])
        return data

    def extract_heuristic(self, paper: ParsedPaper) -> dict[str, Any]:
        data = self._base_data(paper)
        heuristic = self._heuristic_extract(paper)
        for key, rows in heuristic.items():
            data[key] = rows
        data["warnings"].append("未运行 LLM 结构化抽取，自动结构化总结使用本地启发式结果。")
        return data

    def _base_data(self, paper: ParsedPaper) -> dict[str, Any]:
        data = empty_structured_data()
        data["paper_info"] = {
            "title": paper.paper_info.title,
            "title_zh": paper.paper_info.title_zh,
            "authors": paper.paper_info.authors,
            "doi": paper.paper_info.doi,
            "journal": paper.paper_info.journal,
            "year": paper.paper_info.year,
            "abstract": paper.paper_info.abstract,
            "abstract_zh": paper.paper_info.abstract_zh,
        }
        data["sections"] = [
            {"section_title": s.section_title, "section_title_zh": s.section_title_zh}
            for s in paper.sections
        ]
        return data

    def _make_evidence_text(self, paper: ParsedPaper) -> str:
        lines: list[str] = []
        if paper.paper_info.title:
            lines.append(f"标题：{paper.paper_info.title}")
        if paper.paper_info.abstract:
            lines.append(f"摘要：{paper.paper_info.abstract}")
        for s in paper.sections:
            lines.append(f"\n## {s.section_title}")
            for p in s.paragraphs:
                if p.kind == "non_content":
                    continue
                lines.append(f"[p.{p.source_page}] {p.text_original}")
        for f in paper.figures:
            lines.append(f"[p.{f.page}, {f.figure_id}] {f.caption_original}")
        return "\n".join(lines)

    def _make_performance_text(self, paper: ParsedPaper) -> str:
        lines: list[str] = []
        if paper.paper_info.abstract:
            lines.append(f"摘要：{paper.paper_info.abstract}")
        perf_sections = {"results", "discussion", "正文"}
        for s in paper.sections:
            if not any(kw in s.section_title.lower() for kw in perf_sections):
                continue
            for p in s.paragraphs:
                if p.kind == "non_content":
                    continue
                lines.append(f"[p.{p.source_page}] {p.text_original}")
        for f in paper.figures:
            lines.append(f"[p.{f.page}, {f.figure_id}] {f.caption_original}")
        return "\n".join(lines)

    def _make_method_text(self, paper: ParsedPaper) -> str:
        lines: list[str] = []
        method_sections = {
            "methods", "methodology", "experimental", "experimental section",
            "experimental details", "materials and methods", "device fabrication",
            "sample preparation", "materials characterization", "device characterization",
            "solar cell fabrication", "characterization",
        }
        for s in paper.sections:
            if not any(kw in s.section_title.lower() for kw in method_sections):
                continue
            for p in s.paragraphs:
                if p.kind == "non_content":
                    continue
                lines.append(f"[p.{p.source_page}] {p.text_original}")
        return "\n".join(lines)

    def _make_insight_text(self, paper: ParsedPaper) -> str:
        lines: list[str] = []
        if paper.paper_info.abstract:
            lines.append(f"摘要：{paper.paper_info.abstract}")
        insight_sections = {"results", "discussion", "conclusion", "conclusions", "正文"}
        for s in paper.sections:
            if not any(kw in s.section_title.lower() for kw in insight_sections):
                continue
            for p in s.paragraphs:
                if p.kind == "non_content":
                    continue
                lines.append(f"[p.{p.source_page}] {p.text_original}")
        return "\n".join(lines)

    def _heuristic_extract(self, paper: ParsedPaper) -> dict[str, Any]:
        device_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        method_rows: list[dict[str, Any]] = []
        char_rows: list[dict[str, Any]] = []
        innovation_rows: list[dict[str, Any]] = []
        knowledge_rows: list[dict[str, Any]] = []
        perf_re = re.compile(r"(?:(certified|steady-state|champion|average)\s+)?(?:PCE|efficiency|power conversion efficiency)[^\n.;]{0,120}?([>~<≈≥≤]?\s*\d+(?:\.\d+)?)\s*%", re.I)
        reverse_perf_re = re.compile(r"([>~<≈≥≤]?\s*\d+(?:\.\d+)?)\s*%[^\n.;]{0,80}?(?:PCE|efficiency|power conversion efficiency)", re.I)
        voc_re = re.compile(r"\bV(?:OC|oc)?\b[^\d]{0,20}(\d+(?:\.\d+)?)\s*V", re.I)
        jsc_re = re.compile(r"\bJ(?:SC|sc)\b[^\d]{0,20}(\d+(?:\.\d+)?)\s*mA\s*cm[-−]?\s*2", re.I)
        ff_re = re.compile(r"(?:\bFF\b|fill\s+factor)[^\d]{0,20}(\d+(?:\.\d+)?)\s*%", re.I)
        area_re = re.compile(r"(?:active\s+area|area)[^\d]{0,20}(\d+(?:\.\d+)?)\s*cm\s*2", re.I)
        duration_num = r"\d[\d,]*(?:\.\d+)?"
        stability_re = re.compile(rf"((?:retained|maintained|remaining|retain)[^. ]{{0,0}}[^.]{{0,160}}?([>~<≈≥≤]?\s*\d+(?:\.\d+)?)\s*%[^.]{{0,160}}?)(?:after|for)\s+({duration_num})\s*(h|hours|hour|cycles)", re.I)
        reverse_stability_re = re.compile(rf"(?:after|for)\s+({duration_num})\s*(h|hours|hour|cycles)[^. ]{{0,0}}[^.]{{0,180}}?(?:retained|maintained|remaining|retain)[^.]{{0,120}}?([>~<≈≥≤]?\s*\d+(?:\.\d+)?)\s*%", re.I)
        loss_re = re.compile(rf"(?:after|for)\s+({duration_num})\s*(h|hours|hour|cycles)[^. ]{{0,0}}[^.]{{0,180}}?(?:loss|lost|decrease|decreased|degradation)[^.]{{0,80}}?([>~<≈≥≤]?\s*\d+(?:\.\d+)?)\s*%", re.I)
        methods = ["XPS", "UPS", "SEM", "TEM", "HRTEM", "AFM", "KPFM", "TRPL", "PL", "XRD", "GIWAXS", "ToF-SIMS", "SIMS", "EQE", "J-V", "MPP", "EL", "Raman", "FTIR", "UV-vis", "XAS"]
        items: list[tuple[int, str, str, str, str]] = []
        if paper.paper_info.abstract:
            items.append((1, paper.paper_info.abstract, paper.paper_info.abstract_zh or paper.paper_info.abstract, "", "abstract"))
        for s in paper.sections:
            stitle = s.section_title.lower()
            for p in s.paragraphs:
                if p.kind == "non_content":
                    continue
                items.append((p.source_page, p.text_original, p.text_zh or p.text_original, "", stitle))
        for f in paper.figures:
            caption_zh = f.caption_zh or f.caption_original
            items.append((f.page, f.caption_original, caption_zh, f.figure_id, "figure_caption"))
        for page, text_en, text_zh, fig_id, section in items:
            if not text_en:
                continue
            source_zh = self._sentence_around(text_zh, 0) if text_zh else ""
            for m in list(perf_re.finditer(text_en)) + list(reverse_perf_re.finditer(text_en)):
                source = self._sentence_around(text_en, m.start())
                if self._looks_like_stability_sentence(source):
                    continue
                if self._looks_like_reference_to_other_work(source, page):
                    continue
                if self._looks_like_retention_as_pce(source):
                    continue
                if re.search(r"\b(?:relative\s+humidity|temperature|°C|RH)\b", source, re.I) and not re.search(r"\bPCE\b", source, re.I):
                    continue
                op, pce = parse_operator_number(m.group(2) if m.re is perf_re else m.group(1))
                row = self._empty_device_row()
                label = ""
                if m.re is perf_re and m.group(1):
                    label = m.group(1)
                elif re.search(r"\bchampion\b", source, re.I):
                    label = "champion"
                elif re.search(r"\bcertified\b", source, re.I):
                    label = "certified"
                elif re.search(r"\bsteady[- ]state\b", source, re.I):
                    label = "steady-state"
                row.update(
                    {
                        "device_label": label,
                        "sample_role": self._role(source),
                        "pce_percent": pce,
                        "certified_pce_percent": pce if label.lower() == "certified" else None,
                        "steady_state_pce_percent": pce if "steady" in label.lower() else None,
                        "voc_v": self._float_match(voc_re.search(source)),
                        "jsc_ma_cm2": self._float_match(jsc_re.search(source)),
                        "ff_percent": self._float_match(ff_re.search(source)),
                        "active_area_cm2": self._float_match(area_re.search(source)),
                        "source_text": source_zh if len(source_zh) > 20 else source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                if op:
                    row["source_text"] = f"{source} [operator for PCE: {op}]"
                device_rows.append(row)
            for m in stability_re.finditer(text_en):
                source = self._sentence_around(text_en, m.start())
                op, retained = parse_operator_number(m.group(2))
                row = self._empty_stability_row()
                row.update(
                    {
                        "sample_role": self._role(source),
                        "test_type": self._test_type(source),
                        "duration_h": self._to_float_num(m.group(3)) if m.group(4).lower().startswith("h") else None,
                        "cycles": int(self._to_float_num(m.group(3)) or 0) if "cycle" in m.group(4).lower() else None,
                        "retained_pce_operator": op,
                        "retained_pce_percent": retained,
                        "temperature_c": self._temperature(source),
                        "relative_humidity_percent": self._humidity(source),
                        "qualitative_result": "no significant degradation" if re.search(r"no significant degradation", source, re.I) else "",
                        "source_text": source_zh if len(source_zh) > 20 else source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for m in reverse_stability_re.finditer(text_en):
                source = self._sentence_around(text_en, m.start())
                op, retained = parse_operator_number(m.group(3))
                row = self._empty_stability_row()
                row.update(
                    {
                        "sample_role": self._role(source),
                        "test_type": self._test_type(source),
                        "duration_h": self._to_float_num(m.group(1)) if m.group(2).lower().startswith("h") else None,
                        "cycles": int(self._to_float_num(m.group(1)) or 0) if "cycle" in m.group(2).lower() else None,
                        "retained_pce_operator": op,
                        "retained_pce_percent": retained,
                        "temperature_c": self._temperature(source),
                        "relative_humidity_percent": self._humidity(source),
                        "source_text": source_zh if len(source_zh) > 20 else source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for m in loss_re.finditer(text_en):
                source = self._sentence_around(text_en, m.start())
                op, loss = parse_operator_number(m.group(2))
                row = self._empty_stability_row()
                row.update(
                    {
                        "sample_role": self._role(source),
                        "test_type": self._test_type(source),
                        "duration_h": self._to_float_num(m.group(1)) if m.group(2).lower().startswith("h") else None,
                        "cycles": int(self._to_float_num(m.group(1)) or 0) if "cycle" in m.group(2).lower() else None,
                        "retained_pce_operator": "~" if loss is not None else op,
                        "retained_pce_percent": round(100 - loss, 3) if loss is not None else None,
                        "temperature_c": self._temperature(source),
                        "relative_humidity_percent": self._humidity(source),
                        "qualitative_result": f"reported efficiency loss {op}{loss}%" if loss is not None else "",
                        "source_text": source_zh if len(source_zh) > 20 else source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for method in methods:
                if re.search(rf"\b{re.escape(method)}\b", text_en, re.I):
                    char_rows.append({"method": method, "purpose": "", "key_finding": "", "source_text": source_zh[:260], "source_page": page})
            if self._is_method_section(section):
                if any(k in text_en.lower() for k in ["fabricat", "deposited", "spin-coat", "evaporation", "anneal", "sputter", "co-deposit", "ALD"]):
                    method_rows.append({"method_name": self._infer_method_name(text_en), "description": source_zh[:260], "source_page": page})
            if self._is_results_section(section):
                source_en = self._sentence_around(text_en, 0)
                if self._looks_like_innovation(source_en):
                    innovation_rows.append({"innovation": source_zh[:220], "explanation": "", "source_text": source_zh[:220], "source_page": page})
                if self._looks_like_finding(source_en):
                    knowledge_rows.append({"finding": source_zh[:220], "evidence": "", "source_text": source_zh[:220], "source_page": page})
        return {
            "device_performance": self._dedupe_device_rows(self._unique(device_rows, "source_text")),
            "stability_tests": self._dedupe_stability_rows(self._unique(stability_rows, "source_text")),
            "experimental_methods": self._unique(method_rows, "description"),
            "characterization_methods": self._unique(char_rows, "source_text"),
            "key_innovations": self._unique(innovation_rows, "source_text")[:8],
            "new_knowledge": self._unique(knowledge_rows, "source_text")[:10],
        }

    def _empty_device_row(self) -> dict[str, Any]:
        return {"device_label": "", "sample_role": "unknown", "material_system": "", "perovskite_composition": "", "device_structure": "", "substrate": "", "etl": "", "interface_layers": [], "htl": "", "electrode": "", "active_area_cm2": None, "voc_v": None, "jsc_ma_cm2": None, "ff_percent": None, "pce_percent": None, "certified_pce_percent": None, "steady_state_pce_percent": None, "scan_direction": "", "source_text": "", "source_page": None, "source_figure": ""}

    def _empty_stability_row(self) -> dict[str, Any]:
        return {"device_label": "", "sample_role": "unknown", "material_system": "", "device_structure": "", "test_type": "other", "protocol": "", "temperature_c": None, "relative_humidity_percent": None, "light_condition": "", "illumination_intensity": "", "tracking_mode": "", "bias_condition": "", "atmosphere": "", "encapsulated": None, "duration_h": None, "cycles": None, "cycle_profile": "", "initial_pce_percent": None, "retained_pce_operator": "", "retained_pce_percent": None, "final_pce_percent": None, "t80_h": None, "t95_h": None, "qualitative_result": "", "source_text": "", "source_page": None, "source_figure": "", "source_panel": ""}

    def _sentence_around(self, text: str, pos: int) -> str:
        if pos < 0:
            pos = 0
        # Look backward for sentence boundary (period, question mark, exclamation,
        # Chinese punctuation, or double newline indicating paragraph break)
        start = 0
        for idx in range(pos - 1, -1, -1):
            char = text[idx]
            if char == "\n" and idx > 0 and text[idx - 1] == "\n":
                # Double newline = paragraph break
                start = idx + 1
                break
            if char in ".!?。！？" and not self._is_decimal_point(text, idx):
                start = idx + 1
                break
        # Look forward for end of sentence
        end = min(len(text), pos + 320)
        for idx in range(pos, len(text)):
            char = text[idx]
            if char == "\n" and idx + 1 < len(text) and text[idx + 1] == "\n":
                # Double newline = paragraph break
                end = idx
                break
            if char in ".!?。！？" and not self._is_decimal_point(text, idx):
                end = idx + 1
                break
        result = text[start:end].strip(" .;")
        # If the result is very short (< 40 chars) and there are more content nearby,
        # try to expand to the next sentence
        if len(result) < 40 and end < len(text):
            for idx in range(end, min(len(text), end + 160)):
                char = text[idx]
                if char in ".!?。！？" and not self._is_decimal_point(text, idx):
                    result = text[start:idx + 1].strip(" .;")
                    break
        return result

    def _is_decimal_point(self, text: str, idx: int) -> bool:
        return (
            text[idx] == "."
            and idx > 0
            and idx + 1 < len(text)
            and text[idx - 1].isdigit()
            and text[idx + 1].isdigit()
        )

    def _role(self, text: str) -> str:
        if re.search(r"\b(control|reference)\b", text, re.I):
            return "control"
        if re.search(r"\b(target|modified|treated)\b", text, re.I):
            return "target"
        return "unknown"

    def _test_type(self, text: str) -> str:
        low = text.lower()
        if "mpp" in low:
            return "MPP"
        if "thermal" in low or "heat" in low or "damp" in low:
            return "thermal"
        if "reverse" in low and ("bias" in low or "breakdown" in low):
            return "reverse_bias"
        if "light" in low or "illumination" in low or "uv" in low:
            return "illumination"
        return "other"

    def _looks_like_stability_sentence(self, text: str) -> bool:
        return bool(
            re.search(r"\b(retain(?:ed)?|maintain(?:ed)?|remaining|after|aging|stability|MPP tracking|cycles?|hours?|degradation|loss)\b", text, re.I)
            and re.search(r"\b(initial|original)\s+PCE\b|\bof\s+their\s+(?:initial|original)\b|\bof\s+its\s+(?:initial|original)\b|\bafter\b", text, re.I)
        )

    def _looks_like_reference_to_other_work(self, text: str, page: int) -> bool:
        if page > 2:
            return False
        return bool(re.search(
            r"\b(?:have been|has been|previously|reported|demonstrated|achieved|realized|exceeding\s+\d+)\b.{0,80}\b(?:PCE|efficiency)\b"
            r"|(?:PCE|efficiency).{0,40}\b(?:have been|has been|previously|reported|demonstrated|achieved|exceeding)",
            text, re.I,
        ))

    def _looks_like_retention_as_pce(self, text: str) -> bool:
        return bool(
            re.search(r"\b(?:retain(?:ed)?|maintain(?:ed)?|keep(?:ing)?|preserv(?:ed)?)\s+\d+(?:\.\d+)?\s*%.{0,80}?\b(?:of\s+(?:the|their|its)\s+)?(?:original|initial|pristine)\s+(?:PCE|efficiency)", text, re.I)
        )

    def _looks_like_finding(self, text: str) -> bool:
        if len(text) < 60 or len(text) > 500:
            return False
        # Must contain a quantitative or qualitative result with context
        has_result_word = re.search(
            r"\b(improved|enhanced|suppressed|reduced|increased|decreased|boosted|achieved|"
            r"superior|excellent|outstanding|significant|remarkable|notable|prominent|"
            r"best|highest|lowest|record|unprecedented|breakthrough)\b",
            text, re.I,
        )
        # Must have some form of result-claiming language (we found, our results show, etc.)
        has_claim = re.search(
            r"\b(we\s+(?:find|found|show|showed|demonstrat|reveal|identif|confirm|observ|"
            r"discover|quantif|achiev|present|propos|report|develop|introduc))"
            r"|(the\s+(?:results|findings|data|measurements|study|work)\s+(?:indicate|show|"
            r"demonstrate|reveal|suggest|confirm|highlight|prove|establish))"
            r"|(this\s+(?:work|study|paper|approach|strategy|method)\s+(?:demonstrates|shows|"
            r"reveals|presents|proposes|introduces|achieves|provides|offers|enables))"
            r"|(it\s+is\s+(?:found|shown|demonstrated|revealed|observed|confirmed)\s+that)",
            text, re.I,
        )
        if not has_claim:
            return False
        if not has_result_word:
            # If no strong result word, require a numeric finding
            if not re.search(r"\d+(?:\.\d+)?\s*(?:%|eV|nm|µm|cm|mA|V|K|°C|h)", text):
                return False
        return True

    def _looks_like_innovation(self, text: str) -> bool:
        if len(text) < 50 or len(text) > 500:
            return False
        we_verbs = (
            r"demonstrate|report|develop|introduce|design|show|find|identify|confirm|reveal|achieve|present|propose|discover"
            r"|demonstrated|reported|developed|introduced|designed|showed|found|identified|confirmed|revealed|achieved|presented|proposed|discovered"
            r"|apply|applied|use|used|employ|employed|fabricate|fabricated|prepare|prepared|construct|constructed|engineer|engineered"
            r"|establish|established|create|created|synthesize|synthesized|combine|combined|quantif|quantified"
            r"|have\s+(?:developed|demonstrated|designed|introduced|shown|identified|achieved|presented|proposed|discovered|applied|used|employed|fabricated|prepared|constructed|established|created|synthesized|combined|quantified)"
        )
        patterns = [
            rf"\b(we\s+(?:{we_verbs}))",
            r"\b(our\s+(?:results|findings|strategy|approach|method|work|study|device|module)\s+(?:demonstrate|show|reveal|indicate|confirm|suggest|highlight|prove|establish|achieve))",
            r"\b(this\s+(?:work|study|paper|approach|strategy|method)\s+(?:demonstrates|shows|reveals|presents|proposes|introduces|achieves|provides|offers|enables))",
            r"\b(here[,]?\s+we\s+\w+)",
        ]
        return any(re.search(p, text, re.I) for p in patterns)

    def _is_method_section(self, section: str) -> bool:
        method_keywords = {"methods", "methodology", "experimental", "experimental section", "experimental details", "materials and methods", "device fabrication", "sample preparation", "materials characterization", "device characterization", "solar cell fabrication", "characterization"}
        return any(kw in section for kw in method_keywords)

    def _is_results_section(self, section: str) -> bool:
        results_keywords = {"results", "discussion", "conclusion", "正文"}
        return any(kw in section for kw in results_keywords)

    def _infer_method_name(self, text: str) -> str:
        low = text.lower()
        if "fabricat" in low:
            return "器件制备"
        if "spin" in low:
            return "旋涂"
        if "evaporation" in low or "evaporated" in low:
            return "热蒸发"
        if "co-deposit" in low:
            return "共沉积"
        if "ald" in low or "atomic layer" in low:
            return "原子层沉积"
        if "sputter" in low:
            return "溅射"
        return "实验方法"

    def _temperature(self, text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*C", text)
        return float(m.group(1)) if m else None

    def _humidity(self, text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*RH", text, re.I)
        return float(m.group(1)) if m else None

    def _float_match(self, m: re.Match | None) -> float | None:
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _to_float_num(self, s: str) -> float:
        return float(s.replace(",", "")) if s else 0.0

    def _unique(self, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            value = str(row.get(key, ""))[:220]
            if value in seen:
                continue
            seen.add(value)
            out.append(row)
        return out

    def _dedupe_device_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        from collections import defaultdict
        groups: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pce = row.get("pce_percent")
            page = row.get("source_page")
            role = row.get("sample_role", "unknown")
            if pce is None:
                key = (0.0, page or 0, role)
            else:
                rounded = round(pce, 1)
                key = (rounded, page or 0, role)
            groups[key].append(row)
        result: list[dict[str, Any]] = []
        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
                continue
            best = max(group, key=lambda r: (
                sum(1 for k in ["voc_v", "jsc_ma_cm2", "ff_percent", "active_area_cm2"]
                    if r.get(k) is not None),
                len(r.get("source_text", "")),
            ))
            result.append(best)
        result.sort(key=lambda r: (r.get("source_page", 0), -(r.get("pce_percent") or 0)))
        return result

    def _dedupe_stability_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        from collections import defaultdict
        groups: dict[tuple[float | None, float | None, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            retained = row.get("retained_pce_percent")
            if retained is not None:
                retained = round(retained, 1)
            dur = row.get("duration_h")
            page = row.get("source_page", 0)
            key = (retained, dur, page)
            groups[key].append(row)
        result: list[dict[str, Any]] = []
        for group in groups.values():
            best = max(group, key=lambda r: len(r.get("source_text", "")))
            result.append(best)
        result.sort(key=lambda r: r.get("source_page", 0))
        return result
