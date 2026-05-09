from __future__ import annotations

import logging
import re
from typing import Any

from .llm_client import LLMClient
from .models import ParsedPaper, empty_structured_data
from .utils import extract_json_object, parse_operator_number


EXTRACTION_SYSTEM_PROMPT = """你是材料与光伏论文结构化抽取专家。
从论文全文、图注和表格文本中抽取器件性能、稳定性测试、实验方法、器件结构、表征手段、创新点和新知识。
严格输出 JSON，不要输出 Markdown。
如果某字段没有找到，填 null 或空字符串。不要编造数据。
每条 device_performance 和 stability_tests 记录都要尽量附 source_text、source_page、source_figure。
必须尽量覆盖 target/control、champion/average/certified/steady-state、小面积/大面积、不同材料/结构/尺寸，以及图注中出现的数据。"""


SCHEMA_HINT = """输出 JSON schema:
{
  "device_performance": [{"device_label":"","sample_role":"target/control/reference/unknown","material_system":"","perovskite_composition":"","device_structure":"","substrate":"","etl":"","interface_layers":[],"htl":"","electrode":"","active_area_cm2":null,"voc_v":null,"jsc_ma_cm2":null,"ff_percent":null,"pce_percent":null,"certified_pce_percent":null,"steady_state_pce_percent":null,"scan_direction":"","source_text":"","source_page":null,"source_figure":""}],
  "stability_tests": [{"device_label":"","sample_role":"target/control/reference/unknown","material_system":"","device_structure":"","test_type":"MPP/damp_heat/light_dark/storage/thermal/humidity/illumination/other","protocol":"","temperature_c":null,"relative_humidity_percent":null,"light_condition":"","illumination_intensity":"","tracking_mode":"","bias_condition":"","atmosphere":"","encapsulated":null,"duration_h":null,"cycles":null,"cycle_profile":"","initial_pce_percent":null,"retained_pce_operator":"","retained_pce_percent":null,"final_pce_percent":null,"t80_h":null,"t95_h":null,"qualitative_result":"","source_text":"","source_page":null,"source_figure":"","source_panel":""}],
  "experimental_methods": [{"method_name":"","description":"","source_page":null}],
  "characterization_methods": [{"method":"","purpose":"","key_finding":"","source_text":"","source_page":null}],
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
        text = self._make_evidence_text(paper)[: self.max_chars]
        if self.llm.available():
            prompt = f"{SCHEMA_HINT}\n\n论文文本：\n{text}"
            raw = self.llm.chat(EXTRACTION_SYSTEM_PROMPT, prompt, temperature=0)
            parsed = extract_json_object(raw)
            if parsed:
                for key in [
                    "device_performance",
                    "stability_tests",
                    "experimental_methods",
                    "characterization_methods",
                    "key_innovations",
                    "new_knowledge",
                    "limitations",
                ]:
                    if key in parsed and isinstance(parsed[key], list):
                        data[key] = parsed[key]
            else:
                data["warnings"].append("LLM structured extraction did not return valid JSON; heuristic extraction used.")
        heuristic = self._heuristic_extract(paper)
        for key in [
            "device_performance",
            "stability_tests",
            "experimental_methods",
            "characterization_methods",
            "key_innovations",
            "new_knowledge",
        ]:
            if not data.get(key):
                data[key] = heuristic.get(key, [])
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
        data["paper_info"] = paper.paper_info.__dict__.copy()
        data["sections"] = [
            {
                "section_title": s.section_title,
                "section_title_zh": s.section_title_zh,
                "paragraphs": [p.__dict__ for p in s.paragraphs],
            }
            for s in paper.sections
        ]
        data["figures"] = [f.__dict__ for f in paper.figures]
        data["warnings"] = list(paper.warnings)
        return data

    def _make_evidence_text(self, paper: ParsedPaper) -> str:
        parts = [paper.paper_info.title, paper.paper_info.abstract]
        for section in paper.sections:
            parts.append(f"\n## {section.section_title}")
            for p in section.paragraphs:
                if p.kind == "non_content":
                    continue
                parts.append(f"[page {p.source_page}] {p.text_original}")
        for fig in paper.figures:
            parts.append(f"[page {fig.page}] {fig.figure_id}: {fig.caption_original}")
        return "\n".join(part for part in parts if part)

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
        method_keywords = ["fabricat", "deposited", "spin-coat", "evaporation", "anneal", "sputter", "co-deposit", "ALD", "solution"]
        innovation_keywords = ["we demonstrate", "we report", "we develop", "we introduce", "we design", "herein", "strategy", "approach", "novel", "enable"]
        items = [(1, paper.paper_info.abstract, "")] if paper.paper_info.abstract else []
        items += [(p.source_page, p.text_original, "") for s in paper.sections for p in s.paragraphs if p.kind != "non_content"]
        items += [(f.page, f.caption_original, f.figure_id) for f in paper.figures]
        for page, text, fig_id in items:
            if not text:
                continue
            for m in list(perf_re.finditer(text)) + list(reverse_perf_re.finditer(text)):
                source = self._sentence_around(text, m.start())
                if self._looks_like_stability_sentence(source):
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
                        "source_text": source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                if op:
                    row["source_text"] = f"{source} [operator for PCE: {op}]"
                device_rows.append(row)
            for m in stability_re.finditer(text):
                source = self._sentence_around(text, m.start())
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
                        "source_text": source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for m in reverse_stability_re.finditer(text):
                source = self._sentence_around(text, m.start())
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
                        "source_text": source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for m in loss_re.finditer(text):
                source = self._sentence_around(text, m.start())
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
                        "source_text": source,
                        "source_page": page,
                        "source_figure": fig_id,
                    }
                )
                stability_rows.append(row)
            for method in methods:
                if re.search(rf"\b{re.escape(method)}\b", text, re.I):
                    char_rows.append({"method": method, "purpose": "", "key_finding": "", "source_text": self._sentence_around(text, text.lower().find(method.lower())), "source_page": page})
            if any(k in text.lower() for k in method_keywords):
                method_rows.append({"method_name": self._infer_method_name(text), "description": self._sentence_around(text, 0)[:260], "source_page": page})
            if any(k in text.lower() for k in innovation_keywords):
                source = self._sentence_around(text, 0)
                innovation_rows.append({"innovation": source[:220], "explanation": "", "source_text": source, "source_page": page})
            if re.search(r"\b(improve|enhance|suppress|reduce|increase|enable|reveal|confirm|indicate|demonstrate)\w*\b", text, re.I):
                source = self._sentence_around(text, 0)
                knowledge_rows.append({"finding": source[:220], "evidence": "", "source_text": source, "source_page": page})
        return {
            "device_performance": self._unique(device_rows, "source_text"),
            "stability_tests": self._unique(stability_rows, "source_text"),
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
        start = 0
        for idx in range(pos - 1, -1, -1):
            char = text[idx]
            if char not in ".;!?":
                continue
            if self._is_decimal_point(text, idx):
                continue
            start = idx + 1
            break
        end = min(len(text), pos + 320)
        for idx in range(pos, len(text)):
            char = text[idx]
            if char not in ".;!?":
                continue
            if self._is_decimal_point(text, idx):
                continue
            end = idx + 1
            break
        return text[start:end].strip(" .;")

    def _is_decimal_point(self, text: str, idx: int) -> bool:
        return (
            text[idx] == "."
            and idx > 0
            and idx + 1 < len(text)
            and text[idx - 1].isdigit()
            and text[idx + 1].isdigit()
        )

    def _float_match(self, match: re.Match | None) -> float | None:
        return float(match.group(1)) if match else None

    def _to_float_num(self, value: str) -> float | None:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    def _role(self, text: str) -> str:
        if re.search(r"\b(control|reference)\b", text, re.I):
            return "control"
        if re.search(r"\b(target|modified|treated)\b", text, re.I):
            return "target"
        return "unknown"

    def _test_type(self, text: str) -> str:
        low = text.lower()
        if "damp" in low or "humidity" in low:
            return "damp_heat"
        if "mpp" in low or "maximum power" in low:
            return "MPP"
        if "light/dark" in low or "cycling" in low:
            return "light_dark"
        if "thermal" in low or "heat" in low:
            return "thermal"
        if "storage" in low:
            return "storage"
        if "illumination" in low or "light soaking" in low or "1-sun" in low:
            return "illumination"
        return "other"

    def _looks_like_stability_sentence(self, text: str) -> bool:
        return bool(
            re.search(r"\b(retain(?:ed)?|maintain(?:ed)?|remaining|after|aging|stability|MPP tracking|cycles?|hours?|degradation|loss)\b", text, re.I)
            and re.search(r"\binitial\s+PCE\b|\bof\s+their\s+initial\b|\bof\s+its\s+initial\b|\bafter\b", text, re.I)
        )

    def _infer_method_name(self, text: str) -> str:
        low = text.lower()
        if "fabricat" in low:
            return "device fabrication"
        if "spin" in low:
            return "spin coating"
        if "evaporation" in low or "evaporated" in low:
            return "thermal evaporation"
        if "co-deposit" in low:
            return "co-deposition"
        if "ald" in low or "atomic layer" in low:
            return "atomic layer deposition"
        if "sputter" in low:
            return "sputtering"
        return "experimental method"

    def _temperature(self, text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*C", text)
        return float(m.group(1)) if m else None

    def _humidity(self, text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*RH", text, re.I)
        return float(m.group(1)) if m else None

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
