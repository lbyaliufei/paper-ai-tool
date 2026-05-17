from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .config import TERMINOLOGY
from .llm_client import LLMClient
from .models import ParsedPaper


TRANSLATE_SYSTEM_PROMPT = """你是科研论文中英翻译助手。把英文论文段落翻译成中文。
要求：
- 保留专业术语、数字、单位、公式、化学式、变量和 DOI。
- 保留 Fig. / Table 编号和参考文献编号。
- 必须完整翻译输入内容，禁止用“……”或“...”省略。
- 不要添加原文没有的内容。
- 不要解释，只输出译文。"""

CAPTION_SYSTEM_PROMPT = """你是科研论文图注翻译助手。翻译图注为中文。
要求：
- 保留 a, b, c 等 panel 标记。
- 保留 Fig. 1、Table 1 等编号。
- 保留单位、缩写、材料名称、化学式和数字。
- 必须完整翻译输入内容，禁止用“……”或“...”省略。
- 不要解释，只输出译文。"""

BATCH_TRANSLATE_SYSTEM_PROMPT = """你是科研论文中英批量翻译助手。
把每个 <<<ID>>> 后面的英文论文段落翻译成中文。
要求：
- 每段译文前必须保留同样的 <<<ID>>> 标记。
- 保留专业术语、数字、单位、公式、化学式、变量、DOI、Fig. / Table 编号和参考文献编号。
- 必须完整翻译每个 <<<ID>>> 的全部内容，禁止用“……”或“...”省略。
- 不要添加原文没有的内容。
- 不要解释，不要输出额外说明。"""


class Translator:
    def __init__(
        self,
        llm: LLMClient,
        logger: logging.Logger,
        progress: Callable[[str, float], None] | None = None,
        batch_size: int = 10,
        batch_max_chars: int = 9000,
        max_workers: int = 3,
    ):
        self.llm = llm
        self.logger = logger
        self.progress = progress
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars
        self.max_workers = max(1, max_workers)

    def translate_paper(self, paper: ParsedPaper) -> ParsedPaper:
        info = paper.paper_info
        total = sum(len(s.paragraphs) for s in paper.sections)
        done = 0
        self.logger.info("Starting translation for %s paragraphs and %s figures.", total, len(paper.figures))
        self._report("准备批量翻译任务", 0, total)
        # Title is short; use local glossary replacement to avoid a tiny model call.
        translation_items: list[tuple[str, Callable[[str], None], str]] = []
        if info.title:
            translation_items.append((info.title, lambda value: setattr(info, "title_zh", value), "title"))
        if info.abstract:
            translation_items.append((info.abstract, lambda value: setattr(info, "abstract_zh", value), "abstract"))
        for section in paper.sections:
            # Section titles are short and predictable; local glossary replacement avoids many tiny LLM calls.
            section.section_title_zh = self._fallback_translate(section.section_title) if section.section_title else ""
            is_refs = bool(re.search(r"references|bibliography", section.section_title, re.I))
            for para in section.paragraphs:
                if is_refs or para.kind == "non_content" or self._should_skip_translation(para.text_original):
                    para.text_zh = para.text_original
                    done += 1
                elif para.kind == "caption":
                    translation_items.append((para.text_original, lambda value, p=para: setattr(p, "text_zh", value), f"caption_page_{para.source_page}"))
                else:
                    translation_items.append((para.text_original, lambda value, p=para: setattr(p, "text_zh", value), f"paragraph_page_{para.source_page}"))
                if done and (done == 1 or done % 20 == 0 or done == total):
                    self.logger.info("Prepared paragraph %s/%s.", done, total)
                    self._report(f"准备翻译正文 {done}/{total}", done, total)

        for fig in paper.figures:
            translation_items.append((fig.caption_original, lambda value, f=fig: setattr(f, "caption_zh", value), f"figure_{fig.figure_id}"))

        self._translate_items_concurrently(translation_items, already_done=done, total=max(total, len(translation_items)))
        return paper

    def _should_skip_translation(self, text: str) -> bool:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return True
        if re.fullmatch(r"\d+", clean):
            return True
        if len(clean) < 25 and re.search(r"\b(article|doi|volume|issue|page)\b", clean, re.I):
            return True
        if re.search(
            r"©|copyright|all rights reserved|received:|accepted:|published:|correspondence|corresponding author|"
            r"publisher'?s note|author contributions|competing interests|conflict of interest|data availability|"
            r"code availability|supplementary information|www\.|https?://",
            clean,
            re.I,
        ):
            return True
        return False

    def _report(self, message: str, done: int, total: int) -> None:
        if not self.progress:
            return
        if total <= 0:
            self.progress(message, 0.46)
            return
        # Translation owns roughly 45%-62% of the total pipeline progress.
        self.progress(message, 0.45 + min(done / total, 1.0) * 0.17)

    def translate_text(self, text: str) -> str:
        if not text:
            return ""
        if self.llm.available():
            glossary = self._relevant_glossary([text])
            result = self.llm.chat(TRANSLATE_SYSTEM_PROMPT, f"术语表：\n{glossary}\n\n待翻译段落：\n{text}")
            if result.strip():
                return result.strip()
        return self._fallback_translate(text)

    def translate_caption(self, text: str) -> str:
        if not text:
            return ""
        if self.llm.available():
            result = self.llm.chat(CAPTION_SYSTEM_PROMPT, text)
            if result.strip():
                return result.strip()
        return self._fallback_translate(text)

    def translate_texts_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        if not self.llm.available():
            return [self._fallback_translate(text) for text in texts]
        glossary = self._relevant_glossary(texts)
        blocks = "\n\n".join(f"<<<{i}>>>\n{text}" for i, text in enumerate(texts))
        prompt = f"术语表：\n{glossary or '无'}\n\n待翻译段落：\n{blocks}"
        self.logger.info("Batch translation prompt chars=%s, paragraphs=%s.", len(prompt), len(texts))
        result = self.llm.chat(BATCH_TRANSLATE_SYSTEM_PROMPT, prompt)
        parsed = self._parse_marked_translations(result, len(texts))
        if not parsed:
            self.logger.warning("Batch translation returned unparseable marked text; using local fallback for this batch.")
            return [self._fallback_translate(text) for text in texts]
        translations = [translation or self._fallback_translate(text) for translation, text in zip(parsed, texts)]
        return [self._repair_if_truncated(source, translated) for source, translated in zip(texts, translations)]

    def _translate_items_concurrently(self, items: list[tuple[str, Callable[[str], None], str]], already_done: int, total: int) -> None:
        if not items:
            return
        items = self._dedupe_items(items)
        batches = self._make_item_batches(items)
        self.logger.info(
            "Starting concurrent batch translation: items=%s, batches=%s, workers=%s.",
            len(items),
            len(batches),
            self.max_workers,
        )
        completed_items = 0
        if not self.llm.available() or self.max_workers == 1 or len(batches) == 1:
            for batch_index, batch in enumerate(batches, start=1):
                self.logger.info("Translating batch %s/%s with %s items.", batch_index, len(batches), len(batch))
                translations = self.translate_texts_batch([item[0] for item in batch])
                for (_, setter, _), translated in zip(batch, translations):
                    setter(translated)
                completed_items += len(batch)
                self._report(f"批量翻译完成 {min(already_done + completed_items, total)}/{total} 段", min(already_done + completed_items, total), total)
            return

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            future_map = {
                executor.submit(self.translate_texts_batch, [item[0] for item in batch]): (batch_index, batch)
                for batch_index, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_map):
                batch_index, batch = future_map[future]
                try:
                    translations = future.result()
                except Exception as exc:
                    self.logger.exception("Translation batch %s failed; using local fallback: %s", batch_index, exc)
                    translations = [self._fallback_translate(item[0]) for item in batch]
                for (_, setter, _), translated in zip(batch, translations):
                    setter(translated)
                completed_items += len(batch)
                self.logger.info("Completed translation batch %s/%s.", batch_index, len(batches))
                self._report(f"批量翻译完成 {min(already_done + completed_items, total)}/{total} 段", min(already_done + completed_items, total), total)

    def _dedupe_items(self, items: list[tuple[str, Callable[[str], None], str]]) -> list[tuple[str, Callable[[str], None], str]]:
        grouped: dict[str, list[Callable[[str], None]]] = {}
        labels: dict[str, str] = {}
        for text, setter, label in items:
            key = re.sub(r"\s+", " ", text).strip()
            if not key:
                continue
            grouped.setdefault(key, []).append(setter)
            labels.setdefault(key, label)

        deduped: list[tuple[str, Callable[[str], None], str]] = []
        for text, setters in grouped.items():
            def apply_all(value: str, targets=setters) -> None:
                for target in targets:
                    target(value)

            deduped.append((text, apply_all, labels[text]))
        removed = len(items) - len(deduped)
        if removed:
            self.logger.info("Removed %s duplicate translation items before LLM calls.", removed)
        return deduped

    def _make_batches(self, paragraphs) -> list[list]:
        batches = []
        current = []
        current_chars = 0
        for para in paragraphs:
            text_len = len(para.text_original)
            if current and (len(current) >= self.batch_size or current_chars + text_len > self.batch_max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(para)
            current_chars += text_len
        if current:
            batches.append(current)
        return batches

    def _make_item_batches(self, items: list[tuple[str, Callable[[str], None], str]]) -> list[list[tuple[str, Callable[[str], None], str]]]:
        batches: list[list[tuple[str, Callable[[str], None], str]]] = []
        current: list[tuple[str, Callable[[str], None], str]] = []
        current_chars = 0
        for item in items:
            text_len = len(item[0])
            if current and (len(current) >= self.batch_size or current_chars + text_len > self.batch_max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += text_len
        if current:
            batches.append(current)
        return batches

    def _parse_marked_translations(self, text: str, expected: int) -> list[str] | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        matches = list(re.finditer(r"<<<\s*(\d+)\s*>>>", cleaned))
        if not matches:
            return None
        translations = [""] * expected
        for i, match in enumerate(matches):
            idx = int(match.group(1))
            if idx >= expected:
                continue
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            translations[idx] = cleaned[start:end].strip()
        return translations

    def _repair_if_truncated(self, source: str, translation: str) -> str:
        if not self._looks_truncated(source, translation):
            return translation
        self.logger.warning(
            "Detected suspiciously short or ellipsis-ended translation; retrying as single paragraph. source_chars=%s translated_chars=%s",
            len(source),
            len(translation),
        )
        retry = self._retry_single_complete_translation(source)
        if retry and not self._looks_truncated(source, retry):
            return retry
        self.logger.warning("Single paragraph retry still looked incomplete; using local fallback for this paragraph.")
        return self._fallback_translate(source)

    def _looks_truncated(self, source: str, translation: str) -> bool:
        src = re.sub(r"\s+", " ", source).strip()
        dst = re.sub(r"\s+", " ", translation).strip()
        if not src:
            return False
        if not dst:
            return True
        source_has_ellipsis = bool(re.search(r"(\.\.\.|…|……)\s*$", src))
        translated_has_ellipsis = bool(re.search(r"(\.\.\.|…|……)\s*$", dst))
        if translated_has_ellipsis and not source_has_ellipsis:
            return True
        # Exclude non-translateable content (numbers, formulas, DOIs, units) from length ratio.
        # Chinese text is usually shorter than English, but very short output is suspicious.
        non_translate = re.findall(
            r"\b(?:10\.\d{4,9}/[^\s]+|\d+(?:\.\d+)?\s*(?:%|°C|K|eV|nm|µm|cm|mm|mV|V|mA|mW|h|s|cm2|cm-2))"
            r"|[A-Z][a-z]{1,5}\d+(?:-\d+)?(?:\s*[A-Z])?"
            r"|[A-Z]{2,}(?!\w)",
            src,
        )
        sig_len_src = len(src) - sum(len(m) for m in non_translate)
        sig_len_dst = len(dst) - sum(len(m) for m in non_translate)
        if sig_len_src <= 0:
            return False
        ratio = sig_len_dst / sig_len_src
        # Chinese text typically ~70% length of English; <30% is suspicious
        if sig_len_src >= 300 and ratio < 0.30:
            return True
        # For shorter texts, use a more lenient threshold
        if sig_len_src >= 180 and ratio < 0.20:
            return True
        return False

    def _retry_single_complete_translation(self, text: str) -> str:
        if not self.llm.available():
            return ""
        glossary = self._relevant_glossary([text])
        prompt = (
            f"术语表：\n{glossary or '无'}\n\n"
            "请完整翻译下面这一整段英文科研论文内容。不要总结，不要省略，不要使用“……”或“...”。\n\n"
            f"{text}"
        )
        return self.llm.chat(TRANSLATE_SYSTEM_PROMPT, prompt)

    def _relevant_glossary(self, texts: list[str]) -> str:
        combined = "\n".join(texts).lower()
        rows = []
        for en, zh in TERMINOLOGY.items():
            if en.lower() in combined or (en.isupper() and re.search(rf"\b{re.escape(en)}\b", "\n".join(texts))):
                rows.append(f"{en}={zh}")
        return "; ".join(rows)

    def _fallback_translate(self, text: str) -> str:
        translated = text
        for en, zh in sorted(TERMINOLOGY.items(), key=lambda kv: len(kv[0]), reverse=True):
            translated = re.sub(re.escape(en), zh, translated, flags=re.I)
        return translated
