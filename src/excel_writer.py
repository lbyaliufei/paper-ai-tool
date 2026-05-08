from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


class ExcelWriter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write(self, structured: dict[str, Any], output_path: Path) -> None:
        sheets = {
            "paper_info": [structured.get("paper_info", {})],
            "device_performance": structured.get("device_performance", []),
            "stability_tests": structured.get("stability_tests", []),
            "characterization_methods": structured.get("characterization_methods", []),
            "key_innovations": structured.get("key_innovations", []),
            "figures": self._figure_rows(structured.get("figures", [])),
        }
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, rows in sheets.items():
                df = pd.DataFrame(rows or [{}])
                for col in df.columns:
                    df[col] = df[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
                df.to_excel(writer, sheet_name=name[:31], index=False)

    def _figure_rows(self, figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for fig in figures:
            row = dict(fig)
            if row.get("image_base64"):
                row["image_base64"] = f"<base64 length={len(row['image_base64'])}>"
            rows.append(row)
        return rows
