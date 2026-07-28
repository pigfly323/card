"""
Utilities for loading deterministic protocol/contract blocks from `documents/dataprotocol.md`.

This enables prompt templates to reference authoritative contract text via placeholders like:
- {tool_parameters_contract}
- {output_schema_contract}
- {subdb_schema_contract}
- {toolscript_contract}
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict


_SECTION_RE = re.compile(r"^###\s+11\.(\d+)\b")


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "documents" / "dataprotocol.md").is_file():
            return p
    return start


def _read_dataprotocol() -> str:
    repo_root = _find_repo_root(Path(__file__).resolve())
    path = repo_root / "documents" / "dataprotocol.md"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_contract_blocks() -> Dict[str, str]:
    """
    Load Section 11.* contract blocks as a mapping from placeholder name -> markdown text.
    """
    lines = _read_dataprotocol().splitlines(keepends=True)

    by_section: Dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is None:
            return
        text = "".join(buf).rstrip() + "\n"
        by_section[current] = text
        buf = []

    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            flush()
            current = f"11.{m.group(1)}"
            buf = [line]
            continue
        if current is not None:
            buf.append(line)

    flush()

    mapping: Dict[str, str] = {
        "tool_parameters_contract": by_section.get("11.1", ""),
        "output_schema_contract": by_section.get("11.2", ""),
        "subdb_schema_contract": by_section.get("11.3", ""),
        "toolscript_contract": by_section.get("11.4", ""),
    }

    missing = [k for k, v in mapping.items() if not v.strip()]
    if missing:
        raise ValueError(
            "Failed to load required dataprotocol contract blocks from documents/dataprotocol.md: "
            + ", ".join(missing)
        )

    return mapping
