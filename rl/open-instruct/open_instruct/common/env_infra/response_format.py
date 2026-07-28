from __future__ import annotations

import json
from typing import Any, Optional


def env_error(code: str, detail: Optional[str] = None) -> str:
    code_str = str(code or "").strip() or "UNKNOWN"
    if detail is None:
        return f"ENV_ERROR:{code_str}"
    detail_str = str(detail).strip()
    if not detail_str:
        return f"ENV_ERROR:{code_str}"
    return f"ENV_ERROR:{code_str}: {detail_str}"


def env_error_message(code: str, detail: Optional[str] = None) -> str:
    return env_error(code, detail)


def format_message_value(data: Any) -> str:
    """
    Standardize ENV `.call_tool(...)` response message:
      {"status": bool, "message": string}

    - If ToolScript returns data like {"message": "<string>"}, unwrap to "<string>".
    - Otherwise JSON-encode the data to preserve structure for downstream parsing.
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if sorted(data.keys()) == ["message"] and isinstance(data.get("message"), str):
            return data["message"]
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)

