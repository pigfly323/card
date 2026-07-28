from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ToolSubprocessResult:
    ok: bool
    status: Optional[bool] = None
    data: Any = None
    new_subdb: Optional[Dict[str, Any]] = None
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    error: Optional[str] = None
    memory_info: Optional[Dict[str, Any]] = None


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    raw = (stdout or "").strip()
    if not raw:
        raise ValueError("empty stdout")

    # Prefer parsing the entire stdout (allows pretty-printed JSON).
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: parse the last non-empty line (allows logs + final JSON line).
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty stdout")
    obj = json.loads(lines[-1])
    if not isinstance(obj, dict):
        raise ValueError("stdout JSON must be an object")
    return obj


def run_tool_script(
    *,
    script_path: Path,
    subdb: Dict[str, Any],
    input_dict: Dict[str, Any],
    timeout_seconds: int = 120,
    memory_limit_mb: Optional[int] = None,
) -> ToolSubprocessResult:
    if not isinstance(script_path, Path):
        script_path = Path(str(script_path))

    if not script_path.exists():
        return ToolSubprocessResult(
            ok=False, error=f"tool_script_not_found:{script_path}"
        )

    try:
        subdb_json = json.dumps(subdb, ensure_ascii=False)
        input_json = json.dumps(input_dict, ensure_ascii=False)
    except Exception as e:
        return ToolSubprocessResult(ok=False, error=f"json_serialize_failed:{e}")

    input_summary: Dict[str, Any] = {}
    try:
        input_summary = {
            "input_keys": list(input_dict.keys())[:50]
            if isinstance(input_dict, dict)
            else [],
            "input_preview": input_json[:1000],
            "subdb_keys": list(subdb.keys())[:50] if isinstance(subdb, dict) else [],
        }
    except Exception:
        input_summary = {}

    cmd = [
        sys.executable,
        str(script_path),
        "--subdbdict",
        subdb_json,
        "--inputdict",
        input_json,
    ]

    # Ensure repo-local packages  are importable when running tool scripts
    # located under Data/... by injecting a best-effort repo root into PYTHONPATH.
    env = os.environ.copy()
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "src").is_dir():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(parent) + (
                os.pathsep + existing if existing else ""
            )
            break

    # Prepare preexec_fn for memory limiting (Linux/macOS only)
    preexec_fn = None
    if memory_limit_mb and sys.platform != "win32":

        def _limit_memory():
            try:
                import resource

                limit_bytes = int(memory_limit_mb * 1024 * 1024)
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
            except Exception as e:
                sys.stderr.write(f"Error setting memory limit: {e}\n")

        preexec_fn = _limit_memory

    # Use Popen to allow monitoring (fallback or enforcement)
    monitor_thread = None
    oom_detected = False
    peak_rss_mb: Optional[float] = None
    rss_mb_at_kill: Optional[float] = None
    process = None

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=preexec_fn,
        )

        # Start Monitor if memory_limit_mb is set
        if memory_limit_mb:
            import threading
            import time

            def _monitor():
                nonlocal oom_detected, peak_rss_mb, rss_mb_at_kill
                try:
                    import psutil

                    proc = psutil.Process(process.pid)
                    limit_bytes = memory_limit_mb * 1024 * 1024
                    while process.poll() is None:
                        try:
                            rss = proc.memory_info().rss
                            rss_mb = rss / (1024 * 1024)
                            peak_rss_mb = max(peak_rss_mb or 0.0, rss_mb)
                            if rss > limit_bytes:
                                oom_detected = True
                                rss_mb_at_kill = rss_mb
                                process.kill()
                                break
                        except Exception:
                            # Process might have ended
                            break
                        time.sleep(0.1)
                except ImportError:
                    pass

            monitor_thread = threading.Thread(target=_monitor, daemon=True)
            monitor_thread.start()

        try:
            stdout, stderr = process.communicate(
                timeout=int(timeout_seconds) if timeout_seconds else None
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return ToolSubprocessResult(ok=False, error="tool_timeout")

    except Exception as e:
        if process:
            process.kill()
        return ToolSubprocessResult(ok=False, error=f"subprocess_failed:{e}")

    if monitor_thread and monitor_thread.is_alive():
        # Wait for monitor thread to finish if needed, though daemon handles it
        pass

    completed_returncode = process.returncode
    stdout = stdout or ""
    stderr = stderr or ""

    if oom_detected:
        return ToolSubprocessResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed_returncode,
            error="OOM_SIGNAL_ERROR",
            memory_info={
                "limit_mb": memory_limit_mb,
                "reason": "Monitor detected OOM",
                "peak_rss_mb": peak_rss_mb,
                "rss_mb_at_kill": rss_mb_at_kill,
                **input_summary,
            },
        )

    # Check for OOM or other failures
    if completed_returncode != 0:
        # Check for SIGKILL (often OOM Killer) - Return code is negative of signal
        if completed_returncode == -9:
            return ToolSubprocessResult(
                ok=False,
                stdout=stdout,
                stderr=stderr,
                returncode=completed_returncode,
                error="OOM_SIGNAL_ERROR",
                memory_info={
                    "limit_mb": memory_limit_mb,
                    "reason": "SIGKILL",
                    **input_summary,
                },
            )

        # Check stderr for MemoryError (if Python caught it before SIGKILL)
        if "MemoryError" in stderr:
            return ToolSubprocessResult(
                ok=False,
                stdout=stdout,
                stderr=stderr,
                returncode=completed_returncode,
                error="OOM_SIGNAL_ERROR",
                memory_info={
                    "limit_mb": memory_limit_mb,
                    "reason": "MemoryError",
                    **input_summary,
                },
            )

        return ToolSubprocessResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed_returncode,
            error="tool_nonzero_exit",
        )

    try:
        payload = _parse_json_from_stdout(stdout)
    except Exception as e:
        return ToolSubprocessResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed_returncode,
            error=f"tool_stdout_json_parse_failed:{e}",
        )

    status = payload.get("status")
    new_subdb = payload.get("new_subdb")
    if not isinstance(status, bool):
        return ToolSubprocessResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed_returncode,
            error="tool_invalid_status_type",
        )
    if not isinstance(new_subdb, dict):
        return ToolSubprocessResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed_returncode,
            error="tool_invalid_new_subdb_type",
        )

    return ToolSubprocessResult(
        ok=True,
        status=status,
        data=payload.get("data"),
        new_subdb=new_subdb,
        stdout=stdout,
        stderr=stderr,
        returncode=completed_returncode,
    )
