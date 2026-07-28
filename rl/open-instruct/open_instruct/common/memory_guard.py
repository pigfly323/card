import _thread
import functools
import io
import logging
import os
import platform
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

# Try importing resource (Linux/Mac)
try:
    import resource
except ImportError:
    resource = None

# Try importing psutil
try:
    import psutil
except ImportError:
    psutil = None

import tracemalloc

logger = logging.getLogger(__name__)


def _safe_repr(value: Any, max_len: int = 300) -> str:
    """Best-effort repr that stays bounded."""
    try:
        s = repr(value)
    except Exception:
        s = f"<unreprable {type(value)}>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _capture_all_thread_stacks(limit_lines_per_thread: int = 200) -> str:
    """Capture stacks for all threads (best effort)."""
    out = io.StringIO()
    out.write("=== THREAD STACKS (best effort) ===\n")
    try:
        frames = sys._current_frames()  # type: ignore[attr-defined]
    except Exception:
        frames = {}
    for tid, frame in frames.items():
        out.write(f"\n--- thread_id={tid} ---\n")
        try:
            stack = traceback.format_stack(frame)
            if len(stack) > limit_lines_per_thread:
                stack = stack[-limit_lines_per_thread:]
            out.write("".join(stack))
        except Exception:
            out.write("<failed to format stack>\n")
    return out.getvalue()


class MemoryLimitError(Exception):
    """Raised when memory limit is exceeded."""

    def __init__(
        self,
        message: str,
        peak_mb: float,
        limit_mb: float,
        params_snapshot: Optional[Dict] = None,
        diagnostics: Optional[Dict] = None,
        traceback_str: Optional[str] = None,
    ):
        self.message = message
        self.peak_mb = peak_mb
        self.limit_mb = limit_mb
        self.pid = os.getpid()
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.params_snapshot = params_snapshot or {}
        # If raised from a monitor thread, format_exc() can be empty/useless.
        self.traceback_str = (
            traceback_str or traceback.format_exc() or _capture_all_thread_stacks()
        )
        self.diagnostics = diagnostics or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "MEMORY_LIMIT_EXCEEDED",
            "message": self.message,
            "peak_mb": self.peak_mb,
            "limit_mb": self.limit_mb,
            "pid": self.pid,
            "timestamp": self.timestamp,
            "params_snapshot": _safe_repr(self.params_snapshot, max_len=1500),
            "traceback": self.traceback_str,
            "diagnostics": self.diagnostics,
        }


class MemoryMonitor(threading.Thread):
    """
    Background thread to monitor memory usage of the current process.
    Used as a fallback when resource.setrlimit is not available or not sufficient.
    """

    def __init__(
        self,
        limit_mb: float,
        check_interval: float = 0.1,
        detailed_diagnostics: bool = True,
    ):
        super().__init__(daemon=True)
        self.limit_mb = limit_mb
        self.check_interval = check_interval
        self.detailed_diagnostics = detailed_diagnostics
        self._stop_event = threading.Event()
        self._exception: Optional[MemoryLimitError] = None
        self.process = psutil.Process(os.getpid()) if psutil else None
        self.peak_memory = 0.0
        self.last_memory = 0.0

    def run(self):
        if not self.process:
            return

        while not self._stop_event.is_set():
            try:
                # RSS is the Resident Set Size (memory actually used in RAM)
                mem_info = self.process.memory_info()
                current_mb = mem_info.rss / (1024 * 1024)
                self.last_memory = current_mb
                self.peak_memory = max(self.peak_memory, current_mb)

                if current_mb > self.limit_mb:
                    self._raise_oom(current_mb)
                    break

                time.sleep(self.check_interval)
            except Exception:
                # Avoid crashing the monitor thread on transient errors
                pass

    def stop(self):
        self._stop_event.set()

    def check_exception(self):
        if self._exception:
            raise self._exception

    def _raise_oom(self, current_mb: float):
        diagnostics = {}
        if self.detailed_diagnostics:
            try:
                # Capture top memory blocks if tracemalloc is enabled
                if tracemalloc.is_tracing():
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics("lineno")
                    diagnostics["top_allocations"] = [
                        str(stat) for stat in top_stats[:10]
                    ]
            except Exception:
                pass

        self._exception = MemoryLimitError(
            message=f"Memory limit exceeded: {current_mb:.2f}MB > {self.limit_mb:.2f}MB",
            peak_mb=current_mb,
            limit_mb=self.limit_mb,
            diagnostics=diagnostics,
            traceback_str=_capture_all_thread_stacks(),
        )
        # Best effort: interrupt main thread to abort quickly.
        try:
            _thread.interrupt_main()
        except Exception:
            pass


def memory_limit(
    limit_mb: int,
    mode: str = "auto",
    detailed_diagnostics: bool = True,
    params: Optional[Dict[str, Any]] = None,
    params_provider: Optional[Callable[[], Dict[str, Any]]] = None,
):
    """
    Decorator to limit memory usage of a function.

    Args:
        limit_mb: Memory limit in Megabytes.
        mode: 'auto', 'resource', or 'psutil'.
        detailed_diagnostics: Whether to collect detailed info on OOM.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve mode
            effective_mode = mode
            if effective_mode == "auto":
                # Prefer resource if available and not Windows (setrlimit not on Windows)
                if resource and platform.system() != "Windows":
                    effective_mode = "resource"
                else:
                    effective_mode = "psutil"

            # Prepare params snapshot for diagnostics (best effort, bounded)
            params_snapshot: Dict[str, Any] = {}
            try:
                if params_provider is not None:
                    params_snapshot = params_provider() or {}
                elif params is not None:
                    params_snapshot = dict(params)
                else:
                    params_snapshot["args_types"] = [str(type(a)) for a in args]
                    params_snapshot["kwargs_preview"] = {
                        k: _safe_repr(v, max_len=200)
                        for k, v in list(kwargs.items())[:20]
                    }
            except Exception:
                params_snapshot = {}

            if effective_mode == "resource":
                try:
                    assert resource is not None
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    new_limit_bytes = int(limit_mb * 1024 * 1024)

                    # CRITICAL: do not lower hard limit; lowering hard can be irreversible.
                    if hard != resource.RLIM_INFINITY:
                        soft_target = min(new_limit_bytes, hard)
                    else:
                        soft_target = new_limit_bytes

                    resource.setrlimit(resource.RLIMIT_AS, (soft_target, hard))

                    try:
                        return func(*args, **kwargs)
                    except MemoryError:
                        current_usage = 0
                        if psutil:
                            try:
                                current_usage = psutil.Process(
                                    os.getpid()
                                ).memory_info().rss / (1024 * 1024)
                            except Exception:
                                pass

                        raise MemoryLimitError(
                            message=f"Memory limit exceeded (RLIMIT_AS): limit={limit_mb}MB",
                            peak_mb=current_usage,
                            limit_mb=limit_mb,
                            params_snapshot=params_snapshot,
                        )
                    except KeyboardInterrupt:
                        # If we were interrupted by a monitor in fallback paths, propagate.
                        raise
                    finally:
                        # Try to restore limits
                        try:
                            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
                        except ValueError:
                            pass
                except Exception as e:
                    if isinstance(e, MemoryLimitError):
                        raise
                    # If setrlimit fails (e.g. not allowed), fallback to psutil
                    effective_mode = "psutil"

            if effective_mode == "psutil":
                if not psutil:
                    # No protection available
                    return func(*args, **kwargs)

                # Start tracing if requested
                started_tracemalloc = False
                if detailed_diagnostics and not tracemalloc.is_tracing():
                    tracemalloc.start()
                    started_tracemalloc = True

                monitor = MemoryMonitor(
                    limit_mb, detailed_diagnostics=detailed_diagnostics
                )
                monitor.start()

                try:
                    try:
                        return func(*args, **kwargs)
                    except KeyboardInterrupt:
                        # If monitor interrupted main thread, convert to MemoryLimitError.
                        try:
                            monitor.check_exception()
                        except MemoryLimitError as e:
                            e.params_snapshot = params_snapshot
                            raise e
                        raise
                finally:
                    monitor.stop()
                    monitor.join()
                    # Check if monitor detected OOM
                    try:
                        monitor.check_exception()
                    except MemoryLimitError as e:
                        e.params_snapshot = params_snapshot
                        raise e
                    finally:
                        if started_tracemalloc:
                            tracemalloc.stop()

            # Fallthrough
            return func(*args, **kwargs)

        return wrapper

    return decorator
