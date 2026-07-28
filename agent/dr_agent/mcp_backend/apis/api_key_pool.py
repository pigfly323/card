"""
API Key Pool — thread-safe round-robin key manager with rate-limit cooling.

Usage:
    pool = ApiKeyPool.from_env("SERPER_API_KEY_LIST", "SERPER_API_KEY")
    key  = pool.get()          # get next available key
    pool.mark_rate_limited(key)  # call when 429 received; key enters cooldown

Environment variables:
    SERPER_API_KEY_LIST  comma-separated list of keys (takes priority)
    SERPER_API_KEY       single key fallback
    S2_API_KEY_LIST      comma-separated list of keys
    S2_API_KEY           single key fallback

    API_KEY_COOLDOWN_SECONDS  cooldown duration after 429 (default: 60)
"""

import logging
import os
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN = float(os.environ.get("API_KEY_COOLDOWN_SECONDS", "60"))


class ApiKeyPool:
    """
    Thread-safe round-robin API key pool with automatic rate-limit cooldown.

    When a key receives a 429 response, call mark_rate_limited(key) to put
    it in cooldown for API_KEY_COOLDOWN_SECONDS seconds. During cooldown the
    key is skipped; it re-enters rotation automatically after the cooldown.

    If all keys are in cooldown, get() waits up to `wait_timeout` seconds
    for any key to become available, then returns the least-recently-cooled
    key as a last resort.
    """

    def __init__(self, keys: List[str], cooldown_seconds: float = _DEFAULT_COOLDOWN):
        if not keys:
            raise ValueError("ApiKeyPool requires at least one key")
        self._keys = list(dict.fromkeys(k.strip() for k in keys if k.strip()))
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        # Start round-robin from a PID-based offset so that different
        # processes (e.g. multiple MCP server instances) spread load
        # across keys instead of all hammering the same first key.
        self._index = os.getpid() % len(self._keys)
        self._cooled_until: dict[str, float] = {}  # key → epoch when cooldown ends

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        list_env: str,
        single_env: str,
        cooldown_seconds: float = _DEFAULT_COOLDOWN,
    ) -> "ApiKeyPool":
        """
        Build pool from environment variables.

        list_env  — comma-separated list (e.g. SERPER_API_KEY_LIST)
        single_env — single key fallback (e.g. SERPER_API_KEY)
        """
        raw_list = os.environ.get(list_env, "")
        keys = [k.strip() for k in raw_list.split(",") if k.strip()]
        if not keys:
            single = os.environ.get(single_env, "").strip()
            if single:
                keys = [single]
        if not keys:
            raise ValueError(
                f"No API keys found. Set {list_env} or {single_env} environment variable."
            )
        pool = cls(keys, cooldown_seconds)
        logger.info(f"[ApiKeyPool] Loaded {len(keys)} key(s) from {list_env or single_env}")
        return pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, wait_timeout: float = 10.0) -> str:
        """
        Return the next available (non-cooled) key using round-robin.

        If all keys are in cooldown, waits up to wait_timeout seconds for
        any key to recover, then returns the key with the shortest remaining
        cooldown as a last resort.
        """
        deadline = time.monotonic() + wait_timeout
        while True:
            with self._lock:
                key = self._next_available()
                if key is not None:
                    return key
                # All keys cooled — find soonest recovery
                soonest_key, soonest_time = min(
                    self._cooled_until.items(), key=lambda x: x[1]
                )
                wait_remaining = soonest_time - time.monotonic()

            if time.monotonic() >= deadline or wait_remaining <= 0:
                logger.warning(
                    f"[ApiKeyPool] All keys in cooldown; using {soonest_key[:8]}... as fallback"
                )
                return soonest_key

            sleep_time = min(wait_remaining, deadline - time.monotonic(), 1.0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def mark_rate_limited(self, key: str, cooldown_seconds: Optional[float] = None) -> None:
        """Mark a key as rate-limited; it will be skipped until cooldown expires."""
        cd = cooldown_seconds if cooldown_seconds is not None else self._cooldown
        with self._lock:
            self._cooled_until[key] = time.monotonic() + cd
        logger.warning(
            f"[ApiKeyPool] Key {key[:8]}... rate-limited; cooling for {cd:.0f}s "
            f"({self._available_count()} of {len(self._keys)} keys available)"
        )

    def size(self) -> int:
        return len(self._keys)

    def available_count(self) -> int:
        with self._lock:
            return self._available_count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self, key: str) -> bool:
        """Must be called with self._lock held."""
        until = self._cooled_until.get(key, 0.0)
        if until and time.monotonic() < until:
            return False
        # Auto-clear expired cooldown
        if key in self._cooled_until:
            del self._cooled_until[key]
        return True

    def _available_count(self) -> int:
        """Must be called with self._lock held."""
        return sum(1 for k in self._keys if self._is_available(k))

    def _next_available(self) -> Optional[str]:
        """Round-robin scan starting from _index. Must be called with lock held."""
        n = len(self._keys)
        for offset in range(n):
            idx = (self._index + offset) % n
            key = self._keys[idx]
            if self._is_available(key):
                self._index = (idx + 1) % n  # advance cursor past this key
                return key
        return None


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_serper_pool: Optional[ApiKeyPool] = None
_s2_pool: Optional[ApiKeyPool] = None
_pool_lock = threading.Lock()


def _build_pool_with_google_priority(list_env: str, single_env: str) -> ApiKeyPool:
    """
    Build an ApiKeyPool that puts GOOGLE_SERPENT_API_KEY_LIST keys first.

    Order: google keys → remaining keys from list_env / single_env
    This means round-robin naturally prefers google keys; they must all be
    rate-limited before the pool falls back to ordinary keys.
    """
    google_raw = os.environ.get("GOOGLE_SERPENT_API_KEY_LIST", "")
    google_keys = [k.strip() for k in google_raw.split(",") if k.strip()]

    regular_raw = os.environ.get(list_env, "")
    regular_keys = [k.strip() for k in regular_raw.split(",") if k.strip()]
    if not regular_keys:
        single = os.environ.get(single_env, "").strip()
        if single:
            regular_keys = [single]

    # Deduplicate while preserving order: google first, then the rest
    seen: set = set()
    ordered: list = []
    for k in google_keys + regular_keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)

    if not ordered:
        raise ValueError(
            f"No API keys found. Set GOOGLE_SERPENT_API_KEY_LIST, {list_env}, or {single_env}."
        )

    n_google = len([k for k in ordered if k in set(google_keys)])
    logger.info(
        "[ApiKeyPool] %s pool: %d total key(s) (%d google-priority, %d regular)",
        list_env, len(ordered), n_google, len(ordered) - n_google,
    )
    pool = ApiKeyPool(ordered)
    # Override pid-based start index so round-robin always begins with
    # google keys (index 0) rather than a random position that may skip them.
    pool._index = 0
    return pool


def get_serper_pool() -> ApiKeyPool:
    global _serper_pool
    if _serper_pool is None:
        with _pool_lock:
            if _serper_pool is None:
                _serper_pool = _build_pool_with_google_priority(
                    "SERPER_API_KEY_LIST", "SERPER_API_KEY"
                )
    return _serper_pool


def get_s2_pool() -> ApiKeyPool:
    global _s2_pool
    if _s2_pool is None:
        with _pool_lock:
            if _s2_pool is None:
                _s2_pool = _build_pool_with_google_priority(
                    "S2_API_KEY_LIST", "S2_API_KEY"
                )
    return _s2_pool


_draft_answer_pool: Optional[ApiKeyPool] = None


def get_draft_answer_pool() -> ApiKeyPool:
    global _draft_answer_pool
    if _draft_answer_pool is None:
        with _pool_lock:
            if _draft_answer_pool is None:
                _draft_answer_pool = ApiKeyPool.from_env(
                    "DRAFT_ANSWER_API_KEY_LIST", "DRAFT_ANSWER_API_KEY"
                )
    return _draft_answer_pool


# ---------------------------------------------------------------------------
# Google Serper key routing helpers
# ---------------------------------------------------------------------------

_google_serper_keys: Optional[set] = None
_google_key_lock = threading.Lock()


def get_google_serper_keys() -> set:
    """Return set of keys from GOOGLE_SERPENT_API_KEY_LIST (lazy, thread-safe)."""
    global _google_serper_keys
    if _google_serper_keys is None:
        with _google_key_lock:
            if _google_serper_keys is None:
                raw = os.environ.get("GOOGLE_SERPENT_API_KEY_LIST", "")
                _google_serper_keys = {k.strip() for k in raw.split(",") if k.strip()}
                logger.info(
                    "[ApiKeyPool] GOOGLE_SERPENT_API_KEY_LIST: %d key(s) loaded",
                    len(_google_serper_keys),
                )
    return _google_serper_keys


def is_google_serper_key(key: str) -> bool:
    """Return True if key is in GOOGLE_SERPENT_API_KEY_LIST."""
    return bool(key) and key in get_google_serper_keys()
