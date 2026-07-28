import json
import asyncio
import itertools
import re
import threading
import time
import weakref
import logging
import os
from typing import Any, Dict, Optional, List

import jsonlines
from openai import AzureOpenAI, OpenAI, AsyncOpenAI

LOGGER = logging.getLogger(__name__)


class ApiKeyPool:
    """
    Thread-safe round-robin API key pool with 429 backoff per key.

    Keys are loaded from the environment variable OPENAI_API_KEY_POOL
    as a comma-separated list. Falls back to OPENAI_API_KEY if not set.

    Example:
        export OPENAI_API_KEY_POOL="sk-key1,sk-key2,sk-key3"
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._cycle = None
        self._backoff_until: Dict[str, float] = {}  # key -> timestamp when usable again
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        pool_str = os.environ.get("OPENAI_API_KEY_POOL", "")
        if pool_str:
            self._keys = [k.strip() for k in pool_str.split(",") if k.strip()]
        if not self._keys:
            single = os.environ.get("OPENAI_API_KEY", "")
            if single:
                self._keys = [single]
        if not self._keys:
            raise ValueError(
                "No API keys found. Set OPENAI_API_KEY_POOL (comma-separated) or OPENAI_API_KEY."
            )
        # Shuffle the starting position based on process id so that different
        # worker processes (e.g. multiple learners) do not all start from the
        # same key and hammer the same rate-limit bucket.
        import random
        keys = list(self._keys)
        offset = os.getpid() % len(keys)
        keys = keys[offset:] + keys[:offset]
        self._cycle = itertools.cycle(keys)
        self._loaded = True

    def get_key(self) -> str:
        """Return the next available key using round-robin, skipping keys in backoff."""
        with self._lock:
            self._load()
            now = time.monotonic()
            # Try each key at most len(keys) times to find one not in backoff
            for _ in range(len(self._keys)):
                key = next(self._cycle)
                if now >= self._backoff_until.get(key, 0):
                    return key
            # All keys in backoff — return the one with shortest remaining wait
            key = min(self._keys, key=lambda k: self._backoff_until.get(k, 0))
            wait = self._backoff_until.get(key, 0) - now
            if wait > 0:
                LOGGER.warning("All API keys in backoff, waiting %.1fs for key %s...", wait, key[-4:])
                time.sleep(wait)
            return key

    def mark_rate_limited(self, key: str, retry_after: float = 60.0):
        """Mark a key as rate-limited for retry_after seconds."""
        with self._lock:
            self._backoff_until[key] = time.monotonic() + retry_after
            LOGGER.warning("API key ...%s rate-limited, backing off %.0fs", key[-4:], retry_after)


# Module-level singleton for the main key pool
_KEY_POOL = ApiKeyPool()


class _BypassKeyPool:
    """
    Lazy-loaded pool of API keys that bypass content-safety checks (error 450).
    Loaded from OPENAI_API_KEY_BYPASS_POOL (comma-separated).
    Falls back to _KEY_POOL if not configured.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._cycle = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        pool_str = os.environ.get("OPENAI_API_KEY_BYPASS_POOL", "")
        self._keys = [k.strip() for k in pool_str.split(",") if k.strip()]
        if self._keys:
            import random
            keys = list(self._keys)
            offset = os.getpid() % len(keys)
            keys = keys[offset:] + keys[:offset]
            self._cycle = itertools.cycle(keys)
        self._loaded = True

    def get_key(self) -> Optional[str]:
        """Return next bypass key, or None if pool is empty (caller falls back to main pool)."""
        with self._lock:
            self._load()
            if not self._keys:
                return None
            return next(self._cycle)

    def available(self) -> bool:
        with self._lock:
            self._load()
            return bool(self._keys)


_BYPASS_KEY_POOL = _BypassKeyPool()


# Per-event-loop concurrency control for async OpenAI calls to avoid event loop binding issues
_OPENAI_SEMAPHORES = weakref.WeakKeyDictionary()


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


def _build_chat_completions_url(base_url: Optional[str]) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _debug_print_openai_request(
    api_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: Optional[float] = None,
) -> None:
    headers = {
        "Authorization": f"Bearer {_mask_api_key(api_key)}",
        "Content-Type": "application/json",
    }
    print("[RUBRIC_OPENAI_DEBUG] Request URL:", api_url)
    print("[RUBRIC_OPENAI_DEBUG] Request Headers:", json.dumps(headers, ensure_ascii=False))
    print(
        "[RUBRIC_OPENAI_DEBUG] Request Payload:",
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    if timeout is not None:
        print("[RUBRIC_OPENAI_DEBUG] Request Timeout:", timeout)


def _debug_print_openai_response(content: str) -> None:
    print("[RUBRIC_OPENAI_DEBUG] Response Content:", content)


def _get_openai_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent async OpenAI requests."""
    loop = asyncio.get_running_loop()
    sem = _OPENAI_SEMAPHORES.get(loop)
    if sem is None:
        max_concurrent = int(
            os.environ.get(
                "OPENAI_MAX_CONCURRENT_CALLS",
                os.environ.get("LITELLM_MAX_CONCURRENT_CALLS", "256"),
            )
        )
        sem = asyncio.Semaphore(max_concurrent)
        _OPENAI_SEMAPHORES[loop] = sem
    return sem


def _normalize_model_name(model_name: str) -> str:
    # Accept `openai/<model>` format while using OpenAI SDK directly.
    if model_name and model_name.lower().startswith("openai/"):
        return model_name.split("/", 1)[1]
    return model_name


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from a model response.

    Handles common failure modes:
    - <think>...</think> blocks before the JSON (thinking models)
    - Truncated responses (missing closing })
    - Nested JSON / code blocks inside string values (e.g. evidence field)
    - BOM / invisible characters
    """
    if not response:
        return None

    # Strip <think>...</think> blocks first (QwQ-32B, DeepSeek-R1, etc.)
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Strategy: find the FIRST '{' and scan forward counting braces to find the
    # matching '}'. This correctly handles nested objects / arrays inside string
    # values because we only count structural braces, not those inside strings.
    def _extract_by_brace_counting(text: str) -> Optional[str]:
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Truncated response: no matching } found — try to close it
        if depth > 0:
            return text[start:] + '}' * depth
        return None

    for source in (cleaned, response):
        candidate = _extract_by_brace_counting(source)
        if not candidate:
            continue
        # Try direct parse first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # BOM / invisible chars
        try:
            return json.loads(candidate.strip().encode('utf-8').decode('utf-8-sig'))
        except json.JSONDecodeError:
            pass
        # Last resort: extract just score+reasoning via regex (evidence may be malformed)
        m_score = re.search(r'"score"\s*:\s*(\d+)', candidate)
        m_reason = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
        m_evidence = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
        if m_score:
            return {
                "score": int(m_score.group(1)),
                "reasoning": m_reason.group(1) if m_reason else "",
                "evidence": m_evidence.group(1) if m_evidence else "",
            }

    LOGGER.warning(f"Could not decode JSON from response: {repr(response[:300])}")
    return None


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    base_url: Optional[str] = None,
    **chat_kwargs,
) -> str:
    """
    Run OpenAI Chat Completions API directly (without LiteLLM).
    Uses OPENAI_API_KEY_POOL (comma-separated) for key rotation and 429 backoff.
    Falls back to OPENAI_API_KEY if pool is not set.
    base_url overrides env vars when provided explicitly.
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)

    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    debug_request = bool(chat_kwargs.pop("debug_request", False))
    debug_response = bool(chat_kwargs.pop("debug_response", False))

    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    if debug_request:
        _debug_print_openai_request(
            api_url=_build_chat_completions_url(base_url),
            api_key=api_key,
            payload={
                "model": _normalize_model_name(model_name),
                "messages": msgs,
                **chat_kwargs,
            },
        )

    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
    for attempt in range(max_retries):
        api_key = _KEY_POOL.get_key()
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = client.chat.completions.create(
                model=_normalize_model_name(model_name),
                messages=msgs,
                **chat_kwargs,
            )
            break
        except Exception as e:
            err_str = str(e)
            if "450" in err_str or "403" in err_str or "违规信息" in err_str or "sensitive information" in err_str.lower() or "ModelArts.81011" in err_str:
                bypass_key = _BYPASS_KEY_POOL.get_key()
                if bypass_key and bypass_key != api_key:
                    LOGGER.warning(f"run_chatopenai got content policy error, retrying with bypass key ...{bypass_key[-6:]}")
                    try:
                        _client = OpenAI(api_key=bypass_key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"))
                        _resp = _client.chat.completions.create(model=_normalize_model_name(model_name), messages=msgs, **chat_kwargs)
                        return _resp.choices[0].message.content or ""
                    except Exception as _bypass_exc:
                        LOGGER.warning(f"run_chatopenai bypass key also failed: {_bypass_exc}")
                else:
                    LOGGER.warning(f"run_chatopenai got content policy error, no bypass key available, giving up")
                return ""
            if "429" in err_str or "rate limit" in err_str.lower() or "too many requests" in err_str.lower() or "请求次数超过限制" in err_str:
                retry_after = float(os.environ.get("OPENAI_RETRY_AFTER", "60"))
                _KEY_POOL.mark_rate_limited(api_key, retry_after)
                if attempt < max_retries - 1:
                    wait = min(retry_after, 5 * (2 ** attempt))  # exponential backoff, capped at retry_after
                    LOGGER.warning(f"run_chatopenai 429 rate limited (attempt {attempt+1}/{max_retries}), sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                if attempt < max_retries - 1:
                    wait = 2 * (attempt + 1)
                    LOGGER.warning(f"run_chatopenai timeout (attempt {attempt+1}/{max_retries}), sleeping {wait}s")
                    time.sleep(wait)
                    continue
            LOGGER.error(f"Error in run_chatopenai (attempt {attempt+1}/{max_retries}): {e}")
            return ""
    else:
        LOGGER.error(f"run_chatopenai failed after {max_retries} attempts")
        return ""

    content = resp.choices[0].message.content
    content = content if content is not None else ""
    if debug_response:
        _debug_print_openai_response(content)
    return content


async def run_chatopenai_async(
    model_name: str,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    json_mode: bool = False,
    base_url: Optional[str] = None,
    **chat_kwargs,
) -> str:
    """
    Async OpenAI Chat Completions API wrapper.
    Uses per-event-loop semaphores to avoid cross-loop binding issues.
    base_url overrides env vars when provided explicitly.
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)

    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    debug_request = bool(chat_kwargs.pop("debug_request", False))
    debug_response = bool(chat_kwargs.pop("debug_response", False))

    user_prompt_local = user_prompt if user_prompt is not None else ""

    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_local},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt_local}]
    )

    timeout = chat_kwargs.pop("timeout", None)
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
    sem = _get_openai_semaphore()

    api_url = _build_chat_completions_url(base_url)
    LOGGER.debug(
        "run_chatopenai_async start: url=%s model=%s timeout=%s max_retries=%d",
        api_url, model_name, timeout, max_retries,
    )

    for attempt in range(max_retries):
        api_key = _KEY_POOL.get_key()

        if debug_request:
            _debug_print_openai_request(
                api_url=api_url,
                api_key=api_key,
                payload={
                    "model": _normalize_model_name(model_name),
                    "messages": msgs,
                    **chat_kwargs,
                },
                timeout=timeout,
            )

        await sem.acquire()
        try:
            async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
                create_coro = client.chat.completions.create(
                    model=_normalize_model_name(model_name),
                    messages=msgs,
                    timeout=timeout,
                    **chat_kwargs,
                )
                if timeout is not None:
                    resp = await asyncio.wait_for(create_coro, timeout=timeout + 1.0)
                else:
                    resp = await create_coro
            LOGGER.debug("run_chatopenai_async success on attempt %d", attempt + 1)
            break  # success
        except asyncio.TimeoutError:
            LOGGER.warning(
                "run_chatopenai_async timed out (attempt %d/%d) url=%s model=%s key=...%s",
                attempt + 1, max_retries, api_url, model_name, api_key[-6:],
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            LOGGER.error("run_chatopenai_async timed out after %d attempts, giving up", max_retries)
            return ""
        except Exception as e:
            err_str = str(e)
            exc_type = type(e).__name__
            LOGGER.warning(
                "run_chatopenai_async exception (attempt %d/%d) url=%s model=%s key=...%s "
                "exc_type=%s exc=%s",
                attempt + 1, max_retries, api_url, model_name, api_key[-6:],
                exc_type, err_str,
            )
            # Error 450 / 403 = content policy violation on the current key.
            # Switch to a bypass key (OPENAI_API_KEY_BYPASS_POOL) and retry once.
            # If no bypass pool is configured, give up immediately.
            if "450" in err_str or "403" in err_str or "违规信息" in err_str or "sensitive information" in err_str.lower() or "ModelArts.81011" in err_str:
                bypass_key = _BYPASS_KEY_POOL.get_key()
                if bypass_key and bypass_key != api_key:
                    LOGGER.warning(
                        "run_chatopenai_async got content policy error, retrying with bypass key ...%s | url=%s model=%s",
                        bypass_key[-6:], api_url, model_name,
                    )
                    try:
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=timeout)
                        ) as _sess:
                            async with _sess.post(
                                api_url,
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {bypass_key}",
                                },
                                json=payload,
                            ) as _resp:
                                _text = await _resp.text()
                                if _resp.status == 200:
                                    _obj = json.loads(_text)
                                    return _obj["choices"][0]["message"]["content"]
                                LOGGER.warning(
                                    "run_chatopenai_async bypass key also failed (%d): %s",
                                    _resp.status, _text[:200],
                                )
                    except Exception as _bypass_exc:
                        LOGGER.warning("run_chatopenai_async bypass key exception: %s", _bypass_exc)
                else:
                    LOGGER.warning(
                        "run_chatopenai_async got 450, no bypass key available, giving up | url=%s model=%s",
                        api_url, model_name,
                    )
                return ""
            if "429" in err_str or "rate limit" in err_str.lower() or "too many requests" in err_str.lower() or "请求次数超过限制" in err_str:
                retry_after = float(os.environ.get("OPENAI_RETRY_AFTER", "60"))
                _KEY_POOL.mark_rate_limited(api_key, retry_after)
                if attempt < max_retries - 1:
                    wait = min(retry_after, 5 * (2 ** attempt))
                    LOGGER.warning(
                        "run_chatopenai_async 429 rate limited (attempt %d/%d), sleeping %.1fs",
                        attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
            LOGGER.error(
                "Error in run_chatopenai_async (attempt %d/%d): [%s] %s | url=%s model=%s key=...%s",
                attempt + 1, max_retries, exc_type, err_str, api_url, model_name, api_key[-6:],
            )
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                await asyncio.sleep(wait)
                continue
            return ""
        finally:
            sem.release()
    else:
        LOGGER.error(
            "run_chatopenai_async failed after %d attempts: url=%s model=%s",
            max_retries, api_url, model_name,
        )
        return ""

    content = resp.choices[0].message.content
    content = content if content is not None else ""
    if debug_response:
        _debug_print_openai_response(content)
    return content


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)


def run_azure_openai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    **chat_kwargs,
) -> str:
    """
    Run Azure OpenAI model with the given prompts.

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt
        user_prompt: User prompt
        endpoint: Azure OpenAI endpoint URL
        deployment: Azure OpenAI deployment name (defaults to model_name if not specified)
        api_version: Azure OpenAI API version
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """
    # Get Azure credentials from environment
    subscription_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not subscription_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

    # Use provided endpoint or get from environment
    if endpoint is None:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided or set in AZURE_OPENAI_ENDPOINT environment variable"
            )

    # Use provided deployment or default to model_name
    if deployment is None:
        deployment = model_name

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_completion_tokens"] = chat_kwargs.get("max_completion_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)

    # Create Azure OpenAI client
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # Prepare messages
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    response = client.chat.completions.create(
        messages=msgs,
        model=deployment,
        **chat_kwargs,
    )

    return response.choices[0].message.content


def run_litellm(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    **chat_kwargs,
) -> str:
    """
    Run litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

    # Backward-compatible wrapper: keep function name, delegate to OpenAI SDK implementation.
    return run_chatopenai(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **chat_kwargs,
    )


async def run_litellm_async(
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **chat_kwargs,
) -> str:
    """
    Async version of run_litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

    # Backward-compatible wrapper: keep function name, delegate to AsyncOpenAI implementation.
    # If caller provides `messages`, flatten them into a user prompt.
    if messages is not None:
        merged = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            merged.append(f"[{role}] {content}")
        system_prompt_local = None
        user_prompt_local = "\n".join(merged)
    else:
        system_prompt_local = system_prompt
        user_prompt_local = user_prompt if user_prompt is not None else ""

    return await run_chatopenai_async(
        model_name=model_name,
        system_prompt=system_prompt_local,
        user_prompt=user_prompt_local,
        **chat_kwargs,
    )


# ---------------------------------------------------------------------------
# Shared judge agent factory (rubric + citation) backed by config.yaml
# ---------------------------------------------------------------------------

_judge_agents: Dict[str, Any] = {}
_judge_agent_lock = threading.Lock()


def get_judge_agent(llm_backend: str) -> Any:
    """
    Lazy-initialize and return an Agent for the given llm_backend key.
    Priority:
    1. config.yaml (open_instruct/common/config.yaml) — multi-key load balancing
    2. Env-var fallback — OPENAI_API_KEY_POOL / RUBRIC_JUDGE_* / CITATION_JUDGE_* / OPENAI_*
       (mirrors what run_chatopenai_async did before; works on training servers)
    """
    if llm_backend in _judge_agents:
        return _judge_agents[llm_backend]
    with _judge_agent_lock:
        if llm_backend in _judge_agents:
            return _judge_agents[llm_backend]

        agent = None
        # --- attempt 1: config.yaml ---
        try:
            from pathlib import Path
            import open_instruct
            from open_instruct.common.config import load_config
            from open_instruct.common.agent import create_agent_from_config
            config_path = Path(open_instruct.__file__).parent / "common" / "config.yaml"
            config = load_config(str(config_path))
            agent = create_agent_from_config(config, role=llm_backend, llm_backend=llm_backend)
            LOGGER.info("[judge_agent] initialized from config.yaml: backend=%s", llm_backend)
        except Exception as e:
            LOGGER.warning("[judge_agent] config.yaml load failed (%s), falling back to env vars", e)

        # --- attempt 2: env-var fallback (OPENAI_API_KEY_POOL) ---
        if agent is None:
            try:
                from open_instruct.common.agent import Agent
                _is_citation = "citation" in llm_backend
                _model    = (os.environ.get("CITATION_JUDGE_MODEL") if _is_citation
                             else os.environ.get("RUBRIC_JUDGE_MODEL")
                             ) or os.environ.get("OPENAI_MODEL", "deepseek-v3.2-huawei")
                _base_url = (os.environ.get("CITATION_JUDGE_API_BASE") if _is_citation
                             else os.environ.get("RUBRIC_JUDGE_API_BASE")
                             ) or os.environ.get("OPENAI_API_BASE", "")
                # For citation judge, prefer CITATION_JUDGE_API_KEY_LIST (bypass keys) over the
                # general pool so that content-safety errors (450) are avoided from the start.
                if _is_citation:
                    _pool_str = (os.environ.get("CITATION_JUDGE_API_KEY_LIST", "")
                                 or os.environ.get("OPENAI_API_KEY_BYPASS_POOL", "")
                                 or os.environ.get("OPENAI_API_KEY_POOL", "")
                                 or os.environ.get("OPENAI_API_KEY", ""))
                else:
                    _pool_str = os.environ.get("OPENAI_API_KEY_POOL", "") or os.environ.get("OPENAI_API_KEY", "")
                _timeout  = int(os.environ.get("RUBRIC_JUDGE_TIMEOUT", "300"))
                agent_cfg: Dict[str, Any] = {
                    "model":     _model,
                    "base_url":  _base_url,
                    "api_key":   _pool_str.split(",")[0].strip() if _pool_str else "",
                    "timeout_seconds": _timeout,
                    "max_retries": int(os.environ.get("OPENAI_MAX_RETRIES", "10")),
                    # Pass full pool via api_key_list so LLMClient can build a load balancer
                    "api_key_list": [
                        {"base_url": _base_url, "api_key": k.strip()}
                        for k in _pool_str.split(",") if k.strip()
                    ] if _pool_str else [],
                }
                agent = Agent(config=agent_cfg, role_name=llm_backend)
                LOGGER.info("[judge_agent] initialized from env vars: backend=%s model=%s keys=%d",
                            llm_backend, _model, len(agent_cfg["api_key_list"]))
            except Exception as e2:
                LOGGER.error("[judge_agent] env-var fallback also failed: %s", e2)

        _judge_agents[llm_backend] = agent
        return agent


if __name__ == "__main__":
    # Simple test case for run_chatopenai function
    def test_run_chatopenai():
        """Test the run_chatopenai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_chatopenai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require a valid model name and API credentials
            # Uncomment the line below to actually test (requires OPENAI_API_KEY)
            response = run_chatopenai("gpt-3.5-turbo", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_azure():
        """Test the run_azure_openai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_azure_openai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require valid Azure OpenAI credentials
            # Use the correct deployment name that matches test_azure_api.py
            response = run_azure_openai(
                "gpt-4.5-preview", system_prompt, user_prompt, deployment="gpt-4.5-preview-standard"
            )
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_litellm():
        """Test the run_litellm function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = run_litellm("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    async def test_run_litellm_async():
        """Test the run_litellm_async function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm_async function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = await run_litellm_async("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    # test_run_chatopenai()
    # test_run_azure()
    # test_run_litellm()
    
    # Run async test
    import asyncio
    asyncio.run(test_run_litellm_async())
