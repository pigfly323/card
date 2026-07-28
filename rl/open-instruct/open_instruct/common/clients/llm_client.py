"""
LLM Client Module

Provides a unified interface for calling LLM APIs (DeepSeek/OpenAI compatible).
Supports both text completion and JSON-mode responses.
"""

import json
import time
import random
import threading
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests
import urllib3

# Disable SSL warnings for internal APIs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMClientError(Exception):
    """Exception raised for LLM client errors."""
    pass


class ContentPolicyError(LLMClientError):
    """
    Exception raised when the LLM API rejects a request due to content policy
    (HTTP 450 / '违规信息').  Retrying with the same content will never succeed,
    so callers should catch this and skip / penalise the sample rather than
    crashing training.
    """
    pass


class JSONParseError(Exception):
    """Exception raised for JSON parsing errors."""
    pass


@dataclass
class JSONParseFailureRecord:
    """Record for JSON parsing failures."""
    prompt: str
    raw_response: str
    error_message: str
    attempt: int
    timestamp: str


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str
    base_url: str
    api_key: str
    max_context_tokens: int = 32000
    max_response_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60
    max_retries: int = 10
    retry_delay: float = 10.0
    json_parse_retries: int = 3  # JSON parse retry config: 0=no retry, -1=infinite, >0=retry count
    enable_thinking: bool = False


class APILoadBalancer:
    """
    Load balancer for distributing API requests across multiple API configs using Round-Robin with RPM-aware allocation.

    Features:
    - Round-Robin queue-based allocation for even distribution
    - RPM-proportional queue sizing (higher RPM = more slots)
    - Real-time load tracking per API key
    - Thread-safe operations
    """

    def __init__(self, api_configs: List[Dict[str, Any]]):
        """
        Initialize the load balancer.

        Args:
            api_configs: List of API configuration dicts, each with model, base_url, api_key, rpm
        """
        self.api_configs = api_configs
        # Calculate base weights based on RPM (default to 50 if not specified)
        self.base_weights = [config.get("rpm", 50) for config in api_configs]
        self.total_rpm = sum(self.base_weights)

        # Thread-safe tracking of current load per API key
        self._lock = threading.Lock()
        self._request_timestamps = [[] for _ in api_configs]  # Track request timestamps per key
        self._current_load = [0] * len(api_configs)  # Current request count per key
        self._total_requests = 0
        
        # Round-Robin queue: build a queue with RPM-proportional distribution
        # E.g., if RPMs are [50, 100, 50], queue will be [0, 1, 1, 2] (repeating based on ratio)
        self._allocation_queue = self._build_allocation_queue()
        self._queue_index = 0

    def _build_allocation_queue(self) -> List[int]:
        """
        Build a Round-Robin allocation queue where each API appears proportionally to its RPM.
        
        Strategy:
        - Find GCD of all RPM values to determine minimal queue size
        - Each API index appears (rpm / gcd) times in the queue
        - This ensures RPM-proportional distribution while maintaining Round-Robin fairness
        
        Returns:
            List of API indices in Round-Robin order
        """
        if not self.base_weights:
            return []
        
        # Import gcd for calculating greatest common divisor
        from math import gcd
        from functools import reduce
        
        # Find GCD of all RPM values
        rpms = [int(w) for w in self.base_weights if w > 0]
        if not rpms:
            return list(range(len(self.api_configs)))
        
        common_divisor = reduce(gcd, rpms)
        
        # Build queue: each API appears (rpm / gcd) times
        queue = []
        for idx, rpm in enumerate(self.base_weights):
            count = int(rpm / common_divisor) if common_divisor > 0 else 1
            queue.extend([idx] * count)
        
        return queue

    def get_available_api(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """
        Get an API config using Round-Robin allocation from the pre-built queue.

        Round-Robin ensures even distribution while respecting RPM proportions.
        Each API is selected in turn from the queue, cycling back to the start.

        Returns:
            Tuple of (index, api_config) or None if no configs available
        """
        if not self.api_configs or not self._allocation_queue:
            return None

        with self._lock:
            # Get next API from Round-Robin queue
            chosen_idx = self._allocation_queue[self._queue_index]
            
            # Advance queue index (wrap around at end)
            self._queue_index = (self._queue_index + 1) % len(self._allocation_queue)

            # Record this request
            self._current_load[chosen_idx] += 1
            self._total_requests += 1
            self._request_timestamps[chosen_idx].append(time.time())

        return (chosen_idx, self.api_configs[chosen_idx])

    def release_api_key(self, api_index: int) -> None:
        """
        Release an API key after request completion.

        Args:
            api_index: Index of the API config
        """
        with self._lock:
            if 0 <= api_index < len(self._current_load):
                self._current_load[api_index] = max(0, self._current_load[api_index] - 1)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current load statistics.

        Returns:
            Dictionary with load information
        """
        with self._lock:
            stats = {
                "total_requests": self._total_requests,
                "total_rpm_capacity": self.total_rpm,
                "num_keys": len(self.api_configs),
                "current_load": self._current_load.copy(),
                "load_ratios": [
                    self._current_load[i] / self.base_weights[i]
                    if self.base_weights[i] > 0 else 0
                    for i in range(len(self.api_configs))
                ],
            }
        return stats


class LLMClient:
    """
    LLM API client with OpenAI-compatible interface.

    Supports:
    - Text completion
    - JSON-mode completion
    - Retry with exponential backoff
    - Multiple API configs with RPM-based load balancing
    """

    def __init__(self, config: Union[LLMConfig, Dict[str, Any]]):
        """
        Initialize the LLM client.

        Args:
            config: LLMConfig object or dict with configuration
                If dict has 'api_config' key, will use load balancing across multiple APIs
        """
        self._original_config = config

        # Resolve api_configs:
        # 1. pre-processed api_config list (set by Config.get_llm_config)
        # 2. raw api_key_list in the same dict (from config.yaml merged directly)
        # 3. OPENAI_API_KEY_POOL env var (training fallback)
        import os as _os
        _resolved_api_configs: Optional[List[Dict[str, Any]]] = None
        if isinstance(config, dict):
            if config.get("_has_api_config") and config.get("api_config"):
                _resolved_api_configs = config["api_config"]
            elif config.get("api_key_list"):
                _base_url = config.get("base_url", "")
                _model    = config.get("model", "")
                _rpm      = config.get("rpm", 50)
                _resolved_api_configs = [
                    {"model": _model, "base_url": k["base_url"], "api_key": k["api_key"], "rpm": _rpm}
                    for k in config["api_key_list"]
                    if not _base_url or k.get("base_url") == _base_url
                ]
            if not _resolved_api_configs:
                # Fallback: read OPENAI_API_KEY_POOL env var (used in RL training scripts)
                _pool_str = _os.environ.get("OPENAI_API_KEY_POOL", "")
                _base_url = config.get("base_url") or _os.environ.get("OPENAI_API_BASE", "")
                _model    = config.get("model", "")
                _rpm      = config.get("rpm", 50)
                if _pool_str and _base_url:
                    _resolved_api_configs = [
                        {"model": _model, "base_url": _base_url, "api_key": k.strip(), "rpm": _rpm}
                        for k in _pool_str.split(",") if k.strip()
                    ]

        # Check if config has api_configs for load balancing
        if _resolved_api_configs:
            # Use load balancing mode
            self.api_configs = _resolved_api_configs
            self.load_balancer = APILoadBalancer(self.api_configs)
            self.use_load_balancing = True

            base_config = self.api_configs[0] if self.api_configs else {}
            self.config = LLMConfig(
                model=base_config.get("model", config.get("model", "deepseek-chat")),
                base_url=base_config.get("base_url", config.get("base_url", "https://api.deepseek.com/v1")),
                api_key=base_config.get("api_key", config.get("api_key", "")),
                max_context_tokens=config.get("max_context_tokens", 32000),
                max_response_tokens=config.get("max_response_tokens", 4096),
                max_retries=config.get("max_retries", 10),
                retry_delay=config.get("retry_delay", 10.0),
                temperature=config.get("temperature", 0.7),
                timeout_seconds=config.get("timeout_seconds", 60),
                json_parse_retries=config.get("json_parse_retries", 3),
                enable_thinking=config.get('enable_thinking', False)
            )
        else:
            # Single API mode (legacy)
            self.use_load_balancing = False
            self.load_balancer = None
            print(f"Single API key mode (legacy)")

            if isinstance(config, dict):
                self.config = LLMConfig(
                    model=config.get("model", "deepseek-chat"),
                    base_url=config.get("base_url", "https://api.deepseek.com/v1"),
                    api_key=config.get("api_key", ""),
                    max_context_tokens=config.get("max_context_tokens", 32000),
                    max_response_tokens=config.get("max_response_tokens", 4096),
                    max_retries=config.get("max_retries", 10),
                    retry_delay=config.get("retry_delay", 10.0),
                    temperature=config.get("temperature", 0.7),
                    timeout_seconds=config.get("timeout_seconds", 60),
                    json_parse_retries=config.get("json_parse_retries", 3),
                )
            else:
                self.config = config

        # Validate configuration
        if not self.config.api_key:
            raise LLMClientError("API key is required")
        if not self.config.base_url:
            raise LLMClientError("Base URL is required")

        # ── Bypass key pool ──────────────────────────────────────────────────
        # Keys that skip the internal content-safety filter; used as a single
        # retry attempt when the regular pool returns HTTP 450.
        # Priority: bypass_api_config from Config > OPENAI_API_KEY_BYPASS_POOL env.
        import itertools as _itertools
        self._bypass_configs: List[Dict[str, Any]] = []

        if isinstance(config, dict) and config.get("bypass_api_config"):
            self._bypass_configs = list(config["bypass_api_config"])
        else:
            # Env-var fallback: OPENAI_API_KEY_BYPASS_POOL (comma-separated keys)
            _bypass_pool_str = _os.environ.get("OPENAI_API_KEY_BYPASS_POOL", "")
            if _bypass_pool_str:
                _burl = (
                    (config.get("base_url") if isinstance(config, dict) else None)
                    or self.config.base_url
                )
                _bmodel = self.config.model
                self._bypass_configs = [
                    {"model": _bmodel, "base_url": _burl, "api_key": k.strip()}
                    for k in _bypass_pool_str.split(",")
                    if k.strip()
                ]

        if self._bypass_configs:
            self._bypass_cycle = _itertools.cycle(self._bypass_configs)
            self._bypass_lock = threading.Lock()
            print(f"Bypass key pool initialized: {len(self._bypass_configs)} key(s) available for 450 fallback")
        else:
            self._bypass_cycle = None
            self._bypass_lock = threading.Lock()
            print("No bypass key pool configured (450 errors will fail fast)")

    def _get_next_bypass_config(self) -> Optional[Dict[str, Any]]:
        """
        Return the next bypass API config (round-robin, thread-safe).
        Returns None if no bypass pool is configured.
        """
        with self._bypass_lock:
            if self._bypass_cycle is None:
                return None
            return next(self._bypass_cycle)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a text completion.

        Args:
            prompt: User message/prompt
            system_prompt: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: Optional stop sequences

        Returns:
            Generated text response

        Raises:
            LLMClientError: If API call fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self._call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a chat completion from a list of messages.

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: Optional stop sequences

        Returns:
            Generated text response

        Raises:
            LLMClientError: If API call fails
        """
        return self._call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a JSON-mode completion with retry logic for JSON parsing failures.

        Args:
            prompt: User message/prompt (should request JSON output)
            system_prompt: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Parsed JSON response as dict

        Raises:
            JSONParseError: If JSON parsing fails after all retries
            LLMClientError: If API call fails
        """
        messages = []

        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\nYou must respond with valid JSON only. No markdown, no explanation, just pure JSON."
        messages.append({"role": "system", "content": json_system.strip()})
        messages.append({"role": "user", "content": prompt})

        # Determine retry count based on config
        json_parse_retries = self.config.json_parse_retries

        # 0 = no retry, -1 = infinite retry, >0 = specific retry count
        if json_parse_retries == 0:
            # No retry, just try once
            max_attempts = 1
        elif json_parse_retries == -1:
            # Infinite retry (use a large number as practical limit)
            max_attempts = 1000
        else:
            # Specific retry count
            max_attempts = json_parse_retries

        last_error = None
        last_response = None

        for attempt in range(max_attempts):
            try:
                response = self._call_api(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )

                last_response = response

                # Parse JSON response
                try:
                    # Clean up response if it contains markdown code blocks
                    cleaned = response.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()

                    return json.loads(cleaned)
                except json.JSONDecodeError as e:
                    last_error = e
                    error_msg = f"Failed to parse JSON response: {e}"

                    # Save failure record
                    self._save_json_parse_failure(
                        prompt=prompt,
                        raw_response=response,
                        error_message=error_msg,
                        attempt=attempt + 1,
                    )

                    # If no retry or last attempt, raise error
                    if json_parse_retries == 0 or attempt == max_attempts - 1:
                        raise JSONParseError(
                            f"Failed to parse JSON response after {attempt + 1} attempt(s): {error_msg}\n"
                            f"Response: {response[:500]}"
                        )

                    # Wait before retry
                    wait_time = min(self.config.retry_delay * (2 ** attempt), 20.0)
                    time.sleep(wait_time)

            except LLMClientError as e:
                # API call failed, re-raise immediately
                raise

        # Should not reach here, but just in case
        raise JSONParseError(
            f"Failed to parse JSON response after {max_attempts} attempt(s): {last_error}\n"
            f"Response: {last_response[:500] if last_response else 'No response'}"
        )

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Make an API call with retry logic and load-aware API key selection.

        Features:
        - Adaptive API key selection based on current load
        - Thread-safe concurrent request handling
        - Automatic retry with exponential backoff
        - Real-time load tracking

        Args:
            messages: List of message dicts
            temperature: Override default temperature
            max_retries: Maximum number of retries (uses config default if None)
            max_tokens: Max tokens to generate
            stop: Stop sequences
            response_format: Response format specification

        Returns:
            Generated text response

        Raises:
            LLMClientError: If all retries fail
        """
        retries = max_retries if max_retries is not None else self.config.max_retries
        last_error = None
        api_index = None

        for attempt in range(retries):
            try:
                # Get API config with load-aware selection
                if self.use_load_balancing and self.load_balancer:
                    api_result = self.load_balancer.get_available_api()
                    if api_result is None:
                        raise LLMClientError("No available API configs")
                    api_index, api_config = api_result
                    base_url = api_config["base_url"]
                    api_key = api_config["api_key"]
                    model = api_config["model"]
                else:
                    api_index = None
                    base_url = self.config.base_url
                    api_key = self.config.api_key
                    model = self.config.model

                url = f"{base_url.rstrip('/')}/chat/completions"

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "max_tokens": max_tokens if max_tokens is not None else self.config.max_response_tokens,
                }

                # Enable Thinking Mode
                if self.config.enable_thinking:
                    payload["extra_body"] = {"thinking": {"type": "enabled"}}

                if stop:
                    payload["stop"] = stop

                if response_format:
                    payload["response_format"] = response_format

                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout_seconds,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        # Release the API key after successful request
                        if api_index is not None and self.load_balancer:
                            self.load_balancer.release_api_key(api_index)
                        return result["choices"][0]["message"]["content"]

                    # Handle rate limiting
                    if response.status_code == 429:
                        # Release the API key and try another one
                        if api_index is not None and self.load_balancer:
                            self.load_balancer.release_api_key(api_index)
                        wait_time = min(self.config.retry_delay * (2 ** attempt), 20.0)
                        print(f"Rate limited (429), waiting {wait_time:.1f}s before retry {attempt + 1}/{retries}...")
                        time.sleep(wait_time)
                        continue

                    # Handle content policy violation — retrying with the same
                    # key/content will never succeed.  Try once with a bypass
                    # key (content-safety exempt); if that also fails, raise
                    # ContentPolicyError so the caller can skip the sample.
                    if response.status_code == 450:
                        error_detail = response.text[:500]
                        if api_index is not None and self.load_balancer:
                            self.load_balancer.release_api_key(api_index)
                        bypass_cfg = self._get_next_bypass_config()
                        if bypass_cfg:
                            bypass_url = f"{bypass_cfg['base_url'].rstrip('/')}/chat/completions"
                            bypass_payload = {**payload, "model": bypass_cfg["model"]}
                            try:
                                bypass_resp = requests.post(
                                    bypass_url,
                                    headers={
                                        "Authorization": f"Bearer {bypass_cfg['api_key']}",
                                        "Content-Type": "application/json",
                                    },
                                    json=bypass_payload,
                                    timeout=self.config.timeout_seconds,
                                )
                                if bypass_resp.status_code == 200:
                                    print(f"Content policy bypass succeeded with key ...{bypass_cfg['api_key'][-4:]}")
                                    return bypass_resp.json()["choices"][0]["message"]["content"]
                                print(
                                    f"Bypass key also failed (status {bypass_resp.status_code}): "
                                    f"{bypass_resp.text[:200]}"
                                )
                            except Exception as bypass_exc:
                                print(f"Bypass key request failed: {bypass_exc}")
                        else:
                            print(f"Content policy violation (450), no bypass key available: {error_detail}")
                        raise ContentPolicyError(
                            f"API error (status 450): {error_detail}"
                        )

                    # Handle other errors
                    error_detail = response.text[:500]
                    last_error = LLMClientError(
                        f"API error (status {response.status_code}): {error_detail}"
                    )
                    # Release the API key on error
                    if api_index is not None and self.load_balancer:
                        self.load_balancer.release_api_key(api_index)

                    if attempt < retries - 1:
                        wait_time = min(self.config.retry_delay * (2 ** attempt), 20.0)
                        print(f"API error ({response.status_code}), waiting {wait_time:.1f}s before retry {attempt + 1}/{retries}...")
                        time.sleep(wait_time)

                except requests.exceptions.Timeout:
                    last_error = LLMClientError(
                        f"Request timed out after {self.config.timeout_seconds}s"
                    )
                    # Release the API key on timeout
                    if api_index is not None and self.load_balancer:
                        self.load_balancer.release_api_key(api_index)

                    if attempt < retries - 1:
                        wait_time = min(self.config.retry_delay * (2 ** attempt), 20.0)
                        print(f"Request timeout, waiting {wait_time:.1f}s before retry {attempt + 1}/{retries}...")
                        time.sleep(wait_time)

                except requests.exceptions.RequestException as e:
                    last_error = LLMClientError(f"Request failed: {e}")
                    # Release the API key on request exception
                    if api_index is not None and self.load_balancer:
                        self.load_balancer.release_api_key(api_index)

                    if attempt < retries - 1:
                        wait_time = min(self.config.retry_delay * (2 ** attempt), 20.0)
                        time.sleep(wait_time)

            except LLMClientError as e:
                # Re-raise LLMClientError immediately
                raise

        raise last_error or LLMClientError("All retries failed")

    def _save_json_parse_failure(
        self,
        prompt: str,
        raw_response: str,
        error_message: str,
        attempt: int,
    ) -> None:
        """
        Save JSON parsing failure record to a file.

        Args:
            prompt: The original prompt
            raw_response: The raw response from LLM
            error_message: The error message from JSON parsing
            attempt: The attempt number
        """
        try:
            # Create failure record
            record = JSONParseFailureRecord(
                prompt=prompt,
                raw_response=raw_response,
                error_message=error_message,
                attempt=attempt,
                timestamp=datetime.now().isoformat(),
            )

            # Create failure log directory if it doesn't exist
            failure_log_dir = Path("json_parse_failures")
            failure_log_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            failure_file = failure_log_dir / f"failure_{timestamp_str}.json"

            # Save failure record as JSON
            failure_data = {
                "prompt": record.prompt,
                "raw_response": record.raw_response,
                "error_message": record.error_message,
                "attempt": record.attempt,
                "timestamp": record.timestamp,
            }

            with open(failure_file, "w", encoding="utf-8") as f:
                json.dump(failure_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            # Silently fail to avoid disrupting the main flow
            # Log to stderr if needed
            import sys
            print(f"Warning: Failed to save JSON parse failure record: {e}", file=sys.stderr)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Uses a simple heuristic: ~4 characters per token for English,
        ~2 characters per token for Chinese.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars

        # Estimate tokens
        chinese_tokens = chinese_chars / 1.5  # Chinese: ~1.5 chars per token
        other_tokens = other_chars / 4  # English: ~4 chars per token

        return int(chinese_tokens + other_tokens)

    def get_load_stats(self) -> Dict[str, Any]:
        """
        Get current load statistics for all API keys.

        Returns:
            Dictionary with load information including:
            - total_requests: Total requests processed
            - total_rpm_capacity: Total RPM capacity
            - num_keys: Number of API keys
            - current_load: Current load per key
            - load_ratios: Load ratio per key (0-1+)
        """
        if self.use_load_balancing and self.load_balancer:
            return self.load_balancer.get_stats()
        else:
            return {
                "total_requests": 0,
                "total_rpm_capacity": 0,
                "num_keys": 1,
                "current_load": [0],
                "load_ratios": [0.0],
                "mode": "single_key"
            }

    # ------------------------------------------------------------------
    # Async variants — use aiohttp to avoid blocking the event loop.
    # Session is cached per-event-loop so concurrent coroutines within
    # the same asyncio.run() share one TCP connection pool instead of
    # each paying a fresh TCP+TLS handshake.
    # ------------------------------------------------------------------

    def _get_async_session(self, aiohttp_mod: Any, timeout: Any) -> Any:
        """
        Return a cached aiohttp.ClientSession for the currently running event loop.
        Creates a new session (with connection pool) on first use per loop.
        Using the loop id as key ensures each asyncio.run() thread has its own
        session while concurrent coroutines within that run() share it.
        """
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        loop_id = id(loop) if loop else 0

        if not hasattr(self, "_loop_sessions"):
            self._loop_sessions: Dict[int, Any] = {}

        session = self._loop_sessions.get(loop_id)
        if session is None or session.closed:
            connector = aiohttp_mod.TCPConnector(ssl=False, limit=200, limit_per_host=50)
            session = aiohttp_mod.ClientSession(connector=connector, timeout=timeout)
            self._loop_sessions[loop_id] = session
            # Evict stale loop entries to prevent unbounded growth
            stale = [k for k in self._loop_sessions if k != loop_id and self._loop_sessions[k].closed]
            for k in stale:
                del self._loop_sessions[k]
        return session

    async def _call_api_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """Async version of _call_api using aiohttp with per-loop session reuse."""
        import aiohttp
        import asyncio as _asyncio

        retries = self.config.max_retries
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        session = self._get_async_session(aiohttp, timeout)

        last_error: Optional[Exception] = None
        api_index: Optional[int] = None

        for attempt in range(retries):
            if self.use_load_balancing and self.load_balancer:
                api_result = self.load_balancer.get_available_api()
                if api_result is None:
                    raise LLMClientError("No available API configs")
                api_index, api_config = api_result
                base_url = api_config["base_url"]
                api_key  = api_config["api_key"]
                model    = api_config["model"]
            else:
                api_index = None
                base_url = self.config.base_url
                api_key  = self.config.api_key
                model    = self.config.model

            url = f"{base_url.rstrip('/')}/chat/completions"
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.config.max_response_tokens,
            }
            if response_format:
                payload["response_format"] = response_format

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if api_index is not None and self.load_balancer:
                            self.load_balancer.release_api_key(api_index)
                        return data["choices"][0]["message"]["content"]
                    if resp.status == 429:
                        if api_index is not None and self.load_balancer:
                            self.load_balancer.release_api_key(api_index)
                        wait = min(self.config.retry_delay * (2 ** attempt), 20.0)
                        await _asyncio.sleep(wait)
                        last_error = LLMClientError(f"429 rate limited (attempt {attempt + 1})")
                        continue
                    text = await resp.text()
                    if api_index is not None and self.load_balancer:
                        self.load_balancer.release_api_key(api_index)
                    # Content policy violation — no retry with the same key.
                    # Try once with a bypass key (content-safety exempt).
                    if resp.status == 450:
                        bypass_cfg = self._get_next_bypass_config()
                        if bypass_cfg:
                            bypass_url = f"{bypass_cfg['base_url'].rstrip('/')}/chat/completions"
                            bypass_payload = {**payload, "model": bypass_cfg["model"]}
                            try:
                                async with session.post(
                                    bypass_url,
                                    headers={
                                        "Authorization": f"Bearer {bypass_cfg['api_key']}",
                                        "Content-Type": "application/json",
                                    },
                                    json=bypass_payload,
                                ) as bypass_resp:
                                    if bypass_resp.status == 200:
                                        bypass_data = await bypass_resp.json()
                                        print(
                                            f"[async] Content policy bypass succeeded with key "
                                            f"...{bypass_cfg['api_key'][-4:]}"
                                        )
                                        return bypass_data["choices"][0]["message"]["content"]
                                    bypass_text = await bypass_resp.text()
                                    print(
                                        f"[async] Bypass key also failed (status {bypass_resp.status}): "
                                        f"{bypass_text[:200]}"
                                    )
                            except Exception as bypass_exc:
                                print(f"[async] Bypass key request failed: {bypass_exc}")
                        else:
                            print(f"[async] Content policy violation (450), no bypass key available: {text[:200]}")
                        raise ContentPolicyError(f"API error (status 450): {text[:200]}")
                    last_error = LLMClientError(f"API error ({resp.status}): {text[:200]}")
                    wait = min(self.config.retry_delay * (2 ** attempt), 20.0)
                    print(f"[async] API error ({resp.status}), waiting {wait:.1f}s before retry {attempt + 1}/{retries}...")
                    await _asyncio.sleep(wait)
            except LLMClientError:
                # ContentPolicyError / LLMClientError raised deliberately — re-raise
                # immediately without converting to a generic error or retrying.
                raise
            except Exception as e:
                if api_index is not None and self.load_balancer:
                    self.load_balancer.release_api_key(api_index)
                last_error = LLMClientError(f"Async request failed: {e}")
                print(f"[async] Request failed (attempt {attempt + 1}/{retries}): {e}")

        raise last_error or LLMClientError("Async API call failed after retries")

    async def complete_chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async version of complete_chat."""
        return await self._call_api_async(messages, temperature=temperature, max_tokens=max_tokens)

    async def complete_json_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async version of complete_json using aiohttp."""
        json_system = (system_prompt or "") + "\nYou must respond with valid JSON only. No markdown, no explanation, just pure JSON."
        messages = [
            {"role": "system", "content": json_system.strip()},
            {"role": "user",   "content": prompt},
        ]
        raw = await self._call_api_async(
            messages, temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        cleaned = raw.strip()
        for prefix in ("```json", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
