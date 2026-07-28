"""
verify_answer_apis.py

Implements the verify_answer MCP tool backend.

The tool evaluates a draft answer against:
1. Rubric criteria (via LLM judge, reusing RUBRIC_JUDGE_* env vars)
2. Citation quality (via batch LLM judge, reusing CITATION_JUDGE_* env vars)

Returns a plain-text feedback string (not JSON) to keep the model response short
and avoid truncation issues that plagued the old draft_answer JSON format.

Environment variables:
  VERIFY_ANSWER_FEEDBACK_MODE - controls feedback format:

    ``judge`` (default)
        Call LLM judge(s) for every rubric criterion and return a ``Rubric Issues:``
        list of specific per-criterion complaints about the draft answer.

    ``rubrics_as_issues`` (curriculum learning — late-stage training)
        Skip all LLM judge calls.  Return the rubric criteria directly formatted
        as a ``Rubric Issues:`` list that is structurally identical to ``judge``
        mode output, so the model cannot distinguish the two modes.  Each rubric
        entry becomes one issue line: ``[Title] <description>``.  Use this when
        you want the model to internalize rubric-guided reflection without the
        latency and cost of per-call judge inference.

    ``rubrics_only`` (aliases: ``rubrics``, ``raw_rubrics``)
        Skip all LLM judge calls and return the resolved rubric criteria as
        plain human-readable text (section headings + bullet points).  Intended
        for analysis / debugging rather than as model feedback.

Environment variables (shared with RL reward system):
  RUBRIC_JUDGE_MODEL          - model for rubric scoring (default: gpt-4.1)
  RUBRIC_JUDGE_API_BASE       - base URL for rubric judge API
  RUBRIC_JUDGE_TIMEOUT        - per-attempt timeout in seconds (default: 20)
  RUBRIC_JUDGE_MAX_TOKENS     - max tokens for rubric judge (default: 128)
  RUBRIC_JUDGE_API_KEY_LIST   - comma-separated API keys for rubric judge pool
  RUBRIC_JUDGE_API_KEY        - single API key fallback
  RUBRIC_JUDGE_MAX_RETRIES    - max retry attempts on 429/timeout (default: 3)
  RUBRIC_JUDGE_RETRY_AFTER    - max wait between retries in seconds (default: 60)
  RUBRIC_JUDGE_LOG_IO         - if 1/true, log each rubric judge user_prompt + raw model response (INFO)
  RUBRIC_JUDGE_LOG_IO_MAX_CHARS - max chars per field (default: 0 = no truncation); positive = cap

  CITATION_JUDGE_MODEL        - model for citation scoring (falls back to RUBRIC_JUDGE_MODEL)
  CITATION_JUDGE_API_BASE     - base URL for citation judge (falls back to RUBRIC_JUDGE_API_BASE)
  CITATION_JUDGE_API_KEY_LIST - comma-separated API keys for citation judge pool
  CITATION_JUDGE_API_KEY      - single API key fallback (falls back to RUBRIC_JUDGE_API_KEY)
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .rubric_store import RubricStore, normalize_rubric_payload

logger = logging.getLogger(__name__)

_RUBRIC_STORE = RubricStore()

# In-memory adaptive rubric cache: question -> {"positive_rubrics": [...], "negative_rubrics": [...]}
# Updated at runtime by generate_adaptive_rubrics(); merged into rubric resolution.
# FIFO eviction: shares the same max-questions limit as the draft answer cache.
_ADAPTIVE_RUBRIC_CACHE: Dict[str, Dict] = {}
_ADAPTIVE_CACHE_LOCK = threading.Lock()

# Draft answer cache: question -> List[str]
# Accumulates draft answers across verify_answer calls for the same question,
# so rubric generation can compare multiple answers and produce discriminative rubrics.
#
# Eviction policy:
#   Per-question : FIFO, keep at most ADAPTIVE_RUBRIC_MAX_DRAFTS_PER_QUESTION (default 8)
#   Cross-question: FIFO, keep at most ADAPTIVE_RUBRIC_MAX_QUESTIONS distinct questions (default 30000)
_DRAFT_ANSWER_CACHE: Dict[str, List[str]] = {}
_DRAFT_CACHE_MAX_PER_QUESTION = int(os.environ.get("ADAPTIVE_RUBRIC_MAX_DRAFTS_PER_QUESTION", "8"))
_DRAFT_CACHE_MAX_QUESTIONS = int(os.environ.get("ADAPTIVE_RUBRIC_MAX_QUESTIONS", "30000"))
_DRAFT_CACHE_LOCK = threading.Lock()


def _add_draft_answer(question: str, answer: str) -> List[str]:
    """
    Add a draft answer to the cache for this question. Returns current list.

    Per-question FIFO: keeps at most _DRAFT_CACHE_MAX_PER_QUESTION answers.
    Cross-question FIFO: when distinct questions exceed _DRAFT_CACHE_MAX_QUESTIONS,
    evict the oldest question (Python 3.7+ dicts preserve insertion order).
    """
    with _DRAFT_CACHE_LOCK:
        if question not in _DRAFT_ANSWER_CACHE:
            if len(_DRAFT_ANSWER_CACHE) >= _DRAFT_CACHE_MAX_QUESTIONS:
                oldest = next(iter(_DRAFT_ANSWER_CACHE))
                del _DRAFT_ANSWER_CACHE[oldest]
                logger.debug(
                    "[verify_answer] draft cache evicted oldest question (cache_size=%d)",
                    len(_DRAFT_ANSWER_CACHE),
                )
            _DRAFT_ANSWER_CACHE[question] = []

        lst = _DRAFT_ANSWER_CACHE[question]
        if answer not in lst:
            lst.append(answer)
            if len(lst) > _DRAFT_CACHE_MAX_PER_QUESTION:
                _DRAFT_ANSWER_CACHE[question] = lst[-_DRAFT_CACHE_MAX_PER_QUESTION:]
        return list(_DRAFT_ANSWER_CACHE[question])

# General rubric used when no instance-wise rubrics are available (mirrors use_general_rubric=True in RL)
_GENERAL_RUBRIC = {
    "title": "General Quality",
    "description": (
        "(1) Overall Comprehensiveness: The report should cover content as comprehensively as possible\n"
        "(2) Thoroughness of Discussion: Each section should be discussed thoroughly, not just superficially\n"
        "(3) Factuality: There should be minimal factual errors\n"
        "(4) Coherence: The discussion should stay focused and relevant to the topic"
    ),
}

# ---------------------------------------------------------------------------
# Rubric judge system prompt
# ---------------------------------------------------------------------------

# Import rubric judge prompt and normalization helpers from rubric_utils
# so verify_answer scoring is always aligned with the r1 rubric reward pipeline.
try:
    from open_instruct.search_rewards.utils.rubric_utils import (
        _default_feedback_judge_system_prompt as _rubric_judge_system_prompt_fn,
        _default_feedback_judge_user_prompt as _rubric_judge_user_prompt_fn,
        _normalize_feedback_obj,
    )
    _RUBRIC_JUDGE_SYSTEM_PROMPT = _rubric_judge_system_prompt_fn()
except ImportError:
    def _rubric_judge_user_prompt_fn(question: str, response: str, prop: str) -> str:
        return (
            f"<question>{question}</question>\n"
            f"<response>{response}</response>\n"
            f"<criterion>{prop}</criterion>"
        )

    def _normalize_feedback_obj(obj, score_scale: float = 2.0):
        if not isinstance(obj, dict):
            return {"score": 0.0, "reasoning": "", "evidence": ""}
        try:
            raw_score = float(obj.get("score", 0.0))
        except Exception:
            raw_score = 0.0
        return {
            "score": max(0.0, min(1.0, raw_score / score_scale)),
            "reasoning": str(obj.get("reasoning", "") or "").strip(),
            "evidence":  str(obj.get("evidence",  "") or "").strip(),
        }

    _RUBRIC_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. You will be given a user's question (in <question></question> tags), the assistant's response (in <response></response> tags), and a specific criterion to evaluate (in <criterion></criterion> tags).

Evaluate the response based ONLY on the provided criterion and output a JSON object with three exact keys:
1. "score": An integer on a scale of 0 to 2. 0 = poor/fails criterion, 1 = partial/needs improvement, 2 = perfect/fully satisfies criterion.
2. "reasoning": A concise explanation of why the given score was assigned. Describe which parts of the response satisfy or fail the criterion. Be specific and grounded in the response content.
3. "evidence": A short verbatim excerpt (one or two sentences) from the response that most strongly supports the assigned score. Do NOT include code blocks or multi-line formatted text. If score=0 and there is nothing relevant, write an empty string "".

Output valid JSON only in the following format:
{
  "score": x,
  "reasoning": "...",
  "evidence": "..."
}"""

# ---------------------------------------------------------------------------
# Citation batch judge system prompt
# ---------------------------------------------------------------------------

# Import batch citation prompt templates directly from citation_utils
# so the judge prompt is always in sync with the r1 citation reward.
try:
    from open_instruct.search_rewards.utils.citation_utils import (
        _BATCH_CITATION_SYSTEM_PROMPT,
        _BATCH_CITATION_USER_TEMPLATE,
    )
except ImportError:
    # Fallback when running outside the RL training Python path
    _BATCH_CITATION_SYSTEM_PROMPT = (
        "You are an expert citation evaluator. You will receive a question and a list of "
        "(claim, snippet) pairs extracted from an AI assistant's response. For each pair evaluate:\n"
        "1. **Recall**: Does the snippet support the claim?\n"
        '   - "fully", "partially", or "none"\n'
        "2. **Precision**: Is the snippet relevant to the claim?\n"
        '   - "relevant" or "irrelevant"\n\n'
        "Return ONLY a JSON array: "
        '[{"claim_idx": 0, "recall": "fully", "precision": "relevant"}, ...]'
    )
    _BATCH_CITATION_USER_TEMPLATE = (
        "Question: {question}\n\nClaims and their cited snippets:\n{claims_block}\n\nReturn the JSON array now."
    )


# ---------------------------------------------------------------------------
# Lazy-initialized API key pools (one for rubric judge, one for citation judge)
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_rubric_judge_pool = None
_citation_judge_pool = None


def _load_key_pool(list_env: str, fallback_envs: List[str]) -> "ApiKeyPool":
    """
    Build an ApiKeyPool with the following priority:
    1. list_env (comma-separated list, e.g. RUBRIC_JUDGE_API_KEY_LIST)
    2. OPENAI_API_KEY_POOL (shared pool used by RL reward system)
    3. First non-empty value from fallback_envs (single-key fallback)

    Using OPENAI_API_KEY_POOL as a shared fallback ensures that verify_answer
    benefits from the same key rotation as the RL rubric/citation judges,
    reducing the chance of hitting rate limits on a single key.
    """
    from .api_key_pool import ApiKeyPool

    # Priority 1: dedicated list env
    raw = os.environ.get(list_env, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]

    # Priority 2: shared OPENAI_API_KEY_POOL
    if not keys:
        raw_pool = os.environ.get("OPENAI_API_KEY_POOL", "")
        keys = [k.strip() for k in raw_pool.split(",") if k.strip()]

    # Priority 3: single-key fallbacks
    if not keys:
        for env in fallback_envs:
            val = os.environ.get(env, "").strip()
            if val:
                keys = [val]
                break

    if not keys:
        keys = ["EMPTY"]

    return ApiKeyPool(keys)


def _get_rubric_judge_pool():
    global _rubric_judge_pool
    if _rubric_judge_pool is None:
        with _pool_lock:
            if _rubric_judge_pool is None:
                _rubric_judge_pool = _load_key_pool(
                    list_env="RUBRIC_JUDGE_API_KEY_LIST",
                    fallback_envs=["RUBRIC_JUDGE_API_KEY", "DRAFT_ANSWER_API_KEY", "OPENAI_API_KEY"],
                )
                logger.info("[verify_answer] Rubric judge pool: %d key(s)", _rubric_judge_pool.size())
    return _rubric_judge_pool


def _get_citation_judge_pool():
    global _citation_judge_pool
    if _citation_judge_pool is None:
        with _pool_lock:
            if _citation_judge_pool is None:
                _citation_judge_pool = _load_key_pool(
                    list_env="CITATION_JUDGE_API_KEY_LIST",
                    fallback_envs=["CITATION_JUDGE_API_KEY", "RUBRIC_JUDGE_API_KEY", "DRAFT_ANSWER_API_KEY", "OPENAI_API_KEY"],
                )
                logger.info("[verify_answer] Citation judge pool: %d key(s)", _citation_judge_pool.size())
    return _citation_judge_pool


# ---------------------------------------------------------------------------
# Adaptive rubric generation (mirrors RL's _generate_instance_wise_adaptive_rubrics)
# ---------------------------------------------------------------------------

_ADAPTIVE_RUBRIC_GENERATION_PROMPT = """You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a response excellent/problematic
- **Title**: Concise abstract label (general, not question-specific)

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior responses
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines
- Focus ONLY on criteria meaningfully separating quality levels
- Never duplicate overlapping rubrics in meaning/scope
- Never create positive/negative versions of same criterion
- Conservative negative rubrics: identify clear failure modes, not absence of excellence
- Quantity: 1-5 total rubrics (fewer high-quality > many generic)
- Return empty lists if existing rubrics are already comprehensive

## Output Format
```json
{
  "question": "<original question verbatim>",
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""


async def generate_adaptive_rubrics(
    question: str,
    answer_list: List[str],
    existing_rubrics: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Generate instance-wise adaptive rubrics from multiple draft answers.

    Mirrors RL's generate_instance_wise_adaptive_rubrics.
    Passes all cached draft answers for this question so the model can
    compare them and produce discriminative rubrics.
    Also passes all existing rubrics (static + previously generated adaptive)
    so the model can avoid semantic duplicates.

    Returns dict with keys: question, positive_rubrics, negative_rubrics
    or None on failure.
    """
    if not answer_list:
        return None

    model = os.environ.get("RUBRIC_GENERATION_MODEL") or os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1")
    api_base = (
        os.environ.get("RUBRIC_GENERATION_API_BASE")
        or os.environ.get("RUBRIC_JUDGE_API_BASE")
        or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    )
    timeout_s = float(os.environ.get("RUBRIC_GENERATION_TIMEOUT", "60"))
    max_tokens = int(os.environ.get("RUBRIC_GENERATION_MAX_TOKENS", "2048"))

    # Build responses block — truncate each answer to avoid overly long prompts
    max_answer_chars = int(os.environ.get("ADAPTIVE_RUBRIC_MAX_ANSWER_CHARS", "3000"))
    responses_block = ""
    for i, ans in enumerate(answer_list):
        truncated = ans[:max_answer_chars] + ("..." if len(ans) > max_answer_chars else "")
        responses_block += f"Response {i + 1}:\n{truncated}\n\n"

    # Build existing rubrics block — includes static rubrics from rubric store
    # AND previously generated adaptive rubrics from the in-memory cache.
    # This allows the generation model to avoid semantic duplicates.
    existing_parts = []
    if existing_rubrics:
        for cat in ("static_rubrics", "positive_rubrics", "negative_rubrics"):
            for r in existing_rubrics.get(cat, []) or []:
                title = r.get("title", "")
                desc = r.get("description", "")
                existing_parts.append(f"- [{title}] {desc}")
    # Also include previously cached adaptive rubrics for this question
    cached_adaptive = _get_cached_adaptive_rubrics(question)
    for cat in ("positive_rubrics", "negative_rubrics"):
        for r in cached_adaptive.get(cat, []) or []:
            title = r.get("title", "")
            desc = r.get("description", "")
            line = f"- [{title}] {desc}"
            if line not in existing_parts:
                existing_parts.append(line)

    existing_str = (
        "\n\nExisting Rubrics (do NOT generate rubrics that duplicate these):\n"
        + "\n".join(existing_parts)
        if existing_parts else ""
    )

    user_prompt = (
        _ADAPTIVE_RUBRIC_GENERATION_PROMPT
        + f"\nQuestion: {question}\n\nResponses:\n{responses_block}"
        + existing_str
    )

    pool = _get_rubric_judge_pool()
    try:
        raw = await _call_judge_api_with_retry(
            system_prompt="",
            user_prompt=user_prompt,
            model=model,
            api_base=api_base,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            pool=pool,
        )
        obj = _extract_json_from_response(raw)
        if not isinstance(obj, dict):
            logger.warning("[verify_answer] adaptive rubric generation returned non-dict: %s", type(obj))
            return None
        if "positive_rubrics" not in obj and "negative_rubrics" not in obj:
            logger.warning("[verify_answer] adaptive rubric generation missing rubric keys")
            return None
        obj.setdefault("positive_rubrics", [])
        obj.setdefault("negative_rubrics", [])
        logger.info(
            "[verify_answer] adaptive rubrics generated from %d answers: +%d pos -%d neg",
            len(answer_list),
            len(obj["positive_rubrics"]),
            len(obj["negative_rubrics"]),
        )
        return obj
    except Exception as e:
        logger.warning("[verify_answer] adaptive rubric generation failed (skipped): %s", e)
        return None


def _update_adaptive_cache(question: str, adaptive: Dict) -> None:
    """Update in-memory adaptive rubric cache for this question."""
    with _ADAPTIVE_CACHE_LOCK:
        existing = _ADAPTIVE_RUBRIC_CACHE.get(question, {})
        # Merge: append new rubrics, avoid exact-title duplicates
        def _merge_list(old: List, new: List) -> List:
            existing_titles = {r.get("title", "").lower() for r in old}
            merged = list(old)
            for r in new:
                if r.get("title", "").lower() not in existing_titles:
                    merged.append(r)
                    existing_titles.add(r.get("title", "").lower())
            return merged

        _MAX_ACTIVE_RUBRICS = int(os.environ.get("ADAPTIVE_RUBRIC_MAX_ACTIVE_PER_QUESTION", "8"))
        merged = {
            "positive_rubrics": _merge_list(
                existing.get("positive_rubrics", []),
                adaptive.get("positive_rubrics", []),
            )[-_MAX_ACTIVE_RUBRICS:],
            "negative_rubrics": _merge_list(
                existing.get("negative_rubrics", []),
                adaptive.get("negative_rubrics", []),
            )[-_MAX_ACTIVE_RUBRICS:],
        }
        # FIFO eviction when over limit
        if question not in _ADAPTIVE_RUBRIC_CACHE and len(_ADAPTIVE_RUBRIC_CACHE) >= _DRAFT_CACHE_MAX_QUESTIONS:
            oldest = next(iter(_ADAPTIVE_RUBRIC_CACHE))
            del _ADAPTIVE_RUBRIC_CACHE[oldest]
        _ADAPTIVE_RUBRIC_CACHE[question] = merged
        logger.info(
            "[verify_answer] adaptive cache updated for question (len=%d): +%d pos, +%d neg",
            len(question),
            len(merged["positive_rubrics"]),
            len(merged["negative_rubrics"]),
        )


def _get_cached_adaptive_rubrics(question: str) -> Dict:
    """Return cached adaptive rubrics for a question (empty dict if none)."""
    with _ADAPTIVE_CACHE_LOCK:
        return dict(_ADAPTIVE_RUBRIC_CACHE.get(question, {}))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_rubric_judge_model() -> str:
    return os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1")


def _get_citation_judge_model() -> str:
    return os.environ.get("CITATION_JUDGE_MODEL") or _get_rubric_judge_model()


def _get_rubric_judge_api_base() -> str:
    return (
        os.environ.get("RUBRIC_JUDGE_API_BASE")
        or os.environ.get("DRAFT_ANSWER_API_BASE")
        or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    )


def _get_citation_judge_api_base() -> str:
    return (
        os.environ.get("CITATION_JUDGE_API_BASE")
        or _get_rubric_judge_api_base()
    )


# ---------------------------------------------------------------------------
# Rubric / Citation judge Agents (lazy-initialized from config.yaml)
# ---------------------------------------------------------------------------

def _make_judge_agent(llm_backend: str, role: str):
    """Load config.yaml and create an Agent for the given llm_backend."""
    from pathlib import Path
    import open_instruct
    from open_instruct.common.config import load_config
    from open_instruct.common.agent import create_agent_from_config
    config_path = Path(open_instruct.__file__).parent / "common" / "config.yaml"
    config = load_config(str(config_path))
    return create_agent_from_config(config, role=role, llm_backend=llm_backend)


_RUBRIC_JUDGE_AGENT = None
_RUBRIC_JUDGE_AGENT_LOCK = threading.Lock()


def _get_rubric_judge_agent():
    global _RUBRIC_JUDGE_AGENT
    if _RUBRIC_JUDGE_AGENT is not None:
        return _RUBRIC_JUDGE_AGENT
    with _RUBRIC_JUDGE_AGENT_LOCK:
        if _RUBRIC_JUDGE_AGENT is not None:
            return _RUBRIC_JUDGE_AGENT
        try:
            _RUBRIC_JUDGE_AGENT = _make_judge_agent("rubric_judge", "rubric_judge")
            logger.info("[verify_answer] rubric judge agent initialized from config.yaml")
        except Exception as e:
            logger.warning("[verify_answer] failed to init rubric judge agent: %s", e)
            _RUBRIC_JUDGE_AGENT = None
    return _RUBRIC_JUDGE_AGENT


# Citation judge Agent (lazy-initialized, backed by config.yaml citation_judge backend)
_CITATION_JUDGE_AGENT = None
_CITATION_JUDGE_AGENT_LOCK = threading.Lock()


def _get_citation_judge_agent():
    global _CITATION_JUDGE_AGENT
    if _CITATION_JUDGE_AGENT is not None:
        return _CITATION_JUDGE_AGENT
    with _CITATION_JUDGE_AGENT_LOCK:
        if _CITATION_JUDGE_AGENT is not None:
            return _CITATION_JUDGE_AGENT
        try:
            _CITATION_JUDGE_AGENT = _make_judge_agent("citation_judge", "citation_judge")
            logger.info("[verify_answer] citation judge agent initialized from config.yaml")
        except Exception as e:
            logger.warning("[verify_answer][citation] failed to init citation judge agent: %s", e)
            _CITATION_JUDGE_AGENT = None
    return _CITATION_JUDGE_AGENT


# ---------------------------------------------------------------------------
# Core API call with retry + key pool
# ---------------------------------------------------------------------------

async def _call_judge_api_with_retry(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    timeout_s: float,
    pool,
) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint with:
    - Round-robin API key rotation from pool
    - 429 rate-limit handling: mark key as cooled, retry with next key
    - Timeout handling: exponential backoff retry
    - All exceptions are propagated to caller after max_retries exhausted
    """
    max_retries = int(os.environ.get("RUBRIC_JUDGE_MAX_RETRIES", "3"))
    retry_after = float(os.environ.get("RUBRIC_JUDGE_RETRY_AFTER", "60"))
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        api_key = pool.get()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout_s)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status == 429:
                        pool.mark_rate_limited(api_key)
                        last_exc = RuntimeError(f"429 rate limited (key={api_key[:8]}...)")
                        wait = min(retry_after, 5 * (2 ** attempt))
                        logger.warning(
                            "[verify_answer] 429 on attempt %d/%d, waiting %.1fs",
                            attempt + 1, max_retries, wait,
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(wait)
                        continue
                    if resp.status in (450, 403) or (resp.status >= 400 and ("sensitive information" in text.lower() or "ModelArts.81011" in text or "违规信息" in text)):
                        # Content policy violation — retry with a bypass key once.
                        bypass_keys_str = os.environ.get("OPENAI_API_KEY_BYPASS_POOL", "")
                        bypass_keys = [k.strip() for k in bypass_keys_str.split(",") if k.strip() and k.strip() != api_key]
                        if bypass_keys:
                            import random
                            bypass_key = random.choice(bypass_keys)
                            logger.warning(
                                "[verify_answer] content policy error (%d), retrying with bypass key ...%s",
                                resp.status, bypass_key[-6:],
                            )
                            try:
                                bypass_headers = {**headers, "Authorization": f"Bearer {bypass_key}"}
                                async with aiohttp.ClientSession(timeout=client_timeout) as _s:
                                    async with _s.post(url, headers=bypass_headers, json=payload) as _r:
                                        _text = await _r.text()
                                        if _r.status == 200:
                                            return json.loads(_text)["choices"][0]["message"]["content"]
                                        logger.warning("[verify_answer] bypass key also failed (%d)", _r.status)
                            except Exception as _be:
                                logger.warning("[verify_answer] bypass key exception: %s", _be)
                        else:
                            logger.warning("[verify_answer] content policy error, no bypass key configured")
                        last_exc = RuntimeError(f"content policy error {resp.status} (key={api_key[:8]}...)")
                        continue
                    if resp.status >= 400:
                        raise RuntimeError(f"Judge API error ({resp.status}): {text[:500]}")
                    obj = json.loads(text)
                    return obj["choices"][0]["message"]["content"]
        except asyncio.TimeoutError as e:
            last_exc = e
            wait = 2 * (attempt + 1)
            logger.warning(
                "[verify_answer] Timeout on attempt %d/%d, waiting %.1fs",
                attempt + 1, max_retries, wait,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
        except aiohttp.ClientError as e:
            last_exc = e
            logger.warning("[verify_answer] Client error on attempt %d/%d: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

    raise last_exc or RuntimeError("All judge API attempts failed")


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _extract_json_from_response(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from model output.

    Handles thinking-mode models (e.g. QwQ-32B) that emit a <think>...</think>
    block before the JSON output, including cases where the think block is
    truncated (no closing </think> tag due to max_tokens limit).

    Strategy: always search from the END of the text, since the JSON answer
    always comes after the think block regardless of whether it's complete.
    """
    if not text:
        return None

    # The JSON output always appears AFTER the think block.
    # Find the last '{' in the text and try to parse from there forward.
    # This is more robust than stripping think blocks (which can be truncated).
    last_brace = text.rfind("{")
    if last_brace != -1:
        candidate = text[last_brace:]
        # Strip trailing markdown fences if any
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback: strip complete <think>...</think> then try direct parse
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Last resort: find any {...} block in cleaned text
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Rubric normalization
# ---------------------------------------------------------------------------

def _normalize_rubrics_input(rubrics) -> Dict[str, List[Dict]]:
    empty = {"static_rubrics": [], "positive_rubrics": [], "negative_rubrics": []}
    if rubrics is None:
        return empty
    if isinstance(rubrics, dict):
        static = rubrics.get("static_rubrics") or rubrics.get("rubrics", [])
        return {
            "static_rubrics": static if isinstance(static, list) else [],
            "positive_rubrics": rubrics.get("positive_rubrics", []) or [],
            "negative_rubrics": rubrics.get("negative_rubrics", []) or [],
        }
    if isinstance(rubrics, list):
        return {"static_rubrics": rubrics, "positive_rubrics": [], "negative_rubrics": []}
    if not isinstance(rubrics, str) or not rubrics.strip():
        return empty
    try:
        obj = json.loads(rubrics.strip())
        if isinstance(obj, dict):
            static = obj.get("static_rubrics") or obj.get("rubrics", [])
            return {
                "static_rubrics": static if isinstance(static, list) else [],
                "positive_rubrics": obj.get("positive_rubrics", []) or [],
                "negative_rubrics": obj.get("negative_rubrics", []) or [],
            }
        if isinstance(obj, list):
            return {"static_rubrics": obj, "positive_rubrics": [], "negative_rubrics": []}
    except Exception:
        pass
    return {
        "static_rubrics": [{"title": "Custom Rubric", "description": rubrics.strip()}],
        "positive_rubrics": [],
        "negative_rubrics": [],
    }


def resolve_rubrics_for_question(question: Optional[str], rubrics=None) -> Dict:
    """
    Priority:
    1. Explicit rubrics argument (if provided and non-empty)
    2. RubricStore lookup by question
    3. Fall back to general rubric (mirrors use_general_rubric=True in RL training)
    """
    if rubrics is not None and (not isinstance(rubrics, str) or rubrics.strip()):
        parsed = _normalize_rubrics_input(rubrics)
        parsed["source"] = "explicit_input"
        return normalize_rubric_payload(parsed)

    result = _RUBRIC_STORE.get_rubrics(question)
    if result.get("source") not in ("default_static_fallback", "default_static_without_question"):
        return normalize_rubric_payload(result)

    # No per-query rubrics found — use general rubric
    return normalize_rubric_payload(
        {
            "static_rubrics": [_GENERAL_RUBRIC],
            "positive_rubrics": [],
            "negative_rubrics": [],
            "source": "general_rubric_fallback",
        }
    )


# ---------------------------------------------------------------------------
# Citation utilities
# ---------------------------------------------------------------------------

def _extract_claims_and_citation_ids(response: str) -> Dict[str, List[str]]:
    """Extract <cite id=...>claim</cite> pairs from response text."""
    claims = {}
    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"

    cite_matches = re.findall(cite_pattern, response)
    non_cited_parts = re.split(cite_tag_pattern, response)

    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []

    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            for cid in citation_ids.split(","):
                claims[cited_text] = [cid.strip()]

    return claims


def _build_batch_citation_user_prompt(
    question: str,
    claim_pairs: List[Tuple[str, str]],  # (claim_text, snippet) — only cited claims
) -> str:
    """Build user prompt using _BATCH_CITATION_USER_TEMPLATE placeholder (same as citation_utils)."""
    lines = []
    for i, (claim_text, snippet) in enumerate(claim_pairs):
        lines.append(f"[{i}] Claim: {claim_text}")
        lines.append(f"    Snippet: {snippet[:1000]}")
        lines.append("")
    claims_block = "\n".join(lines).rstrip()
    return _BATCH_CITATION_USER_TEMPLATE.format(question=question, claims_block=claims_block)


def _parse_batch_citation_response(raw, n: int) -> List[Tuple[float, float]]:
    """
    Parse the citation judge output.
    Accepts either:
    - str  : raw text response (json_mode=False) — extract JSON array with regex
    - list : already-parsed JSON array (json_mode=True)
    Falls back to (0.0, 0.0) for missing/malformed entries.
    """
    recall_map    = {"fully": 1.0, "partially": 0.5, "none": 0.0}
    precision_map = {"relevant": 1.0, "irrelevant": 0.0}
    results: List[Tuple[float, float]] = [(0.0, 0.0)] * n
    try:
        if isinstance(raw, list):
            # json_mode=True returned an already-parsed array
            parsed = raw
        elif isinstance(raw, dict):
            # json_object mode wrapped the array: {"results": [...]} or first list value
            inner = next((v for v in raw.values() if isinstance(v, list)), None)
            if inner is None:
                return results
            parsed = inner
        else:
            # raw string — extract JSON array with regex
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                return results
            parsed = json.loads(json_match.group(0))
        for item in parsed:
            idx = int(item.get("claim_idx", -1))
            if 0 <= idx < n:
                recall    = recall_map.get(str(item.get("recall",    "")).lower(), 0.0)
                precision = precision_map.get(str(item.get("precision", "")).lower(), 0.0)
                results[idx] = (recall, precision)
    except Exception as e:
        logger.warning("[verify_answer][citation] exception parsing batch response: %s", e)
    return results


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

async def compute_rubric_issues(
    draft_answer: str,
    question: str,
    rubrics: Dict[str, List[Dict]],
) -> List[str]:
    """
    Evaluate draft_answer against each rubric criterion in parallel.
    Returns a list of issue strings for rubrics where score < 2.
    Each string is formatted as "[Title] <issue text>".

    Each rubric call has independent retry logic. A failed call returns None
    (silently skipped) — it never raises and never blocks other rubric calls.
    """
    rubrics = normalize_rubric_payload(rubrics)

    # max_tokens can be overridden per-deployment; agent timeout comes from config.yaml
    max_tokens = int(os.environ.get("RUBRIC_JUDGE_MAX_TOKENS", "2048"))
    agent = _get_rubric_judge_agent()

    all_rubrics: List[Tuple[str, Dict]] = []
    for cat in ("static_rubrics", "positive_rubrics", "negative_rubrics"):
        for r in rubrics.get(cat, []) or []:
            all_rubrics.append((cat, r))

    if not all_rubrics:
        return []

    # Limit the number of rubrics to avoid timeouts on large rubric sets.
    # Select top-k by |weight| (most important rubrics first).
    # Override via VERIFY_ANSWER_MAX_RUBRICS (default: 5).
    max_rubrics = int(os.environ.get("VERIFY_ANSWER_MAX_RUBRICS", "5"))
    if len(all_rubrics) > max_rubrics:
        # Sort by absolute weight descending; rubrics without a weight field
        # default to 1.0 (positive rubric) or -1.0 (negative rubric).
        def _rubric_weight(item: Tuple[str, Dict]) -> float:
            cat, r = item
            w = r.get("weight")
            if w is not None:
                return abs(float(w))
            # infer sign from category if weight not explicitly set
            return 1.0
        all_rubrics_sorted = sorted(all_rubrics, key=_rubric_weight, reverse=True)
        logger.info(
            "[verify_answer] selecting top %d rubrics by |weight| from %d total",
            max_rubrics, len(all_rubrics),
        )
        all_rubrics = all_rubrics_sorted[:max_rubrics]

    async def _score_one(cat: str, rubric: Dict) -> Optional[str]:
        title = str(rubric.get("title", "")).strip() or "Criterion"
        description = str(rubric.get("description", "")).strip()
        weight = float(rubric.get("weight", -1.0 if cat == "negative_rubrics" else 1.0))
        is_negative = (cat == "negative_rubrics") or (weight < 0)
        if not description:
            return None
        # Use same user prompt format as rubric_utils._default_feedback_judge_user_prompt
        user_prompt = _rubric_judge_user_prompt_fn(question, draft_answer, description)
        try:
            if agent is None:
                logger.warning("[verify_answer] rubric judge agent unavailable, skipping '%s'", title)
                return None
            raw = await agent.async_single_turn(
                user_prompt,
                _RUBRIC_JUDGE_SYSTEM_PROMPT,
                None,        # temperature — use agent default (0.0)
                max_tokens,  # max_tokens
                True,        # json_mode
            )
            if os.environ.get("RUBRIC_JUDGE_LOG_IO", "").lower() in ("1", "true", "yes"):
                try:
                    cap = int(os.environ.get("RUBRIC_JUDGE_LOG_IO_MAX_CHARS", "0"))
                except ValueError:
                    cap = 0

                def _log_field(s: str) -> str:
                    if cap <= 0:
                        return s
                    return s[:cap] + ("…" if len(s) > cap else "")

                logger.info(
                    "[verify_answer][rubric_judge_io] title=%r\n--- user_prompt ---\n%s\n--- raw_response ---\n%s",
                    title,
                    _log_field(user_prompt),
                    _log_field(json.dumps(raw) if isinstance(raw, dict) else (raw or "")),
                )
            # json_mode=True → raw is already a parsed dict; otherwise parse from string
            obj = raw if isinstance(raw, dict) else _extract_json_from_response(raw)
            if not obj:
                logger.warning(
                    "[verify_answer] Rubric judge returned unparseable response for '%s'",
                    title,
                )
                return None
            # Normalize internally, then convert back to raw scale (0/1/2) to match the
            # header "score 0=not met, 1=partially met, 2=fully met" shown to the model.
            normalized = _normalize_feedback_obj(obj, score_scale=2.0)
            norm_score = normalized["score"]       # float in [0,1]
            raw_score  = round(norm_score * 2)     # 0, 1, or 2 — matches header scale
            reasoning  = normalized["reasoning"]
            evidence   = normalized["evidence"]
            logger.info("[verify_answer] Rubric '%s' (negative=%s): score=%d/2", title, is_negative, raw_score)
            return json.dumps({
                "title":     title,
                "weight":    weight,    # >0: should satisfy; <0: must avoid
                "score":     raw_score,  # raw scale 0/1/2 matching header description
                "reasoning": reasoning,
                "evidence":  evidence,
            }, ensure_ascii=False)
        except Exception as e:
            # Fully isolated: exception from one rubric never affects others or the caller
            logger.warning("[verify_answer] Rubric judge failed for '%s' (skipped): %s", title, e)
            return None

    tasks = [_score_one(cat, rubric) for cat, rubric in all_rubrics]
    # return_exceptions=True ensures one task failure never cancels siblings
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect all evaluated rubric items.
    # All rubrics (including fully met ones) are shown so the model knows which
    # criteria it satisfied — score 0=not met, 1=partially met, 2=fully met.
    scored_items: List[tuple] = []   # (norm_score, weight) for reward computation
    issues:       List[str]   = []   # feedback lines shown to the model
    failed = 0

    for r in results:
        if isinstance(r, Exception):
            logger.warning("[verify_answer] Unexpected exception in rubric task (ignored): %s", r)
            failed += 1
        elif r is None:
            failed += 1
        else:
            try:
                item = json.loads(r)
                raw_s = float(item.get("score",  0.0))  # raw 0/1/2
                w     = float(item.get("weight", 1.0))
                norm_s = max(0.0, min(1.0, raw_s / 2.0))  # normalize for reward
                scored_items.append((norm_s, w))
                issues.append(r)   # always show every rubric so model sees full picture
            except Exception as parse_err:
                logger.warning("[verify_answer] Failed to parse rubric result (skipped): %s", parse_err)
                failed += 1

    # Raise if every rubric call failed — the judge service is unreachable or broken.
    if not scored_items and all_rubrics:
        raise RuntimeError(
            f"All {len(all_rubrics)} rubric judge call(s) failed. "
            "The judge LLM service may be unavailable or rate-limited."
        )

    if failed:
        logger.warning(
            "[verify_answer] %d/%d rubric calls failed (partial result returned)",
            failed, len(all_rubrics),
        )

    # Compute weighted rubric_reward from all scored rubrics (same formula as r1).
    total_pos_weight = sum(w for _, w in scored_items if w > 0)
    weighted_sum = sum(
        s * w if w > 0 else s * w   # negative w: high score means bad behaviour present
        for s, w in scored_items
    )
    rubric_reward = weighted_sum / max(total_pos_weight, 1.0)
    rubric_reward = max(0.0, min(1.0, rubric_reward))

    # Append a machine-parseable summary line for r0 computation in grpo_lff.py
    issues.append(f"Rubric reward: {rubric_reward:.4f}")

    logger.info(
        "[verify_answer] rubric_reward=%.4f scored=%d perfect_silent=%d feedback=%d failed=%d",
        rubric_reward, len(scored_items),
        len(scored_items) - len([x for x in issues if x.startswith("{")]),
        len([x for x in issues if x.startswith("{")]),
        failed,
    )
    return issues


async def compute_citation_issues(
    draft_answer: str,
    question: str,
    citation_store: Dict[str, str],
) -> List[str]:
    """
    Evaluate citation quality for all <cite> claims in draft_answer via a single
    batch LLM call. Returns issue strings for claims that are not fully supported
    or are irrelevant.

    On any failure (timeout, 429 exhausted, parse error), returns [] — never raises.
    """
    logger.info(
        "[verify_answer][citation] citation_store_size=%d answer_chars=%d",
        len(citation_store),
        len(draft_answer),
    )
    claims = _extract_claims_and_citation_ids(draft_answer)

    # Separate cited and uncited claims
    cited_claims = [(text, ids) for text, ids in claims.items() if ids]
    uncited_claims = [(text, ids) for text, ids in claims.items() if not ids]

    logger.info(
        "[verify_answer][citation] citation_store_size=%d total_claims=%d "
        "cited_claims=%d uncited_claims=%d",
        len(citation_store),
        len(claims),
        len(cited_claims),
        len(uncited_claims),
    )

    # Only evaluate cited claims — uncited claims (no <cite> tags) are ignored entirely.
    if not cited_claims:
        logger.info("[verify_answer][citation] skipped: no cited claims found in answer")
        return []

    def _concat_snippets(ids: List[str]) -> Optional[str]:
        if not ids or not citation_store:
            return None
        parts = [citation_store[cid] for cid in ids if cid in citation_store]
        return "\n\n".join(parts) if parts else None

    # Build eval list from cited claims only.
    # Skip claims whose citation IDs are absent from the store (can't evaluate).
    all_claim_list = list(claims.items())  # preserve order for stable indexing
    eval_pairs: List[Tuple[int, str, str]] = []
    for i, (claim_text, ids) in enumerate(all_claim_list):
        if not ids:
            continue  # uncited claim → skip entirely
        snippet = _concat_snippets(ids)
        if snippet is None:
            continue  # citation ID present but not in store → skip
        eval_pairs.append((i, claim_text, snippet))

    if not eval_pairs:
        logger.info("[verify_answer][citation] skipped: no cited claims with resolvable snippets")
        return []

    # 2048 think budget + 60 tokens per claim for JSON output (matches citation_utils)
    max_tokens = 2048 + 60 * len(eval_pairs)

    pairs_for_prompt = [(ct, sn) for _, ct, sn in eval_pairs]
    user_prompt = _build_batch_citation_user_prompt(question, pairs_for_prompt)

    agent = _get_citation_judge_agent()
    if agent is None:
        logger.warning("[verify_answer] citation judge agent unavailable — skipping citation scoring")
        return []
    try:
        raw = await agent.async_single_turn(
            user_prompt,
            _BATCH_CITATION_SYSTEM_PROMPT,
            None,          # temperature — use agent default (0.0)
            max_tokens,    # max_tokens
            True,          # json_mode
        )
    except Exception as e:
        logger.warning("[verify_answer] Citation batch judge failed (skipped): %s", e)
        return []

    scores = _parse_batch_citation_response(raw, len(eval_pairs))

    # Compute F1 per claim and citation_reward — same formula as r1:
    #   citation_reward = 0.6 * avg_f1 + 0.4 * citation_format_reward
    # All eval_pairs have valid citation IDs (filtered above), so citation_format_reward = 1.0.
    recall_map    = {1.0: "Fully supported", 0.5: "Partially supported", 0.0: "No support"}
    precision_map = {1.0: "Relevant", 0.0: "Irrelevant"}
    f1_scores: List[float] = []
    issues: List[str] = []
    for batch_idx, (orig_idx, claim_text, _snippet) in enumerate(eval_pairs):
        recall, precision = scores[batch_idx]
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        f1_scores.append(f1)
        short_claim   = claim_text[:120] + "..." if len(claim_text) > 120 else claim_text
        recall_label  = recall_map.get(recall,    "No support")
        precision_label = precision_map.get(precision, "Irrelevant")
        issues.append(
            f"Claim {orig_idx + 1}: \"{short_claim}\" [{recall_label}] [{precision_label}]"
        )

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    citation_format_reward = 1.0  # all eval_pairs have valid IDs (filtered above)
    citation_reward = 0.6 * avg_f1 + 0.4 * citation_format_reward
    # Append a machine-parseable summary line for r0 computation in grpo_lff.py
    issues.append(f"Citation reward: {citation_reward:.4f}")

    logger.info(
        "[verify_answer][citation] evaluated=%d avg_f1=%.3f citation_reward=%.3f",
        len(eval_pairs), avg_f1, citation_reward,
    )
    return issues


# ---------------------------------------------------------------------------
# Context citation extraction (mirrors format_utils.extract_citations_from_context)
# ---------------------------------------------------------------------------

def _extract_citations_from_context_for_verify(context: str) -> Dict[str, str]:
    """
    Extract citation snippets from a context string containing <snippets id="...">,
    <snippet id="...">, or <webpage id="..."> tags.
    """
    citations: Dict[str, str] = {}
    pattern1 = r'<snippets?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets?>'
    pattern2 = r'<snippet?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippet?>'
    pattern3 = r'<webpage?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</webpage?>'
    for pattern in (pattern1, pattern2, pattern3):
        for _, snippet_id, content in re.findall(pattern, context, re.DOTALL):
            clean_id = snippet_id.strip()
            clean_content = content.strip()
            if clean_id and clean_content:
                citations[clean_id] = clean_content
    return citations


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

_VERIFY_ANSWER_FOOTER = (
    "The above is a draft of your answer for this round (for reflection only, not the final response). "
    "Do not repeat the `verify_answer` tool calls. "
    "You must now provide your final answer using <answer></answer> tags "
    "with complete, substantive content based on the feedback received."
)


def _format_resolved_rubrics_plain(rubrics: Dict[str, Any]) -> str:
    """Human-readable rubric criteria only (no LLM judge output)."""
    lines: List[str] = []
    sections = (
        ("static_rubrics", "Static criteria"),
        ("positive_rubrics", "Positive criteria"),
        ("negative_rubrics", "Negative criteria"),
    )
    for key, heading in sections:
        items = rubrics.get(key) or []
        if not items:
            continue
        lines.append(f"{heading}:")
        for r in items:
            if not isinstance(r, dict):
                lines.append(f"- {r}")
                continue
            title = str(r.get("title", "")).strip()
            desc = str(r.get("description", "")).strip()
            if title and desc:
                lines.append(f"- [{title}] {desc}")
            elif desc:
                lines.append(f"- {desc}")
            elif title:
                lines.append(f"- [{title}]")
        lines.append("")
    out = "\n".join(lines).strip()
    return out if out else "(No rubric criteria resolved for this question.)"


def _format_rubrics_for_feedback(rubrics: Dict[str, Any]) -> str:
    """
    Format rubrics for inline feedback (disable_rubrics_feedback=True mode).

    Renders static, positive, and negative rubrics with clear polarity labels
    so the model can distinguish excellence indicators from failure modes.
    """
    lines: List[str] = []

    def _render_item(r) -> str:
        if not isinstance(r, dict):
            return f"  - {r}"
        title = str(r.get("title", "")).strip()
        desc = str(r.get("description", "")).strip()
        if title and desc:
            return f"  - [{title}] {desc}"
        return f"  - {desc or title}"

    static_items = rubrics.get("static_rubrics") or []
    positive_items = rubrics.get("positive_rubrics") or []
    negative_items = rubrics.get("negative_rubrics") or []

    if static_items:
        lines.append("General criteria (apply to all responses):")
        lines.extend(_render_item(r) for r in static_items)
        lines.append("")

    if positive_items:
        lines.append("Positive criteria [GOOD — your answer should satisfy these]:")
        lines.extend(_render_item(r) for r in positive_items)
        lines.append("")

    if negative_items:
        lines.append("Negative criteria [BAD — your answer must avoid these]:")
        lines.extend(_render_item(r) for r in negative_items)
        lines.append("")

    out = "\n".join(lines).strip()
    return out if out else "(No rubric criteria resolved for this question.)"


def _format_rubrics_as_issues(rubrics: Dict[str, Any]) -> List[str]:
    """
    Render rubric criteria as a ``Rubric Issues:`` list identical in structure
    to the LLM-judge output, so the model treats them as concrete critique items.

    Each rubric entry becomes one issue line: ``[Title] <description>``.
    Positive rubrics are prefixed with "Should satisfy: ", negative rubrics
    with "Must avoid: ", so the model understands polarity without a section header.
    Static rubrics are emitted as-is.
    """
    issues: List[str] = []

    for r in rubrics.get("static_rubrics") or []:
        if not isinstance(r, dict):
            issues.append(str(r))
            continue
        title = str(r.get("title", "")).strip()
        desc  = str(r.get("description", "")).strip()
        if title and desc:
            issues.append(f"[{title}] {desc}")
        elif desc:
            issues.append(desc)

    for r in rubrics.get("positive_rubrics") or []:
        if not isinstance(r, dict):
            issues.append(f"Should satisfy: {r}")
            continue
        title = str(r.get("title", "")).strip()
        desc  = str(r.get("description", "")).strip()
        if title and desc:
            issues.append(f"[{title}] Should satisfy: {desc}")
        elif desc:
            issues.append(f"Should satisfy: {desc}")

    for r in rubrics.get("negative_rubrics") or []:
        if not isinstance(r, dict):
            issues.append(f"Must avoid: {r}")
            continue
        title = str(r.get("title", "")).strip()
        desc  = str(r.get("description", "")).strip()
        if title and desc:
            issues.append(f"[{title}] Must avoid: {desc}")
        elif desc:
            issues.append(f"Must avoid: {desc}")

    return issues


async def _verify_answer_rubrics_as_issues(
    answer: str,
    question: str,
    resolved: Dict[str, Any],
    rubric_source: str,
    apply_adaptive: bool,
) -> str:
    """
    Return rubric criteria formatted as a ``Rubric Issues:`` list (no LLM judge).

    Output is structurally identical to judge mode so the model cannot distinguish
    the two modes — each rubric entry becomes one ``- [Title] <desc>`` issue line.
    Used for curriculum learning in late-stage RL training to reduce judge latency.
    """
    merged = dict(resolved)
    src = rubric_source

    if apply_adaptive:
        answer_list = _add_draft_answer(question, answer)
        adaptive_result = await generate_adaptive_rubrics(
            question=question,
            answer_list=answer_list,
            existing_rubrics=merged,
        )
        if isinstance(adaptive_result, Exception):
            logger.warning("[verify_answer] rubrics_as_issues adaptive raised: %s", adaptive_result)
            adaptive_result = None
        if adaptive_result is not None:
            _update_adaptive_cache(question, adaptive_result)
            merged = {
                "static_rubrics": merged.get("static_rubrics", []) or [],
                "positive_rubrics": (
                    (merged.get("positive_rubrics", []) or [])
                    + (adaptive_result.get("positive_rubrics", []) or [])
                ),
                "negative_rubrics": (
                    (merged.get("negative_rubrics", []) or [])
                    + (adaptive_result.get("negative_rubrics", []) or [])
                ),
                "source": src + "+adaptive",
            }
            src = merged["source"]
        else:
            cached = _get_cached_adaptive_rubrics(question)
            if cached:
                merged = {
                    "static_rubrics": merged.get("static_rubrics", []) or [],
                    "positive_rubrics": (
                        (merged.get("positive_rubrics", []) or [])
                        + (cached.get("positive_rubrics", []) or [])
                    ),
                    "negative_rubrics": (
                        (merged.get("negative_rubrics", []) or [])
                        + (cached.get("negative_rubrics", []) or [])
                    ),
                    "source": src + "+cached_adaptive",
                }
                src = merged["source"]

    issues = _format_rubrics_as_issues(merged)
    n_static = len(merged.get("static_rubrics") or [])
    n_pos    = len(merged.get("positive_rubrics") or [])
    n_neg    = len(merged.get("negative_rubrics") or [])

    lines = [
        "Rubric criteria for this task:",
    ]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- (No rubric criteria resolved for this question.)")

    logger.info(
        "[verify_answer] rubrics_as_issues: source=%s issues=%d",
        src, len(issues),
    )
    return "\n".join(lines) + "\n\n" + _VERIFY_ANSWER_FOOTER


async def _verify_answer_rubrics_only_plain(
    answer: str,
    question: str,
    resolved: Dict[str, Any],
    rubric_source: str,
    apply_adaptive: bool,
) -> str:
    """
    Return rubric criteria as plain text only. Optionally merge adaptive rubrics
    without calling rubric/citation judge LLMs.
    """
    merged = dict(resolved)
    src = rubric_source

    if apply_adaptive:
        answer_list = _add_draft_answer(question, answer)
        logger.info(
            "[verify_answer] rubrics_only adaptive: cached_drafts=%d",
            len(answer_list),
        )
        adaptive_result = await generate_adaptive_rubrics(
            question=question,
            answer_list=answer_list,
            existing_rubrics=merged,
        )
        if isinstance(adaptive_result, Exception):
            logger.warning(
                "[verify_answer] rubrics_only adaptive rubric task raised: %s",
                adaptive_result,
            )
            adaptive_result = None

        if adaptive_result is not None:
            _update_adaptive_cache(question, adaptive_result)
            merged = {
                "static_rubrics": merged.get("static_rubrics", []) or [],
                "positive_rubrics": (
                    (merged.get("positive_rubrics", []) or [])
                    + (adaptive_result.get("positive_rubrics", []) or [])
                ),
                "negative_rubrics": (
                    (merged.get("negative_rubrics", []) or [])
                    + (adaptive_result.get("negative_rubrics", []) or [])
                ),
                "source": src + "+adaptive",
            }
            src = merged["source"]
        else:
            cached = _get_cached_adaptive_rubrics(question)
            if cached:
                merged = {
                    "static_rubrics": merged.get("static_rubrics", []) or [],
                    "positive_rubrics": (
                        (merged.get("positive_rubrics", []) or [])
                        + (cached.get("positive_rubrics", []) or [])
                    ),
                    "negative_rubrics": (
                        (merged.get("negative_rubrics", []) or [])
                        + (cached.get("negative_rubrics", []) or [])
                    ),
                    "source": src + "+cached_adaptive",
                }
                src = merged["source"]

    body = _format_resolved_rubrics_plain(merged)
    logger.info("[verify_answer] rubrics_only plain text chars=%d", len(body))
    return "Rubric criteria for this task:\n\n" + body + "\n\n" + _VERIFY_ANSWER_FOOTER


async def verify_answer_handler(
    answer: str,
    question: Optional[str],
    rubrics=None,
    citation_store: Optional[Dict[str, str]] = None,
    apply_adaptive_rubric_reward: bool = False,
    disable_rubrics_feedback: bool = False,
) -> str:
    """
    Evaluate a draft answer and return a plain-text feedback string.

    When APPLY_ADAPTIVE_RUBRIC_REWARD=true (mirrors RL training flag):
    1. Cache this draft answer for the question.
    2. Generate instance-wise adaptive rubrics using all cached drafts for
       this question + all existing rubrics (static + previously generated),
       so the model can produce discriminative, non-duplicate rubrics.
    3. Update the in-memory adaptive rubric cache for this question.
    4. Evaluate the answer against the full merged rubric set
       (static + adaptive) to produce comprehensive feedback.

    This function is fully exception-safe: any internal failure falls back to a
    degraded-but-valid feedback string so the MCP tool always returns something
    useful and never blocks the training thread.
    """
    # Fast-path validation (no LLM calls needed)
    if not answer or not answer.strip():
        return "Rubric Issues:\n- (No draft answer provided)\n\n" + _VERIFY_ANSWER_FOOTER

    if not question or not str(question).strip():
        return "Rubric Issues:\n- (Missing question parameter — cannot look up rubrics)\n\n" + _VERIFY_ANSWER_FOOTER

    # Require the draft answer to be wrapped in <answer></answer> tags.
    # VerifyAnswerTool.preprocess now preserves the tags (group(0)).
    # If missing, return a soft-error message so the model can correct its call —
    # raising here would cause ToolError → tenacity retries → RetryError, and the
    # model would never see the guidance message.
    if not re.search(r'<answer>.*?</answer>', answer, re.DOTALL):
        return (
            "No answer found in <answer></answer> tags. "
            "Do not repeat the `verify_answer` tool calls. "
            "You must now provide your final answer using <answer></answer> tags."
        )

    apply_adaptive = apply_adaptive_rubric_reward
    citation_issues: List[str] = []   # populated in both adaptive and non-adaptive paths

    try:
        # Step 1: resolve base rubrics from rubric store
        resolved = resolve_rubrics_for_question(question=question, rubrics=rubrics)
        rubric_source = resolved.get("source", "unknown")

        # disable_rubrics_feedback: skip judge LLMs; emit rubric criteria directly
        # under "Criterion:" with polarity labels so the model can self-assess.
        # if disable_rubrics_feedback:
        #     citation_issues = await compute_citation_issues(
        #         draft_answer=answer,
        #         question=question,
        #         citation_store=citation_store or {},
        #     )
        #     rubric_body = _format_rubrics_for_feedback(resolved)
        #     lines = [
        #         f"[rubric diagnostics] source={rubric_source} mode=disable_rubrics_feedback",
        #         "",
        #         "Criterion:",
        #         rubric_body,
        #         "",
        #         "Citation Issues:",
        #     ]
        #     if citation_issues:
        #         lines.extend(f"- {issue}" for issue in citation_issues)
        #     else:
        #         lines.append("- (No citation issues found)")
        #     lines.append("")
        #     lines.append(_VERIFY_ANSWER_FOOTER)
        #     logger.info(
        #         "[verify_answer] disable_rubrics_feedback: rubric_source=%s citation_issues=%d",
        #         rubric_source,
        #         len(citation_issues),
        #     )
        #     return "\n".join(lines)

        feedback_mode = os.environ.get("VERIFY_ANSWER_FEEDBACK_MODE", "judge").lower().strip()
        if feedback_mode == "rubrics_as_issues":
            return await _verify_answer_rubrics_as_issues(
                answer=answer,
                question=question,
                resolved=resolved,
                rubric_source=rubric_source,
                apply_adaptive=apply_adaptive,
            )
        if feedback_mode in ("rubrics_only", "rubrics", "raw_rubrics"):
            return await _verify_answer_rubrics_only_plain(
                answer=answer,
                question=question,
                resolved=resolved,
                rubric_source=rubric_source,
                apply_adaptive=apply_adaptive,
            )

        if apply_adaptive:
            # Step 2: cache this draft answer
            answer_list = _add_draft_answer(question, answer)
            logger.info(
                "[verify_answer] adaptive mode: question cached_drafts=%d",
                len(answer_list),
            )

            # Step 3: generate adaptive rubrics from all cached drafts + existing rubrics
            # Run concurrently with citation evaluation to save time
            adaptive_task = generate_adaptive_rubrics(
                question=question,
                answer_list=answer_list,
                existing_rubrics=resolved,
            )
            citation_task = compute_citation_issues(
                draft_answer=answer,
                question=question,
                citation_store=citation_store or {},
            )
            gather_result = await asyncio.gather(
                adaptive_task, citation_task, return_exceptions=True
            )
            adaptive_result  = gather_result[0]
            citation_issues  = gather_result[1] if not isinstance(gather_result[1], Exception) else []

            # Step 4: update adaptive cache and merge into resolved rubrics
            if isinstance(adaptive_result, Exception):
                logger.warning("[verify_answer] adaptive rubric task raised: %s", adaptive_result)
                adaptive_result = None

            if adaptive_result is not None:
                _update_adaptive_cache(question, adaptive_result)
                # Merge adaptive rubrics into resolved set for scoring
                resolved = {
                    "static_rubrics": resolved.get("static_rubrics", []) or [],
                    "positive_rubrics": (
                        (resolved.get("positive_rubrics", []) or [])
                        + (adaptive_result.get("positive_rubrics", []) or [])
                    ),
                    "negative_rubrics": (
                        (resolved.get("negative_rubrics", []) or [])
                        + (adaptive_result.get("negative_rubrics", []) or [])
                    ),
                    "source": rubric_source + "+adaptive",
                }
                rubric_source = resolved["source"]
                logger.info(
                    "[verify_answer] merged rubrics: static=%d pos=%d neg=%d",
                    len(resolved["static_rubrics"]),
                    len(resolved["positive_rubrics"]),
                    len(resolved["negative_rubrics"]),
                )
            else:
                # Adaptive generation failed — also include cached adaptive rubrics
                cached = _get_cached_adaptive_rubrics(question)
                if cached:
                    resolved = {
                        "static_rubrics": resolved.get("static_rubrics", []) or [],
                        "positive_rubrics": (
                            (resolved.get("positive_rubrics", []) or [])
                            + (cached.get("positive_rubrics", []) or [])
                        ),
                        "negative_rubrics": (
                            (resolved.get("negative_rubrics", []) or [])
                            + (cached.get("negative_rubrics", []) or [])
                        ),
                        "source": rubric_source + "+cached_adaptive",
                    }
                    rubric_source = resolved["source"]

            # Step 5: score against merged rubrics
            rubric_issues = await compute_rubric_issues(
                draft_answer=answer,
                question=question,
                rubrics=resolved,
            )

        else:
            # Non-adaptive mode: use static rubrics only for feedback.
            # Positive/negative (adaptive) rubrics are excluded so the judge
            # evaluates only against the stable, instance-independent criteria.
            static_only_rubrics = {
                "static_rubrics":   resolved.get("static_rubrics") or [],
                "positive_rubrics": [],
                "negative_rubrics": [],
                "source":           rubric_source + "+static_only",
            }
            rubric_source = static_only_rubrics["source"]
            logger.info(
                "[verify_answer] non-adaptive mode: using static rubrics only (n=%d)",
                len(static_only_rubrics["static_rubrics"]),
            )
            rubric_task = compute_rubric_issues(
                draft_answer=answer,
                question=question,
                rubrics=static_only_rubrics,
            )
            citation_task = compute_citation_issues(
                draft_answer=answer,
                question=question,
                citation_store=citation_store or {},
            )
            results = await asyncio.gather(rubric_task, citation_task, return_exceptions=True)

            rubric_issues   = results[0] if not isinstance(results[0], Exception) else []
            citation_issues = results[1] if not isinstance(results[1], Exception) else []

            if isinstance(results[0], Exception):
                logger.warning("[verify_answer] rubric task raised unexpectedly: %s", results[0])
            if isinstance(results[1], Exception):
                logger.warning("[verify_answer] citation task raised unexpectedly: %s", results[1])

    except Exception as e:
        logger.exception("[verify_answer] Unexpected error in verify_answer_handler: %s", e)
        return (
            "Rubric Issues:\n- (Evaluation service temporarily unavailable)\n\n"
            "Citation Issues:\n- (Evaluation service temporarily unavailable)\n\n"
            + _VERIFY_ANSWER_FOOTER
        )

    lines = []
    lines.append(
        "Rubric scores (score 0=not met, 1=partially met, 2=fully met; "
        "weight>0: criterion to satisfy, weight<0: behaviour to avoid):"
    )
    if rubric_issues:
        for issue in rubric_issues:
            lines.append(issue)
    else:
        lines.append("(No rubric scores available)")

    if citation_issues:
        lines.append("")
        lines.append("Citation scores (Fully supported=2 / Partially supported=1 / No support=0):")
        for issue in citation_issues:
            lines.append(issue)

    lines.append("")
    lines.append(_VERIFY_ANSWER_FOOTER)

    logger.info(
        "[verify_answer] Done: rubric_source=%s rubric_issues=%d adaptive=%s",
        rubric_source, len(rubric_issues), apply_adaptive,
    )
    return "\n".join(lines)
