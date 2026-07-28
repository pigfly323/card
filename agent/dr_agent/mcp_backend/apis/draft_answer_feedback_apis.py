import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from .rubric_store import RubricStore, normalize_rubric_payload

logger = logging.getLogger(__name__)


DRAFT_ANSWER_SYSTEM_PROMPT = """You are an expert evaluator providing training signals for an AI assistant. You will be given a user's question (in <question></question> tags) and the assistant's response (in <response></response> tags).

Core principles:
1) Follow rubric criteria strictly. Evaluate each criterion independently.
2) Separate positive-rubric misses from negative-rubric violations.
3) Prioritize static rubrics first, then adaptive positive/negative rubrics.
4) Ground criticism in the provided question and draft content; do not invent facts.

Output JSON only. Do not include any reasoning, explanation, or preamble before the JSON. Start your response directly with {.
"""

_RUBRIC_STORE = RubricStore()


def _normalize_rubrics_input(rubrics: Optional[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse rubrics string into {static_rubrics, positive_rubrics, negative_rubrics}.
    Accepts:
    - JSON object with keys: static_rubrics/positive_rubrics/negative_rubrics/rubrics
    - JSON array of rubrics (treated as static_rubrics)
    - Plain text (stored as one static rubric)
    """
    empty = {"static_rubrics": [], "positive_rubrics": [], "negative_rubrics": []}
    if rubrics is None:
        return empty
    if isinstance(rubrics, dict):
        obj = rubrics
        static_rubrics = obj.get("static_rubrics")
        if static_rubrics is None:
            static_rubrics = obj.get("rubrics", [])
        return {
            "static_rubrics": static_rubrics if isinstance(static_rubrics, list) else [],
            "positive_rubrics": obj.get("positive_rubrics", []) if isinstance(obj.get("positive_rubrics", []), list) else [],
            "negative_rubrics": obj.get("negative_rubrics", []) if isinstance(obj.get("negative_rubrics", []), list) else [],
        }
    if isinstance(rubrics, list):
        return {"static_rubrics": rubrics, "positive_rubrics": [], "negative_rubrics": []}
    if not isinstance(rubrics, str):
        return empty
    if not rubrics.strip():
        return empty

    text = rubrics.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            static_rubrics = obj.get("static_rubrics")
            if static_rubrics is None:
                static_rubrics = obj.get("rubrics", [])
            positive_rubrics = obj.get("positive_rubrics", [])
            negative_rubrics = obj.get("negative_rubrics", [])
            return {
                "static_rubrics": static_rubrics if isinstance(static_rubrics, list) else [],
                "positive_rubrics": positive_rubrics if isinstance(positive_rubrics, list) else [],
                "negative_rubrics": negative_rubrics if isinstance(negative_rubrics, list) else [],
            }
        if isinstance(obj, list):
            return {"static_rubrics": obj, "positive_rubrics": [], "negative_rubrics": []}
    except Exception:
        pass

    # Fallback: use plain-text rubric as one static criterion.
    return {
        "static_rubrics": [{"title": "Custom Rubric", "description": text}],
        "positive_rubrics": [],
        "negative_rubrics": [],
    }


def _render_criteria_block(name: str, rubrics: List[Dict[str, Any]], start_idx: int = 1) -> str:
    lines = [f"<{name}>"]
    for i, item in enumerate(rubrics, start=start_idx):
        title = str(item.get("title", f"{name}_{i}")).strip()
        desc = str(item.get("description", "")).strip()
        weight = item.get("weight", None)
        if weight is None:
            lines.append(f'<criterion id="{name}-{i}" title="{title}">{desc}</criterion>')
        else:
            lines.append(
                f'<criterion id="{name}-{i}" title="{title}" weight="{weight}">{desc}</criterion>'
            )
    lines.append(f"</{name}>")
    return "\n".join(lines)


def build_draft_answer_user_prompt(
    draft_answer: str,
    question: Optional[str] = None,
    rubrics: Optional[Any] = None,
) -> str:
    parsed = _normalize_rubrics_input(rubrics)
    has_rubrics = bool(
        parsed.get("static_rubrics") or
        parsed.get("positive_rubrics") or
        parsed.get("negative_rubrics")
    )

    sections = []
    if question:
        sections.append(f"<question>\n{question.strip()}\n</question>")

    sections.append(f"<response>\n{draft_answer.strip()}\n</response>")

    if has_rubrics:
        # Rubric-guided evaluation
        static_block   = _render_criteria_block("static_rubrics",   parsed["static_rubrics"],   start_idx=1)
        positive_block = _render_criteria_block("positive_rubrics", parsed["positive_rubrics"], start_idx=1)
        negative_block = _render_criteria_block("negative_rubrics", parsed["negative_rubrics"], start_idx=1)

        sections.insert(2, static_block)   # after question and before response
        sections.insert(3, positive_block)
        sections.insert(4, negative_block)

        output_schema = {
            "feedback": {
                "global_issues": [
                    {
                        "issue": "High-level issue statement",
                        "why_it_matters": "Impact on answer correctness/usefulness",
                        "evidence": "Quote or short pointer from draft answer",
                    }
                ],
                "rubric_issues": [
                    {
                        "criterion_id": "static_rubrics-1",
                        "criterion_title": "criterion title",
                        "rubric_type": "static|positive|negative",
                        "issue": "What is wrong relative to this criterion",
                        "evidence": "Grounded evidence from draft answer",
                        "suggestion": "Concrete fix direction",
                    }
                ],
                "strengths": [
                    "Optional short note on what is already good and should be kept"
                ],
            }
        }
        sections.append(
            "Evaluate the response against the rubric criteria above.\n"
            "Return JSON only. Use this exact top-level schema (no extra top-level keys):\n"
            + json.dumps(output_schema, ensure_ascii=True, indent=2)
        )
    else:
        # No rubrics available — free-form evaluation based on question and response only
        output_schema = {
            "feedback": {
                "global_issues": [
                    {
                        "issue": "High-level issue statement",
                        "why_it_matters": "Impact on answer correctness/usefulness",
                        "evidence": "Quote or short pointer from draft answer",
                    }
                ],
                "rubric_issues": [],
                "strengths": [
                    "Optional short note on what is already good and should be kept"
                ],
            }
        }
        sections.append(
            "No rubric criteria are available. Evaluate the response based solely on how well it answers the question above. "
            "Focus on factual accuracy, completeness, clarity, and relevance.\n"
            "Return JSON only. Use this exact top-level schema (no extra top-level keys):\n"
            + json.dumps(output_schema, ensure_ascii=True, indent=2)
        )

    return "\n\n".join(sections)


def resolve_rubrics_for_question(question: Optional[str], rubrics: Optional[Any] = None) -> Dict[str, Any]:
    """
    Priority:
    1) explicit rubrics argument (if provided)
    2) rubric store lookup by question (exact or normalized match)
    3) no rubrics — return empty so build_draft_answer_user_prompt uses free-form evaluation

    NOTE: default_static_fallback is intentionally NOT used here to avoid prompt explosion
    when the rubric buffer accumulates thousands of static rubrics from all queries.
    """
    # Only treat rubrics as explicit input when it is meaningfully provided.
    # Empty string / whitespace should fallback to store lookup by question.
    if rubrics is not None and (not isinstance(rubrics, str) or rubrics.strip()):
        parsed = _normalize_rubrics_input(rubrics)
        parsed["source"] = "explicit_input"
        return normalize_rubric_payload(parsed)

    result = _RUBRIC_STORE.get_rubrics(question)

    # If the store fell back to default_static_rubrics (no per-query match),
    # return empty rubrics instead to avoid using a potentially huge default set.
    if result.get("source") in ("default_static_fallback", "default_static_without_question"):
        logger.info(
            "resolve_rubrics_for_question: store returned %s, using empty rubrics for free-form evaluation",
            result.get("source"),
        )
        return {
            "static_rubrics": [],
            "positive_rubrics": [],
            "negative_rubrics": [],
            "source": "no_rubrics_free_form",
        }

    return normalize_rubric_payload(result)


def normalize_feedback_payload(raw_text: str) -> Dict[str, Any]:
    import re as _re

    def _try_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "feedback" in parsed:
                return parsed
        except Exception:
            pass
        return None

    # 1. Try raw text first (no thinking mode or already clean)
    result = _try_parse(raw_text)
    if result:
        return result

    # 2. Strip <think>...</think> blocks (Qwen3 thinking mode)
    cleaned = _re.sub(r"<think>.*?</think>", "", raw_text, flags=_re.DOTALL).strip()
    result = _try_parse(cleaned)
    if result:
        return result

    # 3. Strip markdown code fences (```json ... ```)
    cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
    result = _try_parse(cleaned)
    if result:
        return result

    # 4. Extract first {...} block from the text
    m = _re.search(r"\{.*\}", cleaned, flags=_re.DOTALL)
    if m:
        result = _try_parse(m.group(0))
        if result:
            return result

    return {
        "feedback": {
            "global_issues": [
                {
                    "issue": "Failed to parse model feedback JSON.",
                    "why_it_matters": "Feedback cannot be reliably consumed by policy.",
                    "evidence": raw_text[:1200],
                }
            ],
            "rubric_issues": [],
            "strengths": [],
        }
    }


async def call_openai_compatible_chat(
    system_prompt: str,
    user_prompt: str,
) -> str:
    from .api_key_pool import get_draft_answer_pool

    api_base = os.environ.get("DRAFT_ANSWER_API_BASE", os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    model_name = os.environ.get("DRAFT_ANSWER_FEEDBACK_MODEL", "Qwen/Qwen3-32B")
    timeout_s = float(os.environ.get("DRAFT_ANSWER_TIMEOUT", "45"))
    temperature = float(os.environ.get("DRAFT_ANSWER_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("DRAFT_ANSWER_MAX_TOKENS", "4096"))

    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    pool = get_draft_answer_pool()
    max_retries = int(os.environ.get("DRAFT_ANSWER_MAX_RETRIES", str(max(1, pool.size()))))
    retry_after = float(os.environ.get("DRAFT_ANSWER_RETRY_AFTER", "60"))
    last_exc = None

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    for attempt in range(max_retries):
        api_key = pool.get()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    text = await response.text()
                    if response.status == 429:
                        pool.mark_rate_limited(api_key)
                        last_exc = RuntimeError(f"Draft feedback API rate limited (429), key={api_key[:8]}...")
                        if attempt < max_retries - 1:
                            wait = min(retry_after, 5 * (2 ** attempt))  # exponential backoff: 5s, 10s, 20s...
                            await asyncio.sleep(wait)
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"Draft feedback API failed ({response.status}): {text[:1000]}")
                    obj = json.loads(text)
                    return obj["choices"][0]["message"]["content"]
        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)  # linear backoff: 2s, 4s, 6s...
                await asyncio.sleep(wait)
            continue
        except aiohttp.ClientError as e:
            last_exc = e
            continue

    raise last_exc or RuntimeError("All draft answer API keys exhausted or failed")
