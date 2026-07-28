"""
verify_short_answer_apis.py

Factual verifier + reward scorer for short-answer QA tasks.

输出格式与 _verify_answer_native_short（mcp_tools.py）完全对齐，
使 _lff_parse_r0_from_va_output（grpo_lff.py）能正确提取 r0：

    Answer reward: X.XXXX   ← LLM judge 正确性分数 [0,1]
    Format reward: X.XXXX   ← 答案格式 + citation + 搜索调用格式 [0,1]
    Search score:  X.XXXX   ← 上下文搜索轮次奖励 [0,1]
    Rubric reward: X.XXXX   ← r0 综合分 (0.8*ans + 0.1*fmt + 0.1*srch)

    ### Feedback (teacher context — do not copy into final answer):
    <issues or positive feedback>

Environment variables:
  VERIFY_SHORT_ANSWER_MODEL        - judge model (default: RUBRIC_JUDGE_MODEL or gpt-4.1)
  VERIFY_SHORT_ANSWER_API_BASE     - API base URL
  VERIFY_SHORT_ANSWER_MAX_TOKENS   - max response tokens (default: 1024)
  VERIFY_SHORT_ANSWER_TIMEOUT      - per-attempt timeout in seconds (default: 60)
  VERIFY_SHORT_ANSWER_MAX_SP_CHARS - max chars of solution_path sent to judge (default: 12000)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .verify_answer_apis import (
    _call_judge_api_with_retry,
    _extract_json_from_response,
    _get_rubric_judge_pool,
    _RUBRIC_STORE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a factual verifier for a search-and-reasoning agent.
Audit the predicted answer against the evidence in the solution path.
Never state or imply the correct answer. Express every issue as a factual or
evidentiary problem, not as "the answer is wrong".

# Solution path format

The <solution_path> is the agent's tool-interaction trace: alternating
<call_tool> blocks (what the agent searched/browsed) and <tool_output> blocks
(what the tools returned).
Example trace:
<call_tool name="google_search">who sees the ghost of King Hamlet first</call_tool>
<tool_output><snippet id="s1">
Title: Hamlet Act 1 Scene 1
Snippet: The ghost appears silently and is seen by Marcellus, Bernardo, and Horatio.
</snippet></tool_output>
<call_tool name="browse_webpage">https://example.com/hamlet</call_tool>
<tool_output><webpage id="w1">
Title: Hamlet Summary
Snippet: Horatio is the first to confirm the ghost and later tells Hamlet.
</webpage></tool_output>

Each snippet/webpage block has an id (e.g. s1, w1). The predicted answer may
reference these via <cite id="s1">. Trace every factual claim in the answer back
to a specific snippet; if no snippet supports it, it is unsupported.

# Check for these four problems

1. Unsupported claim — a factual assertion in the answer that cannot be traced
   to any retrieved snippet in <tool_output> blocks of the solution path.

2. Citation mismatch — a <cite id="..."> tag where the cited snippet does not
   actually support, or is irrelevant to, the cited claim.

3. Search hallucination — the agent assumed a fact about an entity (type, role,
   nationality, affiliation) without a retrieved snippet confirming it, or
   resolved an ambiguous name to a specific referent without retrieval evidence.

4. Question misinterpretation — the agent answered at the wrong granularity,
   treated a proper noun as a common phrase, or addressed the wrong sub-question.

# Ground truth

Use it as a private oracle to identify which
claims likely need scrutiny — do not echo it in your output.
<ground_truth> may list multiple acceptable alternatives (any one is correct):
when multiple are given, the predicted answer is correct if it matches ANY of them.

# Correctness score

  1.0 — the answer is fully supported by retrieved evidence and matches the ground truth (if provided)
  0.0 — the answer is incorrect or unsupported

# Output schema

{
  "score": 1.0 | 0.0,
  "issues": ["<factual or evidentiary problem>", ...],
  "feedback": "<brief evidence-quality summary when issues is empty; null otherwise>"
}

Field rules:
- score: float, required
- issues: list of strings, each describing one specific problem; empty list if none found
- feedback: non-empty string when issues is empty; null otherwise
Output raw JSON only — no markdown fences, no prose outside the object.
"""

_USER_TEMPLATE = """\
<problem>
{problem}
</problem>

<predicted_answer>
{predicted_answer}
</predicted_answer>

<ground_truth>
{ground_truth}
</ground_truth>

<solution_path>
{solution_path}
</solution_path>"""

_FAILURE_RESPONSE: str = (
    "Answer reward: 0.0000\n"
    "Format reward: 0.0000\n"
    "Search score:  0.0000\n"
    "Rubric reward: 0.0000\n\n"
    "### Feedback (teacher context — do not copy into final answer):\n"
    "Verification service temporarily unavailable.\n"
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_model() -> str:
    return (
        os.environ.get("VERIFY_SHORT_ANSWER_MODEL")
        or os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1")
    )

def _get_api_base() -> str:
    return (
        os.environ.get("VERIFY_SHORT_ANSWER_API_BASE")
        or os.environ.get("RUBRIC_JUDGE_API_BASE")
        or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    )

def _get_max_tokens() -> int:
    return int(os.environ.get("VERIFY_SHORT_ANSWER_MAX_TOKENS", "1024"))

def _get_timeout() -> float:
    return float(os.environ.get("VERIFY_SHORT_ANSWER_TIMEOUT", "60"))

def _get_max_sp_chars() -> int:
    return int(os.environ.get("VERIFY_SHORT_ANSWER_MAX_SP_CHARS", "12000"))

# ---------------------------------------------------------------------------
# Reward scoring helpers（与 _verify_answer_native_short 使用相同的权重和逻辑）
# ---------------------------------------------------------------------------

# r0 = 0.8 * answer_score + 0.1 * format_reward + 0.1 * search_turns_reward
_W_ANSWER  = 0.8
_W_FORMAT  = 0.1
_W_SEARCH  = 0.1

# format_reward 内部权重（与 open_instruct/search_rewards/utils/format_utils.py 对齐）
_W_FMT_ANSWER   = 0.5  # <answer>...</answer> 存在
_W_FMT_CITATION = 0.3  # <cite ...>...</cite> 存在
_W_FMT_QUERY    = 0.2  # 有搜索工具调用记录

# search_turns_reward 上界（3 次及以上搜索 → 满分）
_SEARCH_UPPER_BOUND = 3

# <answer>...</answer> 内容提取正则（用于清洗 predicted_answer）
_ANSWER_EXTRACT_RE = re.compile(
    r'<answer>(.*?)</answer>',
    re.DOTALL | re.IGNORECASE,
)
# \boxed{...} 提取正则（模型输出数学格式时）
_BOXED_RE = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)

# 检索块正则（匹配 <snippet>/<snippets>/<webpage> 块，用于 format/search reward）
_SNIPPET_BLOCK_RE = re.compile(
    r'<(?:snippets?|webpage)\b[^>]*>',
    re.IGNORECASE,
)
# <answer>...</answer> 正则（排除 verify_answer 调用内部的 <answer> 标签）
_ANSWER_TAG_RE = re.compile(
    r'<answer>(.*?)</answer>',
    re.DOTALL | re.IGNORECASE,
)
# <cite ...>...</cite> 正则
_CITE_TAG_RE = re.compile(
    r'<cite\b[^>]*>',
    re.IGNORECASE,
)


def _extract_clean_answer(predicted_answer: str) -> str:
    """清洗 predicted_answer，消除格式噪声，同时保留完整答案语义。

    策略：substitute（替换格式标记）而非 extract（只取部分内容）：
    1. 若含 <answer>...</answer> 标签，取标签内全部内容
    2. 将所有 \\boxed{content} 替换为 content（保留上下文，只去掉 LaTeX 包装）
    3. 去除首尾空白

    示例（substitute 而非 extract）：
      "<answer>\\n\\boxed{heterochromia}\\n</answer>"
        → "heterochromia"                              ✓ 纯短答案，正常提取

      "<answer>Based on Martini's melody, arranged by \\boxed{Peretti, Creatore, Weiss}.</answer>"
        → "Based on Martini's melody, arranged by Peretti, Creatore, Weiss."
          ↑ 保留了 Martini，只去掉 \\boxed{}；judge 能看到完整信息       ✓

      "\\boxed{42}"
        → "42"                                         ✓
    """
    text = predicted_answer or ""
    # Step 1: 取 <answer> 标签内容（若存在）
    m = _ANSWER_EXTRACT_RE.search(text)
    if m:
        text = m.group(1)
    # Step 2: 替换 \boxed{content} → content（保留全文语义）
    text = _BOXED_RE.sub(lambda mo: mo.group(1), text)
    return text.strip()


def _normalize_gt_to_list(ground_truth) -> list:
    """将 ground_truth（list / str / None）归一化为去重的答案列表。

    str 输入会尝试 JSON 解析（如 '["a","b"]' → ["a","b"]），失败则按单值处理。
    """
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        seen, out = set(), []
        for a in ground_truth:
            s = str(a).strip() if a is not None else ""
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        if not s:
            return []
        # 尝试 JSON 解析（train.jsonl 的 ground_truth 常是 JSON 字符串列表）
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return [s]
        return _normalize_gt_to_list(parsed)
    return [str(ground_truth).strip()]


def _format_ground_truth(gt_list: list) -> str:
    """将答案列表格式化为 judge prompt 可读的 <ground_truth> 内容。

    单答案 → 原样返回
    多答案 → 列出所有可接受候选（judge 任意命中其一即判正确）
    """
    if not gt_list:
        return ""
    if len(gt_list) == 1:
        return gt_list[0]
    # 多候选：明确告知 judge 这些是可互换的等价答案
    lines = ["Any of the following is a correct answer:"]
    lines += [f"- {a}" for a in gt_list]
    return "\n".join(lines)


def _compute_format_reward(solution_path: str, predicted_answer: str) -> float:
    """与 open_instruct compute_format_reward 对齐的简化版本。

    answer_format_reward  : predicted_answer 本身包含 <answer> 标签
    citation_format_reward: solution_path 中有 <cite> 引用
    query_format_reward   : solution_path 中有搜索工具调用

    format_reward = 0.5*answer + 0.3*citation + 0.2*query
    """
    # answer_format_reward: predicted_answer 包含 <answer> 标签
    answer_fmt = 1.0 if _ANSWER_TAG_RE.search(predicted_answer) else 0.0

    # citation_format_reward: solution_path 中有 <cite> 引用
    cite_fmt = 1.0 if _CITE_TAG_RE.search(solution_path) else 0.0

    # query_format_reward: solution_path 中有搜索工具调用
    query_fmt = 1.0 if _SEARCH_CALL_RE.search(solution_path) else 0.0

    return _W_FMT_ANSWER * answer_fmt + _W_FMT_CITATION * cite_fmt + _W_FMT_QUERY * query_fmt


def _compute_search_turns_reward(solution_path: str) -> float:
    """与 open_instruct score_num_in_context_search_turns 对齐的简化版本。

    统计 solution_path 中的搜索工具调用次数：
      0 次 → 0.0,  1 次 → 1/3,  2 次 → 2/3,  ≥3 次 → 1.0
    """
    n = len(_SEARCH_CALL_RE.findall(solution_path))
    return min(1.0, n / _SEARCH_UPPER_BOUND)


def _format_output(
    answer_score: float,
    format_reward: float,
    search_turns_reward: float,
    issues: List[str],
    feedback: Optional[str],
    judge_score_available: bool = True,
) -> str:
    """将各维度分数格式化为与 _verify_answer_native_short 完全一致的输出字符串。

    _lff_parse_r0_from_va_output 依赖以下行格式：
        Answer reward: X.XXXX
        Format reward: X.XXXX
        Search score:  X.XXXX
        Rubric reward: X.XXXX
    """
    r0 = max(0.0, min(1.0,
        _W_ANSWER * answer_score
        + _W_FORMAT * format_reward
        + _W_SEARCH * search_turns_reward
    ))

    lines = [
        f"Answer reward: {answer_score:.4f}",
        f"Format reward: {format_reward:.4f}",
        f"Search score:  {search_turns_reward:.4f}",
        f"Rubric reward: {r0:.4f}",
        "",
    ]

    lines.append("### Feedback (teacher context — do not copy into final answer):")
    if issues:
        lines.append("Issues identified in your solution path:")
        for iss in issues:
            lines.append(f"  - {iss}")
    elif feedback and feedback.strip():
        lines.append(feedback.strip())
    else:
        if answer_score >= 1.0:
            lines.append("The extracted answer matches the correct answer.")
        else:
            lines.append("Incorrect: could not verify the answer against retrieved evidence.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

def _validate_and_normalize(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    # score field
    raw_score = obj.get("score")
    try:
        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None

    issues_raw = obj.get("issues", [])
    if not isinstance(issues_raw, list):
        return None

    normalised_issues: List[str] = []
    for item in issues_raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                normalised_issues.append(s)
        elif isinstance(item, dict):
            s = str(item.get("critique", "") or item.get("action", "")).strip()
            if s:
                normalised_issues.append(s)

    feedback = obj.get("feedback")
    if feedback is not None:
        feedback = str(feedback).strip() or None

    return {
        "score":    score,
        "issues":   normalised_issues,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def verify_short_answer_handler(
    problem: str,
    predicted_answer: str,
    ground_truth="",
    solution_path: str = "",
) -> str:
    """
    短答案验证 + 奖励评分。返回格式化字符串，与 _verify_answer_native_short 对齐，
    使 _lff_parse_r0_from_va_output 能正确提取 r0。

    Parameters
    ----------
    problem          : 原始问题
    predicted_answer : 模型预测答案（含 <answer>...</answer> 标签）
    ground_truth     : 参考答案（私有 oracle，不在输出中泄露）
    solution_path    : 完整推理轨迹（CoT + 工具调用 + 工具输出），judge 据此审计证据完整性

    Raises
    ------
    ValueError
        当无法提取到 ground_truth 时（参数未提供且 RubricStore 查询失败）直接抛出，
        不使用 fallback，避免在无 GT 情况下给出误导性分数。

    Returns
    -------
    格式化字符串，包含：
        Answer reward: X.XXXX
        Format reward: X.XXXX
        Search score:  X.XXXX
        Rubric reward: X.XXXX

        ### Feedback (teacher context — do not copy into final answer):
        <issues or positive feedback>
    """
    # ── 1. 清洗 predicted_answer，消除 <answer>/\boxed{} 格式噪声 ──────────
    # 传给 judge 的是干净的答案文本（如 "heterochromia"），
    # 避免 judge 因 LaTeX/XML 格式不一致而给出矛盾的评分。
    # format_reward 仍用原始 predicted_answer 计算（检测 <answer> 标签是否存在）。
    clean_answer = _extract_clean_answer(predicted_answer)

    # ── 2. 本地计算 format_reward 和 search_turns_reward（基于 solution_path）──
    format_reward       = _compute_format_reward(solution_path, predicted_answer)
    search_turns_reward = _compute_search_turns_reward(solution_path)

    if not predicted_answer or not predicted_answer.strip():
        return _format_output(
            answer_score=0.0,
            format_reward=format_reward,
            search_turns_reward=search_turns_reward,
            issues=[],
            feedback="No predicted answer provided — nothing to verify.",
        )

    # ── 3. GT 归一化为候选列表 + 自动填充 ────────────────────────────────
    # ground_truth 参数可能是 str（调用方传入）或来自 store 的 list[str]。
    # 统一归一化为 list[str]，保留多答案候选，不截断。
    gt_list = _normalize_gt_to_list(ground_truth)

    if not gt_list and problem:
        # 参数未提供 GT → 从 RubricStore 查询（返回 list[str] 或 None）
        store_gt = _RUBRIC_STORE.get_ground_truth(problem)
        if store_gt:
            gt_list = list(store_gt)
            logger.info(
                "[verify_short_answer] GT auto-filled from store: %d candidates %r",
                len(gt_list), gt_list[:2],
            )

    # GT 缺失直接抛异常，不用 fallback —— 无 GT 时 judge 无法判定正确性，
    # 给出分数会误导训练。由调用方决定如何处理（跳过/重试/记录）。
    if not gt_list:
        raise ValueError(
            f"[verify_short_answer] cannot extract ground_truth for problem={problem[:80]!r} "
            f"(ground_truth param empty and RubricStore lookup failed). "
            f"Set GT_DATASET_PATH or pass ground_truth explicitly."
        )

    # 格式化为 judge prompt 可读文本（单答案原样，多答案列为等价候选）
    gt_display = _format_ground_truth(gt_list)
    has_sp = bool(solution_path and solution_path.strip())
    logger.info(
        "[verify_short_answer] request: n_gt_candidates=%d has_solution_path=%s "
        "answer_chars=%d sp_chars=%d",
        len(gt_list), has_sp, len(predicted_answer), len(solution_path),
    )

    # ── 4. LLM Judge：基于 solution_path 审计证据完整性 ───────────────────
    max_sp_chars = _get_max_sp_chars()
    sp_text = solution_path.strip() if has_sp else "(no solution path provided)"
    if len(sp_text) > max_sp_chars:
        sp_text = sp_text[:max_sp_chars] + "\n... [truncated]"

    user_prompt = _USER_TEMPLATE.format(
        problem=problem.strip(),
        predicted_answer=clean_answer,          # 清洗后的干净答案，消除格式噪声
        ground_truth=gt_display,
        solution_path=sp_text,
    )

    model    = _get_model()
    api_base = _get_api_base()
    pool     = _get_rubric_judge_pool()

    try:
        raw = await _call_judge_api_with_retry(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            api_base=api_base,
            max_tokens=_get_max_tokens(),
            timeout_s=_get_timeout(),
            pool=pool,
        )
    except Exception as exc:
        logger.warning("[verify_short_answer] API call failed: %s", exc)
        return _FAILURE_RESPONSE

    if os.environ.get("VERIFY_SHORT_ANSWER_LOG_IO", "").lower() in ("1", "true", "yes"):
        logger.info(
            "[verify_short_answer][io]\n--- user_prompt ---\n%s\n--- raw ---\n%s",
            user_prompt,
            raw,
        )

    obj = _extract_json_from_response(raw) if isinstance(raw, str) else raw
    result = _validate_and_normalize(obj)

    if result is None:
        logger.warning(
            "[verify_short_answer] Could not parse judge response: %s",
            str(raw)[:300],
        )
        return _FAILURE_RESPONSE

    # ── 4. 提取 answer_score ─────────────────────────────────────────────
    # 优先使用 judge 返回的 score 字段 [0,1]；
    # 降级：无 issues → 1.0，有 issues → 0.0
    judge_score = result.get("score")
    if judge_score is not None:
        answer_score = max(0.0, min(1.0, float(judge_score)))
    else:
        answer_score = 0.0 if result["issues"] else 1.0

    logger.info(
        "[verify_short_answer] answer_score=%.4f issues=%d fmt=%.3f srch=%.3f clean_answer=%r",
        answer_score, len(result["issues"]), format_reward, search_turns_reward,
        clean_answer[:60],
    )

    return _format_output(
        answer_score=answer_score,
        format_reward=format_reward,
        search_turns_reward=search_turns_reward,
        issues=result["issues"],
        feedback=result["feedback"],
    )
