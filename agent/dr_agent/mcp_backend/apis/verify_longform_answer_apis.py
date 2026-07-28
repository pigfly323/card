"""
verify_longform_answer_apis.py
==============================
Long-form answer verification + reward scoring for the MCP `verify_answer`
backend (long-form / rubric-based deep-research tasks).

Score computation is aligned with ``longform_rubric_rewards.py`` (RL training):
the score range of each component and the combining weights are identical.

    r0 = 0.5 * rubric_reward
       + 0.2 * citation_reward
       + 0.2 * format_reward
       + 0.1 * num_search_turns_reward

Reward components
-----------------
1. rubric_reward   (LLM judge)  — weighted average of per-rubric scores [0,1].
   Each rubric judge call returns {score(0/1/2), reasoning, evidence}; normalized
   to [0,1] by /2. Weighted by rubric["weight"] (positive rubrics).
2. citation_reward (batch LLM judge) — 0.6 * avg_f1 + 0.4 * citation_format_reward.
   Per-claim recall (Fully/Partially/No support) and precision (Relevant/Irrelevant).
3. format_reward   (rule-based) — <answer></answer> tag presence [0,1].
4. num_search_turns_reward (rule-based) — search tool call count [0,1].

Feedback output
---------------
Plain-text string with sections:
    === Long-form Answer Evaluation ===
    Overall Score: 0.xxx  (rubric=.. citation=.. format=.. search=..)
    [Rubric Assessment]   per-rubric score bar + issue/feedback
    [Citation Quality]    per-claim recall/precision + hallucinated ids
    [Format]              <answer> tag found / not found
    [Search]              N search turns
    <!--EVAL_JSON:{...}-->

Reuses judge agents / prompts from verify_answer_apis.py (same RUBRIC_JUDGE_*
and CITATION_JUDGE_* env vars) so judge behaviour is consistent with r1.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rubric judge prompt (aligned with rubric_utils._default_feedback_judge_system_prompt)
# ---------------------------------------------------------------------------
# 与训练侧 rubric_utils.py 的 prompt 统一：score 0/1/2 + reasoning + evidence。
# 唯一区别：增加 "issue" 字段，给出给模型的 feedback（verify_answer 专用）。
# 负权重 rubric 的语义说明也加上，让 judge 正确理解「描述不良行为」的 criterion。
_RUBRIC_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. You will be given a user's question (in <question></question> tags), the assistant's response (in <response></response> tags), a specific criterion to evaluate (in <criterion></criterion> tags), and its weight (in <weight></weight> tags).

Evaluate the response based ONLY on the provided criterion and output a JSON object with four exact keys:
1. "score": An integer on a scale of 0 to 2.
2. "reasoning": A concise explanation of why the given score was assigned. Be specific and grounded in the response content.
3. "evidence": A short verbatim excerpt (one or two sentences) from the response. If score=0 and nothing relevant, write "".
4. "issue": A brief feedback sentence (max 50 words) for the agent.

# Scoring rules

## Positive-weight criteria (weight > 0)
These describe DESIRABLE behavior the response SHOULD exhibit.
  2 = response fully satisfies the criterion
  1 = response partially satisfies the criterion
  0 = response fails the criterion

## Negative-weight criteria (weight < 0)
These describe UNDESIRABLE behavior the response should AVOID.
  2 = response DOES exhibit the undesirable behavior described (bad)
  1 = response partially exhibits the undesirable behavior
  0 = response does NOT exhibit the undesirable behavior (good)

For negative-weight criteria, a good response scores 0 (correctly avoids the bad behavior), and a bad response scores 2 (contains the bad behavior). The "issue" field should tell the agent what to avoid or what was done correctly.

Output valid JSON only:
{
  "score": x,
  "reasoning": "...",
  "evidence": "...",
  "issue": "..."
}"""

_RUBRIC_JUDGE_USER_TEMPLATE = """\
<question>{question}</question>
<response>{response}</response>
<criterion>{criterion}</criterion>
<weight>{weight}</weight>"""


# Reuse judge agents, prompts, and citation extraction from the sibling module
# so judge behaviour stays aligned with the r1 reward pipeline.
# Note: _RUBRIC_JUDGE_SYSTEM_PROMPT is defined locally above (with issue field +
# negative-weight semantics), NOT imported from verify_answer_apis.
from .verify_answer_apis import (
    _BATCH_CITATION_SYSTEM_PROMPT,
    _build_batch_citation_user_prompt,
    _extract_citations_from_context_for_verify,
    _extract_claims_and_citation_ids,
    _extract_json_from_response,
    _get_citation_judge_agent,
    _get_rubric_judge_agent,
    _normalize_feedback_obj,
    _parse_batch_citation_response,
    _RUBRIC_STORE,   # RubricStore singleton (reads RUBRIC_BUFFER_EXPORT_PATH at import)
)

logger = logging.getLogger(__name__)

# Same weights as longform_rubric_rewards.py REWARD_WEIGHTS
_REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}

# Search-turn reward upper bound (same as score_num_in_context_search_turns)
_SEARCH_UPPER_BOUND = 3

# <call_tool name="google_search|snippet_search|browse_webpage"> for search counting
_SEARCH_CALL_RE = re.compile(
    r'<call_tool\s+[^>]*name=["\']?(?:google_search|snippet_search|browse_webpage)["\']?',
    re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CITE_TAG_RE = re.compile(r"<cite\b[^>]*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def _parse_ground_truth(ground_truth: str) -> List[Dict[str, Any]]:
    """Parse the injected <ground_truth> string into a list of rubric dicts.

    Accepts:
      • JSON list of rubrics: [{"description","title","weight"}, ...]
      • JSON object with "rubrics" key: {"query":..., "rubrics":[...]}
    Returns a flat list of {"description","title","weight"} dicts.
    """
    if not ground_truth or not ground_truth.strip():
        return []
    raw = ground_truth.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("rubrics") or []
    else:
        return []

    rubrics: List[Dict[str, Any]] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        desc = str(r.get("description", "")).strip()
        if not desc:
            continue
        rubrics.append({
            "description": desc,
            "title":       str(r.get("title", "")).strip() or "Criterion",
            "weight":      float(r.get("weight", 1.0)),
        })
    return rubrics


# ---------------------------------------------------------------------------
# Rule-based rewards: format + search turns
# ---------------------------------------------------------------------------

def _compute_format_reward(answer: str, solution_path: str) -> Tuple[float, str]:
    """Rule-based format reward, aligned with compute_format_reward.

    answer_format_reward  : answer contains <answer>...</answer> tags (1.0/0.0)
    citation_format_reward: solution_path/answer has <cite> tags (1.0/0.0)
    query_format_reward   : solution_path has search tool calls (1.0/0.0)

    format_reward = 0.5*answer + 0.3*citation + 0.2*query  (same as format_utils)
    """
    has_answer_tag = bool(_ANSWER_TAG_RE.search(answer))
    has_cite = bool(_CITE_TAG_RE.search(answer) or _CITE_TAG_RE.search(solution_path))
    has_search = bool(_SEARCH_CALL_RE.search(solution_path))

    answer_fmt = 1.0 if has_answer_tag else 0.0
    cite_fmt = 1.0 if has_cite else 0.0
    query_fmt = 1.0 if has_search else 0.0

    reward = 0.5 * answer_fmt + 0.3 * cite_fmt + 0.2 * query_fmt

    if has_answer_tag:
        fmt_str = "<answer></answer> tags found"
    else:
        fmt_str = "<answer></answer> tags NOT found"
    if has_cite:
        fmt_str += "; <cite> tags present"
    if has_search:
        fmt_str += "; search calls present"
    return reward, fmt_str


def _compute_search_turns_reward(solution_path: str) -> Tuple[float, int]:
    """Rule-based search-turns reward, aligned with score_num_in_context_search_turns.

    reward = min(1.0, n_search_calls / 3)
    """
    n = len(_SEARCH_CALL_RE.findall(solution_path))
    reward = min(1.0, n / _SEARCH_UPPER_BOUND)
    return reward, n


# ---------------------------------------------------------------------------
# LLM judge: rubric scoring (parallel, one call per rubric)
# ---------------------------------------------------------------------------

async def _score_rubrics(
    answer: str,
    question: str,
    rubrics: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float, bool]:
    """Score the answer against each rubric in parallel.

    Returns
    -------
    (feedback_list, rubric_reward, all_failed)
    feedback_list : per-rubric {title, weight, score(0-1), reasoning, evidence, issue}
    rubric_reward : weighted average [0,1] (aligned with _compute_rubric_scores_and_reward)
    all_failed    : True if every rubric judge call failed
    """
    if not rubrics:
        return [], 0.0, True

    max_tokens = int(os.environ.get("RUBRIC_JUDGE_MAX_TOKENS", "2048"))
    max_rubrics = int(os.environ.get("VERIFY_LONGFORM_MAX_RUBRICS", "8"))
    if len(rubrics) > max_rubrics:
        rubrics = sorted(rubrics, key=lambda r: abs(r.get("weight", 1.0)), reverse=True)[:max_rubrics]
        logger.info("[verify_longform] selected top %d rubrics by |weight|", max_rubrics)

    agent = _get_rubric_judge_agent()

    async def _score_one(rubric: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = rubric["title"]
        description = rubric["description"]
        weight = rubric["weight"]
        if agent is None:
            logger.warning("[verify_longform] rubric judge agent unavailable, skipping '%s'", title)
            return None

        # 与训练侧 rubric_utils 统一的 prompt（0/1/2 score + reasoning + evidence），
        # 增加 issue 字段 + weight 标签（让 judge 区分正/负权重 rubric）。
        user_prompt = _RUBRIC_JUDGE_USER_TEMPLATE.format(
            question=question,
            response=answer,
            criterion=description,
            weight=weight,
        )

        try:
            raw = await agent.async_single_turn(
                user_prompt,
                _RUBRIC_JUDGE_SYSTEM_PROMPT,
                None,        # temperature
                max_tokens,
                True,        # json_mode
            )
        except Exception as e:
            logger.warning("[verify_longform] rubric judge failed for '%s': %s", title, e)
            return None

        obj = raw if isinstance(raw, dict) else _extract_json_from_response(raw)
        if not obj:
            logger.warning("[verify_longform] rubric judge unparseable for '%s'", title)
            return None

        # 用训练侧 _normalize_feedback_obj 归一化（score / 2 → [0,1]）用于 feedback
        normalized = _normalize_feedback_obj(obj, score_scale=2.0)
        norm_score = float(normalized["score"])   # [0,1]，用于 feedback 显示
        reasoning = normalized["reasoning"]
        evidence = normalized["evidence"]

        # 原始 score（0/1/2），用于加权计算（与训练侧 longform_rubric_rewards.py 一致）
        raw_score = float(obj.get("score", 0.0))
        raw_score = max(0.0, min(2.0, raw_score))

        # issue: 优先用 judge 返回的 issue 字段，否则从 score + reasoning 构造
        issue = str(obj.get("issue", "") or "").strip()
        if not issue:
            if weight < 0:
                if raw_score >= 2.0:
                    issue = f"[{title}] {description} — Contains the undesirable behavior described; should be avoided."
                else:
                    issue = f"[{title}] {description} — Correctly avoids this undesirable behavior."
            else:
                if raw_score >= 2.0:
                    issue = f"[{title}] Meets this criterion."
                elif raw_score >= 1.0:
                    issue = f"[{title}] {description} — Partially meets. {reasoning}"
                else:
                    issue = f"[{title}] {description} — Not met. {reasoning}"

        return {
            "title": title,
            "description": description,
            "weight": weight,
            "score": round(norm_score, 4),       # [0,1] for display
            "raw_score": raw_score,               # 0/1/2 for weighted sum
            "reasoning": reasoning,
            "evidence": evidence,
            "issue": issue,
        }

    tasks = [_score_one(r) for r in rubrics]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    feedback_list: List[Dict[str, Any]] = []
    scored: List[Tuple[float, float]] = []   # (raw_score 0/1/2, weight)
    failed = 0

    for r in results:
        if isinstance(r, Exception):
            failed += 1
            continue
        if r is None:
            failed += 1
            continue
        feedback_list.append(r)
        scored.append((r["raw_score"], r["weight"]))

    all_failed = len(rubrics) > 0 and failed == len(rubrics)

    # 加权计算（与训练侧 longform_rubric_rewards.py 一致）：
    # 用原始 score（0/1/2）乘 weight，不预先归一化。
    # 正权重：score=2（满足）→ 加分；负权重：score=2（存在不良行为）→ 扣分。
    total_pos_weight = sum(w for _, w in scored if w > 0)
    total_neg_weight = sum(abs(w) for _, w in scored if w < 0)
    weighted_sum = sum(s * w for s, w in scored)

    # 归一化到 [0, 1]：
    #   max_possible = 2 * total_pos_weight（所有正权重 score=2 + 所有负权重 score=0）
    #   min_possible = -2 * total_neg_weight（所有正权重 score=0 + 所有负权重 score=2）
    max_possible = 2.0 * total_pos_weight
    min_possible = -2.0 * total_neg_weight

    if max_possible > min_possible:
        rubric_reward = (weighted_sum - min_possible) / (max_possible - min_possible)
    else:
        rubric_reward = 0.0
    rubric_reward = max(0.0, min(1.0, rubric_reward))

    logger.info(
        "[verify_longform] rubric_reward=%.4f scored=%d failed=%d all_failed=%s",
        rubric_reward, len(scored), failed, all_failed,
    )
    return feedback_list, rubric_reward, all_failed


# ---------------------------------------------------------------------------
# LLM judge: citation scoring (single batch call)
# ---------------------------------------------------------------------------

async def _score_citations(
    answer: str,
    question: str,
    citation_store: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str], float]:
    """Batch citation scoring, aligned with score_in_context_citations_batch.

    Returns
    -------
    (per_claim, hallucinated_ids, citation_reward)
    per_claim      : [{"claim","citation_ids","recall","precision","f1"}]
    hallucinated_ids: cite ids referenced but not in citation_store
    citation_reward: 0.6 * avg_f1 + 0.4 * citation_format_reward  ([0,1])
    """
    if not citation_store:
        return [], [], 0.0

    claims = _extract_claims_and_citation_ids(answer)

    # Hallucinated: cite ids in answer but not in retrieved snippets
    all_cited_ids = [cid for ids in claims.values() for cid in ids]
    hallucinated = list({cid for cid in all_cited_ids if cid not in citation_store})

    cited_claims = [(text, ids) for text, ids in claims.items() if ids]
    if not cited_claims:
        return [], hallucinated, 0.0

    # Build eval pairs (only claims with resolvable snippets)
    all_claim_list = list(claims.items())
    eval_pairs: List[Tuple[int, str, str]] = []
    for i, (text, ids) in enumerate(all_claim_list):
        if not ids:
            continue
        parts = [citation_store[c] for c in ids if c in citation_store]
        if not parts:
            continue
        eval_pairs.append((i, text, "\n\n".join(parts)))

    if not eval_pairs:
        return [], hallucinated, 0.0

    max_tokens = 2048 + 60 * len(eval_pairs)
    pairs_for_prompt = [(ct, sn) for _, ct, sn in eval_pairs]
    user_prompt = _build_batch_citation_user_prompt(question, pairs_for_prompt)

    agent = _get_citation_judge_agent()
    if agent is None:
        logger.warning("[verify_longform] citation judge agent unavailable")
        return [], hallucinated, 0.0

    try:
        raw = await agent.async_single_turn(
            user_prompt,
            _BATCH_CITATION_SYSTEM_PROMPT,
            None,
            max_tokens,
            True,   # json_mode
        )
    except Exception as e:
        logger.warning("[verify_longform] citation batch judge failed: %s", e)
        return [], hallucinated, 0.0

    scores = _parse_batch_citation_response(raw, len(eval_pairs))

    _recall_label = {1.0: "Fully supported", 0.5: "Partially supported", 0.0: "No support"}
    _precision_label = {1.0: "Relevant", 0.0: "Irrelevant"}

    per_claim: List[Dict[str, Any]] = []
    f1_scores: List[float] = []
    for batch_idx, (orig_idx, claim_text, _sn) in enumerate(eval_pairs):
        recall, precision = scores[batch_idx]
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        f1_scores.append(f1)
        ids = all_claim_list[orig_idx][1]
        per_claim.append({
            "claim": claim_text[:200],
            "citation_ids": ids,
            "recall": _recall_label.get(recall, "No support"),
            "precision": _precision_label.get(precision, "Irrelevant"),
            "f1": round(f1, 4),
        })

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    # citation_format_reward: all eval_pairs have valid IDs → 1.0
    citation_format_reward = 1.0
    citation_reward = 0.6 * avg_f1 + 0.4 * citation_format_reward

    logger.info(
        "[verify_longform] citation_reward=%.4f avg_f1=%.3f claims=%d hallucinated=%d",
        citation_reward, avg_f1, len(per_claim), len(hallucinated),
    )
    return per_claim, hallucinated, citation_reward


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(payload: Dict[str, Any]) -> str:
    """Render feedback as plain text with machine-parseable score lines.

    Output format is designed to be parsed by ``_lff_parse_r0_from_va_output``
    (grpo_lff.py). Each component emits a line the parser recognises:

      Rubric reward: X.XXXX              ← summary (parser uses this directly)
      {"title": "...", "score": N, "weight": N, ...}   ← per-rubric JSON (score 0/1/2)
      Claim N: "..." [recall] [precision]              ← per-claim citation
      Citation reward: X.XXXX            ← (optional, parser recomputes from claims)
      Format reward: X.XXXX              ← (optional, parser recomputes from draft)
      Search score:  X.XXXX              ← (optional, parser recomputes from draft)

    No bars, icons, or markdown — just plain text and the parseable lines.
    """
    lines: List[str] = []

    # ── Rubric section ──
    # Emit one JSON line per rubric (score on raw 0/1/2 scale, matching the
    # judge header "0=not met, 1=partially met, 2=fully met"). The parser
    # divides by 2 to normalise to [0,1].
    # Only keep title/weight/score/issue — concise feedback for the model.
    rubric_feedback = payload.get("rubric_feedback", [])
    for entry in rubric_feedback:
        norm_score = float(entry.get("score", 0.0))   # already [0,1]
        raw_score = round(norm_score * 2)              # back to 0/1/2 for the parser
        rubric_line = {
            "title":  entry.get("title", ""),
            "weight": entry.get("weight", 1.0),
            "score":  raw_score,
            "issue":  entry.get("issue", ""),
        }
        lines.append(json.dumps(rubric_line, ensure_ascii=False))

    # Rubric reward summary line (parser prefers this over per-rubric recompute)
    lines.append(f"Rubric reward: {payload['rubric_reward']:.4f}")
    lines.append("")

    # ── Citation section ──
    ci = payload.get("citation_issues", {})
    hallucinated = ci.get("hallucinated", [])
    per_claim = ci.get("per_claim", [])

    if hallucinated:
        lines.append(f"Hallucinated citations (not in retrieved context): {', '.join(hallucinated)}")

    # Citation reward summary.
    if per_claim:
        lines.append(f"Citation reward: {payload['citation_reward']:.4f}")

    # Per-claim lines: only report claims that are NOT fully supported + relevant
    # (i.e. have issues). Perfect citations are silent — no feedback needed.
    for i, claim_info in enumerate(per_claim, 1):
        recall = claim_info.get("recall", "No support")
        precision = claim_info.get("precision", "Irrelevant")
        if recall == "Fully supported" and precision == "Relevant":
            continue   # perfect citation — skip
        claim_text = claim_info.get("claim", "")
        short = claim_text[:120] + ("..." if len(claim_text) > 120 else "")
        lines.append(f'Claim {i}: "{short}" [{recall}] [{precision}]')

    if not per_claim and not hallucinated:
        lines.append("No citations found in answer (no <cite> tags present).")

    lines.append("")

    # ── Format + search (parser recomputes these from draft_response, but we
    # emit them too for completeness / debugging) ──
    lines.append(f"Format reward: {payload['format_reward']:.4f}")
    lines.append(f"Search score:  {payload['num_search_turns_reward']:.4f}")
    lines.append("")

    # ── Overall ──
    lines.append(f"Overall Score (r0): {payload['r0']:.4f}")
    lines.append(f"  rubric={payload['rubric_reward']:.3f}  citation={payload['citation_reward']:.3f}  "
                 f"format={payload['format_reward']:.3f}  search={payload['num_search_turns_reward']:.3f}")
    lines.append(f"  search_turns={payload.get('num_search_turns', 0)}")

    if payload.get("error"):
        lines.append(f"Error: {payload['error']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def verify_longform_answer_handler(
    question: str,
    answer: str,
    context: str = "",
    solution_path: str = "",
    ground_truth: str = "",
) -> str:
    """Long-form verify_answer: score the draft answer and return feedback.

    Score components and weights are aligned with longform_rubric_rewards.py.

    Parameters
    ----------
    question      : original user question
    answer        : draft answer text (the <answer>...</answer> content)
    context       : retrieved <snippet>/<webpage> blocks (citation source)
    solution_path : agent tool-interaction trace (call_tool + tool_output)
    ground_truth  : JSON string of rubrics [{"description","title","weight"}, ...]

    Returns
    -------
    Plain-text feedback string with rubric/citation/format/search breakdown
    and an embedded ``<!--EVAL_JSON:...-->`` payload.
    """
    if not answer or not answer.strip():
        return _format_output({
            "r0": 0.0, "rubric_reward": 0.0, "citation_reward": 0.0,
            "format_reward": 0.0, "num_search_turns_reward": 0.0,
            "num_search_turns": 0, "rubric_feedback": [], "citation_issues": {},
            "format_feedback": "<answer></answer> tags NOT found",
            "error": "No draft answer provided",
        })

    # ── 1. Rule-based: format + search turns ──
    format_reward, fmt_feedback = _compute_format_reward(answer, solution_path)
    search_turns_reward, num_search_turns = _compute_search_turns_reward(solution_path)

    # ── 2. Parse rubrics from ground_truth ──
    # 优先用 <ground_truth> 注入的 rubrics；若未注入或解析为空，
    # 从 RubricStore 按 question 查询（RubricStore 在 import 时已根据
    # RUBRIC_BUFFER_EXPORT_PATH / RUBRIC_DATASET_PATH 初始化）。
    rubrics = _parse_ground_truth(ground_truth)
    if not rubrics and question:
        store_result = _RUBRIC_STORE.get_rubrics(question)
        static = store_result.get("static_rubrics", []) if store_result else []
        if static:
            rubrics = [
                {
                    "description": str(r.get("description", "")).strip(),
                    "title":       str(r.get("title", "")).strip() or "Criterion",
                    "weight":      float(r.get("weight", 1.0)),
                }
                for r in static
                if isinstance(r, dict) and r.get("description")
            ]
            logger.info(
                "[verify_longform] rubrics from RubricStore: %d (source=%s)",
                len(rubrics), store_result.get("source", "unknown") if store_result else "none",
            )

    # ── 3. Build citation store from context ──
    citation_store = _extract_citations_from_context_for_verify(context) if context else {}

    # ── 4. LLM judges: rubric + citation (run in parallel) ──
    rubric_task = _score_rubrics(answer, question or "", rubrics)
    citation_task = _score_citations(answer, question or "", citation_store)

    try:
        (rubric_feedback, rubric_reward, rubric_all_failed), \
            (per_claim, hallucinated, citation_reward) = await asyncio.gather(
                rubric_task, citation_task
            )
    except Exception as exc:
        logger.exception("[verify_longform] judge tasks failed: %s", exc)
        return _format_output({
            "r0": 0.0, "rubric_reward": 0.0, "citation_reward": 0.0,
            "format_reward": format_reward, "num_search_turns_reward": search_turns_reward,
            "num_search_turns": num_search_turns, "rubric_feedback": [],
            "citation_issues": {}, "format_feedback": fmt_feedback,
            "error": f"judge tasks failed: {exc}",
        })

    # ── 5. Compute r0 (weighted sum, same weights as longform_rubric_rewards.py) ──
    r0 = (
        _REWARD_WEIGHTS["rubric_reward"] * rubric_reward
        + _REWARD_WEIGHTS["citation_reward"] * citation_reward
        + _REWARD_WEIGHTS["format_reward"] * format_reward
        + _REWARD_WEIGHTS["num_search_turns_reward"] * search_turns_reward
    )
    r0 = max(0.0, min(1.0, r0))

    payload = {
        "r0": round(r0, 6),
        "rubric_reward": round(rubric_reward, 4),
        "citation_reward": round(citation_reward, 4),
        "format_reward": round(format_reward, 4),
        "num_search_turns_reward": round(search_turns_reward, 4),
        "num_search_turns": num_search_turns,
        "rubric_feedback": rubric_feedback,
        "citation_issues": {
            "hallucinated": hallucinated,
            "per_claim": per_claim,
        },
        "format_feedback": fmt_feedback,
        "error": None,
    }

    logger.info(
        "[verify_longform] done r0=%.4f rubric=%.3f citation=%.3f fmt=%.3f search=%.3f",
        r0, rubric_reward, citation_reward, format_reward, search_turns_reward,
    )
    return _format_output(payload)
