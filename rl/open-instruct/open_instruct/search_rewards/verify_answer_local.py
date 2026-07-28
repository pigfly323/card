"""
Native verify_answer: computes r0 reward for a draft answer written inside
a verify_answer tool call.  Bypasses the MCP Server — no HTTP round-trip.

Design
------
The model's verify_answer call looks like:

    <call_tool name="verify_answer">
      The draft answer text with <cite id="a1b2">claims</cite>.
      <context><snippet id="a1b2">...</snippet></context>     ← injected by MCPTool
      <solution_path>...</solution_path>                      ← injected by MCPTool
      <ground_truth>answer text</ground_truth>               ← injected by MCPTool
    </call_tool>

We:
  1. Strip the injected blocks → get the raw draft answer text
  2. Parse citations from the <context> block
  3. Build a "synthetic_response" = context_snippets + <answer>draft</answer>
     so that all existing reward utilities work unchanged
  4. Run the same 4 reward components used during training:
       rubric_reward   (use_general_rubric=True — no per-question rubric dicts needed)
       citation_reward (structured: hallucination check + per-claim recall/precision)
       format_reward
       search_turns_reward
  5. Return r0 (weighted sum) + structured feedback for the model

Ground-truth
------------
Because we use use_general_rubric=True the function only needs ``question``
(no specific rubric dicts, no gt_store).  The injected ``<ground_truth>``
answer string is forwarded as a note in the feedback, not used for scoring.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from open_instruct.search_rewards.utils.format_utils import (
    compute_format_reward,
    extract_answer_context_citations,
    extract_citations_from_context,
)
from open_instruct.search_rewards.utils.search_utils import (
    score_num_in_context_search_turns,
)

LOGGER = logging.getLogger(__name__)

# Same weights as in longform_rubric_rewards.py
_REWARD_WEIGHTS = {
    "rubric_reward": 0.5,
    "citation_reward": 0.2,
    "format_reward": 0.2,
    "num_search_turns_reward": 0.1,
}


# ---------------------------------------------------------------------------
# Step 1 – Extract draft answer from inside the verify_answer call
# ---------------------------------------------------------------------------

def _extract_draft_from_verify_call(
    full_response: str,
) -> Tuple[Optional[str], str, Dict[str, str]]:
    """
    Find the LAST <call_tool name="verify_answer"> block in *full_response*
    and return:
      (draft_text, context_text, citations_dict)

    draft_text     – model-written content (injected helper blocks stripped)
    context_text   – raw text of the <context> block (may be empty)
    citations_dict – {snippet_id: snippet_text} parsed from <context>
    """
    patterns = [
        r'<call_tool\s+[^>]*name=["\']?verify_answer["\']?[^>]*>(.*?)(?:</call_tool>|</call>)',
        r'<tool\s+[^>]*name=["\']?verify_answer["\']?[^>]*>(.*?)</tool>',
    ]
    inner: Optional[str] = None
    for pat in patterns:
        matches = list(re.finditer(pat, full_response, re.DOTALL | re.IGNORECASE))
        if matches:
            inner = matches[-1].group(1)
            break

    if inner is None:
        LOGGER.warning("[verify_answer_local] No verify_answer call found in response")
        return None, "", {}

    # Extract <context> block before stripping (citations live here)
    context_match = re.search(r"<context>(.*?)</context>", inner, re.DOTALL)
    context_text = context_match.group(1).strip() if context_match else ""
    citations = extract_citations_from_context(context_text) if context_text else {}

    # Strip injected helper blocks to get the raw draft answer
    draft = inner
    for tag in ("context", "solution_path", "ground_truth"):
        draft = re.sub(rf"<{tag}>.*?</{tag}>", "", draft, flags=re.DOTALL)

    # Also strip the verify_answer opening-tag attributes that may have been
    # reinjected into the content (e.g. question="..." apply_adaptive_rubric_reward="...")
    draft = draft.strip()

    return draft, context_text, citations


def _build_synthetic_response(draft_text: str, citations: Dict[str, str]) -> str:
    """
    Reconstruct a 'normal' response that existing reward utilities accept:

        <snippet id="id1">text</snippet>
        ...
        <answer>draft_text</answer>

    This allows compute_format_reward / extract_answer_context_citations /
    score_in_context_citations_batch to work without modification.
    """
    snippet_blocks = "\n".join(
        f'<snippet id="{sid}">\n{stext}\n</snippet>'
        for sid, stext in citations.items()
    )
    return f"{snippet_blocks}\n<answer>{draft_text}</answer>"


# ---------------------------------------------------------------------------
# Step 2 – Structured citation issues (extends citation_utils)
# ---------------------------------------------------------------------------

async def _compute_citation_issues_detailed(
    question: str,
    synthetic_response: str,
    citations: Dict[str, str],
) -> Dict[str, Any]:
    """
    Returns structured citation feedback:
    {
        "hallucinated":  ["id1", ...]     # cite ids not present in retrieved snippets
        "per_claim": [
            {
                "claim": str,
                "citation_ids": [str],
                "recall":    "fully" | "partially" | "none",
                "precision": "relevant" | "irrelevant",
            }
        ],
        "format_score": float,   # valid / total citations
        "avg_f1": float,
        "citation_reward": float  # 0.6 * avg_f1 + 0.4 * format_score
    }
    """
    from open_instruct.search_rewards.utils.citation_utils import (
        extract_claims_and_corresponding_citation_ids,
        score_citation_format,
        _build_batch_citation_prompt,
        _BATCH_CITATION_SYSTEM_PROMPT,
        _parse_batch_citation_response,
    )
    from open_instruct.search_rewards.utils.run_utils import get_judge_agent

    if not citations:
        return {
            "hallucinated": [],
            "per_claim": [],
            "format_score": 0.0,
            "avg_f1": 0.0,
            "citation_reward": 0.0,
        }

    claims = extract_claims_and_corresponding_citation_ids(synthetic_response)
    format_score = score_citation_format(claims, citations)

    # Hallucinated: cite ids referenced in draft but not in retrieved snippets
    all_cited_ids = [cid for ids in claims.values() for cid in ids]
    hallucinated = list({cid for cid in all_cited_ids if cid not in citations})

    # Only evaluate claims that actually carry cite tags
    cited_claims = [
        (text, ids) for text, ids in claims.items() if ids
    ]

    if not cited_claims:
        return {
            "hallucinated": hallucinated,
            "per_claim": [],
            "format_score": format_score,
            "avg_f1": 0.0,
            "citation_reward": 0.4 * format_score,
        }

    def _concat(ids: List[str]) -> str:
        return "\n\n".join(citations[i] for i in ids if i in citations)

    pairs = [(text, _concat(ids)) for text, ids in cited_claims]
    user_prompt = _build_batch_citation_prompt(question, pairs)

    _recall_label = {1.0: "fully", 0.5: "partially", 0.0: "none"}
    _precision_label = {1.0: "relevant", 0.0: "irrelevant"}

    try:
        agent = get_judge_agent("citation_judge")
        raw = await agent.async_single_turn(
            user_prompt,
            _BATCH_CITATION_SYSTEM_PROMPT,
            None,
            50 * len(pairs) + 64,
        )
        scores = _parse_batch_citation_response(raw, len(pairs))
    except Exception as exc:
        LOGGER.warning(f"[verify_answer_local] Citation judge failed: {exc}")
        scores = [(0.0, 0.0)] * len(pairs)

    per_claim = []
    f1_scores = []
    for (text, ids), (recall, precision) in zip(cited_claims, scores):
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        f1_scores.append(f1)
        per_claim.append({
            "claim": text[:200],
            "citation_ids": ids,
            "recall": _recall_label.get(recall, "none"),
            "precision": _precision_label.get(precision, "irrelevant"),
        })

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    citation_reward = 0.6 * avg_f1 + 0.4 * format_score

    return {
        "hallucinated": hallucinated,
        "per_claim": per_claim,
        "format_score": format_score,
        "avg_f1": avg_f1,
        "citation_reward": citation_reward,
    }


# ---------------------------------------------------------------------------
# Step 3 – Main entry point
# ---------------------------------------------------------------------------

async def compute_verify_answer_feedback(
    full_response: str,
    question: str,
    mcp_parser_name: Optional[str] = None,
    use_general_rubric: bool = True,
    use_likert_rubric: bool = False,
    ground_truth: Optional[Dict[str, Any]] = None,
    pre_extracted_citations: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compute r0 reward for the draft answer inside a verify_answer call.

    Parameters
    ----------
    full_response    : full assistant text (prompt + all generated tokens,
                       including the verify_answer call itself)
    question         : original user question
    mcp_parser_name  : parser name for format/search-turn scoring
    use_general_rubric: score rubric quality with a general 4-criterion rubric
                       (does NOT require per-question rubric dicts)
    use_likert_rubric : use 1-10 Likert scale instead
    ground_truth     : optional full GT dict {query, rubrics, ...}
                       if provided AND rubrics present, use specific rubrics;
                       otherwise falls back to use_general_rubric=True
    pre_extracted_citations : optional {id: text} already extracted from full_response
                       by the caller (e.g. MCPTool._verify_answer_native extracts
                       prior snippets before calling this).  When None the function
                       parses citations itself (from <context> block or full text).

    Returns
    -------
    {
        "r0":                  float,   # weighted reward for draft answer
        "rubric_reward":       float,
        "citation_reward":     float,
        "format_reward":       float,
        "search_turns_reward": float,
        "num_search_turns":    int,
        "rubric_feedback": [           # per-rubric detail
            {"title", "score", "criterion", "reasoning", "evidence"}
        ],
        "citation_issues": {           # structured citation feedback
            "hallucinated": [...],
            "per_claim":    [...],
            "format_score": float,
            "avg_f1":       float,
            "citation_reward": float,
        },
        "draft_answer":        str,
        "all_judges_failed":   bool,
        "error":               str | None,
    }
    """
    result: Dict[str, Any] = {
        "r0": 0.0,
        "rubric_reward": 0.0,
        "citation_reward": 0.0,
        "format_reward": 0.0,
        "search_turns_reward": 0.0,
        "num_search_turns": 0,
        "rubric_feedback": [],
        "citation_issues": {},
        "draft_answer": None,
        "all_judges_failed": False,
        "error": None,
    }

    # ------------------------------------------------------------------
    # 1. Extract draft answer from inside the verify_answer call
    # ------------------------------------------------------------------
    draft_text, context_text, _call_citations = _extract_draft_from_verify_call(full_response)

    if not draft_text:
        result["error"] = "No verify_answer call found or draft answer is empty"
        return result

    result["draft_answer"] = draft_text

    # Resolve citations:
    #   pre_extracted_citations  – passed by MCPTool from prior snippets in full_response
    #                              (when native path is called BEFORE MCP injection)
    #   _call_citations          – parsed from <context> block inside the verify_answer call
    #                              (when the call already has injected context)
    # Priority: pre_extracted > call_context > empty
    if pre_extracted_citations:
        citations = pre_extracted_citations
    elif _call_citations:
        citations = _call_citations
    else:
        # Last resort: scan full response for any snippet/webpage blocks
        from open_instruct.search_rewards.utils.format_utils import extract_citations_from_context
        citations = extract_citations_from_context(full_response)

    # Build synthetic response so existing utilities work
    synthetic_response = _build_synthetic_response(draft_text, citations)

    # ------------------------------------------------------------------
    # 2. Format reward  (uses existing compute_format_reward)
    # ------------------------------------------------------------------
    format_reward = compute_format_reward(
        synthetic_response,
        mcp_parser_name=mcp_parser_name,
    )
    result["format_reward"] = format_reward

    # ------------------------------------------------------------------
    # 3. Search-turns reward  (count searches in context BEFORE this call)
    # ------------------------------------------------------------------
    # Use the full context (everything before verify_answer call) for search counting
    # so we capture all actual searches the model performed
    pre_va_context = full_response
    va_start = re.search(
        r'<call_tool[^>]*name=["\']?verify_answer["\']?', full_response
    )
    if va_start:
        pre_va_context = full_response[: va_start.start()]

    search_turns_reward, num_search_turns = score_num_in_context_search_turns(
        pre_va_context, mcp_parser_name=mcp_parser_name
    )
    result["search_turns_reward"] = search_turns_reward
    result["num_search_turns"] = num_search_turns

    # ------------------------------------------------------------------
    # 4. Rubric reward  (reuse _compute_rubric_scores_and_reward)
    # ------------------------------------------------------------------
    from open_instruct.search_rewards.longform_rubric_rewards import (
        _compute_rubric_scores_and_reward,
    )

    # Decide whether to use specific rubrics or fall back to general rubric
    has_specific_rubrics = (
        ground_truth is not None
        and ground_truth.get("rubrics")
        and not use_general_rubric
        and not use_likert_rubric
    )
    _gt_for_rubric: Dict[str, Any]
    if has_specific_rubrics:
        _gt_for_rubric = ground_truth  # type: ignore[assignment]
        _use_gen = False
        _use_likert_final = False
    else:
        # Fallback: general 4-criterion rubric — no per-question rubric dicts needed
        _gt_for_rubric = {
            "query": question,
            "rubrics": [],
            "rubrics_types": [],
        }
        _use_gen = not use_likert_rubric   # likert takes precedence over general
        _use_likert_final = use_likert_rubric

    try:
        (
            title_scores,
            per_rubric_scores,
            rubric_reward,
            persistent_reward,
            adaptive_reward,
            rubric_feedback_by_title,
            all_judges_failed,
        ) = await _compute_rubric_scores_and_reward(
            response=draft_text,          # score the draft answer
            ground_truth=_gt_for_rubric,
            use_general_rubric=_use_gen,
            use_likert_rubric=_use_likert_final,
        )
    except Exception as exc:
        LOGGER.warning(f"[verify_answer_local] Rubric scoring failed: {exc}")
        rubric_reward = 0.0
        rubric_feedback_by_title = {}
        all_judges_failed = True

    result["rubric_reward"] = rubric_reward
    result["all_judges_failed"] = all_judges_failed

    # Flatten rubric_feedback_by_title → list of feedback entries
    rubric_feedback_list = []
    for rubric_key, entries in rubric_feedback_by_title.items():
        for entry in entries:
            rubric_feedback_list.append({
                "title": entry.get("title") or rubric_key,
                "score": round(float(entry.get("score", 0.0)), 4),
                "criterion": entry.get("criterion", "")[:200],
                "reasoning": entry.get("reasoning", "")[:300],
                "evidence": entry.get("evidence", "")[:150],
            })
    result["rubric_feedback"] = rubric_feedback_list

    # ------------------------------------------------------------------
    # 5. Citation reward (structured)
    # ------------------------------------------------------------------
    citation_issues = await _compute_citation_issues_detailed(
        question=question,
        synthetic_response=synthetic_response,
        citations=citations,
    )
    result["citation_issues"] = citation_issues
    result["citation_reward"] = citation_issues["citation_reward"]

    # ------------------------------------------------------------------
    # 6. r0 = weighted sum  (same weights as training)
    # ------------------------------------------------------------------
    r0 = (
        _REWARD_WEIGHTS["rubric_reward"] * rubric_reward
        + _REWARD_WEIGHTS["citation_reward"] * citation_issues["citation_reward"]
        + _REWARD_WEIGHTS["format_reward"] * format_reward
        + _REWARD_WEIGHTS["num_search_turns_reward"] * search_turns_reward
    )
    result["r0"] = round(r0, 6)

    return result


# ---------------------------------------------------------------------------
# Step 4 – Format output for the model
# ---------------------------------------------------------------------------

def format_verify_answer_output(feedback: Dict[str, Any]) -> str:
    """
    Render structured feedback as human-readable tool output.
    Embeds the raw JSON at the end for downstream parsing by the training loop.

    Format
    ------
    === Draft Answer Evaluation ===
    Overall Score: 0.721

    [Rubric Assessment]
    1. [general_rubric]  ██████░░░░  0.6/1.0
       Criterion: (1) Overall Comprehensiveness...
       Assessment: The response covers the main points...
       Evidence: "...key excerpt..."

    [Citation Quality]
    ✗ Hallucinated citations: xyz123
    ~ "The wall was built..." (recall=partially, precision=relevant)
    ✓ "Construction began..." (recall=fully, precision=relevant)

    <!--EVAL_JSON:{...}-->
    """
    lines: List[str] = ["=== Draft Answer Evaluation ==="]
    lines.append(f"Overall Score: {feedback.get('r0', 0.0):.3f}  "
                 f"(rubric={feedback.get('rubric_reward', 0.0):.2f}  "
                 f"citation={feedback.get('citation_reward', 0.0):.2f}  "
                 f"format={feedback.get('format_reward', 0.0):.2f}  "
                 f"search={feedback.get('search_turns_reward', 0.0):.2f})")
    lines.append("")

    # -- Rubric section --
    rubric_feedback = feedback.get("rubric_feedback", [])
    if rubric_feedback:
        lines.append("[Rubric Assessment]")
        for i, entry in enumerate(rubric_feedback, 1):
            score = float(entry.get("score", 0.0))
            filled = int(score * 10)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"{i}. [{entry.get('title', 'criterion')}]  {bar}  {score:.1f}/1.0")
            criterion = entry.get("criterion", "")
            if criterion:
                lines.append(f"   Criterion: {criterion[:140]}")
            reasoning = entry.get("reasoning", "")
            if reasoning:
                lines.append(f"   Assessment: {reasoning[:240]}")
            evidence = entry.get("evidence", "")
            if evidence:
                lines.append(f"   Evidence: \"{evidence[:160]}\"")
            lines.append("")

    # -- Citation section --
    ci = feedback.get("citation_issues", {})
    if ci:
        lines.append("[Citation Quality]")
        hallucinated = ci.get("hallucinated", [])
        if hallucinated:
            lines.append(f"  ✗ Hallucinated citations (not in retrieved context): "
                         f"{', '.join(hallucinated)}")

        _recall_icon = {"fully": "✓", "partially": "~", "none": "✗"}
        for claim_info in ci.get("per_claim", []):
            icon = _recall_icon.get(claim_info.get("recall", "none"), "?")
            claim_text = claim_info.get("claim", "")
            recall = claim_info.get("recall", "?")
            precision = claim_info.get("precision", "?")
            cite_ids = ", ".join(claim_info.get("citation_ids", []))
            lines.append(
                f"  {icon} [{cite_ids}] \"{claim_text[:80]}...\"  "
                f"recall={recall}, precision={precision}"
            )

        if not hallucinated and not ci.get("per_claim"):
            lines.append("  (No <cite> tags found in draft answer)")
        lines.append("")

    if feedback.get("error"):
        lines.append(f"[Error] {feedback['error']}")

    # Embed raw JSON for training-side parsing
    _json_payload = {
        k: v for k, v in feedback.items()
        if k not in ("rubric_feedback",)  # avoid double-storing long text
    }
    lines.append(f"<!--EVAL_JSON:{json.dumps(_json_payload, ensure_ascii=False)}-->")

    return "\n".join(lines)
