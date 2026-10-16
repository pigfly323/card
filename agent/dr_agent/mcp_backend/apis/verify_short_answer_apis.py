"""
verify_short_answer_apis.py

Short-answer verify_answer backend aligned with verify_longform_answer_apis:

- policy-visible channel: feedback_text only
- hidden training channel: reward_info with r0 and rl_exact_answer_reward

Chosen design for this file:
- The judge model privately sees question / solution_path / predicted_answer /
  acceptable_answers / supporting_facts.
- The judge returns a binary score plus a short self-reflective critique.
- The visible output keeps only textual feedback; reward_info remains hidden and
  parseable for training.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from open_instruct.ground_truth_utils import f1_score, normalize_answer

from .verify_answer_apis import (
    _extract_json_from_response,
    _get_rubric_judge_agent,
    _RUBRIC_STORE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Critique-writing prompt: produces ONLY the visible self-reflection text.
# Scoring is handled separately by _grade_answer (strategy-configurable) — this
# agent does not grade, so its output contains no score field.
_SHORT_JUDGE_SYSTEM_PROMPT = """\
You are a self-verifier for short-answer QA problems.

You will be told whether the current answer is correct:
- If CORRECT: confirm that the retrieved evidence in the solution path supports the current answer. Point to the specific retrieved content (facts, snippets, documents) that backs the answer, in first person, e.g. "The search results confirm that ... so my answer is well supported."
- If INCORRECT: the current answer has been found wrong, so never confirm or praise it. Write a first-person reflection that pins down the concrete failure and the concrete next step, in this order:
  1. Re-read the question and identify which facet it actually asks about, then name the exact claim, entity, date, number, or judgment in the solution that fails. Common failures: answering a different facet, stopping one hop short, or picking the wrong entity among several.
  2. Re-examine the solution path and the supporting facts for evidence that contradicts the answer or already fills the missing piece, and point to it, e.g. "one of the snippets I retrieved already mentions ... which I overlooked" or "the supporting facts point to a different entity than the one I chose".
  3. End with ONE concrete next action in the present tense, e.g. "I need to search for the exact X of Y" or "I should compare the two entities on Z". Never end with a generic vow like "I should verify more carefully" or "I should double-check my sources" without naming what to check.
  For yes/no or either-or questions, say explicitly which entity or which side of the judgment is in doubt rather than discussing definitions in general.
- If UNCERTAIN (no reference was available for the check): judge the answer purely from your own expertise, using only the question, the solution path, and the predicted answer. Point out the most likely defect and give a concrete improvement direction, checking: whether the answer addresses the exact facet the question asks; whether each claim in the answer is actually backed by the retrieved evidence in the solution path (flag unsupported or only partially supported claims); whether the reasoning steps are internally consistent; and whether a hop or entity in the question is left unresolved. Do NOT state that the answer is confirmed or well supported when you have no reference to check it against.

If <acceptable_answers> or <supporting_facts> is marked "(not provided)", you have no reference information for that part — rely on your expertise over the question, the solution path, and the predicted answer as described for the UNCERTAIN case, and do not treat the missing section as an error to mention.

In all cases:
- Do NOT reveal or directly state the correct answer in any part of your response.
- Keep it actionable, compact, and natural.

Hard constraints:
- Treat the inputs as strictly private hidden checks. Do NOT mention or imply the existence of any previous answer, ground truth, or evaluation schema.
- Do NOT quote, closely paraphrase, or reproduce any hidden criterion description.
- Do NOT output bullet points.
- Do NOT provide chain-of-thought; only provide the final concise reflection.
- Prefer general wording like "I should check", "I may be overstating", "I need stronger evidence", "the retrieved evidence supports", rather than specialized evaluation language.
- Keep it to 2-4 sentences and under 120 words.


Return valid raw JSON only with the following field:
{
  "critique": "an actionable first-person reflection of the solution, without revealing the answer"
}
"""


_SHORT_JUDGE_USER_TEMPLATE = """\
<question>
{question}
</question>

<solution_path>
{solution_path}
</solution_path>

<predicted_answer>
{predicted_answer}
</predicted_answer>

<acceptable_answers>
{ground_truth}
</acceptable_answers>

<supporting_facts>
{supporting_facts}
</supporting_facts>

<verdict>
{verdict_line}
</verdict>
"""

# Verdict lines rendered into <verdict>. UNCERTAIN is used when no ground truth
# was available to grade against — the judge must then assess from expertise
# instead of being (mis)told the answer is wrong.
_VERDICT_LINES = {
    "CORRECT": "The current answer has been checked and it is CORRECT.",
    "INCORRECT": "The current answer has been checked and it is INCORRECT.",
    "UNCERTAIN": (
        "No reference answer was available for this check, so the correctness of "
        "the current answer is unknown. Assess it yourself from the question, the "
        "solution path, and the predicted answer."
    ),
}


_VISIBLE_CRITIQUE_META_MARKERS = (
    "acceptable answer",
    "correct answer",
    "reference answer",
    "ground truth",
    "predicted answer",
)

# Grader prompt aligned with the trainer's LLMJudgeVerifier (ground_truth_utils
# GRADER_TEMPLATE) so r0 (draft, graded here) and r1 (final answer, graded in
# training) are scored by the SAME prompt semantics — r1 - r0 then measures the
# true effect of feedback instead of inter-judge disagreement.
# The extraction step is kept even though only the verdict is parsed: it is what
# lets the grader reduce a verbose response ("Yes, both X and Y are documentaries...")
# to the comparable answer ("yes") — without it the grader compares the full
# sentence against a short GT and systematically answers 'no'.
_GRADER_SYSTEM_PROMPT = "You are a precise grading assistant. Output only the requested lines."
_GRADER_USER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

Output ONLY those two lines, nothing else."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSWER_EXTRACT_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)
_SEARCH_CALL_RE = re.compile(
    r'<call_tool\s+[^>]*name=["\']?(?:google_search|snippet_search|browse_webpage)["\']?',
    re.IGNORECASE,
)
_CITE_TAG_RE = re.compile(r'<cite\b[^>]*>', re.IGNORECASE)


def _extract_clean_answer(predicted_answer: str) -> str:
    text = predicted_answer or ""
    m = _ANSWER_EXTRACT_RE.search(text)
    if m:
        text = m.group(1)
    text = _BOXED_RE.sub(lambda mo: mo.group(1), text)
    return text.strip()


def _normalize_gt_to_list(ground_truth) -> List[str]:
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
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return [s]
        return _normalize_gt_to_list(parsed)
    return [str(ground_truth).strip()]


def _format_list_for_prompt(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "\n".join(f"- {x}" for x in items)


def _compute_format_reward(solution_path: str, predicted_answer: str) -> float:
    answer_fmt = 1.0 if _ANSWER_EXTRACT_RE.search(predicted_answer or "") else 0.0
    cite_fmt = 1.0 if _CITE_TAG_RE.search(solution_path or "") else 0.0
    query_fmt = 1.0 if _SEARCH_CALL_RE.search(solution_path or "") else 0.0
    return 0.5 * answer_fmt + 0.3 * cite_fmt + 0.2 * query_fmt


def _compute_search_turns_reward(solution_path: str) -> float:
    n = len(_SEARCH_CALL_RE.findall(solution_path or ""))
    return min(1.0, n / 3.0)


def _compute_rl_exact_answer_reward(clean_answer: str, gt_list: List[str]) -> float:
    """F1-based answer reward, aligned with the trainer's r1 verifier
    (rl_rag_exact_answer, --verifier_strategy f1): max F1 over GT candidates,
    with the same 0.1 floor for well-formatted but wrong answers.

    A hard-EM-only comparison penalizes verbose drafts that contain the correct
    answer (e.g. draft "Deepdale Golf Club is located in Nassau County." vs GT
    "Nassau County"): r0 = 0 while the trainer's F1 gives r1 > 0, systematically
    suppressing A0 and desynchronizing the r0-baseline from r1.
    """
    if not clean_answer or not gt_list:
        return 0.0
    score = max(f1_score(clean_answer, str(gt))["f1"] for gt in gt_list)
    if score == 0:
        # Format-correct floor, mirrors RLRAGExactAnswerVerifier's
        # "if f1 == 0 but format is correct, return 0.1".
        return 0.1
    return float(score)


def _sanitize_supporting_facts(metadata: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    sf = metadata.get("supporting_facts")
    if not isinstance(sf, dict):
        return []

    titles = sf.get("title") if isinstance(sf.get("title"), list) else []
    seen = set()
    ordered_titles: List[str] = []
    for t in titles:
        s = str(t).strip()
        if s and s not in seen:
            seen.add(s)
            ordered_titles.append(s)
    return ordered_titles


def _validate_short_judge_result(obj: Any) -> Optional[str]:
    """Validate the critique agent's JSON output. Returns the critique text or None.

    The critique agent no longer grades (scoring lives in _grade_answer), so
    only the 'critique' field is required. A stale 'score' field — e.g. from a
    judge still following the old prompt — is ignored.
    """
    if not isinstance(obj, dict):
        return None

    critique = str(obj.get("critique", "") or "").strip()
    critique = re.sub(r"\s+", " ", critique)
    if not critique:
        return None
    if len(critique) > 1200:
        critique = critique[:1200].strip()

    return critique


def _normalized_contains(haystack: str, needle: str) -> bool:
    hay = normalize_answer(haystack or "")
    ned = normalize_answer(needle or "")
    if not hay or not ned:
        return False
    return ned in hay


def _gt_ngram_leak(text_norm: str, gt: str, draft_norm: str) -> bool:
    """Detect partial leaks of a multi-word ground truth.

    The full-GT containment check misses critiques that leak only the
    distinctive part of a long GT (observed in HotpotQA: GT "Greenwich Village,
    New York City" leaked as "Greenwich Village"). Flag any run of 3+
    consecutive normalized GT tokens appearing verbatim in the critique but NOT
    already present in the draft — repeating what the draft itself said leaks
    nothing new, so near-correct drafts whose critiques re-quote their entities
    are not rejected. 2-gram overlaps (e.g. "New York") are tolerated to avoid
    false rejections.
    """
    gt_tokens = normalize_answer(str(gt)).split()
    if len(gt_tokens) < 4:
        return False  # short GTs are covered by the full-containment check
    for n in range(3, len(gt_tokens) + 1):
        for i in range(len(gt_tokens) - n + 1):
            gram = " ".join(gt_tokens[i:i + n])
            if gram in text_norm and gram not in draft_norm:
                return True
    return False


def _sanitize_visible_critique(
    critique: str,
    *,
    clean_answer: str,
    gt_list: List[str],
) -> Optional[str]:
    """Return the critique when it is genuine, safe judge output; None otherwise.

    Rejections (not self-reflective, meta-answer markers, echoed predicted
    answer, leaked acceptable answer) mean the visible text is NOT usable
    judge feedback — the caller marks the whole verification as failed
    (feedback_text=None) instead of substituting a canned string.
    """
    text = re.sub(r"\s+", " ", str(critique or "").strip())
    if not text:
        return None

    lower = text.lower()
    if not re.search(r"\b(i|i'm|i’ll|i should|i need|i can|i must|my)\b", lower):
        logger.warning("[verify_short_answer] critique rejected: not self-reflective")
        return None

    if any(marker in lower for marker in _VISIBLE_CRITIQUE_META_MARKERS):
        logger.warning("[verify_short_answer] critique rejected: meta-answer marker detected")
        return None

    clean_answer_norm = normalize_answer(clean_answer or "")
    if len(clean_answer_norm) >= 8 and _normalized_contains(text, clean_answer):
        logger.warning("[verify_short_answer] critique rejected: echoed predicted answer")
        return None

    for gt in gt_list:
        gt_norm = normalize_answer(str(gt))
        if len(gt_norm) >= 4 and _normalized_contains(text, str(gt)):
            logger.warning("[verify_short_answer] critique rejected: leaked acceptable answer")
            return None

    text_norm = normalize_answer(text)
    draft_norm = normalize_answer(clean_answer or "")
    for gt in gt_list:
        if _gt_ngram_leak(text_norm, str(gt), draft_norm):
            logger.warning("[verify_short_answer] critique rejected: leaked part of acceptable answer")
            return None

    return text


def _build_reward_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "verify_short_answer_v2",
        "verifier": "verify_short_answer",
        "r0": float(payload.get("r0", 0.0) or 0.0),
        "rl_exact_answer_reward": float(payload.get("rl_exact_answer_reward", 0.0) or 0.0),
        "format_reward": float(payload.get("format_reward", 0.0) or 0.0),
        "num_search_turns_reward": float(payload.get("num_search_turns_reward", 0.0) or 0.0),
        "judge_score": float(payload.get("judge_score", 0.0) or 0.0),
        "grade_strategy": str(payload.get("grade_strategy", "") or ""),
        "used_ground_truth_candidates": int(payload.get("used_ground_truth_candidates", 0) or 0),
        "reflection_summary": str(payload.get("reflection_summary", "") or ""),
        # True only when the visible feedback is genuine judge-model output.
        # The RL trainer drops trajectories whose verify_answer feedback is
        # not valid (feedback_text=None / feedback_valid=False).
        "feedback_valid": bool(payload.get("feedback_valid")) and not payload.get("error"),
        "error": payload.get("error"),
    }


def _format_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    """feedback_text is the genuine judge reflection, or None on failure.

    None (never a canned fallback string) lets downstream layers distinguish
    real judge feedback from failures by a simple emptiness check.
    """
    feedback_valid = bool(payload.get("feedback_valid")) and not payload.get("error")
    feedback_text = str(payload.get("reflection_summary", "") or "").strip() if feedback_valid else ""
    return {
        "feedback_text": feedback_text or None,
        "reward_info": _build_reward_info(payload),
    }


def _get_grading_strategy() -> str:
    """r0 scoring strategy: 'judge' | 'f1' | 'em'.

    Defaults to 'judge' (aligned with the trainer's --verifier_strategy judge):
    the draft is graded with the same GRADER_TEMPLATE semantics as the final
    answer, so r1 - r0 isolates the effect of feedback rather than measuring
    disagreement between two different judges.
    """
    return os.environ.get("VERIFY_SHORT_ANSWER_STRATEGY", "judge").strip().lower()


async def _grade_answer(
    *,
    problem: str,
    clean_answer: str,
    gt_list: List[str],
) -> Tuple[float, bool]:
    """Grade the draft answer. Returns (score, grade_ok).

    Strategies (VERIFY_SHORT_ANSWER_STRATEGY):
      - judge: LLM grader with the trainer-aligned GRADER_TEMPLATE; binary
               1.0/0.0, 0.1 floor for empty-but-well-formatted output.
      - f1:    token-F1 vs GT candidates (trainer's f1 strategy), 0.1 floor.
      - em:    normalized exact match (trainer's em strategy), no partial credit.
    """
    if not clean_answer or not gt_list:
        return 0.0, False

    strategy = _get_grading_strategy()
    if strategy == "f1":
        return _compute_rl_exact_answer_reward(clean_answer, gt_list), True
    if strategy == "em":
        return float(
            max(normalize_answer(clean_answer) == normalize_answer(str(gt)) for gt in gt_list)
        ), True

    # judge strategy: same prompt semantics as the trainer's LLMJudgeVerifier.
    agent = _get_rubric_judge_agent()
    if agent is None:
        logger.warning("[verify_short_answer] grader agent unavailable")
        return 0.0, False

    # Grade against each GT candidate; take the max (mirrors the trainer's
    # max-over-labels behavior).
    async def _grade_one(gt: str) -> float:
        user_prompt = _GRADER_USER_TEMPLATE.format(
            question=problem.strip(),
            response=clean_answer,
            correct_answer=str(gt),
        )
        # extraction + verdict; 128 is ample headroom and keeps the grader cheap.
        max_tokens = int(os.environ.get("VERIFY_SHORT_ANSWER_GRADER_MAX_TOKENS", "128"))
        try:
            raw = await agent.async_single_turn(
                user_prompt,
                _GRADER_SYSTEM_PROMPT,
                None,
                max_tokens,
                False,
            )
        except Exception as e:
            logger.warning("[verify_short_answer] grader call failed: %s", e)
            return -1.0  # sentinel: call failed (distinguishable from score 0)
        text = raw if isinstance(raw, str) else str(raw or "")
        # Negative look-behind so "incorrect: yes" inside reasoning does not
        # match (only the standalone "correct:" verdict line counts).
        m = re.search(r"(?<![a-z])correct:\s*(yes|no)", text, re.IGNORECASE)
        if m is None:
            logger.warning("[verify_short_answer] grader output unparseable: %r", text[:120])
            return -1.0
        return 1.0 if m.group(1).lower() == "yes" else 0.0

    scores = await asyncio.gather(*(_grade_one(gt) for gt in gt_list))
    valid = [s for s in scores if s >= 0]
    if not valid:
        return 0.0, False
    # A graded 'no' on all candidates with no call failure → 0.0 (still valid).
    return max(valid), True


async def _write_critique(
    *,
    problem: str,
    clean_answer: str,
    gt_list: List[str],
    solution_path: str,
    supporting_facts: List[str],
    grade_score: Optional[float] = None,
) -> Tuple[Optional[str], bool]:
    """Write the self-reflective critique shown to the policy model.

    Returns (critique, critique_ok). critique_ok is True only when the judge
    agent ran, its output parsed, and the critique passed sanitization.

    When grade_score is provided and >= 1.0, the critique agent is told the
    answer is CORRECT and asked to confirm that the retrieved evidence
    supports it (instead of hunting for issues).

    When gt_list is empty (no acceptable answers available), the verdict is
    UNCERTAIN and the judge is asked to diagnose defects from its own expertise
    over question / solution_path / predicted_answer — missing hidden inputs
    are rendered as "(not provided)" rather than blocking the critique.
    """
    if not clean_answer:
        return None, False

    agent = _get_rubric_judge_agent()
    if agent is None:
        logger.warning("[verify_short_answer] critique agent unavailable")
        return None, False

    if grade_score is not None and grade_score >= 1.0:
        verdict = "CORRECT"
    elif not gt_list:
        verdict = "UNCERTAIN"
    else:
        verdict = "INCORRECT"

    user_prompt = _SHORT_JUDGE_USER_TEMPLATE.format(
        question=problem.strip(),
        solution_path=solution_path.strip(),
        predicted_answer=clean_answer,
        ground_truth=_format_list_for_prompt(gt_list) if gt_list else "(not provided)",
        supporting_facts=(
            _format_list_for_prompt(supporting_facts) if supporting_facts else "(not provided)"
        ),
        verdict_line=_VERDICT_LINES[verdict],
    )

    max_tokens = int(os.environ.get("VERIFY_SHORT_ANSWER_MAX_TOKENS", "256"))
    try:
        raw = await agent.async_single_turn(
            user_prompt,
            _SHORT_JUDGE_SYSTEM_PROMPT,
            None,
            max_tokens,
            True,
        )
    except Exception as e:
        logger.warning("[verify_short_answer] critique generation failed: %s", e)
        return None, False

    obj = raw if isinstance(raw, dict) else _extract_json_from_response(raw)
    raw_critique = _validate_short_judge_result(obj)
    if raw_critique is None:
        logger.warning("[verify_short_answer] critique output unparseable")
        return None, False

    critique = _sanitize_visible_critique(
        raw_critique,
        clean_answer=clean_answer,
        gt_list=gt_list,
    )
    if not critique:
        return None, False
    return critique, True


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def verify_short_answer_handler(
    problem: str,
    predicted_answer: str,
    ground_truth="",
    solution_path: str = "",
) -> Dict[str, Any]:
    """Short-answer verify_answer with visible reflection + hidden exact-answer reward."""
    clean_answer = _extract_clean_answer(predicted_answer)
    format_reward = _compute_format_reward(solution_path, predicted_answer)
    search_turns_reward = _compute_search_turns_reward(solution_path)

    gt_list = _normalize_gt_to_list(ground_truth)
    if not gt_list and problem:
        store_gt = _RUBRIC_STORE.get_ground_truth(problem)
        if store_gt:
            gt_list = list(store_gt)

    metadata = _RUBRIC_STORE.get_ground_truth_metadata(problem) if problem else None
    supporting_facts = _sanitize_supporting_facts(metadata)

    if not predicted_answer or not predicted_answer.strip():
        return _format_output(
            {
                "r0": 0.0,
                "rl_exact_answer_reward": 0.0,
                "judge_score": 0.0,
                "format_reward": format_reward,
                "num_search_turns_reward": search_turns_reward,
                "used_ground_truth_candidates": len(gt_list),
                "reflection_summary": None,
                "feedback_valid": False,
                "error": "No predicted answer provided",
            }
        )

    # Score first, then critique: the critique prompt branches on the verdict
    # (CORRECT → confirm retrieved evidence supports the answer; INCORRECT →
    # actionable first-person reflection), so it must wait for the grade.
    grade_score, grade_ok = await _grade_answer(
        problem=problem, clean_answer=clean_answer, gt_list=gt_list,
    )
    if grade_ok and grade_score >= 1.0:
        # Fully correct — skip the feedback model entirely; the critique adds
        # nothing but latency and risk of contradicting a correct answer.
        critique = "My answer is well supported."
        critique_ok = True
    else:
        critique, critique_ok = await _write_critique(
            problem=problem,
            clean_answer=clean_answer,
            gt_list=gt_list,
            solution_path=solution_path,
            supporting_facts=supporting_facts,
            grade_score=grade_score if grade_ok else None,
        )

    rl_exact_answer_reward = _compute_rl_exact_answer_reward(clean_answer, gt_list)
    grade_strategy = _get_grading_strategy()
    if grade_ok:
        r0 = grade_score
    else:
        # Grading failed (agent unavailable / all grader calls errored) —
        # fall back to F1 so r0 is still comparable across the group.
        r0 = rl_exact_answer_reward
        grade_strategy += "+f1_fallback"

    payload = {
        "r0": float(r0),
        "rl_exact_answer_reward": float(rl_exact_answer_reward),
        "judge_score": float(grade_score),
        "format_reward": float(format_reward),
        "num_search_turns_reward": float(search_turns_reward),
        "used_ground_truth_candidates": len(gt_list),
        "grade_strategy": grade_strategy,
        # None when the critique failed / was rejected — feedback_text becomes None.
        "reflection_summary": critique,
        "feedback_valid": critique_ok,
        "error": None,
    }

    logger.info(
        "[verify_short_answer] done r0=%.4f strategy=%s exact=%.3f grade_ok=%s critique_ok=%s fmt=%.3f search=%.3f gt=%d",
        r0,
        grade_strategy,
        rl_exact_answer_reward,
        grade_ok,
        critique_ok,
        format_reward,
        search_turns_reward,
        len(gt_list),
    )
    return _format_output(payload)
