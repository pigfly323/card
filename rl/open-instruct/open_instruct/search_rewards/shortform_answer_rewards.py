"""
shortform_answer_rewards.py

Reward function for short-answer QA tasks (HotpotQA, 2WikiMultiHop, SimpleQA).

Unlike long-form rubric rewards, this reward requires NO LLM judge calls:
correctness is computed by F1 / exact-match string comparison against the
ground-truth answer.

Reward components
-----------------
  answer_reward        (weight 0.8) — token-F1 or exact-match vs GT answer
  format_reward        (weight 0.1) — 1.0 iff final answer is in <answer> tags
  search_turns_reward  (weight 0.1) — search efficiency (fewer turns = higher)

Ground-truth schema (stored as JSON string in the training dataset):
  {"query": "...", "answer": "yes"}              # single string
  {"query": "...", "answer": ["yes", "no"]}      # list of valid answers

For yes/no comparison questions, exact-match is used after normalization.
For factoid span questions, token-F1 is used (handles partial matches such as
"3,677" vs "3,677 seated").
"""

import json
import re
import string
from typing import Any, Dict, List, Optional, Tuple

from open_instruct.search_rewards.utils.format_utils import (
    extract_answer_context_citations,
    compute_format_reward,
)
from open_instruct.search_rewards.utils.search_utils import (
    score_num_in_context_search_turns,
)

# Reward component weights
REWARD_WEIGHTS = {
    "answer_reward":       0.8,
    "format_reward":       0.1,
    "search_turns_reward": 0.1,
}

# Yes/no answers — use exact match (F1 can give partial credit here)
_YES_NO_ANSWERS = {"yes", "no"}


# ---------------------------------------------------------------------------
# Normalisation helpers (SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles / punctuation / extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (standard SQuAD metric)."""
    pred_tokens = _get_tokens(prediction)
    gt_tokens   = _get_tokens(ground_truth)
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 iff normalised prediction == normalised ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def _parse_ground_truth(ground_truth: Any) -> Tuple[List[str], str]:
    """
    Extract the list of valid answers from the ground truth.

    Returns (answers, query) where answers is a non-empty list of strings.
    Handles:
      - JSON string: '{"query": "...", "answer": "yes"}'
      - dict:        {"query": "...", "answer": ["yes", "no"]}
      - raw string:  "yes"
      - list:        ["yes"]
    """
    query = ""
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            # Treat as raw answer string
            return [ground_truth.strip()], query
    else:
        obj = ground_truth

    if isinstance(obj, dict):
        query = str(obj.get("query") or obj.get("Question") or "").strip()
        raw = obj.get("answer") or obj.get("answers") or obj.get("Answer") or ""
    elif isinstance(obj, list):
        raw = obj
    else:
        raw = str(obj).strip()

    if isinstance(raw, list):
        answers = [str(a).strip() for a in raw if str(a).strip()]
    else:
        answers = [str(raw).strip()] if str(raw).strip() else []

    return (answers or [""], query)


# ---------------------------------------------------------------------------
# Answer extraction from model response
# ---------------------------------------------------------------------------

def _extract_answer_text(response: str) -> Optional[str]:
    """
    Extract the final answer from the model response.

    Priority:
      1. Content inside <answer>...</answer> tags (strips think/tool blocks)
      2. Content inside \\boxed{...}
      3. None — no answer found
    """
    _, answer, _ = extract_answer_context_citations(response)
    if answer and answer.strip() not in {"...", "…"}:
        # Strip \boxed{} wrapper if present (added by some models)
        boxed = re.search(r"\\boxed\{(.+?)\}", answer, re.DOTALL)
        return boxed.group(1).strip() if boxed else answer.strip()

    # Fallback: \boxed{} directly in response
    boxed = re.search(r"\\boxed\{(.+?)\}", response, re.DOTALL)
    if boxed:
        return boxed.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Core answer correctness scoring
# ---------------------------------------------------------------------------

def score_answer(prediction: str, answers: List[str]) -> Tuple[float, str]:
    """
    Score prediction against all valid answers, return max score and method.

    For yes/no questions: exact match.
    For factoid questions: token-F1 (handles partial span matches).
    """
    if not prediction or not answers:
        return 0.0, "empty"

    pred_norm = normalize_answer(prediction)

    # Detect yes/no question
    is_yes_no = all(normalize_answer(a) in _YES_NO_ANSWERS for a in answers)

    best = 0.0
    method = "f1"
    for ans in answers:
        if is_yes_no:
            s = compute_exact_match(prediction, ans)
            method = "exact_match"
        else:
            s = compute_f1(prediction, ans)
            method = "f1"
        best = max(best, s)

    return best, method


# ---------------------------------------------------------------------------
# Main reward entry point
# ---------------------------------------------------------------------------

def compute_shortform_answer_reward(
    response: str,
    ground_truth: Any,
    mcp_parser_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute reward for a short-answer QA response.

    Parameters
    ----------
    response      : Full model-generated text (reasoning + tool calls + answer).
    ground_truth  : Ground truth — JSON string, dict, list, or raw string.
    mcp_parser_name: Parser name for search-turn counting (e.g. 'unified').

    Returns
    -------
    dict with keys:
      reward         — final scalar reward in [0, 1]
      answer_reward  — F1 / EM score vs GT
      format_reward  — answer format correctness
      search_turns_reward
      log_values     — dict of all components for W&B logging
      answer_extracted — the answer text found in the response (or None)
      error          — error string if extraction failed
    """
    result: Dict[str, Any] = {
        "reward":              0.0,
        "answer_reward":       0.0,
        "format_reward":       0.0,
        "search_turns_reward": 0.0,
        "answer_extracted":    None,
        "error":               None,
        "log_values":          {},
    }

    # ── 1. Parse ground truth ────────────────────────────────────────────────
    try:
        answers, query = _parse_ground_truth(ground_truth)
    except Exception as e:
        result["error"] = f"GT parsing failed: {e}"
        return result

    # ── 2. Search turns reward (always computed) ─────────────────────────────
    extracted_context, _, _ = extract_answer_context_citations(response)
    search_turns_reward, num_search_turns = score_num_in_context_search_turns(
        extracted_context, mcp_parser_name=mcp_parser_name
    )

    # ── 3. Format reward ─────────────────────────────────────────────────────
    fmt_reward = compute_format_reward(response, mcp_parser_name=mcp_parser_name)

    # ── 4. Answer extraction ─────────────────────────────────────────────────
    prediction = _extract_answer_text(response)
    if prediction is None:
        result["error"] = "No answer found in <answer> tags or \\boxed{}"
        result["format_reward"]       = fmt_reward
        result["search_turns_reward"] = search_turns_reward
        result["reward"] = (
            REWARD_WEIGHTS["format_reward"]       * fmt_reward
            + REWARD_WEIGHTS["search_turns_reward"] * search_turns_reward
        )
        result["log_values"] = {
            "answer_reward":       0.0,
            "format_reward":       fmt_reward,
            "search_turns_reward": search_turns_reward,
            "num_search_turns":    num_search_turns,
            "answer_extracted":    False,
        }
        return result

    result["answer_extracted"] = prediction

    # ── 5. Answer correctness ────────────────────────────────────────────────
    ans_reward, score_method = score_answer(prediction, answers)
    result["answer_reward"] = ans_reward

    # ── 6. Final weighted reward ─────────────────────────────────────────────
    reward = (
        REWARD_WEIGHTS["answer_reward"]       * ans_reward
        + REWARD_WEIGHTS["format_reward"]       * fmt_reward
        + REWARD_WEIGHTS["search_turns_reward"] * search_turns_reward
    )
    result["reward"]              = reward
    result["format_reward"]       = fmt_reward
    result["search_turns_reward"] = search_turns_reward
    result["log_values"] = {
        "answer_reward":       ans_reward,
        "format_reward":       fmt_reward,
        "search_turns_reward": search_turns_reward,
        "num_search_turns":    num_search_turns,
        "answer_extracted":    True,
        "score_method":        score_method,   # "f1" or "exact_match"
        "prediction_norm":     normalize_answer(prediction),
        "gt_answers":          answers,
    }
    return result
