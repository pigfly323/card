# grpo_lff.py — Learning from Feedback (LFF) training script.
# Forked from grpo_fast_rubric.py. Key changes:
#   1. GRPO: symmetric clipping (clip=0.2), sequence-level loss normalization
#   2. Two-segment advantage estimation using verify_answer feedback:
#      Seg-0 (pre-VA+VA_call): A0 = r0 - mean(r0), ratio = π_θ/π_θ_old
#      Seg-1 (post-VA):        A1 = r1 - mean(r0), ratio = π_θ(y1|f)/π_θ(y1)
#      VA_output tokens:       masked, no gradient
#   3. r0 = draft-answer reward (verify_answer call content + prior search context)
#      r1 = final-answer reward (standard verifiable reward)
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass
# isort: on

import asyncio
import atexit
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import yaml
import traceback
import wandb
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
import torch
import torch.utils
import torch.utils.data
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams

from open_instruct.dataset_transformation import (
    DATASET_ORIGIN_KEY,
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_USER_QUERY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    cleanup_all_llm_judge_clients,
    soft_format_reward_func,
    is_a_good_rl_rag_response,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.rl_utils2 import Timer, pack_sequences

# ── LFF: verify_answer response parsing utilities ─────────────────────────────
_LFF_VA_CONTENT_RE = re.compile(
    r'<call_tool\s+[^>]*name=["\']?verify_answer["\']?[^>]*>(.*?)</call_tool>',
    re.DOTALL | re.IGNORECASE,
)
_LFF_VA_CALL_RE = re.compile(
    r'<call_tool\s+[^>]*name=["\']?verify_answer["\']?[^>]*>',
    re.IGNORECASE | re.DOTALL,
)
_LFF_TOOL_OUTPUT_RE = re.compile(
    r'<tool_output>.*?</tool_output>|<output>.*?</output>',
    re.DOTALL | re.IGNORECASE,
)
_LFF_TOOL_OUTPUT_CAPTURE_RE = re.compile(
    r'<tool_output>(.*?)</tool_output>|<output>(.*?)</output>',
    re.DOTALL | re.IGNORECASE,
)
_LFF_THINK_RE = re.compile(
    r'<think>.*?</think>',
    re.DOTALL | re.IGNORECASE,
)
_LFF_ANSWER_RE = re.compile(
    r'<answer\b[^>]*>.*?</answer>',
    re.DOTALL | re.IGNORECASE,
)


def _lff_extract_draft_answer(response: str) -> Optional[str]:
    """Extract content from the FIRST verify_answer call."""
    m = _LFF_VA_CONTENT_RE.search(response)
    return m.group(1).strip() if m else None


# Matches a JSON rubric score line emitted by verify_answer_apis _score_one.
# Each line is a JSON object: {"title": ..., "weight": ..., "score": ..., ...}
_LFF_VA_RUBRIC_JSON_RE = re.compile(r'^\{.*"title".*"score".*\}$')


def _lff_extract_va_tool_output(response: str) -> Optional[str]:
    """
    Return the raw text of the FIRST verify_answer tool_output block.
    Looks for the <tool_output>...</tool_output> that immediately follows the
    first verify_answer <call_tool> tag.
    """
    va_m = _LFF_VA_CALL_RE.search(response)
    if va_m is None:
        return None
    # Find the first <tool_output> after the VA call end
    out_m = re.search(
        r'<tool_output>(.*?)</tool_output>',
        response[va_m.end():],
        re.DOTALL | re.IGNORECASE,
    )
    return out_m.group(1).strip() if out_m else None


def _lff_parse_r0_from_va_output(
    va_output: str,
    draft_response: str,
    ground_truth: dict,
    verification_reward: float = 10.0,
    mcp_parser_name: Optional[str] = None,
) -> Optional[Dict]:
    """
    Compute r0 (draft-answer reward) by parsing the verify_answer tool_output
    instead of calling the judge model a second time.

    Returns a dict with keys:
      r0          : float, final scaled reward in [0, verification_reward]
      rubric_reward, format_reward, search_turns_reward : float components [0,1]
      n_rubrics_parsed : int, number of rubric lines successfully parsed
      n_rubrics_total  : int, total rubrics in ground_truth
      n_rubrics_skipped: int, rubrics missing from output (scored as 0)
      title_scores     : Dict[str, float], per-title normalised score [0,1]
    Returns None if parsing is impossible (no rubrics / no <answer> tag).
    """
    # ground_truth must be a dict; use `is None` rather than `not ground_truth`
    # so that short-answer tasks whose GT is wrapped as {"answer": "92%"} (a
    # non-empty dict) are not rejected by the falsy-empty-dict check.
    if not va_output or ground_truth is None or not isinstance(ground_truth, dict):
        return None

    rubrics = ground_truth.get("rubrics", [])
    if not rubrics:
        # ── Short-answer path ────────────────────────────────────────────────
        # _verify_answer_native_short embeds parseable lines in the VA output:
        #   Answer reward: X.XXXX  ← LLM judge correctness score
        #   Format reward: X.XXXX
        #   Search score:  X.XXXX
        #   Rubric reward: X.XXXX  ← combined r0 (already weighted)
        _answer_reward:       Optional[float] = None
        _format_reward_short: Optional[float] = None
        _search_reward_short: Optional[float] = None
        _rubric_reward_short: Optional[float] = None

        for line in va_output.splitlines():
            line = line.strip()
            if line.startswith("Answer reward:"):
                try:
                    _answer_reward = float(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("Format reward:"):
                try:
                    _format_reward_short = float(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("Search score:"):
                try:
                    _search_reward_short = float(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("Rubric reward:"):
                try:
                    _rubric_reward_short = float(line.split(":", 1)[1])
                except ValueError:
                    pass

        # Return None if none of the short-answer lines were found
        # (allows caller to fall back gracefully)
        if _rubric_reward_short is None and _answer_reward is None:
            return None

        r0_raw   = _rubric_reward_short if _rubric_reward_short is not None else (
            0.8 * (_answer_reward or 0.0)
            + 0.1 * (_format_reward_short or 0.0)
            + 0.1 * (_search_reward_short or 0.0)
        )
        r0_raw   = max(0.0, min(1.0, r0_raw))
        return {
            "r0":                  float(r0_raw * verification_reward),
            "rubric_reward":       float(_answer_reward or 0.0),   # judge score
            "citation_reward":     0.0,
            "format_reward":       float(_format_reward_short or 0.0),
            "search_turns_reward": float(_search_reward_short or 0.0),
            "n_rubrics_parsed":    0,
            "n_rubrics_total":     0,
            "n_rubrics_skipped":   0,
            "title_scores":        {},
        }

    from open_instruct.search_rewards.longform_rubric_rewards import (
        REWARD_WEIGHTS,
        REWARD_WEIGHTS_WITHOUT_CITATION,
    )
    from open_instruct.search_rewards.utils.format_utils import (
        extract_answer_context_citations,
        compute_format_reward,
    )
    from open_instruct.search_rewards.utils.search_utils import score_num_in_context_search_turns

    # ── 1. Parse rubric_reward from tool_output ───────────────────────────────
    # compute_rubric_issues now appends "Rubric reward: X.XXXX" as the last line
    # and only includes JSON lines for imperfect rubrics (score < 1.0 for positive,
    # score > 0.0 for negative). Parse the summary line first; fall back to per-rubric
    # JSON parsing for backward compatibility with older tool_output formats.

    _rubric_reward_from_summary: Optional[float] = None
    title_scores: Dict[str, float] = {}   # lower-cased title → normalised [0,1]
    title_weights: Dict[str, float] = {}  # lower-cased title → weight

    for line in va_output.splitlines():
        line = line.strip()
        # Summary line takes priority
        if line.startswith("Rubric reward:"):
            try:
                _rubric_reward_from_summary = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            t = str(obj.get("title", "")).strip().lower()
            # score is raw 0/1/2 (matches header "2=fully met"); divide by 2 to normalize
            norm_score = max(0.0, min(1.0, float(obj.get("score", 0.0)) / 2.0))
            w = float(obj.get("weight", 1.0))
            if t:
                title_scores[t] = norm_score
                title_weights[t] = w
        except (json.JSONDecodeError, ValueError):
            continue

    n_parsed  = len(title_scores)
    n_total   = len(rubrics)
    n_skipped = sum(
        1 for r in rubrics
        if str(r.get("title", "")).strip().lower() not in title_scores
    )

    # ── 2. Compute weighted rubric_reward ─────────────────────────────────────
    if _rubric_reward_from_summary is not None:
        # Use the pre-computed aggregate (includes all rubrics, even perfect-silent ones)
        rubric_reward = max(0.0, min(1.0, _rubric_reward_from_summary))
    else:
        # Fallback: compute from per-rubric JSON lines (older format or partial output)
        total_pos_weight = 0.0
        weighted_sum = 0.0
        for rubric in rubrics:
            weight = float(rubric.get("weight", 1.0))
            title  = str(rubric.get("title", "")).strip().lower()
            norm_score = title_scores.get(title, 0.0)
            if weight > 0:
                total_pos_weight += weight
                weighted_sum += norm_score * weight
            else:
                # Negative-weight: norm_score=1.0 means bad behaviour IS present → penalty
                weighted_sum += norm_score * weight
        rubric_reward = weighted_sum / max(total_pos_weight, 1.0)
        rubric_reward = max(0.0, min(1.0, rubric_reward))

    # ── 3. Format + search-turns reward from draft_response ───────────────────
    # Use the same scoring logic as r1: format_utils handles missing <answer> tags
    # by returning 0 for answer_format_reward. Do NOT return None here — that would
    # fallback r0 to r1, preventing the negative A0 penalty the model deserves.
    extracted_context, extracted_answer, _ = extract_answer_context_citations(draft_response)
    format_reward = compute_format_reward(draft_response, mcp_parser_name=mcp_parser_name)
    search_turns_reward, _ = score_num_in_context_search_turns(
        extracted_context or "", mcp_parser_name=mcp_parser_name
    )

    # ── 4. Citation reward from verify_answer tool_output ────────────────────
    # compute_citation_issues now always emits one line per cited claim:
    #   Claim N: "..." [recall_label] [precision_label]
    # recall_label ∈ {"Fully supported", "Partially supported", "Not supported"}
    # precision_label ∈ {"Relevant", "Irrelevant"}
    # Score: Fully supported+Relevant→1.0, Partially→0.5, otherwise→0.0
    _CITATION_LINE = re.compile(
        r"^Claim\s+\d+:.*?\[(?P<recall>[^\]]+)\]\s*\[(?P<precision>[^\]]+)\]",
        re.IGNORECASE,
    )
    citation_scores: list = []
    for line in va_output.splitlines():
        m = _CITATION_LINE.match(line.strip())
        if not m:
            continue
        recall = m.group("recall").strip()
        precision = m.group("precision").strip()
        if recall == "Fully supported" and precision != "Irrelevant":
            citation_scores.append(1.0)
        elif recall == "Partially supported" and precision != "Irrelevant":
            citation_scores.append(0.5)
        else:
            citation_scores.append(0.0)

    has_citation = len(citation_scores) > 0
    citation_reward = (sum(citation_scores) / len(citation_scores)) if has_citation else 0.0

    # ── 5. Combine using same weights as r1 ───────────────────────────────────
    weights = REWARD_WEIGHTS if has_citation else REWARD_WEIGHTS_WITHOUT_CITATION
    reward = weights["rubric_reward"] * rubric_reward + weights["format_reward"] * format_reward + weights["num_search_turns_reward"] * search_turns_reward
    if has_citation:
        reward += weights["citation_reward"] * citation_reward
    reward = max(0.0, min(1.0, reward))

    return {
        "r0":                  float(reward * verification_reward),
        "rubric_reward":       float(rubric_reward),
        "citation_reward":     float(citation_reward),
        "format_reward":       float(format_reward),
        "search_turns_reward": float(search_turns_reward),
        "n_rubrics_parsed":    n_parsed,
        "n_rubrics_total":     n_total,
        "n_rubrics_skipped":   n_skipped,
        "title_scores":        dict(title_scores),
    }


def _lff_ensure_answer_tags(answer_text: str) -> str:
    answer_text = answer_text.strip()
    if not _LFF_ANSWER_RE.search(answer_text):
        answer_text = f"<answer>{answer_text}</answer>"
    return answer_text


def _lff_build_draft_response(response: str, draft_answer: str) -> str:
    """
    Build a synthetic response for r0 scoring:
    keep everything up to the first verify_answer call (including search context),
    then append the draft answer as-is (preserving any <answer> tags the model wrote).
    The draft_answer is the raw content of the verify_answer call — if the model
    wrapped it in <answer></answer> tags, those are preserved so reward_fn can
    extract them normally. If not, reward_fn will return ~0 (no answer found).
    """
    m = _LFF_VA_CONTENT_RE.search(response)
    if m is None:
        return _lff_ensure_answer_tags(draft_answer)
    prefix = response[:m.start()].rstrip()
    draft_answer = _lff_ensure_answer_tags(draft_answer)
    return f"{prefix}\n{draft_answer}"


def _lff_extract_va_events(response: str) -> List[Dict[str, Optional[Union[int, str]]]]:
    """
    Extract verify_answer call/output events in textual order.

    Each event contains:
      - call_start / call_end: char span of the <call_tool ...verify_answer...> block
      - call_content: raw inner content of the verify_answer call
      - call_content_wrapped: same content, guaranteed to contain <answer> tags
      - output_text: raw text inside the first following <tool_output>/<output>
      - output_start / output_end: char span of that output block, if present
    """
    events: List[Dict[str, Optional[Union[int, str]]]] = []
    for m in _LFF_VA_CONTENT_RE.finditer(response):
        after = response[m.end():]
        out_m = _LFF_TOOL_OUTPUT_CAPTURE_RE.search(after)
        output_text: Optional[str] = None
        output_start: Optional[int] = None
        output_end: Optional[int] = None
        if out_m is not None:
            output_text = (out_m.group(1) if out_m.group(1) is not None else out_m.group(2))
            output_text = output_text.strip() if output_text is not None else None
            output_start = m.end() + out_m.start()
            output_end = m.end() + out_m.end()

        call_content = (m.group(1) or "").strip()
        events.append(
            {
                "call_start": m.start(),
                "call_end": m.end(),
                "call_content": call_content,
                "call_content_wrapped": _lff_ensure_answer_tags(call_content),
                "output_text": output_text,
                "output_start": output_start,
                "output_end": output_end,
            }
        )
    return events


def _lff_multi_va_debug_summary(
    response: str,
    events: List[Dict[str, Optional[Union[int, str]]]],
    max_preview_chars: int = 220,
) -> str:
    """Compact debug summary for the first two verify_answer events."""
    def _clip(x: Optional[Union[int, str]]) -> str:
        s = "" if x is None else str(x)
        s = s.replace("\n", "\\n")
        return s[:max_preview_chars] + ("..." if len(s) > max_preview_chars else "")

    parts = [f"n_va_events={len(events)}"]
    for idx, ev in enumerate(events[:2], start=1):
        call_raw = ev.get("call_content")
        call_wrapped = ev.get("call_content_wrapped")
        out_text = ev.get("output_text")
        parts.append(
            f"va{idx}: call_len={len(str(call_raw or ''))} "
            f"wrapped_len={len(str(call_wrapped or ''))} "
            f"feedback_len={len(str(out_text or ''))} "
            f"call={_clip(call_wrapped)} "
            f"feedback={_clip(out_text)}"
        )
    parts.append(f"response_prefix={_clip(response[:600])}")
    return "\n".join(parts)


def _lff_build_token_char_offsets(
    response_ids: List[int],
    tokenizer,
) -> List[Tuple[int, int]]:
    """
    Build approximate char offsets for each token in `response_ids` using the
    tokenizer's own prefix decoding on the EXACT token sequence.

    This is intentionally slower than using offset_mapping, but it is robust to
    the tool XML spans in generated responses and does not require a fast
    tokenizer.  Response lengths are short enough that O(T^2) prefix decoding is
    acceptable here.
    """
    offsets: List[Tuple[int, int]] = []
    prev_text = ""
    for t in range(len(response_ids)):
        cur_text = tokenizer.decode(response_ids[: t + 1], skip_special_tokens=True)
        offsets.append((len(prev_text), len(cur_text)))
        prev_text = cur_text
    return offsets


def _lff_char_pos_to_token_start(
    char_pos: int,
    token_offsets: List[Tuple[int, int]],
    fallback_token_idx: int,
) -> int:
    """Map a decoded-text char position to the first token whose span reaches it."""
    if char_pos <= 0:
        return 0
    for tok_idx, (_, end) in enumerate(token_offsets):
        if end > char_pos:
            return tok_idx
    return fallback_token_idx


def _lff_encode_response_with_tool_mask(
    response_text: str,
    tokenizer,
    tool_char_spans: List[Tuple[int, int]],
) -> Tuple[List[int], List[int]]:
    """
    Encode a rewritten decoded response and reconstruct a response-level tool mask
    from character spans.

    Returns:
      response_ids, tool_mask
    where tool_mask[t] == 0 for tokens whose decoded char span overlaps any span in
    `tool_char_spans`, and 1 otherwise.
    """
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None and (
        len(response_ids) == 0 or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids.append(tokenizer.eos_token_id)
    token_offsets = _lff_build_token_char_offsets(response_ids, tokenizer)
    tool_mask: List[int] = []
    for st, ed in token_offsets:
        is_tool = any(st < span_ed and ed > span_st for span_st, span_ed in tool_char_spans)
        tool_mask.append(0 if is_tool else 1)
    return response_ids, tool_mask


def _lff_find_lp_without_mask_start(
    response_ids: List[int],
    decoded_response: str,
    tokenizer,
    call_start_token: int,
) -> int:
    """
    For non-degenerate responses, build the *student* no-feedback context by
    removing not only the verify_answer call/output, but also the immediately
    preceding think block that announces or prepares the VA call.

    Returns the token start of the last <think>...</think> block before the
    FIRST verify_answer call. Falls back to `call_start_token` when alignment is
    unavailable or such a think block does not exist.
    """
    va_m = _LFF_VA_CALL_RE.search(decoded_response)
    if va_m is None:
        return call_start_token

    think_before_start_char: Optional[int] = None
    for m in _LFF_THINK_RE.finditer(decoded_response[:va_m.start()]):
        think_before_start_char = m.start()

    if think_before_start_char is None:
        return call_start_token

    token_offsets = _lff_build_token_char_offsets(response_ids, tokenizer)
    return _lff_char_pos_to_token_start(
        think_before_start_char,
        token_offsets,
        fallback_token_idx=call_start_token,
    )


def _slice_trace_dict_lists(
    trace_info: Optional[Dict],
    indices,
    expected_len: int,
) -> Optional[Dict]:
    """Slice list-valued trace fields whose length matches `expected_len`."""
    if trace_info is None:
        return None
    if isinstance(indices, torch.Tensor):
        idx_list = [int(x) for x in indices.cpu().tolist()]
    else:
        idx_list = [int(x) for x in list(indices)]
    sliced = {}
    for k, v in trace_info.items():
        if isinstance(v, list) and len(v) == expected_len:
            sliced[k] = [v[i] for i in idx_list]
        else:
            sliced[k] = v
    return sliced


@dataclass
class _RewardPrescoreItem:
    """Per-response async prescore payload produced before group aggregation."""

    response_idx: int
    query_ids: List[int]
    response_ids: List[int]
    mask: List[int]
    decoded_response: str
    decoded_response_raw: str
    decoded_query: str
    ground_truth: Union[str, List[str], Dict]
    dataset: Union[str, List[str]]
    finish_reason: str
    num_calls: int
    timeout: bool
    tool_error: str
    tool_output: str
    tool_runtime: float
    tool_called: bool
    good_output: bool
    score: float
    per_func_reward: Dict[str, float]
    reward_log_values: Dict[str, object]
    judge_failed: bool
    metrics: Dict[str, object]


def _merge_reward_log_value_entries(values: List[object]):
    """Merge per-response reward log values into batch-shaped arrays/lists."""
    if not values:
        return []
    first = values[0]
    if isinstance(first, dict):
        merged: Dict[str, List[object]] = defaultdict(list)
        for item in values:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                merged[k].append(v)
        return dict(merged)
    return list(values)


def _maybe_numeric_hist_array(value) -> Optional[np.ndarray]:
    """Return a numeric ndarray suitable for TensorBoard histogram, else None."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, list):
        if len(value) == 0:
            return None
        try:
            arr = np.array(value)
        except Exception:
            return None
    else:
        return None

    if arr.size == 0:
        return None
    if arr.dtype == object:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(float, copy=False)


def _maybe_numeric_scalar(value) -> Optional[float]:
    """Return a Python float suitable for TensorBoard scalar logging, else None.

    Accepts:
      - Python int/float / numpy scalar
      - torch scalar tensor
      - length-1 numeric list / ndarray

    Rejects:
      - non-numeric containers
      - multi-element lists/arrays (these should go to histogram, not scalar)
      - dict / object arrays
    """
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            try:
                return float(value.detach().cpu().item())
            except Exception:
                return None
        return None

    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        if value.dtype == object or not np.issubdtype(value.dtype, np.number):
            return None
        try:
            return float(value.reshape(-1)[0])
        except Exception:
            return None

    if isinstance(value, list):
        if len(value) != 1:
            return None
        try:
            arr = np.array(value)
        except Exception:
            return None
        if arr.size != 1 or arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            return None
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return None

    return None


def _lff_build_packed_data(
    packed_sequences,
    responses: List[List[int]],
    masks: List[List[int]],
    tokenizer,
    lff_A0: "np.ndarray",
    lff_A1: "np.ndarray",
    lff_has_va: List[bool],
    advantages_dapo: "np.ndarray",
    is_degen: Optional["np.ndarray"] = None,
    force_full_a0: Optional["np.ndarray"] = None,
    r1_scores: Optional["np.ndarray"] = None,
    r0_scores: Optional["np.ndarray"] = None,
    enable_va_reweight: bool = True,
    enable_rs: bool = True,
) -> None:
    """
    Build LFF per-token data and store on packed_sequences:
      - packed_sequences.advantages: per-token, with A0 for seg-0 and A1 for seg-1
      - packed_sequences.va_verify_answer_mask: 0 at VA tool_output span (for lp_without forward)
      - packed_sequences.lff_seg1_mask: 1 for seg-1 tokens (post-VA model tokens)
      - packed_sequences.lff_any_va: bool

    Segment definitions per response:
      Seg-0: tokens up to (but not including) VA tool_output span
             + VA call tokens (tool_mask==0, before output_start)
             advantage = A0_i
      VA output: tool_output tokens [output_start, output_end)
             advantage = 0, masked from loss
      Seg-1: tokens after VA tool_output (tool_mask==1 and t >= output_end)
             advantage = A1_i, ratio = distillation
             EXCEPTION: if is_degen[i]=True, seg-1 uses standard ratio (no distillation)
      full-trajectory-A0 (rewritten degen draft): all model tokens use A0_i
      no-VA: all tokens use advantages_dapo[i]
    """
    N = len(responses)
    any_va = any(lff_has_va)

    # Per-response masks (indexed by response position 0..N-1)
    # va_verify_answer_mask_by_resp: 0 at the region removed from the *student*
    # no-feedback context (think_before_va + VA call + VA output for non-degen).
    va_verify_answer_mask_by_resp: Dict[int, List[int]] = {}
    # first_va_span_mask_by_resp: 0 only at the FIRST VA call/output span. These
    # tokens are excluded from the policy-gradient loss because they are not
    # learnable natural-language continuation targets for LFF.
    first_va_span_mask_by_resp:    Dict[int, List[int]] = {}
    # lp_without_removed_mask_by_resp: 0 at the entire region removed from the
    # student no-feedback context (think_before_va + FIRST VA call/output).
    # Used to keep RS trajectory construction consistent with lp_without.
    lp_without_removed_mask_by_resp: Dict[int, List[int]] = {}
    seg1_mask_by_resp:             Dict[int, List[int]] = {}  # 1 at seg-1 tokens
    noise_mask_by_resp:            Dict[int, List[int]] = {}  # 1 at repeated VA call+output tokens
    # rs_seg1_mask: 1 at seg-1 tokens of non-degen responses where r1 > r0
    # These use rejection-sampling SFT loss: -r1 * lp_without (no ratio, no clip)
    rs_seg1_mask_by_resp:  Dict[int, List[int]] = {}
    rs_r1_by_resp:         Dict[int, float] = {}   # r1 weight for each rs response
    A0_by_resp:   Dict[int, float] = {}
    A1_by_resp:   Dict[int, float] = {}
    adv_by_resp:  Dict[int, float] = {}  # for no-VA tokens

    for i in range(N):
        response_ids = list(responses[i])
        tool_mask_i  = list(masks[i]) if masks and i < len(masks) else [1] * len(response_ids)
        r_len = len(response_ids)
        adv_by_resp[i] = float(advantages_dapo[i])

        _force_full_a0 = (
            force_full_a0 is not None and i < len(force_full_a0) and bool(force_full_a0[i])
        )
        if _force_full_a0:
            A0_by_resp[i] = float(lff_A0[i])
            A1_by_resp[i] = float(lff_A0[i])
            continue

        if not lff_has_va[i]:
            # no-VA: standard DAPO advantage for all tokens
            A0_by_resp[i] = float(advantages_dapo[i])
            A1_by_resp[i] = float(advantages_dapo[i])
            continue

        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        va_span = _lff_find_va_span(tool_mask_i, decoded)
        if va_span is None:
            # VA call found but tool_output too short/missing → fallback to DAPO
            A0_by_resp[i] = float(advantages_dapo[i])
            A1_by_resp[i] = float(advantages_dapo[i])
            continue

        call_start, call_end, output_start, output_end = va_span
        # call_start: start of a_t (VA call tokens, tool_mask==0)
        # call_end  = output_start: boundary between a_t and s_t
        # output_end: end of s_t (VA tool_output)

        # seg1_mask: 1 for post-VA model-generated tokens (a_{t+1}, ..., y)
        # These are the tokens after s_t ends (t >= output_end) with tool_mask==1
        _response_is_degen = (is_degen is not None and i < len(is_degen) and bool(is_degen[i]))
        seg1 = [
            1 if (t >= output_end and tool_mask_i[t] == 1) else 0
            for t in range(r_len)
        ]
        first_va_span_mask_by_resp[i] = [
            0 if (call_start <= t < output_end) else 1
            for t in range(r_len)
        ]
        if _response_is_degen:
            # Degenerate response (r1 < r0 or short final answer):
            # - seg-1 tokens added to noise_mask → excluded from loss entirely
            # - va_verify_answer_mask NOT stored → lp_without forward skipped in train()
            if i not in noise_mask_by_resp:
                noise_mask_by_resp[i] = [0] * r_len
            for t in range(r_len):
                if seg1[t] == 1:
                    noise_mask_by_resp[i][t] = 1
            seg1_mask_by_resp.pop(i, None)
            # Intentionally skip va_verify_answer_mask_by_resp[i] so that
            # lp_without forward is not triggered for this response in train().
        else:
            # Non-degenerate: optionally build va_verify_answer_mask and seg1_mask.
            # enable_va_reweight=False skips the lp_without forward entirely, so we
            # omit both masks.  enable_rs=False skips building the RS masks.
            # When BOTH are disabled, neither the distillation ratio nor RS loss is
            # computed — effectively reducing to standard DAPO on all tokens.
            # Note: lp_without forward is needed by BOTH VA-reweight (for Δℓ_t) and
            # RS (for log P_S(τ')).  We trigger it whenever at least one is enabled.
            _need_lp_without = enable_va_reweight or enable_rs
            if _need_lp_without:
                # Mask out the think block that directly precedes the FIRST
                # verify_answer call, PLUS the VA call/output themselves.  This
                # creates a more coherent student no-feedback context while still
                # preserving post-feedback reasoning tokens for distillation.
                # 0 = masked (attend disabled), 1 = keep
                lp_without_start = _lff_find_lp_without_mask_start(
                    response_ids=response_ids,
                    decoded_response=decoded,
                    tokenizer=tokenizer,
                    call_start_token=call_start,
                )
                _lp_without_removed = [
                    0 if (lp_without_start <= t < output_end) else 1
                    for t in range(r_len)
                ]
                va_verify_answer_mask_by_resp[i] = _lp_without_removed
                lp_without_removed_mask_by_resp[i] = _lp_without_removed
            if enable_va_reweight:
                seg1_mask_by_resp[i] = seg1

            # Rejection-sampling distillation: for responses where r1 > r0,
            # compute log π_θ(τ') over the FULL trajectory τ' (not just seg-1).
            if enable_rs and r1_scores is not None and r0_scores is not None:
                _r1_i = float(r1_scores[i])
                _r0_i = float(r0_scores[i])
                if _r1_i > _r0_i:
                    # Full τ' mask: model-generated tokens excluding the SAME
                    # region removed from lp_without, i.e. think_before_va +
                    # FIRST VA call/output.  This keeps the RS SFT trajectory
                    # consistent with the no-feedback student context.
                    _lp_without_removed = lp_without_removed_mask_by_resp.get(i)
                    rs_full = [
                        1 if (
                            tool_mask_i[t] == 1 and
                            not (_lp_without_removed is not None and t < len(_lp_without_removed)
                                 and _lp_without_removed[t] == 0)
                        )
                        else 0
                        for t in range(r_len)
                    ]
                    rs_seg1_mask_by_resp[i] = rs_full
                    rs_r1_by_resp[i] = _r1_i

        A0_by_resp[i] = float(lff_A0[i])
        A1_by_resp[i] = float(lff_A1[i])

        # ── Handle repeated VA calls (noise tokens) ───────────────────────────
        # Only the FIRST verify_answer call is valid. Subsequent calls and their
        # outputs are noise tokens: mask them from both the loss and the
        # without-feedback attention context.
        # Strategy: find all VA spans after the first output_end and mark them
        # as noise (advantage=0, excluded from response_mask_bool in train()).
        noise = [0] * r_len  # 1 = noise token (repeated VA call or output)

        # Find all tool spans in the response
        all_spans_full = []
        in_t, st = False, 0
        for t in range(r_len):
            if not in_t and tool_mask_i[t] == 0:
                in_t, st = True, t
            elif in_t and tool_mask_i[t] == 1:
                all_spans_full.append((st, t))
                in_t = False
        if in_t:
            all_spans_full.append((st, r_len))

        # Count tool_output blocks before the first VA call (already used above)
        n_before = len(list(_LFF_TOOL_OUTPUT_RE.finditer(decoded[:call_start]
            if 'call_start' in dir() else decoded[:output_start])))

        # Find all verify_answer calls in decoded text
        all_va_calls = list(_LFF_VA_CALL_RE.finditer(decoded))
        # We already processed the first one (index 0).
        # For each subsequent VA call (index >= 1):
        #   - find the corresponding call span (tool_mask==0 block)
        #   - find the corresponding tool_output span (next tool_mask==0 block)
        #   - mark both as noise
        for va_idx in range(1, len(all_va_calls)):
            va_char_pos = all_va_calls[va_idx].start()
            # Count tool outputs before this repeated call
            n_outputs_before = len(list(_LFF_TOOL_OUTPUT_RE.finditer(decoded[:va_char_pos])))
            # The call tokens: tool span just before the output
            # (span at n_outputs_before - 1 if adjacent, else scan backwards)
            # The output tokens: tool span at n_outputs_before
            if n_outputs_before < len(all_spans_full):
                out_s, out_e = all_spans_full[n_outputs_before]
                # Mark output span as noise
                for t in range(out_s, out_e):
                    noise[t] = 1
                # Mark preceding call span as noise (if adjacent)
                if n_outputs_before > 0:
                    c_s, c_e = all_spans_full[n_outputs_before - 1]
                    if c_e == out_s:
                        for t in range(c_s, c_e):
                            noise[t] = 1
                else:
                    # Scan backwards from out_s for contiguous tool_mask==0
                    cs = out_s
                    while cs > 0 and tool_mask_i[cs - 1] == 0:
                        cs -= 1
                    for t in range(cs, out_s):
                        noise[t] = 1

        if any(noise):
            noise_mask_by_resp[i] = noise

    n_noise = sum(1 for v in noise_mask_by_resp.values() if any(v))
    print(
        f"[LFF build] N={N} any_va={any_va} "
        f"n_with_span={len(va_verify_answer_mask_by_resp)} "
        f"n_with_repeated_va_noise={n_noise}",
        flush=True,
    )

    n_rs = sum(1 for v in rs_seg1_mask_by_resp.values() if any(v))
    print(
        f"[LFF build] N={N} any_va={any_va} "
        f"n_with_span={len(va_verify_answer_mask_by_resp)} "
        f"n_with_repeated_va_noise={n_noise} "
        f"n_rs_distill(r1>r0)={n_rs}",
        flush=True,
    )

    # ── Pack into packed_sequences layout ─────────────────────────────────────
    packed_va_masks   = []
    packed_seg1       = []
    packed_noise      = []
    packed_adv_new    = []
    packed_rs_seg1    = []   # 1 at seg-1 of rs responses (r1 > r0, non-degen)
    packed_rs_r1      = []   # per-token r1 weight for rs responses (0 elsewhere)

    for packed_response_mask, packed_adv in zip(
        packed_sequences.response_masks,
        packed_sequences.advantages,
    ):
        rm      = packed_response_mask.tolist()
        seq_len = len(rm)

        va_attn   = [1] * seq_len   # 0 = mask a_t+s_t from attention (for lp_without)
        seg1_m    = [0] * seq_len   # 1 = seg-1 token
        noise_m   = [0] * seq_len   # 1 = noise token (repeated VA call/output)
        rs_seg1_m = [0] * seq_len   # 1 = rs seg-1 token (r1 > r0)
        rs_r1_m   = [0.0] * seq_len # r1 score weight for rs tokens
        token_adv = packed_adv.clone()

        resp_token_pos: Dict[int, int] = {}
        for t, resp_idx_val in enumerate(rm):
            if resp_idx_val == 0:
                continue
            pos      = resp_token_pos.get(resp_idx_val, 0)
            orig_idx = resp_idx_val - 1

            # va_verify_answer_mask (0 at removed student-context region for
            # lp_without forward)
            vm = va_verify_answer_mask_by_resp.get(orig_idx)
            if vm is not None and pos < len(vm):
                va_attn[t] = vm[pos]

            # seg1 mask
            sm = seg1_mask_by_resp.get(orig_idx)
            if sm is not None and pos < len(sm):
                seg1_m[t] = sm[pos]

            # noise mask (repeated VA calls/outputs → excluded from loss)
            nm = noise_mask_by_resp.get(orig_idx)
            if nm is not None and pos < len(nm):
                noise_m[t] = nm[pos]

            # rs seg-1 mask and r1 weight (rejection sampling, r1 > r0 responses only)
            rsm = rs_seg1_mask_by_resp.get(orig_idx)
            if rsm is not None and pos < len(rsm) and rsm[pos] == 1:
                rs_seg1_m[t] = 1
                rs_r1_m[t] = rs_r1_by_resp.get(orig_idx, 0.0)

            # per-token advantage
            if orig_idx in A0_by_resp:
                vm_here = va_verify_answer_mask_by_resp.get(orig_idx)
                first_va_here = first_va_span_mask_by_resp.get(orig_idx)
                sm_here = seg1_mask_by_resp.get(orig_idx)
                nm_here = noise_mask_by_resp.get(orig_idx)

                if nm_here is not None and pos < len(nm_here) and nm_here[pos] == 1:
                    # Noise token (repeated VA call/output) → 0, no gradient
                    token_adv[t] = 0.0
                elif first_va_here is not None and pos < len(first_va_here) and first_va_here[pos] == 0:
                    # FIRST VA call/output tokens → 0, no gradient.  Keep this
                    # separate from va_verify_answer_mask so that non-degenerate
                    # think_before_va tokens can still use A0 while being removed
                    # from the no-feedback student context.
                    token_adv[t] = 0.0
                elif vm_here is None:
                    # no-VA or no-span: use DAPO advantage
                    token_adv[t] = adv_by_resp[orig_idx]
                elif sm_here is not None and pos < len(sm_here) and sm_here[pos] == 1:
                    # seg-1 token
                    token_adv[t] = A1_by_resp[orig_idx]
                else:
                    # seg-0 token
                    token_adv[t] = A0_by_resp[orig_idx]

            resp_token_pos[resp_idx_val] = pos + 1

        packed_va_masks.append(torch.tensor(va_attn,  dtype=torch.long))
        packed_seg1.append(    torch.tensor(seg1_m,   dtype=torch.long))
        packed_noise.append(   torch.tensor(noise_m,  dtype=torch.long))
        packed_adv_new.append(token_adv)
        packed_rs_seg1.append(torch.tensor(rs_seg1_m, dtype=torch.long))
        packed_rs_r1.append(  torch.tensor(rs_r1_m,   dtype=torch.float32))

    packed_sequences.advantages            = packed_adv_new
    packed_sequences.va_verify_answer_mask = packed_va_masks
    packed_sequences.lff_seg1_mask         = packed_seg1
    packed_sequences.lff_noise_mask        = packed_noise   # repeated VA noise
    packed_sequences.lff_rs_seg1_mask      = packed_rs_seg1  # rs seg-1 (r1>r0)
    packed_sequences.lff_rs_r1             = packed_rs_r1    # per-token r1 weight
    packed_sequences.lff_any_va            = any_va


def _lff_find_va_span(
    tool_mask: List[int],
    decoded_response: str,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find token spans for the FIRST verify_answer call and its tool_output.

    Returns (call_start, call_end, output_start, output_end) where:
      - [call_start, call_end):   a_t  — the verify_answer <call_tool>...</call_tool> tokens
                                         (tool_mask==0, immediately before the tool_output)
      - [output_start, output_end): s_t — the verify_answer <tool_output>...</tool_output> tokens
                                         (tool_mask==0, the feedback)

    Segment definitions:
      seg-0: response tokens with t < call_start  → pre-VA model output + earlier tool outputs
             PLUS the VA call tokens [call_start, call_end) are also in seg-0 context
             (the model generated a_t, so a_t ∈ π_θ(y0))
      seg-1: response tokens with t >= output_end → post-VA model output and final answer
             For lp_without, we mask BOTH a_t AND s_t from context, i.e. we remove
             the entire [call_start, output_end) span from the attention context.

    Returns None if not found or tool_output too short.
    """
    r_len = len(tool_mask)
    if r_len == 0:
        return None
    va_matches = list(_LFF_VA_CALL_RE.finditer(decoded_response))
    if not va_matches:
        return None
    va_call_char_pos = va_matches[0].start()
    n_outputs_before_va = len(list(_LFF_TOOL_OUTPUT_RE.finditer(decoded_response[:va_call_char_pos])))

    # Build all tool spans (tool_mask==0 blocks) from mask
    all_spans = []
    in_tool, span_start = False, 0
    for t in range(r_len):
        if not in_tool and tool_mask[t] == 0:
            in_tool, span_start = True, t
        elif in_tool and tool_mask[t] == 1:
            all_spans.append((span_start, t))
            in_tool = False
    if in_tool:
        all_spans.append((span_start, r_len))

    # The VA tool_output is the n-th tool span (0-indexed)
    if n_outputs_before_va >= len(all_spans):
        return None
    output_start, output_end = all_spans[n_outputs_before_va]

    # Validate tool_output length
    min_len = int(os.environ.get("VA_MIN_OUTPUT_LEN", "200"))
    all_texts = list(_LFF_TOOL_OUTPUT_RE.finditer(decoded_response))
    va_text = all_texts[n_outputs_before_va].group(0) if n_outputs_before_va < len(all_texts) else ""
    if len(va_text) < min_len:
        return None

    # The VA call (a_t) is the tool span immediately BEFORE the tool_output.
    # The call_tool tag is model-generated (tool_mask==0 while inside the tag).
    # In the token sequence: [...model tokens...][call_tool tokens][tool_output tokens][...]
    # The call tokens are the tool span at index n_outputs_before_va - 1,
    # BUT only if it ends exactly at output_start (i.e. it's adjacent).
    # If not adjacent, it means there are model-generated tokens between call and output
    # (shouldn't happen in practice), so we fall back to using output_start as call boundary.
    call_start = output_start  # fallback: no separate call span found
    if n_outputs_before_va > 0:
        prev_span_start, prev_span_end = all_spans[n_outputs_before_va - 1]
        if prev_span_end == output_start:
            # Adjacent tool spans: the previous one is the VA call (a_t)
            call_start = prev_span_start
    else:
        # The VA output is the first tool span; look for model-generated call tokens
        # by finding the last tool_mask==0 block before output_start
        # (these are the <call_tool name="verify_answer">...</call_tool> tokens)
        # Walk backwards from output_start to find contiguous tool_mask==0
        cs = output_start
        while cs > 0 and tool_mask[cs - 1] == 0:
            cs -= 1
        if cs < output_start:
            call_start = cs

    return call_start, output_start, output_start, output_end
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
    download_latest_checkpoint_from_gs,
    extract_user_query,
    get_beaker_whoami,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    sync_gs_bucket,
)
from open_instruct.tool_utils.tool_actor import ToolActor, TOOL_CLASS_REGISTRY
from open_instruct.tool_utils.tool_proxy import ToolProxy
from open_instruct.vllm_utils3 import create_vllm_engines, init_process_group
from open_instruct.search_utils.mcp_tools import MCP_TOOL_REGISTRY

api = HfApi()
INVALID_LOGPROB = 1.0


def _build_ray_runtime_env_for_proxy() -> Dict[str, Dict[str, str]]:
    """
    Build Ray runtime_env to propagate proxy settings while forcing localhost direct access.
    """
    no_proxy_required = ["127.0.0.1", "localhost", "0.0.0.0", "::1", ".local"]
    current_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    no_proxy_entries = [x.strip() for x in current_no_proxy.split(",") if x.strip()]
    for item in no_proxy_required:
        if item not in no_proxy_entries:
            no_proxy_entries.append(item)
    merged_no_proxy = ",".join(no_proxy_entries)

    env_vars: Dict[str, str] = {
        "NO_PROXY": merged_no_proxy,
        "no_proxy": merged_no_proxy,
    }

    # Preserve existing proxy settings if present.
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value

    # Propagate PYTHONPATH so Ray actors on worker nodes use the same local
    # package versions instead of falling back to site-packages.
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath

    # Propagate MCP connection settings so worker node ToolActors can reach
    # the MCP server running on the head node (not 127.0.0.1).
    for key in [
        "MCP_HOST",
        "MCP_TRANSPORT",
        "MCP_TRANSPORT_HOST",
        "MCP_TRANSPORT_PORT",
        "SERPER_API_KEY",
        "SERPER_API_KEY_LIST",
        "SERPER_BASE_URL",
        "S2_API_KEY",
        "S2_API_KEY_LIST",
        "JINA_API_KEY",
        "UNIVERSAL_SEARCH_SOURCES",
        "API_TIMEOUT",
        "API_KEY_COOLDOWN_SECONDS",
        # LLM inference endpoint
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        # Judge / generation model names
        "RUBRIC_JUDGE_MODEL",
        "RUBRIC_GENERATION_MODEL",
        "CITATION_JUDGE_MODEL",
        "DRAFT_ANSWER_FEEDBACK_MODEL",
        "CRITIQUE_JUDGE_MODEL",
        "RUBRIC_JUDGE_TIMEOUT",
        "RUBRIC_GENERATION_TIMEOUT",
        "RUBRIC_JUDGE_MAX_TOKENS",
        "DRAFT_ANSWER_API_BASE",
        "DRAFT_ANSWER_API_KEY",
        # Per-judge API base URLs for routing to separate vLLM instances
        "RUBRIC_JUDGE_API_BASE",
        "CITATION_JUDGE_API_BASE",
        "RUBRIC_GENERATION_API_BASE",
        "UNIVERSAL_SEARCH_SOURCES",
        "OPENALEX_EMAIL",
        "API_TIMEOUT",
        "RUBRIC_BUFFER_EXPORT_PATH"
    ]:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return {"env_vars": env_vars}


def _normalize_query_for_index(query: str) -> str:
    return " ".join(query.lower().strip().split())


def export_rubric_buffer_snapshot(rubric_buffer: Optional[Dict], output_dir: str, training_step: int) -> None:
    """
    Export rubric_buffer to a file so external services (e.g., MCP backend) can
    retrieve instance-wise rubrics by query.

    File path:
    - RUBRIC_BUFFER_EXPORT_PATH if set
    - else <output_dir>/rubric_buffer_snapshot.json
    """
    if not rubric_buffer:
        return
    every_n = int(os.environ.get("RUBRIC_BUFFER_EXPORT_EVERY_STEPS", "1"))
    if every_n > 1 and training_step % every_n != 0:
        return

    export_path = os.environ.get(
        "RUBRIC_BUFFER_EXPORT_PATH",
        os.path.join(output_dir, "rubric_buffer_snapshot.json"),
    )
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    payload: Dict[str, Union[int, float, Dict, List]] = {
        "training_step": training_step,
        "updated_at": time.time(),
        "query_index": {},
        "normalized_query_index": {},
        "default_static_rubrics": [],
    }

    static_union = {}
    for query, buf in rubric_buffer.items():
        if not isinstance(buf, dict):
            continue
        static_rubrics = buf.get("static_rubrics", []) or []
        active_rubrics = buf.get("active_rubrics", []) or []
        positive_rubrics = []
        negative_rubrics = []
        for rubric in active_rubrics:
            weight = float(rubric.get("weight", 1.0))
            if weight < 0:
                negative_rubrics.append(rubric)
            else:
                positive_rubrics.append(rubric)

        payload["query_index"][query] = {
            "static_rubrics": static_rubrics,
            "positive_rubrics": positive_rubrics,
            "negative_rubrics": negative_rubrics,
        }
        payload["normalized_query_index"][_normalize_query_for_index(query)] = query

        for rubric in static_rubrics:
            key = f"{rubric.get('title', '')}||{rubric.get('description', '')}"
            if key not in static_union:
                static_union[key] = rubric

    payload["default_static_rubrics"] = list(static_union.values())

    tmp_path = export_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, export_path)


@dataclass
class Args:
    # Dataset
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_rl_rag_v1", "rlvr_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: Optional[str] = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    shuffle_eval_dataset: bool = False
    """Whether to shuffle the evaluation dataset."""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""

    # Dataset system prompt injection
    system_prompt_file: Optional[str] = None
    """If set, add or replace system prompts in the dataset with the given system prompt."""
    enable_draft_answer: bool = False
    """Whether to expose and enable verify_answer MCP tool for the agent."""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """RUNTIME VALUE: A unique name of this run"""

    # Optimizer
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""
    fused_optimizer: bool = False
    """Whether to use fused optimizer"""

    # Batch sizes
    per_device_train_batch_size: int = 1
    """The forward batch size per device (local_micro_batch_size)"""
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """RUNTIME VALUE: The number of processes (GPUs) to use"""
    num_training_steps: Optional[int] = None
    """RUNTIME VALUE: The number of training_steps to train"""
    num_evals: int = 10
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """RUNTIME VALUE: The frequency of evaluation steps"""
    save_freq: int = -1
    """How many train steps to save the model"""
    allow_world_padding: bool = False
    """Whether to allow world padding. This is useful for model sweeps, but wastes compute."""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 4
    """the number of samples to generate per prompt during rollout, useful for easy-star"""
    stop_strings: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""

    # Algorithm
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""
    async_steps: int = 1
    """Number of steps ahead to generate responses. Only used when async_mode is True."""
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.2
    """the upper clip range (symmetric with clip_lower for standard GRPO)."""
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    """the KL estimator to use"""
    pack_length: int = 512
    """the length of the pack (you should prob set to the max length of the model)"""
    masked_mean_axis: Optional[int] = None
    """the axis to compute the mean of the masked values"""
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: Optional[int] = None
    """How many training steps to take before updating the reference policy."""
    advantage_normalization_type: Literal["standard", "centered"] = "standard"
    """The type of advantage normalization to use. Standard normalization is the default: it subtracts the mean and
    divides by the standard deviation. Centered normalization is the same but subtracts the mean only (e.g., used in
    DR.GRPO https://arxiv.org/pdf/2503.20783)."""
    mask_truncated_completions: bool = False
    """Whether to discard responses that hit max_tokens (finish_reason != 'stop'). Disabled in standard GRPO."""

    # ── LFF (Learning from Feedback) parameters ───────────────────────────────
    lff_enabled: bool = True
    """Enable LFF two-segment advantage estimation."""
    lff_enable_va_reweight: bool = True
    """Ablation flag: enable the seg-1 IS-reweighting (distillation ratio exp(Δℓ_t)).
    Set False to disable the VA reweight component while keeping other LFF features.
    Ablation: lff_enable_va_reweight=True,  lff_enable_rs=False → VA reweight only.
              lff_enable_va_reweight=False, lff_enable_rs=True  → RS distillation only."""
    lff_enable_rs: bool = True
    """Ablation flag: enable the rejection-sampling distillation loss (L_RS = -r1·log P_S(τ')).
    Set False to disable RS while keeping VA reweighting and other LFF features.
    Ablation: lff_enable_va_reweight=True,  lff_enable_rs=False → VA reweight only.
              lff_enable_va_reweight=False, lff_enable_rs=True  → RS distillation only."""
    lff_rs_weight: float = 1.0
    """Scaling factor λ_RS applied to the RS distillation loss: L_RS = -λ_RS · r1 · log P_S(τ').
    Since r1 is in [0, 10] (reward_mult=10), set to 0.1 to keep the effective gradient
    magnitude comparable to the RL policy gradient loss. Default 1.0 for backward compatibility."""
    lff_va_call_bonus: float = 0.0
    """Bonus added to scores of responses that called verify_answer at least once,
    applied before group normalization. Counteracts the model's incentive to skip
    VA calls for more stable advantages. Recommended: 0.3-0.5. Set 0 to disable."""
    lff_degenerate_final_ans_penalty: float = 0.5
    """Penalty subtracted from r1 when the final answer is degenerate (shorter than
    lff_degenerate_final_ans_min_len chars after receiving VA feedback). Set to 0 to disable."""
    lff_degenerate_final_ans_min_len: int = 50
    """Minimum character length of the final answer (inside <answer>...</answer>) to be considered
    non-degenerate. Answers shorter than this (e.g. '...') receive the penalty and skip distillation."""
    lff_seg1_w_min: float = 0.2
    """Lower clamp for seg-1 distillation ratio exp(Δℓ_t).
    Based on observed delta distribution (seg1_delta_min mean≈-2.5, p5≈-3.1):
    effective token-level lower bound set at exp(-1.5)≈0.22 → rounded to 0.2.
    Tokens where feedback suppresses the token (Δ < 0) are down-weighted but not zeroed out."""
    lff_seg1_w_max: float = 2.0
    """Upper clamp for seg-1 distillation ratio exp(Δℓ_t).
    Based on observed delta distribution (exp(seg1_delta_mean) p95≈1.82):
    95% of steps have mean weight below 1.82; w_max=2.0 covers normal tokens cleanly.
    Combined with the |Δ|>5 noise filter, this prevents gradient explosion from
    outlier tokens (observed delta_max mean≈15.7 → exp≈6.6M without clamping)."""

    # Reward
    # -- r1 style format reward
    apply_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 1.0
    """the reward value for R1 style format reward"""
    apply_rl_rag_format_reward: bool = False
    """whether to add the RL-RAG style format reward--only responses with a valid answer and at least one valid query will be rewarded"""
    additive_format_reward: bool = False
    """whether to add the format reward to the final reward"""

    # -- verifiable reward
    apply_verifiable_reward: bool = True
    """whether to apply verifiable reward"""
    verification_reward: float = 10.0
    """the reward value for verifiable responses"""
    verifier_strategy: str = "judge"
    """The strategy to use for verification in rl_rag_exact_answer verifier: 'f1', 'em', or 'judge'"""

    # -- llm verifiers
    llm_judge_model: str = "qwen3-instruct"
    """the model to use for the llm judge"""
    llm_judge_max_tokens: int = 2048
    """the max tokens to use for the llm judge"""
    llm_judge_temperature: float = 1.0
    """the temperature to use for the llm judge"""
    llm_judge_timeout: int = 60
    """the timeout to use for the llm judge"""
    llm_judge_max_context_length: int = 2048
    """the max context length to use for the llm judge"""

    # -- code verifier
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    """the api url to use for the code verifier"""
    code_max_execution_time: float = 1.0
    """the max execution time to use for the code verifier"""

    # -- non stop penalty
    non_stop_penalty: bool = False
    """whether to penalize responses which did not finish generation"""
    non_stop_penalty_value: float = 0.0
    """the reward value for responses which did not finish generation"""

    # -- tool error penalty
    tool_error_penalty_value: float = 0.0
    """deduction applied to scores[i] when a tool error (e.g. 'No valid tool calls found') is
    detected; result is clamped to 0. Set to e.g. 1.0 to penalize malformed tool calls."""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: List[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    vllm_gpu_memory_utilization: float = 0.9
    """vLLM GPU memory utilization"""
    vllm_enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    vllm_top_p: float = 1.0
    """vLLM top p for nucleus sampling"""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "output"
    """Where to save the model"""
    save_traces: bool = False
    """Whether to save learning data traces"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    checkpoint_state_freq: int = -1
    """How often to save the model checkpoint, optimizer states, and lr scheduler states (in steps)"""
    checkpoint_state_dir: Optional[str] = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: Optional[str] = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""

    # Training rollout logging
    log_training_rollouts: bool = False
    """Whether to log training rollouts to wandb"""
    log_training_rollouts_freq: int = 10
    """How often to log training rollouts (every N training steps)"""
    num_training_rollouts_to_log: int = 16
    """Number of training rollouts to log each time"""
    log_direction_agreement: bool = False
    """Whether to log direction agreement between fine-grained rewards and global reward"""

    # Tool settings
    tools: Optional[List[str]] = None
    """If set, use the tool mapped to the string. Currently only supports `search`, `code`, and `mcp`"""
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    """Maximum number of tool calls allowed. If a list is provided, it must have length 1 (applies to all tools) or same length as tools (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs. By default off. Useful to force the model to use the tool(s)."""
    tool_max_concurrency: int = 512
    """The maximum number of concurrent tool calls allowed across all rollouts per tool."""


    ### begin per-tool settings ###
    # code-tool specific settings
    code_tool_api_endpoint: Optional[str] = None

    # search-tool specific settings
    # rl-rag tool settings. These are shared across different tools.
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: Optional[str] = None
    """The API endpoint for the search engine."""
    use_massive_ds: bool = False
    """Whether to use massive ds for search. Only matters for non-MCP search tool."""

    # mcp-tool specific settings
    mcp_tool_names: Optional[str] = "snippet_search,google_search,browse_webpage,verify_answer"
    """The names (comma separated) of the MCP tool to use. Valid tools are: snippet_search, google_search, browse_webpage, verify_answer."""
    mcp_parser_name: Optional[str] = None
    """The name of the MCP parser to use."""
    mcp_server_command: Optional[str] = None
    """Command to run MCP server subprocess when use_mcp_tools is enabled. Example: 'uv run python -m dr_agent.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp'. If not set, will not launch the MCP stuff on its own."""
    mcp_host: Optional[str] = None
    """The host of the MCP server. Note it should match the host of the MCP server command if specified there."""
    mcp_port: Optional[int] = None
    """The port of the MCP server. Note it should match the port of the MCP server command if specified there."""
    mcp_timeout: int = 180
    """The timeout for the MCP tool."""
    base_url: Optional[str] = None
    """The base URL for the MCP tool, used for massive_serve."""
    ### end per-tool settings ###


    # Reward function override
    overwrite_reward_fn_tag: Optional[str] = None
    """If set, force all datasets to use this specific reward function type instead of dataset-based selection"""
    use_general_rubric: bool = False
    """Whether to use the general rubric for evaluation instead of specific rubrics"""
    evaluate_closed_book_answer: bool = False
    """Whether to evaluate the closed book answer"""
    no_citation_reward: bool = False
    """Whether to disable citation reward (sets citation_reward=0 and reweights rubric:0.6 format:0.2 search:0.2)"""
    apply_adaptive_rubric_reward: bool = False
    """Whether to apply adaptive rubric reward"""
    normalize_rubric_scores: bool = False
    """Whether to normalize the rubric scores per rubric"""
    use_rubric_buffer: bool = False
    """Whether to use the rubric buffer"""
    max_active_rubrics: int = 5
    """Maximum number of active rubrics to keep in the buffer"""
    use_static_rubrics_as_persistent_rubrics: bool = True
    """Whether to use the static rubrics as persistent rubrics"""
    add_static_rubrics_to_active_rubrics_every_n_steps: int = 10
    """How often to add the static rubrics to the active rubrics"""

    def __post_init__(self):
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            print("WARNING: num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE. ")
        assert (
            self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty
        ), "At least one reward must be applied!"
        assert (
            self.pack_length >= self.max_prompt_token_length + self.response_length
        ), "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("`checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")
        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{self.checkpoint_state_dir}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{self.checkpoint_state_dir}"
        if self.gs_checkpoint_state_dir is not None:
            download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
        if self.checkpoint_state_dir is not None:
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in TOOL_CLASS_REGISTRY:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: {', '.join(TOOL_CLASS_REGISTRY.keys())}")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"
        if self.tools and "mcp" in self.tools:
            if self.mcp_tool_names is None:
                raise ValueError("mcp_tool_names must be provided when mcp is in tools")
            self.mcp_tool_names = [n.strip() for n in self.mcp_tool_names.split(",") if n.strip()]
            if self.enable_draft_answer:
                if "verify_answer" not in self.mcp_tool_names:
                    self.mcp_tool_names.append("verify_answer")
            else:
                self.mcp_tool_names = [n for n in self.mcp_tool_names if n != "verify_answer"]
            for mcp_tool_name in self.mcp_tool_names:
                if mcp_tool_name not in MCP_TOOL_REGISTRY:
                    raise ValueError(f"MCP tool {mcp_tool_name} is not supported. Supported tools are: {', '.join(MCP_TOOL_REGISTRY.keys())}")
        if self.system_prompt_file and self.system_prompt_file.endswith(".yaml"):
            no_draft_candidate = self.system_prompt_file.replace(".yaml", "_no_draft_answer.yaml")
            if not self.enable_draft_answer:
                if Path(no_draft_candidate).exists():
                    print(
                        f"[VerifyAnswer] enable_draft_answer=False, switch system prompt to: {no_draft_candidate}"
                    )
                    self.system_prompt_file = no_draft_candidate
                else:
                    print(
                        f"[VerifyAnswer] WARNING: enable_draft_answer=False but no_draft system prompt not found: "
                        f"{no_draft_candidate}. Falling back to removing verify_answer from tool list only. "
                        f"The system prompt still contains verify_answer description — model may still attempt to call it."
                    )

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return ((values * mask).sum(axis=axis) / mask.sum(axis=axis)).mean()
    else:
        return (values * mask).sum() / mask.sum()


class MetricsTracker:
    """A simple class to prellocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: torch.device = torch.device("cuda")):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def add(self, name: str, value: torch.tensor):
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1

        self.metrics[self.names2idx[name]] = value
        return self

    def get_metrics_list(self) -> dict[str, float]:
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


def collate_fn(tensors_list: List[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    if pin_memory:
        padded_tensor = padded_tensor.pin_memory()
    return padded_tensor


def to_device_inplace(tensors_list: List[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


class ShufflingIterator:
    def __init__(self, data: np.ndarray, batch_size: int, seed: Optional[int] = None):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)

        # Ensure the effective dataset size is divisible by batch_size
        self.effective_size = len(self.data) - (len(self.data) % batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        if self.index >= self.effective_size:
            self.index = 0
            self.rng.shuffle(self.data)

        end_index = self.index + self.batch_size
        batch = self.data[self.index : end_index].tolist()
        self.index = end_index

        return batch


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(
        self,
        args: Args,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ):
        # ------------------------------------------------------------
        # Monkey patch to load checkpoints with `weights_only=False`
        # otherwise it errors out with:
        # `_pickle.UnpicklingError: Weights only load failed. ` with pytorch 2.6.0
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine
        from deepspeed.utils import logger

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load

        # ------------------------------------------------------------
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)
        deepspeed.init_distributed()

        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=False,
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # @vwxyzjn: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        print(f"{dschf=}")

        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        # AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        # AdamOptimizer = FusedAdam
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        # self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate)
        self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0.0:
            warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        optimization_steps_done = 0
        if args.checkpoint_state_dir:
            # check if the dir exists
            if not os.path.exists(args.checkpoint_state_dir):
                print(f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!")
            else:
                path, states = self.model.load_checkpoint(
                    args.checkpoint_state_dir,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False,
                )
                if path is None:
                    raise ValueError(f"Failed to load checkpoint from {args.checkpoint_state_dir}")
                optimization_steps_done = states["training_step"]
                print(
                    f"{self.rank=}: Loaded checkpoint from {args.checkpoint_state_dir} with {optimization_steps_done=}"
                )
        self.model.train()

        # reference model
        ds_config = get_eval_ds_config(
            offload=False,
            # inference model only has stage 3 (sharding) or stage 0 (no sharding)
            # stage 2 is optimizer sharding which doesn't apply to inference
            stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        print(f"{dschf=}")

        self.ref_policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.ref_policy)
        self.ref_policy, *_ = deepspeed.initialize(model=self.ref_policy, config=ds_config)
        self.ref_policy.eval()
        self.local_metrics = MetricsTracker(max_metrics=32, device=self.device)
        return optimization_steps_done

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
    ) -> torch.Tensor:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        # NOTE: the [:-1] and [1:] are because the logits and generated tokens are off by 1 in index
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])
        return logprob

    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.args.vllm_num_engines,
                self.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = self.args.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            self.model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )
            ray.get(refs)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # clear vllm cache if we need to
        cache_reset_refs = []
        if self.args.vllm_enable_prefix_caching and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        if self.args.gather_whole_model:
            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=self.args.deepspeed_stage == 3):
                for name, param in model.named_parameters():
                    count += 1  # empty_cache at last param
                    # Fire all vllm engines for broadcast
                    if torch.distributed.get_rank() == 0:
                        shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                            )
                            for engine in self.vllm_engines
                        ]
                        refss.extend(refs)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        else:  # broadcast each parameter independently
            for name, param in model.named_parameters():
                count += 1
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]
                    refss.extend(refs)
                with deepspeed.zero.GatheredParameters([param], enabled=self.args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        if torch.distributed.get_rank() == 0:
            ray.get(refss)
        if self.args.vllm_enable_prefix_caching and torch.distributed.get_rank() == 0:
            ray.get(cache_reset_refs)

    def update_ref_policy(self):
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters(
                    [param, ref_param],
                    modifier_rank=0,
                ):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def train(
        self,
        collated_query_responses,
        collated_tool_masks,
        collated_attention_masks,
        collated_position_ids,
        collated_advantages,
        collated_response_masks,
        pad_token_id: int,
        num_mini_batches: int,
        va_verify_answer_masks: Optional[List] = None,
        va_post_va_mask: Optional[Dict] = None,
        # LFF fields
        lff_seg1_masks:  Optional[List] = None,
        lff_noise_masks: Optional[List] = None,
        lff_any_va: bool = False,
        # RS distillation fields
        lff_rs_seg1_masks: Optional[List] = None,
        lff_rs_r1:         Optional[List] = None,
    ):
        args = self.args
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_tool_masks, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_advantages, self.device)
        to_device_inplace(collated_response_masks, self.device)
        accumulation_steps = math.ceil(len(collated_query_responses) / num_mini_batches - 0.5)
        leftover = len(collated_query_responses) % accumulation_steps
        if leftover > 0:
            collated_query_responses = collated_query_responses[0:-leftover]
            collated_tool_masks = collated_tool_masks[0:-leftover]
            collated_attention_masks = collated_attention_masks[0:-leftover]
            collated_position_ids = collated_position_ids[0:-leftover]
            collated_advantages = collated_advantages[0:-leftover]
            collated_response_masks = collated_response_masks[0:-leftover]

        # Calculate the logprob of the reference policy
        collated_ref_logprobs = []
        with Timer("Inference Calculation", noop=self.rank != 0):
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
                    tool_mask = collated_tool_masks[i]
                    attention_mask = collated_attention_masks[i]
                    position_id = collated_position_ids[i]
                    response_mask = collated_response_masks[i]
                    ref_logprob = self.forward(
                        self.ref_policy,
                        query_response,
                        attention_mask,
                        position_id,
                        pad_token_id,
                        args.temperature,
                    )
                    if args.mask_tool_use and args.tool_use:
                        # mask logprobs for tool tokens
                        response_mask = response_mask.bool() & tool_mask.bool()
                    else:
                        response_mask = response_mask.bool()
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()
        # ── LFF: without-feedback forward (for seg-1 distillation ratio) ──────
        # Compute π_θ(y1|x, r) by masking VA tool_output in attention_mask.
        # va_verify_answer_mask: 0 at VA tool_output positions, 1 elsewhere.
        # This is the SAME mechanism as the original VA batch forward, repurposed
        # for LFF seg-1 ratio: ratio = π_θ(y1|x,r,f) / π_θ(y1|x,r).
        #
        # All ranks run the same number of forward passes (ZeRO-3 requirement).
        va_lp_wo_cache: Dict[int, torch.Tensor] = {}
        _lff_enabled = getattr(args, "lff_enabled", False) and lff_any_va
        _lff_masks_available = bool(
            _lff_enabled and va_verify_answer_masks and len(va_verify_answer_masks) > 0
        )
        # Broadcast flag so all ranks agree on branch taken.
        _va_flag_tensor = torch.tensor(int(_lff_masks_available), dtype=torch.long, device=self.device)
        torch.distributed.broadcast(_va_flag_tensor, src=0)
        _va_masks_available = bool(_va_flag_tensor.item())

        with torch.no_grad():
            for mb_idx in range(len(collated_query_responses)):
                _mb_attn_va = (
                    va_verify_answer_masks[mb_idx]
                    if (_va_masks_available
                        and mb_idx < len(va_verify_answer_masks)
                        and va_verify_answer_masks[mb_idx] is not None)
                    else None
                )
                # ZeRO-3 requires ALL ranks to execute the same number of forward passes.
                # We always run the forward here; when _mb_attn_va is None (all responses
                # in this minibatch are degenerate) we fall back to the standard attention
                # mask so the result equals a no-op, but the collective allgather stays
                # in sync across all ranks. The seg1_mask in the loss loop ensures
                # degenerate tokens don't contribute to the gradient regardless.
                mb_qr  = collated_query_responses[mb_idx].to(self.device)
                mb_pos = collated_position_ids[mb_idx].to(self.device)
                if _mb_attn_va is not None:
                    mb_attn_va = _mb_attn_va.to(self.device)
                else:
                    mb_attn_va = collated_attention_masks[mb_idx].to(self.device)
                lp_wo = self.forward(
                    self.model, mb_qr, mb_attn_va, mb_pos, pad_token_id, args.temperature,
                )
                # Only cache when the mask actually differs from attention_mask
                # (i.e. there are real non-degenerate VA responses in this minibatch).
                va_lp_wo_cache[mb_idx] = lp_wo.detach() if _mb_attn_va is not None else None

        local_step = 0
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            old_logprobs = [None for _ in range(len(collated_query_responses))]
            kl1_stats = torch.zeros(len(collated_query_responses))
            kl2_stats = torch.zeros(len(collated_query_responses))
            kl3_stats = torch.zeros(len(collated_query_responses))
            kl4_stats = torch.zeros(len(collated_query_responses))
            kl_loss_stats = torch.zeros(len(collated_query_responses))
            pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
            pg_loss_stats = torch.zeros(len(collated_query_responses))
            loss_stats = torch.zeros(len(collated_query_responses))
            ratio_stats = torch.zeros(len(collated_query_responses))
            # LFF monitoring
            _va_delta_vals: List[float] = []  # seg-1 distillation delta (lp_new - lp_without)
            _va_w_vals:     List[float] = []  # unused in LFF (kept for compat)
            # RS monitoring
            _rs_n_responses: int = 0          # number of responses with RS loss this step
            _rs_token_count: int = 0          # total RS tokens (numerator)
            _rs_total_tokens: int = 0         # total valid tokens (denominator)
            _rs_loss_vals:       List[float] = []   # RS loss values at RS positions
            _rs_lp_without_vals: List[float] = []   # lp_without at RS positions
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_ref_logprob = collated_ref_logprobs[i]
                    mb_query_responses = collated_query_responses[i]
                    mb_tool_mask = collated_tool_masks[i]
                    mb_advantages = collated_advantages[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
                    # if masking snippets, do it here.
                    if args.mask_tool_use and args.tool_use:
                        mb_response_masks_bool = mb_response_masks[:, 1:].bool() & mb_tool_mask[:, 1:].bool()
                    # LFF: exclude noise tokens (repeated VA calls/outputs) from loss.
                    # These tokens have advantage=0 already, but excluding them from
                    # the denominator of token-level normalization is cleaner.
                    if (lff_noise_masks and i < len(lff_noise_masks)
                            and lff_noise_masks[i] is not None):
                        _noise_bool = lff_noise_masks[i].to(self.device)[:, 1:].bool()
                        mb_response_masks_bool = mb_response_masks_bool & ~_noise_bool
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_new_logprobs = self.forward(
                        self.model,
                        mb_query_responses,
                        mb_attention_mask,
                        mb_position_id,
                        pad_token_id,
                        args.temperature,
                    )
                    mb_new_logprobs = torch.masked_fill(mb_new_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)

                    # Cache the old logprobs
                    with torch.no_grad():
                        if epoch_idx == 0:
                            old_logprobs[i] = mb_new_logprobs
                        mb_old_logprobs = old_logprobs[i].detach()

                    # ── LFF: two-segment ratio computation ────────────────────
                    # Seg-0 (pre-VA + VA call): ratio = π_θ(y0) / π_θ_old(y0)
                    # Seg-1 (post-VA):          ratio = π_θ(y1|f) / π_θ(y1)
                    #   where π_θ(y1) = lp_without (VA output masked in attention)
                    # VA output tokens: advantage=0 (set in _lff_build_packed_data),
                    #   excluded from response_mask → no gradient contribution.

                    # Standard ratio (used for seg-0 and no-VA responses)
                    logprobs_diff = mb_new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logprobs_diff)

                    # Seg-1 distillation ratio (overrides standard ratio at seg-1 positions)
                    # Controlled by lff_enable_va_reweight; when False the ratio stays ρ_t
                    # for all positions (standard DAPO), but lp_without is still computed if
                    # RS is enabled (both share the same masked-attention forward pass).
                    ratio_final = ratio.clone()
                    lp_wo_mb = va_lp_wo_cache.get(i) if va_lp_wo_cache else None
                    # Shared noise mask: tokens where |lp_with - lp_without| > 5 are
                    # noise (lp_without ≈ -inf, e.g. feedback echoes).  Build this
                    # whenever lp_without is available so BOTH VA reweighting and RS
                    # can exclude them, even if only one branch is enabled.
                    #
                    # IMPORTANT: keep two versions of the delta:
                    #   - _dist_diff_live  = mb_new_logprobs - lp_without (live wrt numerator)
                    #   - _dist_diff_stats = detached copy for noise filtering / logging
                    # This makes seg-1 ratio a true token-level reweight with gradient,
                    # analogous to PPO's live numerator over a frozen denominator.
                    _shared_noisy_mask = None
                    _dist_diff_live = None
                    _dist_diff_stats = None
                    if lp_wo_mb is not None and _va_masks_available:
                        _dist_diff_live = mb_new_logprobs - lp_wo_mb
                        _dist_diff_stats = _dist_diff_live.detach()
                        _shared_noisy_mask = _dist_diff_stats.abs() > 5.0
                    _enable_va_rw = getattr(args, "lff_enable_va_reweight", True)
                    if lp_wo_mb is not None and _va_masks_available and lff_seg1_masks and _enable_va_rw:
                        _mb_seg1 = (
                            lff_seg1_masks[i].to(self.device)
                            if i < len(lff_seg1_masks) and lff_seg1_masks[i] is not None
                            else None
                        )
                        if _mb_seg1 is not None:
                            # seg-1 positions (shifted by 1 for logprob alignment)
                            _seg1_pos = _mb_seg1[:, 1:].bool()
                            # distillation ratio: exp(lp_new - lp_without) at seg-1
                            assert _dist_diff_live is not None and _dist_diff_stats is not None
                            _va_delta_vals.extend(_dist_diff_stats[_seg1_pos & mb_response_masks_bool].tolist())

                            # Tokens where |delta| > 5 have numerically unreliable lp_without
                            # (near-zero probability under no-feedback context). Treat them as
                            # noise: exclude from loss entirely and fall back to standard ratio.
                            _dist_noisy = _shared_noisy_mask
                            assert _dist_noisy is not None
                            _noisy_seg1 = _seg1_pos & _dist_noisy
                            if _noisy_seg1.any():
                                mb_response_masks_bool = mb_response_masks_bool & ~_noisy_seg1

                            # Apply distillation ratio to reliable seg-1 tokens,
                            # clamped to [w_min, w_max] instead of the GRPO clip range.
                            # Values calibrated from observed delta distribution:
                            #   seg1_delta_mean p95≈0.60 → exp(delta_mean) p95≈1.82 → w_max=2.0
                            #   seg1_delta_min  mean≈-2.5 → token lower bound exp(-1.5)≈0.22 → w_min=0.2
                            #   seg1_delta_max  mean≈15.7 (handled by |Δ|>5 noise filter first)
                            _w_min = getattr(args, "lff_seg1_w_min", 0.2)
                            _w_max = getattr(args, "lff_seg1_w_max", 2.0)
                            _reliable_seg1 = _seg1_pos & ~_dist_noisy
                            ratio_final[_reliable_seg1] = torch.exp(
                                _dist_diff_live[_reliable_seg1]
                            ).clamp(_w_min, _w_max)

                    # Seg-0 and no-VA tokens: standard GRPO symmetric clipping.
                    # Seg-1 tokens already clamped to [w_min, w_max] above;
                    # applying the GRPO clip on top would narrow them back to [0.8, 1.2],
                    # so we preserve ratio_final for seg-1 and only clip seg-0/no-VA.
                    _seg1_written = (
                        lff_seg1_masks[i].to(self.device)[:, 1:].bool()
                        if (lff_seg1_masks and i < len(lff_seg1_masks)
                            and lff_seg1_masks[i] is not None)
                        else None
                    )
                    ratio_clipped = torch.where(
                        _seg1_written if _seg1_written is not None
                            else torch.zeros_like(ratio_final, dtype=torch.bool),
                        ratio_final,   # seg-1: already w_min/w_max clamped, skip GRPO clip
                        torch.clamp(ratio_final, 1.0 - args.clip_lower, 1.0 + args.clip_higher),
                    )
                    # ── end LFF ratio ──────────────────────────────────────────

                    # ── [DEBUG] Print ratio and advantage stats on first step ──
                    if epoch_idx == 0 and i == 0 and self.rank == 0:
                        with torch.no_grad():
                            _valid_m = mb_response_masks_bool
                            _adv_valid = mb_advantages[:, 1:][_valid_m]
                            _r0_valid  = ratio[_valid_m]
                            _rf_valid  = ratio_final[_valid_m]
                            _seg1_debug = None
                            if lp_wo_mb is not None and lff_seg1_masks and i < len(lff_seg1_masks) \
                                    and lff_seg1_masks[i] is not None:
                                _seg1_debug = lff_seg1_masks[i].to(self.device)[:, 1:].bool()
                            _n_seg1_tokens = int(_seg1_debug[_valid_m].sum().item()) if _seg1_debug is not None else 0
                            _n_seg0_tokens = int(_valid_m.sum().item()) - _n_seg1_tokens
                            _seg0_ratio = ratio[_valid_m & (~_seg1_debug if _seg1_debug is not None
                                          else torch.ones_like(_valid_m))]
                            _seg1_ratio = ratio_final[_valid_m & _seg1_debug] \
                                if _seg1_debug is not None and _n_seg1_tokens > 0 \
                                else torch.tensor([])
                            print(
                                f"\n[LFF RATIO/ADV DEBUG] epoch={epoch_idx} mb={i} "
                                f"valid_tokens={int(_valid_m.sum().item())} "
                                f"seg0_tokens={_n_seg0_tokens} seg1_tokens={_n_seg1_tokens}\n"
                                f"  seg-0 ratio (π_θ/π_θ_old): "
                                f"mean={_seg0_ratio.mean().item():.4f}  "
                                f"min={_seg0_ratio.min().item():.4f}  "
                                f"max={_seg0_ratio.max().item():.4f}\n"
                                + (
                                    f"  seg-1 ratio (π_θ(y1|f)/π_θ(y1)): "
                                    f"mean={_seg1_ratio.mean().item():.4f}  "
                                    f"min={_seg1_ratio.min().item():.4f}  "
                                    f"max={_seg1_ratio.max().item():.4f}\n"
                                    if _seg1_ratio.numel() > 0 else
                                    "  seg-1 ratio: N/A (no seg-1 tokens)\n"
                                )
                                + f"  advantage (all valid): "
                                f"mean={_adv_valid.mean().item():.4f}  "
                                f"min={_adv_valid.min().item():.4f}  "
                                f"max={_adv_valid.max().item():.4f}  "
                                f"pos_rate={(_adv_valid > 0).float().mean().item():.2f}",
                                flush=True,
                            )

                    pg_losses  = -mb_advantages[:, 1:] * ratio_final
                    pg_losses2 = -mb_advantages[:, 1:] * ratio_clipped
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    # KL regularization (applied to seg-0; zero for seg-1 where
                    # distillation ratio already handles regularization)
                    ref_logprobs_diff = (mb_new_logprobs - mb_ref_logprob).clamp(-40.0, 40.0)
                    kl1 = ref_logprobs_diff
                    kl2 = (ref_logprobs_diff) ** 2 / 2
                    kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff
                    kl4 = ratio * ref_logprobs_diff
                    if args.kl_estimator == "kl1":
                        kl = kl1
                    elif args.kl_estimator == "kl2":
                        kl = kl2
                    elif args.kl_estimator == "kl3":
                        kl = kl3
                    elif args.kl_estimator == "kl4":
                        kl = kl4

                    # Zero out KL for seg-1 tokens (distillation ratio regularizes them)
                    if lp_wo_mb is not None and lff_seg1_masks and i < len(lff_seg1_masks) \
                            and lff_seg1_masks[i] is not None:
                        _seg1_bool = lff_seg1_masks[i].to(self.device)[:, 1:].bool()
                        kl = kl.masked_fill(_seg1_bool, 0.0)

                    loss_combined = pg_loss_max + (args.beta * kl)

                    # ── RS distillation loss (rejection sampling, r1 > r0) ─────
                    # Controlled by lff_enable_rs; set False to ablate RS while keeping
                    # VA reweighting.  lff_rs_seg1_masks will also be empty when
                    # enable_rs=False was passed to _lff_build_packed_data.
                    _enable_rs = getattr(args, "lff_enable_rs", True)
                    if (_enable_rs
                            and lff_rs_seg1_masks and i < len(lff_rs_seg1_masks)
                            and lff_rs_seg1_masks[i] is not None
                            and lp_wo_mb is not None):
                        _mb_rs_seg1 = lff_rs_seg1_masks[i].to(self.device)[:, 1:].bool()
                        _mb_rs_r1   = lff_rs_r1[i].to(self.device)[:, 1:] \
                            if (lff_rs_r1 and i < len(lff_rs_r1)
                                and lff_rs_r1[i] is not None) else None
                        if _mb_rs_seg1.any() and _mb_rs_r1 is not None:
                            # RS loss = -λ_RS · r1 · lp_without at rs positions
                            _lff_rs_weight = getattr(args, "lff_rs_weight", 1.0)
                            _rs_loss = -_lff_rs_weight * _mb_rs_r1 * lp_wo_mb  # (batch, seq-1)
                            # Exclude noise tokens (|lp_with - lp_without| > 5) from RS loss.
                            # These tokens have lp_without ≈ -inf (e.g. feedback-echoing words)
                            # and would produce exploding gradients via -r1 * lp_without.
                            _rs_active = _mb_rs_seg1.clone()
                            if _shared_noisy_mask is not None:
                                _rs_active = _rs_active & ~_shared_noisy_mask
                                mb_response_masks_bool = mb_response_masks_bool & ~_shared_noisy_mask
                            loss_combined = torch.where(
                                _rs_active,
                                _rs_loss,
                                loss_combined,
                            )
                            # ── RS monitoring ──────────────────────────────────
                            with torch.no_grad():
                                _rs_valid = _mb_rs_seg1 & mb_response_masks_bool
                                if _rs_valid.any():
                                    # count distinct responses with ≥1 RS token
                                    _rs_n_responses += int(
                                        _rs_valid.any(dim=-1).sum().item()
                                    )
                                    _rs_tok = int(_rs_valid.sum().item())
                                    _rs_token_count  += _rs_tok
                                    _rs_total_tokens += int(mb_response_masks_bool.sum().item())
                                    _rs_loss_vals.extend(
                                        _rs_loss[_rs_valid].tolist()
                                    )
                                    _rs_lp_without_vals.extend(
                                        lp_wo_mb[_rs_valid].tolist()
                                    )

                    # GRPO sequence-level normalization: average loss per sequence,
                    # then average across sequences in the minibatch.
                    loss = masked_mean(loss_combined, mb_response_masks_bool)

                    loss = loss / accumulation_steps
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        # NOTE: in packed implementation, kl calculation are averages over response tokens
                        kl1_stats[i] = masked_mean(kl1, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl2_stats[i] = masked_mean(kl2, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl3_stats[i] = masked_mean(kl3, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl4_stats[i] = masked_mean(kl4, mb_response_masks_bool, args.masked_mean_axis).float()
                        if args.kl_estimator == "kl1":
                            kl_loss_stats[i] = kl1_stats[i] * args.beta
                        elif args.kl_estimator == "kl2":
                            kl_loss_stats[i] = kl2_stats[i] * args.beta
                        elif args.kl_estimator == "kl3":
                            kl_loss_stats[i] = kl3_stats[i] * args.beta
                        elif args.kl_estimator == "kl4":
                            kl_loss_stats[i] = kl4_stats[i] * args.beta
                        pg_clipfrac_stats[i] = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks_bool, args.masked_mean_axis
                        )
                        pg_loss_stats[i] = masked_mean(pg_loss_max, mb_response_masks_bool, args.masked_mean_axis)
                        loss_stats[i] = loss
                        ratio_stats[i] = masked_mean(ratio, mb_response_masks_bool, args.masked_mean_axis)

            with torch.no_grad():
                self.local_metrics.add("objective/kl_avg", kl1_stats.mean())
                self.local_metrics.add("objective/kl2_avg", kl2_stats.mean())
                self.local_metrics.add("objective/kl3_avg", kl3_stats.mean())
                self.local_metrics.add("objective/kl4_avg", kl4_stats.mean())
                self.local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
                self.local_metrics.add("loss/kl_avg", kl_loss_stats.mean())
                self.local_metrics.add("loss/total_avg", loss_stats.mean())
                self.local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
                self.local_metrics.add("val/ratio", ratio_stats.mean())
                self.local_metrics.add("val/ratio_var", ratio_stats.var())
                self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
                # Always add these metrics so all ranks have identical key sets.
                # Missing values default to 0 (no seg-1 tokens were processed).
                _va_t = torch.tensor(_va_delta_vals, dtype=torch.float32) if _va_delta_vals else torch.zeros(1)
                self.local_metrics.add("va/delta_avg", _va_t.mean())
                self.local_metrics.add("va/delta_max", _va_t.max())
                self.local_metrics.add("va/delta_min", _va_t.min())
                _va_w_t = torch.tensor(_va_w_vals, dtype=torch.float32) if _va_w_vals else torch.zeros(1)
                self.local_metrics.add("va/w_avg", _va_w_t.mean())
                self.local_metrics.add("va/w_max", _va_w_t.max())
                self.local_metrics.add("va/w_min", _va_w_t.min())
                _d = torch.tensor(_va_delta_vals, dtype=torch.float32) if _va_delta_vals else torch.zeros(1)
                self.local_metrics.add("lff/seg1_delta_mean", _d.mean())
                self.local_metrics.add("lff/seg1_delta_max",  _d.max())
                self.local_metrics.add("lff/seg1_delta_min",  _d.min())
                # ── RS distillation metrics ──────────────────────────────────
                # Always add so all ranks have identical key sets.
                self.local_metrics.add(
                    "lff/rs_n_responses",
                    torch.tensor(float(_rs_n_responses)),
                )
                self.local_metrics.add(
                    "lff/rs_token_ratio",
                    torch.tensor(
                        _rs_token_count / max(_rs_total_tokens, 1),
                        dtype=torch.float32,
                    ),
                )
                _rs_loss_t = torch.tensor(_rs_loss_vals, dtype=torch.float32) \
                    if _rs_loss_vals else torch.zeros(1)
                self.local_metrics.add("lff/rs_loss_avg", _rs_loss_t.mean())
                _rs_lp_t = torch.tensor(_rs_lp_without_vals, dtype=torch.float32) \
                    if _rs_lp_without_vals else torch.zeros(1)
                self.local_metrics.add("lff/rs_lp_without_avg", _rs_lp_t.mean())
                return self.local_metrics.get_metrics_list()

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: Dict[str, str]) -> None:
        args = self.args
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)
        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            if args.gs_bucket_path is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir,
                    args.gs_checkpoint_state_dir,
                )

    def save_model(self, output_dir: str) -> None:
        model_to_save = self.model
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv

        if self.rank == 0:
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            # save tokenizer
            self.tokenizer.save_pretrained(output_dir)

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                step_dir,
                leaderboard_name,
                args.oe_eval_max_length,
                wandb_url,
                training_step,
                args.oe_eval_tasks,
                args.stop_strings,
                args.gs_bucket_path,
                args.eval_priority,
            )


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: List[int],
        single_gpu_mode: bool,
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 2
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, 0, None, None)

        self.models.append(master_policy)
        master_addr, master_port = ray.get(master_policy.get_master_addr_port.remote())

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        assert get_bundle_index(0, [7, 8, 4]) == 0
        assert get_bundle_index(1, [7, 8, 4]) == 0
        assert get_bundle_index(7, [7, 8, 4]) == 1
        assert get_bundle_index(8, [7, 8, 4]) == 1
        assert get_bundle_index(9, [7, 8, 4]) == 1
        assert get_bundle_index(16, [7, 8, 4]) == 2

        # Setup worker models
        for rank in range(1, world_size):
            print(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, 0, master_addr, master_port)
            self.models.append(worker_policy)


def vllm_generate_thread(
    vllm_engines: List[ray.actor.ActorHandle],
    generation_config: SamplingParams,
    eval_generation_config: SamplingParams,
    inference_results_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    eval_prompt_token_ids: Optional[List[int]],
    evaluation_inference_results_Q: Queue,
    eval_freq: int,
    resume_training_step: int = 1,
    tool_use: bool = False,
):
    def _flatten_engine_outputs(outputs):
        response_ids = []
        finish_reasons = []  # either "stop" or "length"
        masks = []
        num_calls = []
        timeouts = []
        tool_errors = []
        tool_outputs = []
        tool_runtimes = []
        tool_calleds = []
        for output in outputs:
            for out in output.outputs:
                response_ids.append(list(out.token_ids))
                finish_reasons.append(out.finish_reason)
                if tool_use:
                    masks.append(out.mask)
                    num_calls.append(out.num_calls)
                    timeouts.append(out.timeout)
                    tool_errors.append(out.tool_error)
                    tool_outputs.append(out.tool_output)
                    tool_runtimes.append(out.tool_runtime)
                    tool_calleds.append(out.tool_called)
        # if not using the tool, mask is all 1s
        if not tool_use:
            masks = [[1] * len(response_ids[i]) for i in range(len(response_ids))]
            num_calls = [0] * len(response_ids)
            timeouts = [0] * len(response_ids)
            tool_errors = [""] * len(response_ids)
            tool_outputs = [""] * len(response_ids)
            tool_runtimes = [0] * len(response_ids)
            tool_calleds = [False] * len(response_ids)
        return (
            response_ids,
            finish_reasons,
            masks,
            (num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds),
        )

    def generate_with_engines(prompts: List[List[int]], sampling_params: SamplingParams):
        # Split queries between engines
        queries_per_engine = (len(prompts) + len(vllm_engines) - 1) // len(vllm_engines)
        split_queries = [prompts[i : i + queries_per_engine] for i in range(0, len(prompts), queries_per_engine)]
        # Generate responses in parallel across engines
        futures = [
            vllm_engine.generate.remote(sampling_params=sampling_params, prompt_token_ids=queries, use_tqdm=False)
            for vllm_engine, queries in zip(vllm_engines, split_queries)
        ]
        # Gather all responses (timeout prevents permanent hang if a ToolActor crashes)
        all_outputs = ray.get(futures)
        response_ids = []
        finish_reasons = []
        masks = []
        num_calls = []
        timeouts = []
        tool_errors = []
        tool_outputs = []
        tool_runtimes = []
        tool_calleds = []
        for outputs in all_outputs:
            _response_ids, _finish_reasons, _masks, _infos = _flatten_engine_outputs(outputs)
            _num_calls, _timeouts, _tool_errors, _tool_outputs, _tool_runtimes, _tool_calleds = _infos
            response_ids.extend(_response_ids)
            finish_reasons.extend(_finish_reasons)
            masks.extend(_masks)
            num_calls.extend(_num_calls)
            timeouts.extend(_timeouts)
            tool_errors.extend(_tool_errors)
            tool_outputs.extend(_tool_outputs)
            tool_runtimes.extend(_tool_runtimes)
            tool_calleds.extend(_tool_calleds)
        return (
            response_ids,
            finish_reasons,
            masks,
            (num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds),
        )

    def generate_with_engines_streaming(prompts: List[List[int]], sampling_params: SamplingParams, training_step: int):
        queries_per_engine = (len(prompts) + len(vllm_engines) - 1) // len(vllm_engines)
        split_queries = [prompts[i : i + queries_per_engine] for i in range(0, len(prompts), queries_per_engine)]
        pending = []
        prompt_offset = 0
        for engine_idx, (vllm_engine, queries_chunk) in enumerate(zip(vllm_engines, split_queries)):
            if len(queries_chunk) == 0:
                continue
            fut = vllm_engine.generate.remote(
                sampling_params=sampling_params,
                prompt_token_ids=queries_chunk,
                use_tqdm=False,
            )
            pending.append(
                {
                    "future": fut,
                    "engine_idx": engine_idx,
                    "prompt_offset": prompt_offset,
                    "prompt_count": len(queries_chunk),
                }
            )
            prompt_offset += len(queries_chunk)
        total_expected_responses = len(prompts) * int(getattr(sampling_params, "n", 1))
        while pending:
            ready_refs, _ = ray.wait([item["future"] for item in pending], num_returns=1)
            ready_ref = ready_refs[0]
            ready_idx = next(i for i, item in enumerate(pending) if item["future"] == ready_ref)
            item = pending.pop(ready_idx)
            outputs = ray.get(ready_ref)
            response_ids, finish_reasons, masks, infos = _flatten_engine_outputs(outputs)
            response_offset = item["prompt_offset"] * int(getattr(sampling_params, "n", 1))
            inference_results_Q.put(
                {
                    "training_step": training_step,
                    "engine_idx": item["engine_idx"],
                    "prompt_offset": item["prompt_offset"],
                    "prompt_count": item["prompt_count"],
                    "response_offset": response_offset,
                    "expected_total_responses": total_expected_responses,
                    "response_ids": response_ids,
                    "finish_reasons": finish_reasons,
                    "masks": masks,
                    "infos": infos,
                }
            )

    for training_step in range(resume_training_step, num_training_steps + 1):
        items = param_prompt_Q.get()
        if items is None:
            break
        _, g_queries_list = items  # first element is always None in LFF (no mixed-VA)

        with Timer("🔥 Generation time"):
            generate_with_engines_streaming(g_queries_list, generation_config, training_step)

        # Evaluate the model
        if eval_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
            response_ids, finish_reasons, masks, info = generate_with_engines(
                eval_prompt_token_ids, eval_generation_config
            )
            evaluation_inference_results_Q.put((response_ids, finish_reasons, masks, info))


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: Queue,
    packed_sequences_Q: Queue,
    queries_prompt_Q: Queue,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    rubric_buffer: Optional[Dict] = None,
):
    def _count_tool_calls_in_text(text: str, tool_name: str) -> int:
        if not text:
            return 0
        # Be tolerant to single/double quotes in tool call attributes.
        return text.count(f'<call_tool name="{tool_name}"') + text.count(f"<call_tool name='{tool_name}'")

    def _start_heartbeat(label: str, interval_s: float = 60.0):
        stop_evt = threading.Event()
        start_t = time.time()

        def _beat():
            while not stop_evt.wait(interval_s):
                print(
                    f"[Data Preparation Thread] ⏳ still in {label} | "
                    f"elapsed={time.time() - start_t:.1f}s",
                    flush=True,
                )

        t = threading.Thread(target=_beat, daemon=True)
        t.start()
        return stop_evt, start_t

    prescore_workers = max(1, min(32, int(os.environ.get("LFF_REWARD_PRESCORE_WORKERS", "8"))))
    prescore_executor = ThreadPoolExecutor(max_workers=prescore_workers)

    try:
        for training_step in range(1, num_training_steps + 1):
            # Get next batch of prompts and responses
            items = queries_prompt_Q.get()
            # Tuple: (queries, gt, ds, rq, None)  — trailing None reserved for future use
            queries, ground_truths, datasets, raw_user_query, _ = items
            use_prescore_pipeline_for_step = not bool(getattr(args, "apply_adaptive_rubric_reward", False))

            # ------------------------------------------------------------------------------------------------
            # Pack sequences: expand each prompt N times
            if args.num_samples_per_prompt_rollout > 1:
                queries       = [q  for q  in queries       for _ in range(args.num_samples_per_prompt_rollout)]
                ground_truths = [gt for gt in ground_truths for _ in range(args.num_samples_per_prompt_rollout)]
                datasets      = [ds for ds in datasets      for _ in range(args.num_samples_per_prompt_rollout)]
                raw_user_query= [rq for rq in raw_user_query for _ in range(args.num_samples_per_prompt_rollout)]
            with Timer("🚀 [Data Preparation Thread] Getting response ids"):
                expected_total_responses = len(queries)
                responses = [None] * expected_total_responses
                finish_reasons = [None] * expected_total_responses
                masks = [None] * expected_total_responses
                num_calls = [None] * expected_total_responses
                timeouts = [None] * expected_total_responses
                tool_errors = [None] * expected_total_responses
                tool_outputs = [None] * expected_total_responses
                tool_runtimes = [None] * expected_total_responses
                tool_calleds = [None] * expected_total_responses
                prescore_futures_by_idx = {}
                received_count = 0
                _chunk_count = 0
                while received_count < expected_total_responses:
                    chunk = inference_results_Q.get()
                    if not isinstance(chunk, dict):
                        # backward-compat fallback for older producer shape
                        _responses, _finish_reasons, _masks, _infos = chunk
                        _num_calls, _timeouts, _tool_errors, _tool_outputs, _tool_runtimes, _tool_calleds = _infos
                        for i in range(len(_responses)):
                            responses[i] = _responses[i]
                            finish_reasons[i] = _finish_reasons[i]
                            masks[i] = _masks[i]
                            num_calls[i] = _num_calls[i]
                            timeouts[i] = _timeouts[i]
                            tool_errors[i] = _tool_errors[i]
                            tool_outputs[i] = _tool_outputs[i]
                            tool_runtimes[i] = _tool_runtimes[i]
                            tool_calleds[i] = _tool_calleds[i]
                        received_count = len(_responses)
                        break

                    response_offset = int(chunk["response_offset"])
                    _responses = chunk["response_ids"]
                    _finish_reasons = chunk["finish_reasons"]
                    _masks = chunk["masks"]
                    _num_calls, _timeouts, _tool_errors, _tool_outputs, _tool_runtimes, _tool_calleds = chunk["infos"]
                    _chunk_count += 1
                    print(
                        f"[Data Preparation Thread] 📥 generation chunk | step={training_step} "
                        f"chunk={_chunk_count} engine={chunk.get('engine_idx', 'na')} "
                        f"responses={len(_responses)} offset={response_offset}/{expected_total_responses}",
                        flush=True,
                    )
                    for local_i in range(len(_responses)):
                        global_i = response_offset + local_i
                        if global_i >= expected_total_responses:
                            raise RuntimeError(
                                f"[Data Preparation Thread] Received response index {global_i} >= expected_total_responses={expected_total_responses}"
                            )
                        _resp_ids = list(_responses[local_i])
                        _mask_ids = list(_masks[local_i])
                        _finish_reason = _finish_reasons[local_i]
                        if _finish_reason == "stop" and (
                            len(_resp_ids) == 0 or _resp_ids[-1] != tokenizer.eos_token_id
                        ):
                            _resp_ids.append(tokenizer.eos_token_id)
                            _mask_ids.append(1)
                        responses[global_i] = _resp_ids
                        finish_reasons[global_i] = _finish_reason
                        masks[global_i] = _mask_ids
                        num_calls[global_i] = _num_calls[local_i]
                        timeouts[global_i] = _timeouts[local_i]
                        tool_errors[global_i] = _tool_errors[local_i]
                        tool_outputs[global_i] = _tool_outputs[local_i]
                        tool_runtimes[global_i] = _tool_runtimes[local_i]
                        tool_calleds[global_i] = _tool_calleds[local_i]
                        received_count += 1
                        if use_prescore_pipeline_for_step:
                            _decoded_query = raw_user_query[global_i]
                            _decoded_response = tokenizer.decode(_resp_ids, skip_special_tokens=True)
                            infos_single = (
                                [_num_calls[local_i]],
                                [_timeouts[local_i]],
                                [_tool_errors[local_i]],
                                [_tool_outputs[local_i]],
                                [_tool_runtimes[local_i]],
                                [_tool_calleds[local_i]],
                            )
                            prescore_futures_by_idx[global_i] = prescore_executor.submit(
                                _prescore_one_response_sync,
                                response_idx=global_i,
                                query_ids=queries[global_i],
                                response_ids=_resp_ids,
                                mask=_mask_ids,
                                decoded_response=_decoded_response,
                                ground_truth=ground_truths[global_i],
                                dataset=datasets[global_i],
                                finish_reason=_finish_reason,
                                infos_single=infos_single,
                                decoded_query=_decoded_query,
                                rubric_buffer=rubric_buffer,
                                training_step=training_step,
                            )
                good_outputs = [
                    len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                    for i in range(len(tool_outputs))
                ]
                infos = (num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds)

            with Timer("🔥 [Data Preparation Thread] Decoding responses", noop=True):
                decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
                decoded_responses_raw = list(decoded_responses)
                # TODO: make this configurable wrt what is in the raw query.
                # currently, is just the first user turn. 
                decoded_queries = raw_user_query
                stop_rate = sum(int(finish_reason == "stop") for finish_reason in finish_reasons) / len(finish_reasons)
            
                # DEBUG: Log a few sample responses to check for answer tags
                if training_step % 10 == 0 and args.log_training_rollouts:  # Log every 10 steps
                    print(f"[DEBUG] Sample responses at step {training_step}:")
                    for i in range(min(3, len(decoded_responses))):
                        print(f"  Response {i}: {decoded_responses[i][:200]}...{decoded_responses[i][-200:]}")
                        has_answer_tags = any(tag in decoded_responses[i] for tag in ['</answer0>', '</answer1>', '</answer>', '<answer>'])
                        print(f"  Has answer tags: {has_answer_tags}")
                    print()

            # Tool usage observability metrics:
            # per-step average calls per response and non-empty result rate per tool.
            verify_answer_avg_calls = 0.0
            google_search_avg_calls = 0.0
            snippet_search_avg_calls = 0.0
            other_search_mcp_avg_calls = 0.0
            browse_webpage_avg_calls = 0.0

            # Per-tool non-empty result rate: fraction of calls that returned non-empty tool_output
            # Parsed from <tool_output>...</tool_output> blocks in decoded responses.
            tool_nonempty_rates: dict = {}
            # Overall failure/empty rates across all tool calls
            overall_tool_call_failure_rate = 0.0
            overall_tool_output_empty_rate = 0.0

            def _count_nonempty_tool_outputs(text: str, tool_name: str) -> tuple[int, int]:
                """
                Returns (num_calls, num_nonempty_outputs) for a given tool in one response.
                Matches pairs of <call_tool name="tool_name">...</call_tool> and the
                immediately following <tool_output>...</tool_output>.
                """
                import re as _re
                # Find all call positions for this tool
                call_pattern = _re.compile(
                    rf'<call_tool\s+name=["\']?{_re.escape(tool_name)}["\']?[^>]*>',
                    _re.DOTALL,
                )
                output_pattern = _re.compile(r'<tool_output>(.*?)</tool_output>', _re.DOTALL)

                calls = list(call_pattern.finditer(text))
                outputs = list(output_pattern.finditer(text))
                num_calls = len(calls)
                if num_calls == 0:
                    return 0, 0

                # For each call, find the first tool_output that comes after it
                num_nonempty = 0
                for call_match in calls:
                    call_end = call_match.end()
                    # Find the first output after this call
                    next_output = next((o for o in outputs if o.start() > call_end), None)
                    if next_output is not None:
                        content = next_output.group(1).strip()
                        if content:
                            num_nonempty += 1
                return num_calls, num_nonempty

            if len(decoded_responses) > 0:
                mcp_tools_list = args.mcp_tool_names if isinstance(args.mcp_tool_names, list) else []
                tracked_tools = {"verify_answer", "google_search", "snippet_search", "browse_webpage"}
                other_search_tool_names = [
                    name for name in mcp_tools_list
                    if name not in tracked_tools
                ]

                # Accumulators: calls and nonempty counts per tool
                tool_calls_total: dict = {t: 0 for t in tracked_tools}
                tool_nonempty_total: dict = {t: 0 for t in tracked_tools}
                other_calls_total = 0
                other_nonempty_total = 0

                for text in decoded_responses:
                    for tool_name in tracked_tools:
                        c, ne = _count_nonempty_tool_outputs(text, tool_name)
                        tool_calls_total[tool_name] += c
                        tool_nonempty_total[tool_name] += ne
                    for tool_name in other_search_tool_names:
                        c, ne = _count_nonempty_tool_outputs(text, tool_name)
                        other_calls_total += c
                        other_nonempty_total += ne

                n = len(decoded_responses)
                verify_answer_avg_calls  = tool_calls_total["verify_answer"]  / n
                google_search_avg_calls  = tool_calls_total["google_search"]  / n
                snippet_search_avg_calls = tool_calls_total["snippet_search"] / n
                browse_webpage_avg_calls = tool_calls_total["browse_webpage"] / n
                other_search_mcp_avg_calls = other_calls_total / n

                # Non-empty output rate per tool (calls with non-empty output / total calls)
                for tool_name in tracked_tools:
                    total = tool_calls_total[tool_name]
                    tool_nonempty_rates[f"val/tool_nonempty_rate/{tool_name}"] = (
                        tool_nonempty_total[tool_name] / total if total > 0 else float("nan")
                    )
                if other_calls_total > 0:
                    tool_nonempty_rates["val/tool_nonempty_rate/other"] = other_nonempty_total / other_calls_total

                # Overall rates across all tool calls
                all_calls = sum(tool_calls_total.values()) + other_calls_total
                all_nonempty = sum(tool_nonempty_total.values()) + other_nonempty_total
                if all_calls > 0:
                    overall_tool_output_empty_rate = 1.0 - all_nonempty / all_calls

                # Tool call failure rate: calls that triggered tool_error (from infos)
                total_tool_errors = sum(1 for e in tool_errors if e and len(e) > 0)
                total_tool_calleds = sum(1 for c in tool_calleds if c)
                overall_tool_call_failure_rate = (
                    total_tool_errors / total_tool_calleds if total_tool_calleds > 0 else 0.0
                )

                print(
                    f"[Step {training_step}] Tool stats | "
                    f"failure_rate={overall_tool_call_failure_rate:.3f} "
                    f"empty_rate={overall_tool_output_empty_rate:.3f} | "
                    + " ".join(
                        f"{t}(calls={tool_calls_total[t]/n:.2f}, nonempty={tool_nonempty_rates.get(f'val/tool_nonempty_rate/{t}', float('nan')):.2f})"
                        for t in ["google_search", "snippet_search", "browse_webpage", "verify_answer"]
                    )
                )

            with Timer("💰 [Data Preparation Thread] Calculating rewards and advantages"):
                print(
                    f"[Data Preparation Thread] ▶ reward/adv start | "
                    f"step={training_step} n_responses={len(decoded_responses)}",
                    flush=True,
                )
                use_prescore_pipeline = use_prescore_pipeline_for_step
                # ── [DEBUG] Print one sample final response before r1 scoring ──
                if decoded_responses:
                        _dbg_idx = next(
                            (j for j, r in enumerate(decoded_responses) if "verify_answer" not in r),
                            0
                        )
                        _dbg_resp = decoded_responses[_dbg_idx]
                        _dbg_ans_m = re.search(r'<answer>(.*?)</answer>', _dbg_resp, re.DOTALL)
                        _dbg_ans = _dbg_ans_m.group(1).strip()[:300] if _dbg_ans_m else "[no <answer> tag]"
                        print(
                            f"\n[LFF DEBUG r1] step={training_step} sample_idx={_dbg_idx} "
                            f"(no-VA response)\n"
                            f"  response_prefix (200): {repr(_dbg_resp[:200])}\n"
                            f"  <answer> content (300): {repr(_dbg_ans)}",
                            flush=True,
                        )
                _hb_evt, _hb_start = _start_heartbeat(
                    f"{'reward_prescore' if use_prescore_pipeline else 'reward_fn'}(step={training_step})",
                    interval_s=60.0,
                )
                try:
                    if use_prescore_pipeline:
                        if len(prescore_futures_by_idx) != len(decoded_responses):
                            raise RuntimeError(
                                f"[Data Preparation Thread] prescore futures incomplete: "
                                f"got {len(prescore_futures_by_idx)} expected {len(decoded_responses)}"
                            )
                        prescored_items = [prescore_futures_by_idx[i].result() for i in range(len(decoded_responses))]
                        prescored_items.sort(key=lambda x: x.response_idx)
                        scores = [item.score for item in prescored_items]
                        reward_metrics = {}
                        reward_metric_lists: Dict[str, List[object]] = defaultdict(list)
                        reward_log_value_lists: Dict[str, List[object]] = defaultdict(list)
                        judge_failed = []
                        for item in prescored_items:
                            judge_failed.append(bool(item.judge_failed))
                            for k, v in item.reward_log_values.items():
                                reward_log_value_lists[k].append(v)
                            for key, value in item.metrics.items():
                                if key in {"__judge_failed__", "__reward_log_values__"}:
                                    continue
                                if key.startswith("objective/"):
                                    reward_metric_lists[key].append(value)
                                elif key.startswith("val/"):
                                    reward_metric_lists[key].append(value)
                        for key, values in reward_log_value_lists.items():
                            reward_metrics[f"objective/reward_log_values/{key}"] = _merge_reward_log_value_entries(values)
                        for key, values in reward_metric_lists.items():
                            try:
                                arr = np.array(values, dtype=np.float32)
                                reward_metrics[key] = float(arr.mean()) if arr.size > 0 else 0.0
                            except Exception:
                                reward_metrics[key] = values
                        reward_metrics["__judge_failed__"] = judge_failed
                        print(
                            f"[Data Preparation Thread] ✅ reward prescore done | "
                            f"step={training_step} elapsed={time.time() - _hb_start:.1f}s workers={prescore_workers}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Data Preparation Thread] ↪ adaptive-rubric enabled; fallback to batch reward_fn | step={training_step}",
                            flush=True,
                        )
                        scores, reward_metrics, rubric_buffer = asyncio.run(
                            reward_fn(
                                responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, decoded_queries,
                                rubric_buffer=rubric_buffer, is_training=True, training_step=training_step,
                                masks=masks, query_ids_list=queries,
                            )
                        )
                        print(
                            f"[Data Preparation Thread] ✅ reward_fn done | "
                            f"step={training_step} elapsed={time.time() - _hb_start:.1f}s",
                            flush=True,
                        )
                finally:
                    _hb_evt.set()
            
                # Add static rubrics to active rubrics every N steps when use_static_rubrics_as_persistent_rubrics is False
                if (rubric_buffer is not None and 
                    not args.use_static_rubrics_as_persistent_rubrics and 
                    training_step % args.add_static_rubrics_to_active_rubrics_every_n_steps == 0):
                
                    added_count = 0
                    for query, buffer_data in rubric_buffer.items():
                        static_rubrics = buffer_data.get("static_rubrics", [])
                        active_rubrics = buffer_data.get("active_rubrics", [])
                    
                        # Check each static rubric and add if not already in active rubrics
                        for static_rubric in static_rubrics:
                            # Check if this rubric is already in active rubrics (avoid duplicates)
                            if static_rubric not in active_rubrics:
                                active_rubrics.append(static_rubric)
                                added_count += 1
                
                    if added_count > 0:
                        print(f"[Static Rubric Addition] Added {added_count} static rubrics to active rubrics at step {training_step}")

                # Export rubric buffer snapshot for external feedback services (e.g., MCP draft_answer).
                if rubric_buffer is not None:
                    try:
                        export_rubric_buffer_snapshot(
                            rubric_buffer=rubric_buffer,
                            output_dir=args.output_dir,
                            training_step=training_step,
                        )
                    except Exception as e:
                        print(f"[Rubric Export] Failed to export rubric buffer snapshot: {e}")
            
                # ── LFF: verify_answer call bonus ────────────────────────────────────
                # Add a small bonus to scores of responses that called verify_answer.
                # This offsets the model's tendency to learn "skip VA → stable advantage"
                # and ensures verify_answer calls remain rewarded throughout training.
                # Applied before group normalization so it shifts the group mean correctly.
                _va_bonus = getattr(args, "lff_va_call_bonus", 0.0) if getattr(args, "lff_enabled", False) else 0.0
                if _va_bonus > 0:
                    _n_bonus = 0
                    for _bi, _resp in enumerate(decoded_responses):
                        if "verify_answer" in _resp:
                            scores[_bi] = scores[_bi] + _va_bonus
                            _n_bonus += 1
                    if _n_bonus > 0:
                        print(
                            f"[LFF va_bonus] step={training_step} "
                            f"bonus={_va_bonus} applied to {_n_bonus}/{len(scores)} VA responses",
                            flush=True,
                        )

                scores = np.array(scores)
                # Judge-agent timeout / API failure responses must not participate in
                # group normalisation or policy updates.
                _jf_raw = reward_metrics.pop("__judge_failed__", None)
                if _jf_raw is not None and len(_jf_raw) == len(scores):
                    _judge_failed = np.array(_jf_raw, dtype=bool)
                    _n_judge_failed = int(_judge_failed.sum())
                    if _n_judge_failed > 0:
                        _keep_idx = np.where(~_judge_failed)[0]
                        print(
                            f"[LFF Reward] Dropping {_n_judge_failed}/{len(scores)} judge-failed responses "
                            f"before advantage computation",
                            flush=True,
                        )
                        scores = scores[_keep_idx]
                        responses = [responses[i] for i in _keep_idx]
                        masks = [masks[i] for i in _keep_idx]
                        decoded_responses = [decoded_responses[i] for i in _keep_idx]
                        decoded_responses_raw = [decoded_responses_raw[i] for i in _keep_idx]
                        queries = [queries[i] for i in _keep_idx]
                        ground_truths = [ground_truths[i] for i in _keep_idx]
                        datasets = [datasets[i] for i in _keep_idx]
                        finish_reasons = [finish_reasons[i] for i in _keep_idx]
                        decoded_queries = [decoded_queries[i] for i in _keep_idx]
                        num_calls = [num_calls[i] for i in _keep_idx]
                        timeouts = [timeouts[i] for i in _keep_idx]
                        tool_errors = [tool_errors[i] for i in _keep_idx]
                        tool_outputs = [tool_outputs[i] for i in _keep_idx]
                        tool_runtimes = [tool_runtimes[i] for i in _keep_idx]
                        tool_calleds = [tool_calleds[i] for i in _keep_idx]
                        good_outputs = [good_outputs[i] for i in _keep_idx]
                scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
                mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
                mean_grouped_rewards = np.repeat(mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
                std_grouped_rewards = scores_per_prompt.std(axis=-1)
                std_grouped_rewards = np.repeat(std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
                if args.advantage_normalization_type == "standard":
                    advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
                elif args.advantage_normalization_type == "centered":
                    advantages = scores - mean_grouped_rewards
                else:
                    raise ValueError(f"Invalid advantage normalization type: {args.advantage_normalization_type}")

                # ── LFF: compute r0 (draft-answer reward) and two-segment advantages ──
                # This block runs after standard advantages are computed.
                # r1 = scores (final answer reward, already computed above)
                # r0 = draft answer reward (verify_answer call content + search context)
                lff_r0 = None          # np.ndarray[N], None if lff disabled or no VA
                lff_A0 = None          # np.ndarray[N], segment-0 advantages
                lff_A1 = None          # np.ndarray[N], segment-1 advantages
                lff_has_va = None      # List[bool][N]
                lff_force_full_a0 = None  # np.ndarray[N], rewritten degen draft uses A0 on all tokens

                if getattr(args, "lff_enabled", False):
                    args._lff_trace_info = None  # default; populated below if _any_va
                    # Identify which responses have a valid (first) verify_answer call
                    _has_va = ["verify_answer" in r for r in decoded_responses]
                    _any_va = any(_has_va)

                    if _any_va:
                        # ── Build r0: draft-answer reward ─────────────────────────
                        # For VA responses: r0 = reward of the draft answer submitted to verify_answer
                        #   (context = search history up to VA call + draft in <answer> tags)
                        # For no-VA responses: r0 = r1 (their final answer IS their only answer)
                        #   We do NOT score them separately; we simply assign r0[i] = r1[i]
                        #   so they participate correctly in the group mean used as A1 baseline.
                        _draft_decoded = []
                        _draft_answer_raw: List[Optional[str]] = []
                        _va_events_by_resp: List[List[Dict[str, Optional[Union[int, str]]]]] = []
                        _multi_va_rewritten_to_single: List[int] = []
                        _effective_r1_from_second_va: List[int] = []
                        # Per-response LFF stats for trace logging
                        _lff_trace_draft_len:    List[Optional[int]] = []   # chars in draft answer
                        _lff_trace_final_len:    List[Optional[int]] = []   # chars in final answer
                        _lff_trace_feedback_len: List[Optional[int]] = []   # chars in verify_answer tool_output
                        _lff_trace_va_count:     List[int]           = []   # number of verify_answer calls
                        _lff_trace_second_draft_len: List[Optional[int]] = []
                        _lff_trace_second_feedback_len: List[Optional[int]] = []
                        _lff_trace_effective_r1_parse_ok: List[int] = []
                        _lff_trace_multi_va_summary: List[Optional[str]] = []
                        _FINAL_ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
                        _TOOL_OUT_RE  = re.compile(
                            r'(?:<tool_output>|<tool_result>)(.*?)(?:</tool_output>|</tool_result>)',
                            re.DOTALL | re.IGNORECASE,
                        )
                        for resp, is_va in zip(decoded_responses, _has_va):
                            _events = _lff_extract_va_events(resp) if is_va else []
                            _va_events_by_resp.append(_events)
                            # Count all verify_answer calls (including noisy repeated ones)
                            _lff_trace_va_count.append(len(_events))
                            _multi_va_rewritten_to_single.append(0)
                            _effective_r1_from_second_va.append(0)
                            _lff_trace_effective_r1_parse_ok.append(0)

                            # Final answer length (last <answer> block in full response)
                            _all_ans = _FINAL_ANS_RE.findall(resp)
                            _lff_trace_final_len.append(len(_all_ans[-1]) if _all_ans else None)
                            if len(_events) >= 2:
                                _second_wrapped = _events[1].get("call_content_wrapped") or ""
                                _second_out = _events[1].get("output_text") or ""
                                _lff_trace_second_draft_len.append(len(_second_wrapped))
                                _lff_trace_second_feedback_len.append(len(_second_out))
                                _lff_trace_multi_va_summary.append(_lff_multi_va_debug_summary(resp, _events))
                            else:
                                _lff_trace_second_draft_len.append(None)
                                _lff_trace_second_feedback_len.append(None)
                                _lff_trace_multi_va_summary.append(None)

                            if is_va:
                                draft = _events[0].get("call_content") if _events else None
                                if draft is not None:
                                    _draft_decoded.append(_lff_build_draft_response(resp, draft))
                                    _draft_answer_raw.append(draft)
                                    _lff_trace_draft_len.append(len(draft))
                                    # Feedback = first tool_output after the verify_answer call
                                    _first_out = _events[0].get("output_text") if _events else None
                                    _lff_trace_feedback_len.append(len(_first_out) if _first_out else None)
                                else:
                                    # VA keyword present but extraction failed → treat as no-VA
                                    _draft_decoded.append(None)
                                    _draft_answer_raw.append(None)
                                    _lff_trace_draft_len.append(None)
                                    _lff_trace_feedback_len.append(None)
                            else:
                                _draft_decoded.append(None)  # will use r1 directly
                                _draft_answer_raw.append(None)
                                _lff_trace_draft_len.append(None)
                                _lff_trace_feedback_len.append(None)

                        # ── r0 scoring: parse verify_answer tool_output (no second judge call) ──
                        _va_indices = [i for i, d in enumerate(_draft_decoded) if d is not None]

                        # Start with r0 = r1 for all (no-VA responses keep their r1)
                        lff_r0 = scores.copy()

                        # Per-response r0 parse diagnostics (for trace logging)
                        _lff_r0_rubric_reward:       List[Optional[float]] = [None] * len(decoded_responses)
                        _lff_r0_citation_reward:     List[Optional[float]] = [None] * len(decoded_responses)
                        _lff_r0_format_reward:        List[Optional[float]] = [None] * len(decoded_responses)
                        _lff_r0_search_turns_reward:  List[Optional[float]] = [None] * len(decoded_responses)
                        _lff_r0_n_parsed:             List[Optional[int]]   = [None] * len(decoded_responses)
                        _lff_r0_n_skipped:            List[Optional[int]]   = [None] * len(decoded_responses)
                        _lff_r0_parse_ok:             List[int]             = [0]   * len(decoded_responses)

                        if _va_indices:
                            _n_parse_ok = 0
                            _n_parse_fallback = 0
                            _n_multi_va_effective_r1 = 0
                            _n_multi_va_samples = 0
                            for orig_i in _va_indices:
                                resp         = decoded_responses[orig_i]
                                draft_resp   = _draft_decoded[orig_i]
                                gt_raw       = ground_truths[orig_i]
                                # Unwrap list wrapper then parse JSON string if needed
                                _gt_unwrapped = gt_raw[0] if isinstance(gt_raw, list) else gt_raw
                                try:
                                    gt = json.loads(_gt_unwrapped) if isinstance(_gt_unwrapped, str) else _gt_unwrapped
                                except Exception:
                                    gt = _gt_unwrapped
                                # Normalise to dict so _lff_parse_r0_from_va_output can
                                # check rubrics / short-answer format uniformly.
                                # Short-answer GT is often a plain string (e.g. "92%") or
                                # list; wrap it to preserve the short-answer path in the
                                # parser (which only needs the dict to have no "rubrics" key).
                                if not isinstance(gt, dict):
                                    if isinstance(gt, list):
                                        gt = {"answer": gt[0]} if gt else {"answer": ""}
                                    else:
                                        gt = {"answer": str(gt)}

                                va_out = _lff_extract_va_tool_output(resp)
                                parse_result = None
                                if va_out:
                                    parse_result = _lff_parse_r0_from_va_output(
                                        va_output=va_out,
                                        draft_response=draft_resp,
                                        ground_truth=gt,
                                        verification_reward=args.verification_reward,
                                        mcp_parser_name=getattr(args, "mcp_parser_name", None),
                                    )

                                if parse_result is not None:
                                    lff_r0[orig_i] = parse_result["r0"]
                                    _lff_r0_rubric_reward[orig_i]      = parse_result["rubric_reward"]
                                    _lff_r0_citation_reward[orig_i]    = parse_result["citation_reward"]
                                    _lff_r0_format_reward[orig_i]       = parse_result["format_reward"]
                                    _lff_r0_search_turns_reward[orig_i] = parse_result["search_turns_reward"]
                                    _lff_r0_n_parsed[orig_i]            = parse_result["n_rubrics_parsed"]
                                    _lff_r0_n_skipped[orig_i]           = parse_result["n_rubrics_skipped"]
                                    _lff_r0_parse_ok[orig_i]            = 1
                                    _n_parse_ok += 1
                                else:
                                    # Fallback: r0 stays as r1 (already set above)
                                    _n_parse_fallback += 1

                                # Multi-VA conservative rewrite: if a second verify_answer
                                # exists, use its submitted answer as the final answer and
                                # parse the SECOND feedback as the effective final score.
                                _events = _va_events_by_resp[orig_i]
                                if len(_events) >= 2:
                                    _n_multi_va_samples += 1
                                    _second = _events[1]
                                    _second_answer_wrapped = _second.get("call_content_wrapped")
                                    _second_output = _second.get("output_text")
                                    _tool_error_i = tool_errors[orig_i] if orig_i < len(tool_errors) else ""
                                    _generation_side_second_va_rewrite = (
                                        isinstance(_tool_error_i, str)
                                        and "[verify_answer_second_call_rewritten_to_final_answer]" in _tool_error_i
                                    )
                                    _second_call_start = _second.get("call_start")
                                    _first_call_start = _events[0].get("call_start")
                                    _first_call_end = _events[0].get("call_end")
                                    _multi_va_summary = _lff_multi_va_debug_summary(resp, _events)
                                    if not isinstance(_second_answer_wrapped, str) or not _second_answer_wrapped.strip():
                                        raise RuntimeError(
                                            "[LFF multi-VA] Found a second verify_answer call but failed to extract "
                                            f"its submitted answer at step={training_step} sample_idx={orig_i}.\n"
                                            f"{_multi_va_summary}"
                                        )
                                    if not isinstance(_second_output, str) or not _second_output.strip():
                                        if _generation_side_second_va_rewrite:
                                            # Generation-side source fix in tool_vllm.py rewrites the SECOND
                                            # verify_answer call directly into the final answer and stops the
                                            # rollout. In this valid case there is intentionally NO second
                                            # tool_output feedback block. The final answer has already been
                                            # scored by reward_fn into scores[orig_i], so we only need to
                                            # rewrite the trajectory into a single-VA form for LFF.
                                            if not isinstance(_second_call_start, int):
                                                raise RuntimeError(
                                                    "[LFF multi-VA] Generation-side second-VA rewrite marker was present, "
                                                    "but failed to locate the second call span "
                                                    f"at step={training_step} sample_idx={orig_i}.\n"
                                                    f"tool_error={_tool_error_i!r}\n"
                                                    f"{_multi_va_summary}"
                                                )

                                            _prefix_before_second = resp[:_second_call_start].rstrip()
                                            _single_va_response = f"{_prefix_before_second}\n{_second_answer_wrapped}"

                                            _first_tool_spans: List[Tuple[int, int]] = []
                                            _first_output_start = _events[0].get("output_start")
                                            _first_output_end = _events[0].get("output_end")
                                            if isinstance(_first_call_start, int) and isinstance(_first_call_end, int):
                                                _first_tool_spans.append((_first_call_start, _first_call_end))
                                            if isinstance(_first_output_start, int) and isinstance(_first_output_end, int):
                                                _first_tool_spans.append((_first_output_start, _first_output_end))

                                            decoded_responses[orig_i] = _single_va_response
                                            responses[orig_i], masks[orig_i] = _lff_encode_response_with_tool_mask(
                                                response_text=_single_va_response,
                                                tokenizer=tokenizer,
                                                tool_char_spans=_first_tool_spans,
                                            )
                                            # Keep scores[orig_i] unchanged: reward_fn has already scored the
                                            # rewritten final answer produced from the second verify_answer call.
                                            _multi_va_rewritten_to_single[orig_i] = 1
                                            _effective_r1_from_second_va[orig_i] = 1
                                            print(
                                                f"[LFF multi-VA] step={training_step} sample_idx={orig_i} "
                                                "second_feedback_missing_but_generation_rewrite_detected=1 "
                                                "using_reward_fn_final_score=1",
                                                flush=True,
                                            )
                                            continue
                                        raise RuntimeError(
                                            "[LFF multi-VA] Found a second verify_answer call but its feedback "
                                            f"tool_output is missing/empty at step={training_step} sample_idx={orig_i}.\n"
                                            f"tool_error={_tool_error_i!r}\n"
                                            f"{_multi_va_summary}"
                                        )
                                    if not isinstance(_second_call_start, int):
                                        raise RuntimeError(
                                            "[LFF multi-VA] Found a second verify_answer call but failed to locate "
                                            f"its call span at step={training_step} sample_idx={orig_i}.\n"
                                            f"{_multi_va_summary}"
                                        )

                                    _prefix_before_second = resp[:_second_call_start].rstrip()
                                    _single_va_response = f"{_prefix_before_second}\n{_second_answer_wrapped}"
                                    second_parse = _lff_parse_r0_from_va_output(
                                        va_output=_second_output,
                                        draft_response=_single_va_response,
                                        ground_truth=gt,
                                        verification_reward=args.verification_reward,
                                        mcp_parser_name=getattr(args, "mcp_parser_name", None),
                                    )
                                    if second_parse is None:
                                        raise RuntimeError(
                                            "[LFF multi-VA] Failed to parse an effective final reward from the "
                                            f"SECOND verify_answer feedback at step={training_step} sample_idx={orig_i}.\n"
                                            f"rewritten_single_va_prefix={_single_va_response[:600]!r}\n"
                                            f"{_multi_va_summary}"
                                        )

                                    _first_tool_spans: List[Tuple[int, int]] = []
                                    _first_output_start = _events[0].get("output_start")
                                    _first_output_end = _events[0].get("output_end")
                                    if isinstance(_first_call_start, int) and isinstance(_first_call_end, int):
                                        _first_tool_spans.append((_first_call_start, _first_call_end))
                                    if isinstance(_first_output_start, int) and isinstance(_first_output_end, int):
                                        _first_tool_spans.append((_first_output_start, _first_output_end))
                                    decoded_responses[orig_i] = _single_va_response
                                    responses[orig_i], masks[orig_i] = _lff_encode_response_with_tool_mask(
                                        response_text=_single_va_response,
                                        tokenizer=tokenizer,
                                        tool_char_spans=_first_tool_spans,
                                    )
                                    scores[orig_i] = second_parse["r0"]
                                    _multi_va_rewritten_to_single[orig_i] = 1
                                    _effective_r1_from_second_va[orig_i] = 1
                                    _lff_trace_effective_r1_parse_ok[orig_i] = 1
                                    _n_multi_va_effective_r1 += 1

                            print(
                                f"\n[LFF r0 parse] step={training_step} "
                                f"n_va={len(_va_indices)} "
                                f"n_multi_va={_n_multi_va_samples} "
                                f"parse_ok={_n_parse_ok} "
                                f"fallback_to_r1={_n_parse_fallback} "
                                f"multi_va_effective_r1={_n_multi_va_effective_r1}",
                                flush=True,
                            )

                        # ── Compute group-level baseline from r0 ──────────────────
                        # ── Degenerate final answer detection ─────────────────────
                        # A VA response whose final <answer> is shorter than the threshold
                        # (e.g. "...") is considered degenerate:
                        #   1. Apply a penalty to its effective r1 score so A1 is strongly negative
                        #   2. Mark it so _lff_build_packed_data skips distillation ratio for seg-1
                        _degen_min_len = getattr(args, "lff_degenerate_final_ans_min_len", 50)
                        _degen_penalty = getattr(args, "lff_degenerate_final_ans_penalty", 0.5)
                        _is_degen = np.zeros(len(decoded_responses), dtype=bool)
                        lff_force_full_a0 = np.zeros(len(decoded_responses), dtype=bool)
                        _n_degen_short = 0
                        _n_degen_regressed = 0
                        if _degen_penalty > 0:
                            for _di, (_fl, _hv) in enumerate(zip(_lff_trace_final_len, _has_va)):
                                if _hv and (_fl is None or _fl < _degen_min_len):
                                    _is_degen[_di] = True
                                    _n_degen_short += 1

                        # r1 < r0: feedback caused quality regression.
                        # Skip seg-1 distillation for these responses to avoid reinforcing
                        # the degraded post-feedback reasoning and answer.
                        # They still get a negative A1 signal (r1 - mean(r0) < 0) from
                        # the standard DAPO ratio, which correctly penalizes the bad outcome.
                        for _di, _hv in enumerate(_has_va):
                            if _hv and not _is_degen[_di] and scores[_di] < lff_r0[_di]:
                                _is_degen[_di] = True
                                _n_degen_regressed += 1

                        _n_degen = int(_is_degen.sum())
                        _degen_rewrite_count = 0
                        if _n_degen > 0:
                            # Degenerate response handling (two distinct cases):
                            #
                            # Case A — short final answer (fl < min_len):
                            #   - seg-1 excluded from loss (noise mask)
                            #   - scores replaced by r0 (effective signal is draft only)
                            #   - extra penalty applied to scores_for_A1 so A0 is pushed negative,
                            #     strongly discouraging this degenerate behaviour
                            #
                            # Case B — r1 < r0 (feedback caused quality regression):
                            #   - seg-1 excluded from loss (no distillation on bad post-VA content)
                            #   - scores kept as original r1 (NOT replaced by r0)
                            #     Rationale: keeping r1 in the group mean lets the negative advantage
                            #     A0 = (r0 - mean_group) / std naturally penalise the draft-call pair
                            #     via the standard DAPO signal, without over-suppressing the VA
                            #     incentive.  Replacing with r0 was artificially inflating the group
                            #     mean, making VA responses look better than they are and masking the
                            #     true learning signal.
                            #   - no extra penalty (the negative A0 from DAPO is sufficient)
                            scores_for_A1 = scores.copy()
                            for _di in np.where(_is_degen)[0]:
                                _fl = _lff_trace_final_len[_di]
                                _is_short = _fl is None or _fl < _degen_min_len
                                _draft_rewritten = False
                                _draft_text = _draft_decoded[_di]
                                if _draft_text is not None:
                                    _draft_token_ids = tokenizer.encode(_draft_text, add_special_tokens=False)
                                    if len(_draft_token_ids) > 0:
                                        if tokenizer.eos_token_id is not None and _draft_token_ids[-1] != tokenizer.eos_token_id:
                                            _draft_token_ids.append(tokenizer.eos_token_id)
                                        responses[_di] = list(_draft_token_ids)
                                        masks[_di] = [1] * len(_draft_token_ids)
                                        decoded_responses[_di] = _draft_text
                                        _draft_rewritten = True
                                        lff_force_full_a0[_di] = True
                                        _degen_rewrite_count += 1
                                        # The rewritten draft trajectory should no longer be
                                        # treated as a VA rollout by downstream string-based
                                        # diagnostics.
                                        _has_va[_di] = False
                                if _is_short:
                                    # Short answer: replace scores with r0 AND add penalty
                                    scores[_di] = lff_r0[_di]
                                    scores_for_A1[_di] = max(0.0, lff_r0[_di] - _degen_penalty)
                                # r1<r0 (regressed): scores and scores_for_A1 keep original r1
                                # seg-1 is still excluded from loss via noise_mask (set earlier)
                            print(
                                f"[LFF degen] step={training_step} "
                                f"n_degen_short={_n_degen_short}(scores=r0,penalty={_degen_penalty}) "
                                f"n_degen_regressed={_n_degen_regressed}(scores=r1,no_penalty) "
                                f"rewritten_to_draft={_degen_rewrite_count} "
                                f"total={_n_degen} "
                                f"(all degen: rewritten to full draft trajectory with A0; "
                                f"short also uses scores=r0+penalty; regressed keeps scores=r1)",
                                flush=True,
                            )
                        else:
                            scores_for_A1 = scores

                        _n_multi_va_rewritten = sum(_multi_va_rewritten_to_single)
                        if _n_multi_va_rewritten > 0:
                            print(
                                f"[LFF multi-VA] step={training_step} "
                                f"rewritten_to_single_va={_n_multi_va_rewritten}/{sum(1 for c in _lff_trace_va_count if c >= 2)} "
                                f"effective_r1_from_second_va={sum(_effective_r1_from_second_va)} "
                                f"parse_ok={sum(_lff_trace_effective_r1_parse_ok)}",
                                flush=True,
                            )

                        # Group mean of r0 is the baseline for both A0 and A1:
                        #   A0_i = (r0_i - mean(r0)) / std(r0)   [draft quality vs group]
                        #   A1_i = (r1_i - mean(r0)) / std(r1)   [final quality vs draft baseline]
                        # A1 uses std(r1) instead of std(r0) so that when r0 has low variance
                        # (e.g. all fallback to r1), A1 remains well-scaled and doesn't explode.
                        _r0_per_prompt = lff_r0.reshape(-1, args.num_samples_per_prompt_rollout)
                        _r0_group_mean = np.repeat(_r0_per_prompt.mean(axis=1), args.num_samples_per_prompt_rollout)
                        _r0_group_std  = np.repeat(_r0_per_prompt.std(axis=1),  args.num_samples_per_prompt_rollout)

                        _r1_per_prompt = scores_for_A1.reshape(-1, args.num_samples_per_prompt_rollout)
                        _r1_group_std  = np.repeat(_r1_per_prompt.std(axis=1),  args.num_samples_per_prompt_rollout)

                        lff_A0 = (lff_r0        - _r0_group_mean) / np.maximum(_r0_group_std, 1.0)
                        lff_A1 = (scores_for_A1 - _r0_group_mean) / np.maximum(_r1_group_std, 1.0)
                        lff_has_va = _has_va

                        _va_arr    = np.array([lff_r0[j] for j, v in enumerate(_has_va) if v])
                        _nova_arr  = np.array([lff_r0[j] for j, v in enumerate(_has_va) if not v])
                        _r1_va     = np.array([scores[j]  for j, v in enumerate(_has_va) if v])
                        _r1_nova   = np.array([scores[j]  for j, v in enumerate(_has_va) if not v])
                        _A0_va     = np.array([lff_A0[j]  for j, v in enumerate(_has_va) if v])
                        _A1_va     = np.array([lff_A1[j]  for j, v in enumerate(_has_va) if v])
                        _A_nova    = np.array([lff_A1[j]  for j, v in enumerate(_has_va) if not v])
                        def _arr_stats(arr):
                            if len(arr) == 0:
                                return "n/a (empty)"
                            return f"mean={arr.mean():.3f}  min={arr.min():.3f}  max={arr.max():.3f}"

                        print(
                            f"\n[LFF ADVANTAGE DEBUG] step={training_step} "
                            f"N={len(decoded_responses)} n_va={sum(_has_va)} n_nova={sum(not v for v in _has_va)}\n"
                            f"  r0_va   : {_arr_stats(_va_arr)}  (draft answer reward)\n"
                            f"  r0_nova : {_arr_stats(_nova_arr)} (= r1 for no-VA)\n"
                            f"  r1_va   : {_arr_stats(_r1_va)}   (final answer reward, VA group)\n"
                            f"  r1_nova : {_arr_stats(_r1_nova)}  (final answer reward, no-VA group)\n"
                            f"  r1-r0 (VA improvement): mean={(_r1_va - _va_arr).mean():.3f}\n"
                            f"  r0_group_std: mean={_r0_group_std.mean():.4f}  min={_r0_group_std.min():.4f}  "
                            f"(A0 normalizer)\n"
                            f"  r1_group_std: mean={_r1_group_std.mean():.4f}  min={_r1_group_std.min():.4f}  "
                            f"(A1 normalizer)\n"
                            f"  A0_va   : mean={_A0_va.mean():.3f}   pos_rate={(_A0_va > 0).mean():.2f}  "
                            f"(seg-0 adv, draft vs r0 baseline)\n"
                            f"  A1_va   : mean={_A1_va.mean():.3f}   pos_rate={(_A1_va > 0).mean():.2f}  "
                            f"(seg-1 adv, final vs r0 baseline, normalized by r1 std)\n"
                            f"  A_nova  : {'n/a (no no-VA responses)' if len(_A_nova) == 0 else f'mean={_A_nova.mean():.3f}   pos_rate={(_A_nova > 0).mean():.2f}'}  "
                            f"(DAPO adv, no-VA group)",
                            flush=True,
                        )

                        # Stash LFF per-response stats for trace logging below
                        args._lff_trace_info = {
                            # ── Reward scores ──────────────────────────────────────
                            "lff/r0_scores":      lff_r0.tolist(),
                            "lff/r1_scores":      scores.tolist(),
                            "lff/r0_group_mean":  _r0_group_mean.tolist(),
                            "lff/r0_group_std":   _r0_group_std.tolist(),
                            "lff/r1_group_std":   _r1_group_std.tolist(),
                            "lff/r1_minus_r0":    (scores - lff_r0).tolist(),
                            # ── Advantage components ───────────────────────────────
                            "lff/A0":             lff_A0.tolist(),
                            "lff/A1":             lff_A1.tolist(),
                            # ── VA call metadata ───────────────────────────────────
                            "lff/has_va":         [int(v) for v in _has_va],
                            "lff/va_call_count":  _lff_trace_va_count,
                            "lff/draft_ans_len":  _lff_trace_draft_len,
                            "lff/final_ans_len":  _lff_trace_final_len,
                            "lff/feedback_len":   _lff_trace_feedback_len,
                            "lff/second_draft_ans_len": _lff_trace_second_draft_len,
                            "lff/second_feedback_len":  _lff_trace_second_feedback_len,
                            "lff/multi_va_summary": _lff_trace_multi_va_summary,
                            "lff/multi_va_rewritten_to_single": _multi_va_rewritten_to_single,
                            "lff/effective_r1_from_second_va": _effective_r1_from_second_va,
                            "lff/effective_r1_parse_ok": _lff_trace_effective_r1_parse_ok,
                            # ── Degenerate response flags ──────────────────────────
                            "lff/is_degen":       _is_degen.tolist(),
                            "lff/is_degen_short": [
                                int(_hv and (_fl is None or _fl < _degen_min_len))
                                for _hv, _fl in zip(_has_va, _lff_trace_final_len)
                            ],
                            "lff/is_degen_regressed": [
                                int(_hv and not (_fl is None or _fl < _degen_min_len) and scores[_di] < lff_r0[_di])
                                for _di, (_hv, _fl) in enumerate(zip(_has_va, _lff_trace_final_len))
                            ],
                            "lff/degen_rewritten_to_draft": [
                                int(bool(_is_degen[_di]) and _draft_decoded[_di] is not None)
                                for _di in range(len(_is_degen))
                            ],
                            # ── r0 parse diagnostics (from verify_answer tool_output) ─
                            # parse_ok=1: r0 came from parsed VA output; 0: fell back to r1
                            "lff/r0_parse_ok":            _lff_r0_parse_ok,
                            "lff/r0_rubric_reward":        _lff_r0_rubric_reward,
                            "lff/r0_citation_reward":      _lff_r0_citation_reward,
                            "lff/r0_format_reward":        _lff_r0_format_reward,
                            "lff/r0_search_turns_reward":  _lff_r0_search_turns_reward,
                            "lff/r0_n_rubrics_parsed":     _lff_r0_n_parsed,
                            "lff/r0_n_rubrics_skipped":    _lff_r0_n_skipped,
                        }
                    else:
                        args._lff_trace_info = None

                # Record advantage statistics for VA rollouts (responses that called verify_answer).
                # This validates the hypothesis that verify_answer enhances final answer quality:
                # if VA rollouts have systematically higher advantage than no-VA rollouts,
                # the feedback is genuinely improving response quality.
                va_advantages = [
                    advantages[i]
                    for i, resp in enumerate(decoded_responses)
                    if "verify_answer" in resp
                ]
                no_va_advantages = [
                    advantages[i]
                    for i, resp in enumerate(decoded_responses)
                    if "verify_answer" not in resp
                ]
                if va_advantages:
                    va_adv_arr = np.array(va_advantages)
                    print(
                        f"[VA Advantage] step={training_step} "
                        f"n_va={len(va_advantages)} "
                        f"adv_mean={va_adv_arr.mean():.4f} "
                        f"adv_min={va_adv_arr.min():.4f} "
                        f"adv_max={va_adv_arr.max():.4f} "
                        f"adv_positive_rate={( va_adv_arr > 0).mean():.3f}",
                        flush=True,
                    )
                if no_va_advantages:
                    no_va_adv_arr = np.array(no_va_advantages)
                    print(
                        f"[NoVA Advantage] step={training_step} "
                        f"n_no_va={len(no_va_advantages)} "
                        f"adv_mean={no_va_adv_arr.mean():.4f} "
                        f"adv_min={no_va_adv_arr.min():.4f} "
                        f"adv_max={no_va_adv_arr.max():.4f}",
                        flush=True,
                    )

            # Log training rollouts if enabled
            training_rollouts_data = None
            if args.log_training_rollouts and training_step % args.log_training_rollouts_freq == 0 and args.with_tracking:
                print(f"[Data Preparation Thread] 📊 Preparing training rollouts for logging at step {training_step}")
            
                # Select a subset of samples for logging
                num_to_log = min(args.num_training_rollouts_to_log, len(responses))
                if num_to_log > 0:
                    indices_to_log = np.random.choice(len(responses), size=num_to_log, replace=False)
                
                    training_rollouts_data = {
                        "prompt": [extract_user_query(tokenizer.decode(queries[i], skip_special_tokens=True)) for i in indices_to_log],
                        "response": [decoded_responses[i] for i in indices_to_log],
                        "scores": [float(scores[i]) for i in indices_to_log],
                        "advantages": [float(advantages[i]) for i in indices_to_log],
                        "ground_truth": [str(ground_truths[i]) for i in indices_to_log],
                        "finish_reason": [finish_reasons[i] for i in indices_to_log],
                        "dataset": [datasets[i] for i in indices_to_log],
                        "training_step": training_step,
                    }

            with Timer("📦 [Data Preparation Thread] Filtering sequences"):
                print(
                    f"[Data Preparation Thread] ▶ filtering start | step={training_step} n_scores={len(scores)}",
                    flush=True,
                )
                # Here we get the max possible score for each prompt, and see how many prompts are unsolved
                max_possible_score = 0
                if args.apply_verifiable_reward:
                    max_possible_score += args.verification_reward
                if args.apply_r1_style_format_reward and args.additive_format_reward:
                    max_possible_score += args.r1_style_format_reward
                unsolved_batch_size_ratio = ((scores != max_possible_score) > 0).sum() / len(scores)
                # In standard GRPO, if the std of grouped rewards is 0, the batch has
                # zero gradient.  Under LFF, however, rewritten degenerate samples may
                # still have non-zero A0 signal through r0 even when r1/scores are
                # constant within the group, so use r0-group std when available.
                if lff_r0 is not None:
                    _lff_r0_per_prompt = lff_r0.reshape(-1, args.num_samples_per_prompt_rollout)
                    non_zero_std_mask = _lff_r0_per_prompt.std(axis=-1) != 0
                else:
                    non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
                real_batch_size_ratio = non_zero_std_mask.sum() * args.num_samples_per_prompt_rollout / len(scores)
                expanded_mask = np.repeat(non_zero_std_mask, args.num_samples_per_prompt_rollout)
                non_zero_gradient_index = np.where(expanded_mask)[0]
                _pre_filter_len = len(decoded_responses)
                advantages = advantages[non_zero_gradient_index]
                scores = scores[non_zero_gradient_index]
                responses = [responses[i] for i in non_zero_gradient_index]
                masks = [masks[i] for i in non_zero_gradient_index]
                decoded_responses = [decoded_responses[i] for i in non_zero_gradient_index]
                decoded_responses_raw = [decoded_responses_raw[i] for i in non_zero_gradient_index]
                queries = [queries[i] for i in non_zero_gradient_index]
                ground_truths = [ground_truths[i] for i in non_zero_gradient_index]
                datasets = [datasets[i] for i in non_zero_gradient_index]
                finish_reasons = [finish_reasons[i] for i in non_zero_gradient_index]
                decoded_queries = [decoded_queries[i] for i in non_zero_gradient_index]
                num_calls = [num_calls[i] for i in non_zero_gradient_index]
                timeouts = [timeouts[i] for i in non_zero_gradient_index]
                tool_errors = [tool_errors[i] for i in non_zero_gradient_index]
                tool_outputs = [tool_outputs[i] for i in non_zero_gradient_index]
                tool_runtimes = [tool_runtimes[i] for i in non_zero_gradient_index]
                tool_calleds = [tool_calleds[i] for i in non_zero_gradient_index]
                good_outputs = [good_outputs[i] for i in non_zero_gradient_index]
                args._lff_trace_info = _slice_trace_dict_lists(
                    getattr(args, "_lff_trace_info", None),
                    non_zero_gradient_index,
                    expected_len=_pre_filter_len,
                )
                # LFF: keep A0/A1/has_va in sync with non_zero_gradient_index
                if lff_A0 is not None:
                    lff_A0 = lff_A0[non_zero_gradient_index]
                    lff_A1 = lff_A1[non_zero_gradient_index]
                    lff_has_va = [lff_has_va[i] for i in non_zero_gradient_index]
                    if lff_force_full_a0 is not None:
                        lff_force_full_a0 = lff_force_full_a0[non_zero_gradient_index]
                    if lff_r0 is not None:
                        lff_r0 = lff_r0[non_zero_gradient_index]
                    if "_is_degen" in dir():
                        _is_degen = _is_degen[non_zero_gradient_index]
                if args.mask_truncated_completions:
                    stop_idxes = torch.tensor([i for i in range(len(finish_reasons)) if finish_reasons[i] == "stop"], dtype=torch.long)
                    _pre_stop_filter_len = len(decoded_responses)
                    scores = scores[stop_idxes]
                    advantages = advantages[stop_idxes]
                    responses = [responses[i] for i in stop_idxes]
                    masks = [masks[i] for i in stop_idxes]
                    decoded_responses = [decoded_responses[i] for i in stop_idxes]
                    decoded_responses_raw = [decoded_responses_raw[i] for i in stop_idxes]
                    queries = [queries[i] for i in stop_idxes]
                    ground_truths = [ground_truths[i] for i in stop_idxes]
                    datasets = [datasets[i] for i in stop_idxes]
                    finish_reasons = [finish_reasons[i] for i in stop_idxes]
                    decoded_queries = [decoded_queries[i] for i in stop_idxes]
                    num_calls = [num_calls[i] for i in stop_idxes]
                    timeouts = [timeouts[i] for i in stop_idxes]
                    tool_errors = [tool_errors[i] for i in stop_idxes]
                    tool_outputs = [tool_outputs[i] for i in stop_idxes]
                    tool_runtimes = [tool_runtimes[i] for i in stop_idxes]
                    tool_calleds = [tool_calleds[i] for i in stop_idxes]
                    good_outputs = [good_outputs[i] for i in stop_idxes]
                    args._lff_trace_info = _slice_trace_dict_lists(
                        getattr(args, "_lff_trace_info", None),
                        stop_idxes,
                        expected_len=_pre_stop_filter_len,
                    )
                    # LFF: keep A0/A1/has_va in sync with stop_idxes
                    if lff_A0 is not None:
                        stop_np = stop_idxes.numpy()
                        lff_A0 = lff_A0[stop_np]
                        lff_A1 = lff_A1[stop_np]
                        lff_has_va = [lff_has_va[i] for i in stop_np]
                        if lff_force_full_a0 is not None:
                            lff_force_full_a0 = lff_force_full_a0[stop_np]
                        if lff_r0 is not None:
                            lff_r0 = lff_r0[stop_np]
                        if "_is_degen" in dir():
                            _is_degen = _is_degen[stop_np]
                print(
                    f"[Data Preparation Thread] ✅ filtering done | step={training_step} kept={len(responses)}",
                    flush=True,
                )

            with Timer("📦 [Data Preparation Thread] Packing sequences"):
                print(
                    f"[Data Preparation Thread] ▶ packing start | step={training_step} kept={len(responses)}",
                    flush=True,
                )
                packed_sequences = pack_sequences(
                    queries=queries,
                    responses=responses,
                    masks=masks,
                    pack_length=args.pack_length,
                    pad_token_id=tokenizer.pad_token_id,
                )
                num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
                # Vectorized advantage calculation: create a lookup array where each index corresponds to a response mask value
                # and each value is the corresponding advantage score: index 0 is set to 0 since response masks start from 1 (1-indexed)
                lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
                lookup_advantages[1:] = advantages
                packed_advantages = [
                    torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                    for packed_mask in packed_sequences.response_masks
                ]
                packed_sequences.advantages = packed_advantages

                # ── LFF: build per-token advantages and masks ─────────────────────
                # Run before the SD hook so that packed_sequences.advantages contains
                # LFF segment advantages, and va_verify_answer_mask / lff masks are
                # stored for train() to use.
                if getattr(args, "lff_enabled", False) and lff_A0 is not None:
                    _lff_build_packed_data(
                        packed_sequences=packed_sequences,
                        responses=responses,
                        masks=masks,
                        tokenizer=tokenizer,
                        lff_A0=lff_A0,
                        lff_A1=lff_A1,
                        lff_has_va=lff_has_va,
                        advantages_dapo=advantages,  # standard DAPO advantage for no-VA responses
                        is_degen=_is_degen if "_is_degen" in dir() else None,
                        force_full_a0=lff_force_full_a0,
                        r1_scores=scores,      # effective scores (for rs: r1 > r0 check)
                        r0_scores=lff_r0,      # draft scores
                        enable_va_reweight=getattr(args, "lff_enable_va_reweight", True),
                        enable_rs=getattr(args, "lff_enable_rs", True),
                    )

                # Extension hook: allow subclasses / SD variants to inject per-token advantages
                # before collation.  If _sd_per_token_advantages_hook is set on args, call it.
                # Signature: hook(packed_sequences, responses, masks, advantages, non_zero_gradient_index) -> None
                # The hook modifies packed_sequences.advantages in-place.
                _sd_hook = getattr(args, "_sd_per_token_advantages_hook", None)
                if _sd_hook is not None:
                    try:
                        _sd_hook(packed_sequences, responses, masks, advantages, non_zero_gradient_index)
                    except Exception as _hook_err:
                        print(f"[SD hook] Error in per-token advantages hook (skipped): {_hook_err}")
                print(
                    f"[Data Preparation Thread] ✅ packing done | step={training_step} packed={len(packed_sequences.query_responses)}",
                    flush=True,
                )

            # if we have less batches than world size, we need to pad out so each world is fine
            # ideally, you should avoid this since its wasting computation.
            if args.allow_world_padding:
                with Timer("🤺 [Data Preparation Thread] Padding sequences for world size"):
                    shortfall = args.world_size - len(packed_sequences.query_responses)
                    if shortfall > 0:
                        print(
                            f"Padding {shortfall} sequences for world size. In future, you should adjust your compute this."
                        )
                        # construct "dummy" sequences for padding out the world size
                        dummy_qr = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], dtype=torch.long)
                        dummy_tool_mask = torch.zeros_like(dummy_qr)
                        dummy_attention = torch.tensor([1, 1], dtype=torch.long)
                        dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
                        dummy_response_mask = torch.zeros_like(dummy_qr)
                        dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
                        # pad out the world size
                        for _ in range(shortfall):
                            packed_sequences.query_responses.append(dummy_qr)
                            packed_sequences.tool_masks.append(dummy_tool_mask)
                            packed_sequences.attention_masks.append(dummy_attention)
                            packed_sequences.position_ids.append(dummy_position_ids)
                            packed_sequences.response_masks.append(dummy_response_mask)
                            packed_sequences.advantages.append(dummy_advantage)

            with Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker"):
                print(
                    f"[Data Preparation Thread] ▶ collate start | step={training_step}",
                    flush=True,
                )
                B = (
                    len(packed_sequences.query_responses) // args.world_size
                )  # essentially doing `drop_last=True`, which is fine.
                collated_data = []
                for i in range(args.world_size):
                    per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
                    per_device_packed_tool_masks = packed_sequences.tool_masks[B * i : B * (i + 1)]
                    per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
                    per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
                    per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
                    per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]

                    # Shuffle the batch and collate the data
                    b_inds = np.random.permutation(len(per_device_packed_query_responses))
                    collated_query_responses = []
                    collated_tool_masks = []
                    collated_attention_masks = []
                    collated_position_ids = []
                    collated_response_masks = []
                    collated_advantages = []
                    for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                        micro_range = b_inds[j : j + args.per_device_train_batch_size]
                        collated_query_responses.append(
                            collate_fn(
                                [per_device_packed_query_responses[idx] for idx in micro_range], tokenizer.pad_token_id
                            )
                        )
                        collated_tool_masks.append(
                            collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0)
                        )
                        collated_attention_masks.append(
                            collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0)
                        )
                        collated_position_ids.append(
                            collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0)
                        )
                        collated_response_masks.append(
                            collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0)
                        )
                        collated_advantages.append(
                            collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0)
                        )
                    # VA weight data: verify_answer_mask (packed, same shape as attention_masks)
                    # and post_va_mask (per response, keyed by orig_idx).
                    # Only present when va_weight_enabled.
                    packed_va_verify_answer_mask = getattr(packed_sequences, "va_verify_answer_mask", None) or []
                    va_post_va_mask              = getattr(packed_sequences, "va_post_va_mask",        None) or {}

                    # LFF data: seg1_mask and noise_mask (packed, same shape as attention_masks).
                    packed_lff_seg1_mask  = getattr(packed_sequences, "lff_seg1_mask",  None) or []
                    packed_lff_noise_mask = getattr(packed_sequences, "lff_noise_mask", None) or []
                    lff_any_va            = getattr(packed_sequences, "lff_any_va",     False)

                    def _collate_packed_mask(packed_mask_list, pad_value=1):
                        """Collate a packed mask list with the SAME b_inds shuffle."""
                        result = []
                        if not packed_mask_list:
                            return result
                        per_device = packed_mask_list[B * i : B * (i + 1)]
                        for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                            micro_range = b_inds[j : j + args.per_device_train_batch_size]
                            result.append(
                                collate_fn([per_device[idx] for idx in micro_range], pad_value)
                            )
                        return result

                    # Collate va_verify_answer_mask (1=attend, 0=mask VA output).
                    # Degenerate responses have no va_verify_answer_mask entry, so their
                    # packed mask is all-1 (same as attention_mask) — that is fine because
                    # train() always runs forward for every minibatch to keep all ranks in
                    # sync under ZeRO-3 (NCCL requires identical collective call counts).
                    collated_va_verify_answer_masks = _collate_packed_mask(packed_va_verify_answer_mask, 1)
                    if collated_va_verify_answer_masks:
                        assert len(collated_va_verify_answer_masks) == len(collated_query_responses), (
                            f"va_verify_answer_masks count {len(collated_va_verify_answer_masks)} != "
                            f"collated_query_responses count {len(collated_query_responses)}"
                        )

                    # Collate LFF masks (0-padded: excluded positions don't affect loss)
                    collated_lff_seg1_masks  = _collate_packed_mask(packed_lff_seg1_mask,  0)
                    collated_lff_noise_masks = _collate_packed_mask(packed_lff_noise_mask, 0)
                    # RS distillation: seg-1 mask and r1 weights for r1>r0 responses
                    packed_lff_rs_seg1 = getattr(packed_sequences, "lff_rs_seg1_mask", None) or []
                    packed_lff_rs_r1   = getattr(packed_sequences, "lff_rs_r1",        None) or []
                    collated_lff_rs_seg1_masks = _collate_packed_mask(packed_lff_rs_seg1, 0)
                    # r1 weights use float collation (pad with 0.0)
                    collated_lff_rs_r1 = []
                    if packed_lff_rs_r1:
                        per_device_rs_r1 = packed_lff_rs_r1[B * i : B * (i + 1)]
                        for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                            micro_range = b_inds[j : j + args.per_device_train_batch_size]
                            collated_lff_rs_r1.append(
                                collate_fn([per_device_rs_r1[idx] for idx in micro_range], 0.0)
                            )

                    collated_data.append(
                        {
                            "collated_query_responses": collated_query_responses,
                            "collated_tool_masks": collated_tool_masks,
                            "collated_attention_masks": collated_attention_masks,
                            "collated_position_ids": collated_position_ids,
                            "collated_advantages": collated_advantages,
                            "collated_response_masks": collated_response_masks,
                            "va_verify_answer_masks": collated_va_verify_answer_masks,
                            "va_post_va_mask":        va_post_va_mask,
                            # LFF fields
                            "lff_seg1_masks":      collated_lff_seg1_masks,
                            "lff_noise_masks":     collated_lff_noise_masks,
                            "lff_any_va":          lff_any_va,
                            # RS distillation fields
                            "lff_rs_seg1_masks":   collated_lff_rs_seg1_masks,
                            "lff_rs_r1":           collated_lff_rs_r1,
                        }
                    )
                print(
                    f"[Data Preparation Thread] ✅ collate done | step={training_step} B={B} workers={len(collated_data)}",
                    flush=True,
                )

            # Create a result package with metrics and data
            sequence_lengths = np.array([len(response) for response in responses])
            sequence_length_solved = (
                np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == max_possible_score])
            )
            sequence_length_unsolved = (
                np.array([]) if np.all(scores == max_possible_score) else np.array(sequence_lengths[scores == 0])
            )
            if len(responses) > 0:
                metrics = {
                    "scores": np.array(scores).mean(),
                    "real_batch_size_ratio": real_batch_size_ratio,
                    "unsolved_batch_size_ratio": unsolved_batch_size_ratio,
                    "packed_ratio": len(packed_sequences.query_responses) / len(responses),
                    "val/sequence_lengths": sequence_lengths.mean(),
                    "val/sequence_lengths_min": sequence_lengths.min(),
                    "val/sequence_lengths_max": sequence_lengths.max(),
                    "val/sequence_lengths_unsolved": (
                        0 if len(sequence_length_unsolved) == 0 else sequence_length_unsolved.mean()
                    ),
                    "val/sequence_lengths_solved": 0 if len(sequence_length_solved) == 0 else sequence_length_solved.mean(),
                    "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
                    "val/sequence_lengths_solved_hist": sequence_length_solved,
                    "val/stop_rate": stop_rate,
                    "val/advantages_mean": advantages.mean(),
                    "val/advantages_min": advantages.min(),
                    "val/advantages_max": advantages.max(),
                    "val/advantages_hist": advantages,
                    "val/num_calls_rate": np.array(num_calls).mean(),
                    "val/timeouts_rate": np.array(timeouts).mean(),
                    "val/tool_errors_rate": np.array([len(item) > 0 for item in tool_errors]).mean(),
                    "val/good_outputs_rate": np.array(good_outputs).mean(),
                    "val/tool_runtimes_rate": np.array(tool_runtimes).mean(),
                    "val/tool_calleds_rate": np.array(tool_calleds).mean(),
                    "val/verify_answer_calls_avg_per_response": verify_answer_avg_calls,
                    "val/google_search_calls_avg_per_response": google_search_avg_calls,
                    "val/snippet_search_calls_avg_per_response": snippet_search_avg_calls,
                    "val/browse_webpage_calls_avg_per_response": browse_webpage_avg_calls,
                    "val/other_search_mcp_calls_avg_per_response": other_search_mcp_avg_calls,
                    "val/tool_call_failure_rate": overall_tool_call_failure_rate,
                    "val/tool_output_empty_rate": overall_tool_output_empty_rate,
                    **tool_nonempty_rates,
                    **reward_metrics,
                    # VA advantage metrics: validate that verify_answer enhances final answer quality.
                    # If VA rollouts have systematically higher advantage than no-VA rollouts,
                    # the feedback is genuinely improving response quality.
                    **(
                        {
                            "val/va_advantage_mean":          float(np.array(va_advantages).mean()),
                            "val/va_advantage_min":           float(np.array(va_advantages).min()),
                            "val/va_advantage_max":           float(np.array(va_advantages).max()),
                            "val/va_advantage_positive_rate": float((np.array(va_advantages) > 0).mean()),
                            "val/va_rollout_count":           float(len(va_advantages)),
                        }
                        if va_advantages else {}
                    ),
                    **(
                        {
                            "val/no_va_advantage_mean": float(np.array(no_va_advantages).mean()),
                            "val/no_va_advantage_min":  float(np.array(no_va_advantages).min()),
                            "val/no_va_advantage_max":  float(np.array(no_va_advantages).max()),
                            "val/no_va_rollout_count":  float(len(no_va_advantages)),
                        }
                        if no_va_advantages else {}
                    ),
                }
            else:
                print("No responses to evaluate. Not logging any metrics. We will end up skipping this step.")
                metrics = {}

            if args.save_traces:
                # Save decoded strings for human readability instead of token ids.
                decoded_responses_for_traces = list(decoded_responses)
                traces = {
                    "scores": scores.tolist(),
                    "advantages": advantages.tolist() if isinstance(advantages, np.ndarray) else list(advantages),
                    "finish_reasons": finish_reasons,
                    # raw_responses = decoded rollouts BEFORE any LFF degen rewrite.
                    # responses       = actual training trajectories AFTER filtering / rewrite.
                    "raw_responses": decoded_responses_raw,
                    "responses": decoded_responses_for_traces,
                    "response_was_rewritten": [
                        int(raw != cur) for raw, cur in zip(decoded_responses_raw, decoded_responses_for_traces)
                    ],
                    "queries": decoded_queries,
                    "ground_truths": ground_truths,
                    "datasets": datasets,
                    "training_step": training_step,
                    "num_calls": num_calls,
                    "timeouts": timeouts,
                    "tool_errors": tool_errors,
                    "tool_outputs": tool_outputs,
                    "tool_runtimes": tool_runtimes,
                    "tool_calleds": tool_calleds,
                    "good_outputs": good_outputs,
                    "has_verify_answer_raw": [int("verify_answer" in r) for r in decoded_responses_raw],
                    "has_verify_answer_training": [int("verify_answer" in r) for r in decoded_responses_for_traces],
                    **reward_metrics,
                }
                # Attach per-rubric score/reasoning/evidence from the reward judge.
                # rubric_feedback_by_title: {rubric_hash: [{criterion, score, reasoning, evidence}, ...]}
                _rubric_fb = log_values.get("rubric_feedback_by_title") if "log_values" in dir() else None
                if _rubric_fb:
                    traces["rubric_feedback_by_title"] = _rubric_fb
                # Attach LFF per-response stats when available
                _lff_trace = getattr(args, "_lff_trace_info", None)
                if _lff_trace is not None:
                    traces.update(_lff_trace)
                    _trace_multi_va = traces.get("lff/multi_va_rewritten_to_single", [])
                    _trace_degen = traces.get("lff/is_degen", [])
                    print(
                        f"[TRACE] step={training_step} save_traces=1 "
                        f"n_samples={len(traces.get('responses', []))} "
                        f"n_multi_va_rewritten={sum(_trace_multi_va) if _trace_multi_va else 0} "
                        f"n_degen={sum(_trace_degen) if _trace_degen else 0}",
                        flush=True,
                    )
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                    json.dump(traces, f)
                    f.write("\n")

            # Put the packed sequences and metrics into the output queue
            # sd_logprob_data is optionally populated by grpo_rubric_sd.py's _sd_hook
            _sd_logprob_data = getattr(args, "_sd_logprob_data_for_step", None)
            print(
                f"[Data Preparation Thread] ▶ queue put start | step={training_step}",
                flush=True,
            )
            packed_sequences_Q.put(
                {
                    "packed_sequences": packed_sequences,  # for debugging purposes
                    "collated_data": collated_data,
                    "metrics": metrics,
                    "responses_count": len(responses),
                    "num_new_tokens": num_new_tokens,
                    "B": B,
                    "training_rollouts_data": training_rollouts_data,
                    "sd_logprob_data": _sd_logprob_data,  # None unless SD is enabled
                    "training_step": training_step,
                }
            )
            print(
                f"[Data Preparation Thread] ✅ queue put done | step={training_step}",
                flush=True,
            )
            # Clear the per-step SD data to avoid stale values
            if hasattr(args, "_sd_logprob_data_for_step"):
                args._sd_logprob_data_for_step = None
    finally:
        prescore_executor.shutdown(wait=True)



def launch_mcp_subprocess(run_mcp_command: str, output_dir: str) -> Optional[subprocess.Popen]:
    """
    Launch MCP server subprocess if use_mcp_tools is enabled.
    
    Args:
        run_mcp_command: Command to run MCP server
        output_dir: Base output directory for logs
        
    Returns:
        Popen object if launched, None otherwise
    """
    print(f"🚀 Launching MCP server subprocess: {run_mcp_command}")
    
    # Debug: Check if fastmcp command exists
    try:
        import shutil
        fastmcp_path = shutil.which("fastmcp")
        if fastmcp_path:
            print(f"✅ Found fastmcp at: {fastmcp_path}")
        else:
            print("⚠️  Warning: fastmcp command not found in PATH")
    except Exception as e:
        print(f"⚠️  Warning: Could not check for fastmcp: {e}")
    
    # Debug: Show current working directory
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Create log files for MCP server output
    mcp_logs_dir = os.path.join(output_dir, "mcp_logs")
    os.makedirs(mcp_logs_dir, exist_ok=True)
    mcp_stdout = open(os.path.join(mcp_logs_dir, "mcp_server_stdout.log"), "w")
    mcp_stderr = open(os.path.join(mcp_logs_dir, "mcp_server_stderr.log"), "w")
    
    mcp_process = subprocess.Popen(
        [run_mcp_command],
        shell=True,
        stdout=mcp_stdout,
        stderr=mcp_stderr,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    )
    try:
        # Launch subprocess in new process group to avoid signal interference
        
        # Give the server time to start
        time.sleep(3)
        
        # Check if process is still running
        if mcp_process.poll() is None:
            print(f"✅ MCP server started successfully (PID: {mcp_process.pid})")
            print(f"📋 MCP server logs: {os.path.relpath(mcp_logs_dir)}/mcp_server_stdout.log, {os.path.relpath(mcp_logs_dir)}/mcp_server_stderr.log")
            
            # Register cleanup function
            def cleanup_mcp():
                if mcp_process.poll() is None:
                    print("🧹 Cleaning up MCP server...")
                    try:
                        # Kill the entire process group
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                        time.sleep(2)
                        if mcp_process.poll() is None:
                            os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                mcp_stdout.close()
                mcp_stderr.close()
            
            atexit.register(cleanup_mcp)
            return mcp_process
            
        else:
            print(f"❌ MCP server failed to start (exit code: {mcp_process.returncode})")
            # Read any error output
            mcp_stderr.close()
            mcp_stdout.close()
            try:
                with open(os.path.join(mcp_logs_dir, "mcp_server_stderr.log"), "r") as f:
                    stderr_content = f.read().strip()
                    if stderr_content:
                        print(f"📋 MCP server stderr: {stderr_content}")
                with open(os.path.join(mcp_logs_dir, "mcp_server_stdout.log"), "r") as f:
                    stdout_content = f.read().strip()
                    if stdout_content:
                        print(f"📋 MCP server stdout: {stdout_content}")
            except Exception as e:
                print(f"⚠️  Could not read MCP server logs: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching MCP server: {e}")
        mcp_stdout.close()
        mcp_stderr.close()
        return None


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig, reward_fn: Callable):
    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        print(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    )
    # Allow explicit --eval_freq to override the computed value; otherwise derive from --num_evals
    if args.eval_freq is None:
        args.eval_freq = max(1, args.num_training_steps // args.num_evals) if args.num_evals > 0 else args.num_training_steps
    args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    args.tool_use = (args.tools is not None and len(args.tools) > 0)

    # ------------------------------------------------------------
    # Setup experiment tracking and seeds
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))
    if args.with_tracking:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ------------------------------------------------------------
    # Set up datasets
    system_prompt_text = None
    additional_question_instructions = None
    if args.system_prompt_file is not None:
        # just raw text file
        if args.system_prompt_file.endswith(".txt"):
            path = Path(args.system_prompt_file)
            if path.exists():
                system_prompt_text = path.read_text().strip()
            print(f"Using system prompt from {args.system_prompt_file}:\n############\n{system_prompt_text}\n############\n")
        elif args.system_prompt_file.endswith(".yaml"):
            assert Path(args.system_prompt_file).exists()
            with open(args.system_prompt_file, "r", encoding="utf-8") as file:
                prompt = yaml.safe_load(file)
                system_prompt_text = prompt["system_prompt"]
                additional_question_instructions = prompt["additional_instructions"]
                print(f"Using system prompt from {args.system_prompt_file}:\n############\n{system_prompt_text}\n############\n")
                print(f"Using additional question instructions from {args.system_prompt_file}:\n############\n{additional_question_instructions}\n############\n")
        else:
            raise ValueError(f"System prompt file {args.system_prompt_file} does not exist or is not a yaml/txt file.")
    transform_fn_args = [
        {
            "system_prompt_text": system_prompt_text,
            "additional_question_instructions": additional_question_instructions,
        },
        {
            "max_token_length": args.max_token_length,
            "max_prompt_token_length": args.max_prompt_token_length,
        },
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)

    # LFF uses a single dataset: all rollouts use the VA system prompt.
    # r0 is computed for responses that called verify_answer;
    # responses that did not call verify_answer use r0 = r1 (fallback to DAPO).
    eval_dataset = None
    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            args.dataset_mixer_eval_list,
            args.dataset_mixer_eval_list_splits,
            tc,
            args.dataset_transform_fn,
            transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_eval_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
    if args.cache_dataset_only:
        return
    
    # Initialize rubric buffer
    rubric_buffer = None
    if args.apply_adaptive_rubric_reward and args.use_rubric_buffer:
        print("🚨 Applying adaptive rubric reward")
        print("Example of ground truth:")
        print(train_dataset[0][GROUND_TRUTHS_KEY])
        # Initialize rubric buffer with active and inactive rubrics for each query
        rubric_buffer = {}
        for ex in train_dataset:
            _gt_raw = ex[GROUND_TRUTHS_KEY]
            _gt_raw = _gt_raw[0] if isinstance(_gt_raw, list) else _gt_raw
            gt = json.loads(_gt_raw) if isinstance(_gt_raw, str) else _gt_raw
            rubric_buffer[gt['query']] = {
                'active_rubrics': [] if args.use_static_rubrics_as_persistent_rubrics else gt['rubrics'], 
                'inactive_rubrics': [],
                'persistent_rubrics': gt['rubrics'] if args.use_static_rubrics_as_persistent_rubrics else [],  # Forced to use every time
                'static_rubrics': gt['rubrics'],  # Keep a copy of original GT rubrics
            }
    

    # ------------------------------------------------------------
    # Runtime setups and quick logging
    pprint([args, model_config])

    # ------------------------------------------------------------
    # Launch MCP subprocess if needed
    mcp_process = None
    if args.tools and "mcp" in args.tools:
        if args.mcp_server_command is None:
            print("mcp_server_command is not provided when mcp is in tools; please make sure to launch the MCP server manually.")
        else:
            mcp_process = launch_mcp_subprocess(args.mcp_server_command, args.output_dir)
            if mcp_process is None:
                raise RuntimeError("Failed to launch MCP server subprocess")

    # ------------------------------------------------------------
    # Create the model and optimizer
    ray_runtime_env = _build_ray_runtime_env_for_proxy()
    try:
        ray.init(
            dashboard_host="0.0.0.0",
            runtime_env=ray_runtime_env,
        )  # enable debugging from a different machine (e.g., phobos)
    except:
        ray_temp_dir = os.path.join("/tmp", f"ray-{os.getpid()}")
        ray.init(
            dashboard_host="0.0.0.0",
            _temp_dir=ray_temp_dir,
            runtime_env=ray_runtime_env,
        )  # enable debugging from a different machine (e.g., phobos)
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 2} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
    )
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )
    max_len = args.max_prompt_token_length + args.response_length
    tool_objects = {}
    tool_max_conc = args.tool_max_concurrency

    # one additional thing: set mcp host if we are in a beaker job
    if os.environ.get("BEAKER_LEADER_REPLICA_IP") is not None:
        args.mcp_host = os.environ.get("BEAKER_LEADER_REPLICA_IP")
        print(f"🚨 Setting MCP host to {args.mcp_host} based on BEAKER_LEADER_REPLICA_IP")
        # 0.0.0.0 is a bind address, not a client target.
        if args.mcp_host == "0.0.0.0":
            args.mcp_host = "127.0.0.1"

    def _register_actor_backed_tool(class_path: str, init_kwargs: dict):
        actor = ToolActor.options(max_concurrency=tool_max_conc).remote(class_path=class_path, init_kwargs=init_kwargs)
        start = ray.get(actor.get_start_str.remote())
        stop_strings = ray.get(actor.get_stop_strings.remote())
        # If the tool provides multiple stop strings, register each as an entry.
        for end_str in stop_strings:
            tool_objects[end_str] = ToolProxy(actor_handle=actor, start_str=start, end_str=end_str)

    # first, handle the "regular" tools of search and code via actors.
    if args.tools:
        for tool in args.tools:
            class_path = TOOL_CLASS_REGISTRY.get(tool.lower(), None)
            if class_path is None:
                raise ValueError(f"Unknown tool: {tool}")
            # Pass the entire args namespace; ToolActor will filter valid kwargs
            _register_actor_backed_tool(class_path=class_path, init_kwargs=vars(args))

    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tools=tool_objects,
        max_tool_calls=args.max_tool_calls,
    )
    resume_training_step = ray.get(inits)[0] + 1
    episode = (resume_training_step - 1) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    print("======== ✅ all models and vLLM engines initialized =========")

    ray.get([m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models])
    print("======== ✅ model update group setup successfully =========")
    if resume_training_step > 1:
        print(f"Resuming training from step {resume_training_step}... Broadcasting weights to vLLM engines.")
        with Timer("[Main Thread] 🔄 Loading weights using shared memory"):
            ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])

    # Setup training
    stop_strings = [] if args.stop_strings is None else args.stop_strings
    if args.tool_use:
        stop_strings += list(tool_objects.keys())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.6,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=1,  # since we are doing greedy sampling, don't need to generate more
        stop=stop_strings,
    )
    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)

    inference_results_Q = Queue(maxsize=args.async_steps)
    param_prompt_Q = Queue(maxsize=args.async_steps)
    evaluation_inference_results_Q = Queue(maxsize=1)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    queries_prompt_Q = Queue(maxsize=args.async_steps)

    eval_prompt_token_ids = None
    eval_ground_truths = None
    if eval_dataset is not None:
        eval_prompt_token_ids = eval_dataset[INPUT_IDS_PROMPT_KEY]
        eval_ground_truths = eval_dataset[GROUND_TRUTHS_KEY]
        eval_dataset_names = eval_dataset[DATASET_SOURCE_KEY]
    thread = threading.Thread(
        target=vllm_generate_thread,
        args=(
            vllm_engines,
            generation_config,
            eval_generation_config,
            inference_results_Q,
            param_prompt_Q,
            args.num_training_steps,
            eval_prompt_token_ids,
            evaluation_inference_results_Q,
            args.eval_freq,
            resume_training_step,
            args.tool_use,
        ),
    )
    thread.start()
    print("======== ✅ vllm generate thread starts =========")

    packing_thread = threading.Thread(
        target=data_preparation_thread,
        args=(
            reward_fn,
            inference_results_Q,
            packed_sequences_Q,
            queries_prompt_Q,
            args,
            tokenizer,
            args.num_training_steps,
            rubric_buffer,
        ),
    )
    packing_thread.start()
    print("======== ✅ data preparation thread starts =========")

    def _load_batch(dataloader_idx):
        """Load one batch from train_dataset and prepare for both threads."""
        data     = train_dataset[dataloader_idx]
        queries  = data[INPUT_IDS_PROMPT_KEY]
        gt       = data[GROUND_TRUTHS_KEY]
        ds       = data[DATASET_SOURCE_KEY]
        rq       = data[RAW_USER_QUERY]
        # queries_prompt_Q tuple: (queries, gt, ds, rq, None)
        # The trailing None signals to data_preparation_thread that there is
        # no mixed-VA split (standard expand by num_samples_per_prompt_rollout).
        return queries, (queries, gt, ds, rq, None)

    # Send initial data to both threads
    _dataloader_idx = next(iter_dataloader)
    queries_next, queries_prompt_data = _load_batch(_dataloader_idx)
    queries_prompt_Q.put(queries_prompt_data)
    param_prompt_Q.put((None, queries_next))

    num_total_tokens = 0
    start_time = time.time()
    try:
        for training_step in range(resume_training_step, args.num_training_steps + 1):
            print("-" * 100)
            episode += (
                args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
            )  # each sample is an episode

            # ------------------------------------------------------------------------------------------------
            # Sync weights and send the next batch of prompts to vLLM
            if args.async_mode:
                if training_step != 1:
                    _dataloader_idx = next(iter_dataloader)
                    with Timer("[Main Thread] 🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                    queries_next, queries_prompt_data = _load_batch(_dataloader_idx)
                queries_prompt_Q.put(queries_prompt_data)
                param_prompt_Q.put((None, queries_next))
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    _dataloader_idx = next(iter_dataloader)
                    with Timer("🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                    queries_next, queries_prompt_data = _load_batch(_dataloader_idx)
                    queries_prompt_Q.put(queries_prompt_data)
                    param_prompt_Q.put((None, queries_next))

            # ------------------------------------------------------------------------------------------------
            # Get the packed sequences with advantages from the packing thread
            skip_batch = False
            with Timer("[Main Thread] 📦 Getting packed sequences from thread"):
                # Poll with timeout so we can detect data_preparation_thread crashes.
                # If the thread dies, the queue will never receive new data and the
                # main loop would block forever — instead we raise to stop training.
                _poll_interval = 5.0  # seconds between liveness checks
                packed_data = None
                while packed_data is None:
                    try:
                        packed_data = packed_sequences_Q.get(timeout=30)
                    except Exception:
                        # Queue.get(timeout=...) raises queue.Empty on timeout
                        if not packing_thread.is_alive():
                            raise RuntimeError(
                                "data_preparation_thread has died unexpectedly. "
                                "Stopping training to avoid training on stale data. "
                                "Check the thread's traceback above for the root cause."
                            )
                        # Thread still alive, keep waiting
                        print("[Main Thread] ⏳ Waiting for data_preparation_thread...")
                data_thread_metrics = packed_data["metrics"]
                B = packed_data["B"]
                collated_data = packed_data["collated_data"]
                num_total_tokens += packed_data["num_new_tokens"]
                training_rollouts_data = packed_data.get("training_rollouts_data")
                sd_logprob_data = packed_data.get("sd_logprob_data")   # None unless SD enabled
                if B == 0:
                    print("[Main Thread] 🤡 After packing, there is not enough data to train. Will save though.")
                    skip_batch = True

            # ------------------------------------------------------------------------------------------------
            # Train the model
            update_ref_policy_future = []
            if not skip_batch:
                with Timer("[Main Thread] 🗡️ Training"):
                    metrics_list: List[dict[str, float]] = ray.get(
                        [
                            policy_group.models[i].train.remote(
                                **collated_data[i],
                                pad_token_id=tokenizer.pad_token_id,
                                num_mini_batches=args.num_mini_batches,
                            )
                            for i in range(args.world_size)
                        ]
                    )
                    if (
                        args.ref_policy_update_freq is not None
                        and training_step % args.ref_policy_update_freq == 0
                        and args.alpha > 0
                    ):
                        update_ref_policy_future.extend(
                            [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
                        )

                    all_keys = set(k for m in metrics_list for k in m)
                    average_metrics = {k: sum(m.get(k, 0) for m in metrics_list) / len(metrics_list) for k in all_keys}
                    metrics = {
                        "episode": episode,
                        "training_step": training_step,
                        "val/num_total_tokens": num_total_tokens,
                        "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
                        "tokens_per_second": num_total_tokens / (time.time() - start_time),
                        **data_thread_metrics,
                        **average_metrics,
                    }
                    scalar_metrics = {}
                    for key, value in metrics.items():
                        scalar_value = _maybe_numeric_scalar(value)
                        if scalar_value is not None:
                            writer.add_scalar(key, scalar_value, episode)
                            scalar_metrics[key] = scalar_value
                        hist_arr = _maybe_numeric_hist_array(value)
                        if hist_arr is not None and hist_arr.size > 0:
                            writer.add_histogram(key, hist_arr, episode)
                    print_rich_single_line_metrics(scalar_metrics)

            # Log training rollouts if available (thread-safe logging in main thread)
            if training_rollouts_data is not None and args.with_tracking:
                print(f"[Main Thread] 📊 Logging {len(training_rollouts_data['prompt'])} training rollouts to wandb")
                train_df = pd.DataFrame(training_rollouts_data)
                if args.with_tracking:
                    wandb.log({"training_rollouts": wandb.Table(dataframe=train_df)}, step=episode)

            # Save SD logprob data + heatmaps at save_freq intervals
            if sd_logprob_data and args.save_freq > 0 and training_step % args.save_freq == 0:
                import json as _json
                # 1. Save raw logprob records (JSONL)
                sd_logprob_path = os.path.join(args.output_dir, f"sd_logprobs_step{training_step}.jsonl")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(sd_logprob_path, "w") as _f:
                    for _rec in sd_logprob_data:
                        _f.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
                print(f"[SD] Saved {len(sd_logprob_data)} logprob records to {sd_logprob_path}")

                # 2. Save token-level credit heatmaps (PNG + optional wandb)
                try:
                    from open_instruct.grpo_rubric_sd import save_sd_heatmaps
                    save_sd_heatmaps(
                        logprob_records=sd_logprob_data,
                        output_dir=args.output_dir,
                        training_step=training_step,
                        tokenizer=tokenizer,
                        with_tracking=args.with_tracking,
                    )
                except Exception as _he:
                    print(f"[SD heatmap] Failed (non-fatal): {_he}")

            if args.save_freq > 0 and training_step % args.save_freq == 0:
                with Timer("[Main Thread] 🗡️ Saving model"):
                    checkpoint_dir = f"{args.output_dir}_checkpoints"
                    step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                    print(f"Saving model at step {training_step} to {step_dir}")
                    ray.get([policy_group.models[i].save_model.remote(step_dir) for i in range(args.world_size)])
                    if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                        leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                        for i in range(args.world_size):
                            policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                                step_dir, leaderboard_name, wandb_url, training_step
                            )
                if skip_batch:
                    continue
            if (
                args.checkpoint_state_freq > 0
                and training_step % args.checkpoint_state_freq == 0
                and args.checkpoint_state_dir is not None
            ):
                with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                    client_state = {"training_step": training_step}
                    ray.get(
                        [
                            policy_group.models[i].save_checkpoint_state.remote(
                                args.checkpoint_state_dir, client_state
                            )
                            for i in range(args.world_size)
                        ]
                    )
                    print(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

            if len(update_ref_policy_future) > 0:
                with Timer("[Main Thread] 🔃 Updating reference policy"):
                    ray.get(update_ref_policy_future)

            # ------------------------------------------------------------------------------------------------
            # Optionally evaluate the model
            try:
                # timeout 0.01 if this is the last training step or we're not evaluating
                # otherwise, wait to get the last evaluation generations (long timeout just in case)
                timeout = 0.01 if (training_step < args.num_training_steps or args.eval_freq < 0) else 100
                eval_responses, eval_finish_reasons, masks, eval_infos = evaluation_inference_results_Q.get(
                    timeout=timeout
                )
                print("[Main Thread] 📊 Evaluation responses received")

                eval_sequence_lengths = np.array([len(response) for response in eval_responses])
                eval_decoded_responses = tokenizer.batch_decode(eval_responses, skip_special_tokens=True)
                eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_finish_reasons) / len(
                    eval_finish_reasons
                )

                # get and log evaluation metrics
                eval_original_dataset_names = eval_dataset[DATASET_ORIGIN_KEY]

                
                eval_scores, eval_reward_metrics, _ = asyncio.run(
                    reward_fn(
                        eval_responses,
                        eval_decoded_responses,
                        eval_ground_truths,
                        eval_dataset_names,
                        eval_finish_reasons,
                        eval_infos,
                        source_datasets=eval_original_dataset_names,
                        rubric_buffer=None,  # No rubric buffer for evaluation
                        is_training=False,
                        training_step=training_step,
                    )
                )
                eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
                
                # Calculate per-dataset eval scores
                per_dataset_scores = {}
                if eval_dataset_names is not None:
                    from collections import defaultdict
                    dataset_scores = defaultdict(list)
                    
                    # Debug: Print the actual dataset names we're seeing
                    print(f"[DEBUG] eval_dataset_names type: {type(eval_dataset_names)}")
                    print(f"[DEBUG] First few eval_dataset_names: {eval_dataset_names[:5] if len(eval_dataset_names) > 0 else 'empty'}")
                    unique_names = set()
                    for name in eval_dataset_names:
                        if isinstance(name, list):
                            unique_names.add("_".join(str(x) for x in name))
                        else:
                            unique_names.add(str(name))
                    print(f"[DEBUG] Unique dataset names found: {unique_names}")
                    
                    for score, dataset_name in zip(eval_scores, eval_dataset_names):
                        # Convert dataset_name to string if it's a list
                        if isinstance(dataset_name, list):
                            dataset_name_str = "_".join(str(x) for x in dataset_name)
                        else:
                            dataset_name_str = str(dataset_name)
                        dataset_scores[dataset_name_str].append(score)
                    
                    dataset_means = []
                    for dataset_name, scores in dataset_scores.items():
                        dataset_mean = np.array(scores).mean()
                        per_dataset_scores[f"eval/scores_{dataset_name}"] = dataset_mean
                        dataset_means.append(dataset_mean)
                    
                    # Add macro average (equal weight to each test set)
                    if dataset_means:
                        per_dataset_scores["eval/scores_macro"] = np.array(dataset_means).mean()
                
                eval_metrics = {
                    "eval/scores": np.array(eval_scores).mean(),
                    "eval/sequence_lengths": eval_sequence_lengths.mean(),
                    "eval/sequence_lengths_min": eval_sequence_lengths.min(),
                    "eval/sequence_lengths_max": eval_sequence_lengths.max(),
                    "eval/stop_rate": eval_stop_rate,
                    **eval_reward_metrics,
                    **per_dataset_scores,
                }
                print_rich_single_line_metrics(eval_metrics)
                for key, value in eval_metrics.items():
                    scalar_value = _maybe_numeric_scalar(value)
                    if scalar_value is not None:
                        writer.add_scalar(key, scalar_value, episode)
                table = {}
                table["prompt"] = tokenizer.batch_decode(eval_prompt_token_ids)
                table["response"] = eval_decoded_responses
                table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                table["scores"] = eval_scores
                table["ground_truth"] = eval_ground_truths
                if eval_dataset_names is not None:
                    table["dataset"] = eval_dataset_names
                df = pd.DataFrame(table)
                if args.with_tracking:
                    wandb.log({"sample_completions": wandb.Table(dataframe=df)})
                else:
                    print_rich_table(df.iloc[:1])
                del table
            except Empty:
                print("[Main Thread] 🙈 Evaluation responses not received")

        print(f"Saving final model at step {training_step} to {args.output_dir}")
        with Timer("[Main Thread] 🗡️ Saving model"):
            ray.get([policy_group.models[i].save_model.remote(args.output_dir) for i in range(args.world_size)])
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = args.hf_repo_revision
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        args.output_dir, leaderboard_name, wandb_url, training_step
                    )

    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        print(traceback.format_exc())
        try:
            asyncio.run(cleanup_all_llm_judge_clients())
            print("✅ LLM judge clients cleaned up")
        except Exception as cleanup_error:
            print(f"Warning: Error during LLM judge cleanup: {cleanup_error}")

        # Clean up MCP subprocess
        if mcp_process is not None:
            try:
                print("🧹 Cleaning up MCP server subprocess...")
                if mcp_process.poll() is None:
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                    time.sleep(2)
                    if mcp_process.poll() is None:
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                print("✅ MCP server subprocess cleaned up")
            except (OSError, ProcessLookupError) as cleanup_error:
                print(f"Warning: Error during MCP cleanup: {cleanup_error}")

        ray.shutdown()
        os._exit(1)
        raise  # Re-raise the exception after shutdown

    # Clean up threads
    thread.join()
    print("======== ✅ vllm generate thread ends =========")
    packing_thread.join()
    print("======== ✅ data preparation thread ends =========")

    try:
        asyncio.run(cleanup_all_llm_judge_clients())
        print("✅ LLM judge clients cleaned up")
    except Exception as cleanup_error:
        print(f"Warning: Error during LLM judge cleanup: {cleanup_error}")

    ray.shutdown()

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    print("finished training")

    # Note: MCP subprocess cleanup is handled by atexit handler registered in launch_mcp_subprocess
    if mcp_process is not None:
        print("🧹 MCP server subprocess will be cleaned up by atexit handler")

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        print("Pushing model to hub")
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    reward_fn_mapping = build_all_verifiers(args)

    if args.apply_adaptive_rubric_reward:
        from open_instruct.search_rewards.utils.rubric_utils import _generate_instance_wise_adaptive_rubrics, update_ground_truths_with_adaptive_rubrics
        from open_instruct.search_rewards.longform_rubric_rewards import create_rubric_key

    async def reward_fn(
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[str],
        finish_reasons: List[str],
        infos: List[List[int]],
        queries: Optional[List[str]] = None,
        source_datasets: Optional[List[str]] = None,
        rubric_buffer: Optional[Dict[str, List[Dict[str, float]]]] = None,
        is_training: bool = True,
        training_step: Optional[int] = None,
        **kwargs,  # absorb extra args (e.g. masks, query_ids_list) passed by data_preparation_thread
    ) -> List[float]:
        prescore_mode = bool(kwargs.get("prescore_mode", False))
        _timer_noop = prescore_mode
        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
        good_outputs = [
            len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
            for i in range(len(tool_outputs))
        ]
        scores = [0] * len(decoded_responses)
        metrics = {}
        log_values = {}

        if args.apply_r1_style_format_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating format reward", noop=_timer_noop):
                format_scores = soft_format_reward_func(decoded_responses, args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()
        elif args.apply_rl_rag_format_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating RL-RAG format reward", noop=_timer_noop):
                rl_rag_format_scores = is_a_good_rl_rag_response(decoded_responses)
                if len(rl_rag_format_scores) != len(scores):
                    raise ValueError(f"{len(rl_rag_format_scores)=} != {len(scores)=}")
                for i in range(len(rl_rag_format_scores)):
                    scores[i] = rl_rag_format_scores[i] + scores[i]
                metrics["val/rl_rag_format_scores"] = np.array(rl_rag_format_scores).mean()

        if args.apply_adaptive_rubric_reward and is_training:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating adaptive rubric reward", noop=_timer_noop):
                adaptive_rubric_scores = await _generate_instance_wise_adaptive_rubrics(decoded_responses, ground_truths, args.num_samples_per_prompt_rollout, rubric_buffer=rubric_buffer)
                # Update ground truths with adaptive rubrics and incorporate rubric buffer
                ground_truths, valid_adaptive_rubric_rate, avg_num_ground_truths, avg_num_adaptive_rubrics, avg_num_active_buffer_rubrics, rubric_buffer, skipped_count = update_ground_truths_with_adaptive_rubrics(ground_truths, adaptive_rubric_scores, args.num_samples_per_prompt_rollout, rubric_buffer=rubric_buffer)
                metrics["objective/valid_adaptive_rubric_rate"] = valid_adaptive_rubric_rate
                metrics["objective/avg_num_ground_truths"] = avg_num_ground_truths
                metrics["objective/avg_num_adaptive_rubrics"] = avg_num_adaptive_rubrics
                metrics["objective/avg_num_active_buffer_rubrics"] = avg_num_active_buffer_rubrics
                metrics["objective/skipped_adaptive_rubrics"] = skipped_count
                
        if args.apply_verifiable_reward:
            import time as _time
            _score_t0 = _time.perf_counter()
            if not prescore_mode:
                print(f"\n[Reward] step={training_step} scoring {len(decoded_responses)} responses ...", flush=True)
            with Timer("[Data Preparation Thread] Calculating rewards -- 🏆 Applying verifiable reward", noop=_timer_noop):
                verifiable_rewards, per_func_rewards, log_values = await apply_verifiable_reward(
                    reward_fn_mapping,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=args.verification_reward,
                    queries=queries,
                    overwrite_reward_fn_tag=args.overwrite_reward_fn_tag,
                    tool_errors=tool_errors,
                    tool_error_penalty_value=args.tool_error_penalty_value,
                )
                
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                # slightly complex combo of good outputs and additive format reward
                for i in range(len(verifiable_rewards)):
                    if not args.only_reward_good_outputs or (good_outputs[i] and args.only_reward_good_outputs):
                        if (args.apply_r1_style_format_reward or args.apply_rl_rag_format_reward) and args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif (args.apply_r1_style_format_reward or args.apply_rl_rag_format_reward) and not args.additive_format_reward:
                            # Check which format reward was applied
                            if args.apply_r1_style_format_reward:
                                scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                            elif args.apply_rl_rag_format_reward:
                                scores[i] = verifiable_rewards[i] if rl_rag_format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                _score_elapsed = _time.perf_counter() - _score_t0
                _nonzero = int((np_verifiable_rewards > 0.0).sum())
                if not prescore_mode:
                    print(
                        f"[Reward] step={training_step} scored {len(verifiable_rewards)} responses in {_score_elapsed:.1f}s | "
                        f"mean={np_verifiable_rewards.mean():.3f} non-zero={_nonzero}/{len(verifiable_rewards)}",
                        flush=True,
                    )
                # log anything additional
                for key, value in log_values.items():
                    if key in ["rubric_scores_by_title", "per_rubric_rewards", "rubric_feedback_by_title"]:
                        # do not log rubric scores by title or per_rubric_rewards (used for buffer management)
                        continue
                    try:
                        arr = np.array(value)
                        if arr.size == 0:
                            continue
                        if np.issubdtype(arr.dtype, np.number):
                            metrics[f"objective/reward_log_values/{key}"] = arr.mean()
                    except Exception:
                        continue
                # reshuffle around per_func rewards
                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                # log per function rewards
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()
                
                # log per original source dataset, if provided
                if source_datasets is not None and len(source_datasets) == len(verifiable_rewards):
                    source_to_values = defaultdict(list)
                    for src, val in zip(source_datasets, verifiable_rewards):
                        source_to_values[src].append(val)
                    for src, vals in source_to_values.items():
                        arr = np.array(vals)
                        metrics[f"objective/source/{src}_verifiable_reward"] = arr.mean()
                        metrics[f"objective/source/{src}_verifiable_correct_rate"] = (arr > 0.0).mean()
                
                # log direction agreement
                if args.log_direction_agreement:
                    from open_instruct.search_rewards.utils._direction_agreement import compute_direction_agreement
                    with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating direction agreement", noop=_timer_noop):
                        direction_agreement_dict = compute_direction_agreement(log_values, verifiable_rewards)
                        for key, value in direction_agreement_dict.items():
                            metrics[f"analysis/direction_agreement/{key}"] = value

                metrics["__judge_failed__"] = log_values.get("__judge_failed__", [False] * len(decoded_responses))

        # this gets applied at the very end since it replaces (rather than adds to) the existing reward.
        if args.non_stop_penalty:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🦖 Applying non stop penalty", noop=_timer_noop):
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = args.non_stop_penalty_value

        if is_training:
            with Timer("[Rubric-wise Normalization Computation] Compute normalization for each rubric inside the reward function", noop=_timer_noop):
                rubric_scores_by_title = log_values.get("rubric_scores_by_title", [])
                if rubric_scores_by_title:
                    print(f"[RubricNorm] step={training_step} rubric_scores_by_title: {len(rubric_scores_by_title)}/{len(decoded_responses)} responses have per-rubric scores", flush=True)

                # Check lengths
                print(f"[DEBUG] Investigating whether there is a length mismatch:")
                print(f"  - decoded_responses: {len(decoded_responses)}")
                print(f"  - ground_truths: {len(ground_truths)}")
                print(f"  - datasets: {len(datasets) if hasattr(datasets, '__len__') else 'N/A'}")
                print(f"  - scores: {len(scores)}")
                print(f"  - rubric_scores_by_title: {len(rubric_scores_by_title)}")

                # Examine rubric_scores_by_title in detail
                print(f"[DEBUG] Complete rubric_scores_by_title content:")
                for i, entry in enumerate(rubric_scores_by_title):
                    print(f"  [{i}]: {entry}")

                print(f"[DEBUG] Summary of rubric_scores_by_title:")
                non_empty_count = sum(1 for entry in rubric_scores_by_title if entry)
                empty_count = sum(1 for entry in rubric_scores_by_title if not entry)
                print(f"  - Total entries: {len(rubric_scores_by_title)}")
                print(f"  - Non-empty entries: {non_empty_count}")
                print(f"  - Empty entries: {empty_count}")

                if rubric_scores_by_title:
                    all_keys = set()
                    for entry in rubric_scores_by_title:
                        if entry:
                            all_keys.update(entry.keys())
                    print(f"  - Unique rubric keys found: {len(all_keys)} keys")
                    # Show first few keys for debugging
                    if all_keys:
                        sample_keys = sorted(list(all_keys))[:5]
                        print(f"  - Sample keys: {sample_keys}")

                # Check ALL unique datasets to identify routing
                if hasattr(datasets, '__len__') and len(datasets) > 0:
                    from collections import Counter
                    flat_ds = [d[0] if isinstance(d, list) else d for d in datasets]
                    ds_counts = Counter(flat_ds)
                    print(f"[DEBUG] All datasets in batch: {dict(ds_counts)}")

                # Strategy: Apply normalization to available rubric scores, then map them back
                # Step 1: Collect all available rubric scores for normalization
                rubric_key_scores = {}
                for response_scores in rubric_scores_by_title:
                    if response_scores:
                        for rubric_key, score in response_scores.items():
                            if rubric_key not in rubric_key_scores:
                                rubric_key_scores[rubric_key] = []
                            rubric_key_scores[rubric_key].append(score)
                
                # Step 2: Compute per-rubric-key mean and std from available scores
                rubric_key_stats = {}
                for rubric_key, scores_list in rubric_key_scores.items():
                    scores_array = np.array(scores_list)
                    rubric_key_stats[rubric_key] = {
                        "mean": scores_array.mean(),
                        "std": scores_array.std()
                    }
                
                print(f"[DEBUG] Computed stats for {len(rubric_key_stats)} unique rubric keys")

                # Step 3: Create a full-length advantages array (initialized to 0)
                final_advantages = [0.0] * len(scores)
                original_scores = scores.copy()

                # Step 4: Apply per rubric group advantages
                print(f"[DEBUG] Applying advantages to first {len(rubric_scores_by_title)} responses")

                for i, response_scores in enumerate(rubric_scores_by_title):
                    ground_truth_item = ground_truths[i]
                    if isinstance(ground_truth_item, list):
                        ground_truth_item = ground_truth_item[0]
                    test_case = json.loads(ground_truth_item) if isinstance(ground_truth_item, str) else ground_truth_item
                    query = test_case.get("query", "")
                    rubrics = test_case.get("rubrics", [])
                    
                    # Create rubric_key->weight mapping
                    rubric_key_weights = {}
                    for rubric in rubrics:
                        # Import the helper function here to avoid circular imports
                        from open_instruct.search_rewards.longform_rubric_rewards import create_rubric_key
                        rubric_key = create_rubric_key(query, rubric)
                        if rubric_key not in rubric_key_weights:
                            rubric_key_weights[rubric_key] = []
                        rubric_key_weights[rubric_key].append(rubric["weight"])
                    
                    # Average weights for rubric keys with multiple rubrics (though this should be rare now)
                    for rubric_key in rubric_key_weights:
                        rubric_key_weights[rubric_key] = np.mean(rubric_key_weights[rubric_key])
                    
                    if args.normalize_rubric_scores:
                        # Compute per-rubric weighted normalized scores
                        weighted_scores = []
                        total_weight = 0.0
                        for rubric_key, score in response_scores.items():
                            if rubric_key in rubric_key_stats:
                                stats = rubric_key_stats[rubric_key]
                                weight = rubric_key_weights.get(rubric_key, 1.0)
                                
                                if stats["std"] > 0:
                                    normalized_score = (score - stats["mean"]) / stats["std"]
                                else:
                                    normalized_score = 0.0
                                
                                weighted_scores.append(normalized_score * weight)
                                if weight > 0:
                                    total_weight += weight

                                if i == 0:
                                    print(f"[DEBUG] Response {i}: rubric_key_stats = {stats}")
                                    print(f"[DEBUG] Response {i}: rubric_key_weights = {weight}")
                                    print(f"[DEBUG] Response {i}: response_scores = {score}")
                                    print(f"[DEBUG] Response {i}: weighted_scores = {weighted_scores}")

                        # Weighted average
                        final_advantage = sum(weighted_scores) / max(total_weight, 1.0)
                        final_advantages[i] = final_advantage
                        scores[i] = final_advantage
                        print(f"[DEBUG] Response {i}: Per-rubric normalized advantage = {final_advantage:.4f}")
                
                # Log average of rubric means and stds
                if rubric_key_stats:
                    avg_mean = np.mean([stats["mean"] for stats in rubric_key_stats.values()])
                    avg_std = np.mean([stats["std"] for stats in rubric_key_stats.values()])
                    metrics["rubric_keys/avg_mean"] = avg_mean
                    metrics["rubric_keys/avg_std"] = avg_std
                    
                    num_all_zero_rubrics_ratio = sum(1 for stats in rubric_key_stats.values() if stats["mean"] == 0 and stats["std"] == 0) / len(rubric_key_stats)
                    num_all_same_value_non_zero_rubrics_ratio = sum(1 for stats in rubric_key_stats.values() if stats["mean"] > 0 and stats["std"] == 0) / len(rubric_key_stats)
                    metrics["rubric_keys/num_all_zero_rubrics_ratio"] = num_all_zero_rubrics_ratio
                    metrics["rubric_keys/num_all_same_value_non_zero_rubrics_ratio"] = num_all_same_value_non_zero_rubrics_ratio
                    
                    if args.normalize_rubric_scores:
                        advantage_corr = np.corrcoef(final_advantages, original_scores)[0, 1]
                        metrics["rubric_keys/advantage_corr"] = advantage_corr
                    
                    
                print(f"[DEBUG] Applied rubric normalization: {len([a for a in final_advantages if a != 0])} non-zero advantages")
            
            with Timer("[Adaptive Rubric Filtering Thread] Filtering adaptive rubric rewards"):
                # TODO: implement a function that selects a fixed number of rubrics from adaptive rubrics for next round of evaluation.
                # The idea is to select the adaptive rubrics that has the highest std of scores of the current round of scoring.
                # TODO: once selected, keep the rubrics with high std in "active_rubrics" and move the rest to the "inactive_rubrics" field of the rubric buffer
                if args.apply_adaptive_rubric_reward and rubric_buffer is not None:
                    # Collect rubrics to move based on their stats (much smaller iteration)
                    rubrics_to_deactivate = []  # (query, rubric) pairs to move to inactive
                    rubrics_by_query_std = {}   # query -> [(rubric, std)] for ranking
                    
                    # Iterate through the smaller rubric_key_stats instead of large rubric_buffer
                    for rubric_key, stats in rubric_key_stats.items():
                        # Find which query and rubric this key corresponds to
                        for query, buffer_data in rubric_buffer.items():
                            active_rubrics = buffer_data.get("active_rubrics", [])
                            for rubric in active_rubrics:
                                if create_rubric_key(query, rubric) == rubric_key:
                                    if stats["std"] == 0:
                                        # Mark for deactivation
                                        rubrics_to_deactivate.append((query, rubric))
                                    else:
                                        # Track for potential ranking/filtering
                                        if query not in rubrics_by_query_std:
                                            rubrics_by_query_std[query] = []
                                        rubrics_by_query_std[query].append((rubric, stats["std"]))
                                    break
                            else:
                                continue
                            break
                    
                    # Move zero-std rubrics to inactive
                    moved_count = 0
                    for query, rubric in rubrics_to_deactivate:
                        buffer_data = rubric_buffer[query]
                        if rubric in buffer_data["active_rubrics"]:
                            buffer_data["active_rubrics"].remove(rubric)
                            buffer_data["inactive_rubrics"].append(rubric)
                            moved_count += 1
                    
                    # Apply max_active_rubrics cap per query
                    capped_count = 0
                    for query, rubric_std_pairs in rubrics_by_query_std.items():
                        buffer_data = rubric_buffer[query]
                        active_rubrics = buffer_data["active_rubrics"]
                        
                        if len(active_rubrics) > args.max_active_rubrics:
                            # Sort by std descending and keep only top max_active_rubrics
                            rubric_std_pairs.sort(key=lambda x: x[1], reverse=True)
                            
                            # Find rubric keys to keep (top std ones)
                            rubric_keys_to_keep = set(create_rubric_key(query, rubric) for rubric, _ in rubric_std_pairs[:args.max_active_rubrics])
                            
                            # Move excess rubrics to inactive
                            new_active = []
                            for rubric in active_rubrics:
                                rubric_key = create_rubric_key(query, rubric)
                                if rubric_key in rubric_keys_to_keep or rubric_key not in rubric_key_stats:
                                    # Keep high-std rubrics or rubrics without stats
                                    new_active.append(rubric)
                                else:
                                    # Move low-std rubrics to inactive
                                    buffer_data["inactive_rubrics"].append(rubric)
                                    capped_count += 1
                            
                            buffer_data["active_rubrics"] = new_active
                    
                    if moved_count > 0 or capped_count > 0:
                        print(f"[Adaptive Rubric Filtering] Moved {moved_count} zero-std rubrics and {capped_count} low-std rubrics to inactive")

        metrics["__reward_log_values__"] = log_values

        return scores, metrics, rubric_buffer

    async def _prescore_one_response_async(
        response_idx: int,
        query_ids: List[int],
        response_ids: List[int],
        mask: List[int],
        decoded_response: str,
        ground_truth,
        dataset,
        finish_reason: str,
        infos_single,
        decoded_query: str,
        rubric_buffer: Optional[Dict[str, List[Dict[str, float]]]] = None,
        training_step: Optional[int] = None,
    ) -> _RewardPrescoreItem:
        scores, metrics, _ = await reward_fn(
            [response_ids],
            [decoded_response],
            [ground_truth],
            [dataset],
            [finish_reason],
            infos_single,
            queries=[decoded_query],
            rubric_buffer=rubric_buffer,
            is_training=True,
            training_step=training_step,
            masks=[mask],
            query_ids_list=[query_ids],
            prescore_mode=True,
        )
        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos_single
        reward_log_values = dict(metrics.get("__reward_log_values__", {}) or {})
        judge_failed_values = metrics.get("__judge_failed__", [False])
        judge_failed = bool(judge_failed_values[0]) if judge_failed_values else False
        return _RewardPrescoreItem(
            response_idx=response_idx,
            query_ids=list(query_ids),
            response_ids=list(response_ids),
            mask=list(mask),
            decoded_response=decoded_response,
            decoded_response_raw=decoded_response,
            decoded_query=decoded_query,
            ground_truth=ground_truth,
            dataset=dataset,
            finish_reason=finish_reason,
            num_calls=int(num_calls[0]),
            timeout=bool(timeouts[0]),
            tool_error=tool_errors[0],
            tool_output=tool_outputs[0],
            tool_runtime=float(tool_runtimes[0]),
            tool_called=bool(tool_calleds[0]),
            good_output=bool(len(tool_outputs[0]) > 0 and tool_calleds[0] and not timeouts[0] and not tool_errors[0]),
            score=float(scores[0]),
            per_func_reward={},
            reward_log_values=reward_log_values,
            judge_failed=judge_failed,
            metrics=dict(metrics),
        )

    def _prescore_one_response_sync(*args, **kwargs) -> _RewardPrescoreItem:
        return asyncio.run(_prescore_one_response_async(*args, **kwargs))

    main(args, tokenizer_config, model_config, reward_fn)
