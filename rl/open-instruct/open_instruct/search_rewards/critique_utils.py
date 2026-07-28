"""
Critique generation utilities for multi-round critique RL training.

Provides:
- Judge-model critique generation (rubric-grounded, search-strategy-aware)
- Self-critique generation (using current policy weights via vLLM)
- Critique injection helpers (build user turn, strip for SFT)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

JUDGE_CRITIQUE_SYSTEM_PROMPT = (
    "You are an expert research evaluator. Your job is to give concise, "
    "actionable feedback on a deep research answer to help the researcher "
    "improve their search strategy and final answer."
)

JUDGE_CRITIQUE_USER_TEMPLATE = """\
Question: {question}

Current Answer:
{answer}

Evaluation criteria (subset):
{rubric_summary}

Provide a critique with the following sections:
1. MISSING EVIDENCE: What specific facts or evidence are absent? Be concrete.
2. WEAK CITATIONS: Which claims lack proper source support?
3. SEARCH GAPS: What relevant topics or angles were not searched?
4. ACTIONABLE NEXT STEPS: Give 2-3 specific search queries that would most improve the answer.

Keep the critique under 300 words. Focus on search strategy improvements, not writing style."""

SELF_CRITIQUE_SYSTEM_PROMPT = (
    "You are reviewing your own research answer to identify gaps and improve it."
)

SELF_CRITIQUE_USER_TEMPLATE = """\
Question: {question}

Your Previous Answer:
{answer}

Identify the most important gaps in your research:
1. What key evidence or facts are missing?
2. What search queries would most strengthen your answer?
3. Which claims need better citation support?

Be specific and actionable. Keep your critique under 200 words."""

# Appended to the agent system prompt to encourage hiding critique references
CRITIQUE_HIDING_INSTRUCTION = (
    "\n\n## Handling Feedback\n"
    "If you receive feedback in <critique> tags, use it silently to improve your "
    "search strategy and answer. Do not reference the critique directly (e.g., avoid "
    "\"Based on the feedback...\"). Instead, reframe insights as your own reasoning, "
    "e.g., \"I should investigate...\" or \"The question may also require...\"."
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CritiqueResult:
    text: str                          # The critique text
    source: str                        # "judge" or "self"
    token_ids: list[int] = field(default_factory=list)    # For self-critique RL
    logprobs: list[float] = field(default_factory=list)   # For self-critique RL
    prompt_token_ids: list[int] = field(default_factory=list)  # Prompt used for self-critique


# ---------------------------------------------------------------------------
# Rubric summarization helper
# ---------------------------------------------------------------------------

def _summarize_rubrics(ground_truth: dict, max_rubrics: int = 3) -> str:
    """Extract a short rubric summary for the critique prompt."""
    rubric_items = []

    if "Answer Critical" in ground_truth:
        for r in ground_truth["Answer Critical"][:max_rubrics]:
            handle = r.get("Handle", "")
            ingredient = r.get("Ingredient", "")
            rubric_items.append(f"- {handle}: {ingredient[:120]}")
    elif "rubric" in ground_truth:
        for r in ground_truth["rubric"][:max_rubrics]:
            rubric_type = r.get("type", "")
            rubric_text = r.get("rubric_item", "")
            rubric_items.append(f"- {rubric_type}: {rubric_text[:120]}")
    elif "rubrics" in ground_truth:
        rubrics = ground_truth["rubrics"]
        if isinstance(rubrics, list):
            for r in rubrics[:max_rubrics]:
                if isinstance(r, dict):
                    desc = r.get("description", r.get("rubric_item", str(r)))
                    rubric_items.append(f"- {desc[:150]}")
                else:
                    rubric_items.append(f"- {str(r)[:150]}")

    if not rubric_items:
        return "Provide a comprehensive, well-cited answer to the question."

    return "\n".join(rubric_items)


# ---------------------------------------------------------------------------
# Judge critique generation
# ---------------------------------------------------------------------------

async def generate_judge_critique(
    question: str,
    answer: str,
    ground_truth: dict,
    judge_model: str,
    openai_api_base: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    max_rubrics: int = 3,
    max_tokens: int = 512,
    timeout: float = 30.0,
) -> str:
    """
    Call a judge model to generate a rubric-grounded, search-strategy-aware critique.

    Args:
        question: The research question.
        answer: The model's current answer (extracted from <answer> tags).
        ground_truth: Dict containing rubrics (Answer Critical / rubric / rubrics).
        judge_model: Model name to use for critique generation.
        openai_api_base: OpenAI-compatible API base URL.
        openai_api_key: API key.
        max_rubrics: Number of rubric criteria to include in the prompt.
        max_tokens: Max tokens for the critique response.
        timeout: Request timeout in seconds.

    Returns:
        Critique text string.
    """
    rubric_summary = _summarize_rubrics(ground_truth, max_rubrics=max_rubrics)
    answer_truncated = answer[:3000] if len(answer) > 3000 else answer

    user_content = JUDGE_CRITIQUE_USER_TEMPLATE.format(
        question=question,
        answer=answer_truncated,
        rubric_summary=rubric_summary,
    )

    client_kwargs = {}
    if openai_api_base:
        client_kwargs["base_url"] = openai_api_base
    if openai_api_key:
        client_kwargs["api_key"] = openai_api_key

    client = openai.AsyncOpenAI(**client_kwargs)

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_CRITIQUE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            ),
            timeout=timeout,
        )
        critique_text = response.choices[0].message.content.strip()
        return critique_text
    except asyncio.TimeoutError:
        logger.warning(f"Judge critique timed out after {timeout}s")
        return ""
    except Exception as e:
        logger.warning(f"Judge critique failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Self-critique generation (uses current policy via vLLM)
# ---------------------------------------------------------------------------

async def generate_self_critique(
    question: str,
    answer: str,
    vllm_engine,  # Ray actor (LLMEngine or AsyncLLMEngine)
    tokenizer,
    renderer,     # dr_agent renderer with build_generation_prompt / parse_response
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> CritiqueResult:
    """
    Generate a self-critique using the current policy weights via vLLM.

    The self-critique prompt is built as a chat conversation and tokenized
    using the model's renderer. Token IDs and logprobs are returned so the
    critique tokens can be trained with RL.

    Returns:
        CritiqueResult with text, token_ids, logprobs, prompt_token_ids.
    """
    import ray
    from vllm import SamplingParams

    answer_truncated = answer[:2000] if len(answer) > 2000 else answer
    user_content = SELF_CRITIQUE_USER_TEMPLATE.format(
        question=question,
        answer=answer_truncated,
    )

    conversation = [
        {"role": "system", "content": SELF_CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt_tokens = renderer.build_generation_prompt(conversation)
    if hasattr(prompt_tokens, "to_ints"):
        prompt_token_ids = prompt_tokens.to_ints()
    elif isinstance(prompt_tokens, list):
        prompt_token_ids = prompt_tokens
    else:
        prompt_token_ids = tokenizer.encode(user_content, add_special_tokens=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        logprobs=1,
        skip_special_tokens=False,
    )

    try:
        outputs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ray.get(
                vllm_engine.generate.remote(
                    sampling_params=sampling_params,
                    prompt_token_ids=[prompt_token_ids],
                    use_tqdm=False,
                )
            ),
        )
        if not outputs or not outputs[0].outputs:
            return CritiqueResult(text="", source="self")

        out = outputs[0].outputs[0]
        token_ids = list(out.token_ids)
        logprobs_raw = out.logprobs or []
        logprobs = []
        for i, lp_dict in enumerate(logprobs_raw):
            if i < len(token_ids) and lp_dict is not None:
                tok = token_ids[i]
                lp = lp_dict.get(tok, None)
                logprobs.append(float(lp.logprob) if lp is not None else 0.0)
            else:
                logprobs.append(0.0)

        critique_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return CritiqueResult(
            text=critique_text,
            source="self",
            token_ids=token_ids,
            logprobs=logprobs,
            prompt_token_ids=prompt_token_ids,
        )
    except Exception as e:
        logger.warning(f"Self-critique generation failed: {e}")
        return CritiqueResult(text="", source="self")


# ---------------------------------------------------------------------------
# Critique injection helpers
# ---------------------------------------------------------------------------

def build_critique_user_turn(critique_text: str) -> str:
    """
    Wrap critique text in <critique> tags and format as a user turn message
    that asks the agent to continue research with the feedback.
    """
    return (
        f"<critique>\n{critique_text}\n</critique>\n\n"
        "Please continue your research to address the above feedback and provide an improved answer."
    )


def build_critique_turn_tokens(
    critique_text: str,
    tokenizer,
    renderer=None,
) -> list[int]:
    """
    Tokenize the critique user turn using the model's chat template.
    Returns token IDs for the full user turn (including role markers).
    """
    user_content = build_critique_user_turn(critique_text)
    conversation = [{"role": "user", "content": user_content}]

    # Use renderer if available to apply chat template correctly
    try:
        # Build as continuation (no system prompt, just the new user turn)
        tokens = renderer.build_generation_prompt(conversation)
        if hasattr(tokens, "to_ints"):
            return tokens.to_ints()
        elif isinstance(tokens, list):
            return tokens
    except Exception:
        pass

    # Fallback: apply chat template directly
    try:
        formatted = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        return formatted
    except Exception:
        return tokenizer.encode(user_content, add_special_tokens=False)


# ---------------------------------------------------------------------------
# SFT trajectory stripping
# ---------------------------------------------------------------------------

def strip_critique_turns_from_text(full_text: str) -> str:
    """
    Remove <critique>...</critique> user turn blocks from decoded text.
    Also removes the follow-up instruction line.

    Used to create clean SFT trajectories from critique-augmented episodes.
    """
    # Remove the full critique user turn including the instruction line
    pattern = (
        r"<critique>.*?</critique>\s*\n?"
        r"Please continue your research to address the above feedback "
        r"and provide an improved answer\.\s*\n?"
    )
    cleaned = re.sub(pattern, "", full_text, flags=re.DOTALL)
    return cleaned


def strip_critique_turns_from_tokens(
    full_token_ids: list[int],
    tokenizer,
) -> list[int]:
    """
    Remove critique turn tokens from a token sequence by decoding, stripping,
    and re-encoding. Returns the stripped token IDs.

    Note: Re-encoding may not be perfectly token-identical to the original
    for the non-critique portions, but is close enough for SFT training.
    """
    full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)
    stripped_text = strip_critique_turns_from_text(full_text)
    stripped_ids = tokenizer.encode(stripped_text, add_special_tokens=False)
    return stripped_ids


def compute_stripped_sft_weights(
    query_token_ids: list[int],
    stripped_token_ids: list[int],
) -> list[float]:
    """
    Compute SFT loss weights for stripped trajectory tokens.
    Query tokens (prefix matching query_token_ids) get weight 0.
    Response tokens get weight 1.
    """
    query_len = len(query_token_ids)
    total_len = len(stripped_token_ids)
    weights = [0.0] * min(query_len, total_len) + [1.0] * max(0, total_len - query_len)
    return weights
