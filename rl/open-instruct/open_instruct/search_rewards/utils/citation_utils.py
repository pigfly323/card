import json
import re
import os
import asyncio
from typing import Dict, List, Tuple

from open_instruct.search_rewards.utils.run_utils import get_judge_agent

_CITATION_JUDGE_TIMEOUT_S = float(os.environ.get("CITATION_JUDGE_TIMEOUT", "120"))


def _get_citation_judge_model() -> str:
    return os.environ.get("CITATION_JUDGE_MODEL") or os.environ.get(
        "RUBRIC_JUDGE_MODEL", "qwen3-instruct"
    )


citation_recall_has_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


citation_recall_no_citation_prompt = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{full_response}
</response>

<statement>
{statement}
</statement>"""


citation_precision_prompt = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almost unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


def extract_claims_and_corresponding_citation_ids(
    response: str, 
    split_non_cited_parts_by_newlines: bool = False,
    split_non_cited_parts_by_sentences: bool = False,
    ) -> Dict[str, List[str]]:
    """
    Example response:
    "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. <cite id=a1b2c3d4>Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north. <cite id=e5f6g7h8>The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite> Contrary to popular belief, <cite id=i9j0k1l2>the Great Wall is not visible from space with the naked eye, a myth that has been debunked by astronauts and satellite imagery.</cite>"
    """
    claims = {}

    # Use findall to get all cite tags and their content
    cite_pattern = r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>"
    cite_matches = re.findall(cite_pattern, response)
    
    # Split the response by cite tags to get non-cited text
    cite_tag_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    non_cited_parts = re.split(cite_tag_pattern, response)
    
    # Further split non-cited parts by newlines if requested
    if split_non_cited_parts_by_newlines:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"\n", part))
        non_cited_parts = further_split_parts
    
    # Further split non-cited parts by sentences if requested
    if split_non_cited_parts_by_sentences:
        further_split_parts = []
        for part in non_cited_parts:
            further_split_parts.extend(re.split(r"[.!?]", part))
        non_cited_parts = further_split_parts
    
    # Add non-cited text (parts between cite tags)
    for part in non_cited_parts:
        part = part.strip()
        if part:
            claims[part] = []
    
    # Add cited text with their citation IDs
    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            # When there are multiple citations (e.g., <cite id="a1b2c3d4,e5f6g7h8">), separately record the citations
            for citation_id in citation_ids.split(","):
                claims[cited_text] = [citation_id]

    return claims
    
    
def score_in_context_citations(question: str, response: str, citations: Dict[str, str]) -> Dict[str, float]:
    """
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param response:
    :param citations:
    :return: final weighted score with and without the static components
    """
    if not citations:
        return 0
    
    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

    claims = extract_claims_and_corresponding_citation_ids(response)

    citation_format_reward = score_citation_format(claims, citations)
    

    avg_f1 = 0
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        avg_f1 += score_citation_f1(question, claim_text, concatenated_citations, response)
    if len(claims) > 0:
        avg_f1 /= len(claims)

    return 0.6 * avg_f1 + 0.4 * citation_format_reward


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> Dict[str, float]:
    """
    Check if the model has hallucinated citations.
    """
    all_citations = []
    for claim in claims.items():
        all_citations.extend(claim[1])
    all_citations = list(set(all_citations))
    if len(all_citations) == 0:
        # If there are no citations, return 0
        return 0
    valid_citations = [citation for citation in all_citations if citation in citations]
    return len(valid_citations) / len(all_citations)


def score_citation_f1(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall = score_citation_recall(question, claim, concatenated_citations, full_response)
    precision = score_citation_precision(question, claim, concatenated_citations)
    # avoid division by zero
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def score_citation_recall(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return score_no_citation_recall(question, claim, full_response)
    else:
        return score_with_citation_recall(question, claim, concatenated_citations)


def score_with_citation_recall(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    try:
        response = get_judge_agent("citation_judge").single_turn(user_prompt, None, None, 800)
    except Exception as e:
        print(f"Error scoring citation recall: {e}")
        return 0.0
    return extract_recall_rating_from_response(response)


def score_no_citation_recall(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    try:
        response = get_judge_agent("citation_judge").single_turn(user_prompt, None, None, 800)
    except Exception as e:
        print(f"Error scoring no-citation recall: {e}")
        return 0.0
    # "yes" means it is a factual claim, but no citation is provided.
    return 1 - extract_yes_no_from_response(response)


def score_citation_precision(question: str, claim: str, concatenated_citations: str) -> float:
    if len(concatenated_citations) == 0:
        return 1
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    try:
        response = get_judge_agent("citation_judge").single_turn(user_prompt, None, None, 800)
    except Exception as e:
        print(f"Error scoring citation precision: {e}")
        return 0.0
    return extract_relevant_rating_from_response(response)


def extract_recall_rating_from_response(response: str) -> float:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "fully supported":
            return 1.0
        elif extracted_text == "partially supported":
            return 0.5
        elif extracted_text == "no support":
            return 0.0
        else:
            # If the extracted text doesn't match any expected values, return 0 as default
            return 0.0
    else:
        return 0.0


def extract_yes_no_from_response(response: str) -> int:
    yes_no = re.search(r"Need Citation: \[\[(.*)\]\]", response)
    if yes_no:
        extracted_text = yes_no.group(1).strip().lower()
        if extracted_text == "yes":
            return 1
        elif extracted_text == "no":
            return 0
        else:
            # If the extracted text is neither "yes" nor "no", return 0 as default
            return 0
    else:
        return 0


def extract_relevant_rating_from_response(response: str) -> int:
    rating = re.search(r"Rating: \[\[(.*)\]\]", response)
    if rating:
        extracted_text = rating.group(1).strip().lower()
        if extracted_text == "relevant":
            return 1
        elif extracted_text == "unrelevant":
            return 0
        else:
            # If the extracted text is neither "relevant" nor "unrelevant", return 0 as default
            return 0
    else:
        return 0


def run_llm_judge(user_prompt: str, model_name: str = "gpt-4.1", deployment: str = "gpt-4.1-standard") -> str:
    try:
        return get_judge_agent("citation_judge").single_turn(user_prompt, None, None, 800)
    except Exception as e:
        print(f"Error in run_llm_judge: {e}")
        return ""


# Async versions of citation scoring functions

async def score_with_citation_recall_async(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_recall_has_citation_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    try:
        response = await get_judge_agent("citation_judge").async_single_turn(
            user_prompt, None, None, 800,
        )
    except Exception as e:
        print(f"Error scoring citation recall: {e}")
        return 0.0
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_no_citation_recall_async(question: str, claim: str, full_response: str) -> float:
    user_prompt = citation_recall_no_citation_prompt.format(
        question=question, statement=claim, full_response=full_response
    )
    try:
        response = await get_judge_agent("citation_judge").async_single_turn(
            user_prompt, None, None, 800,
        )
    except Exception as e:
        print(f"Error scoring no citation recall: {e}")
        return 0.0
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_citation_recall_async(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    if len(concatenated_citations) == 0:
        return await score_no_citation_recall_async(question, claim, full_response)
    else:
        return await score_with_citation_recall_async(question, claim, concatenated_citations)


async def score_citation_precision_async(question: str, claim: str, concatenated_citations: str) -> float:
    user_prompt = citation_precision_prompt.format(
        question=question, statement=claim, concatenated_cited_snippets=concatenated_citations
    )
    try:
        response = await get_judge_agent("citation_judge").async_single_turn(
            user_prompt, None, None, 800,
        )
    except Exception as e:
        print(f"Error scoring citation precision: {e}")
        return 0.0
    if "[[Fully supported]]" in response:
        return 1.0
    elif "[[Partially supported]]" in response:
        return 0.5
    else:
        return 0.0


async def score_citation_f1_async(question: str, claim: str, concatenated_citations: str, full_response: str) -> float:
    recall_task = score_citation_recall_async(question, claim, concatenated_citations, full_response)
    precision_task = score_citation_precision_async(question, claim, concatenated_citations)
    
    recall, precision = await asyncio.gather(recall_task, precision_task)
    
    # avoid division by zero
    if recall + precision == 0:
        return 0
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


async def score_in_context_citations_async(question: str, response: str, citations: Dict[str, str]) -> float:
    """
    Async version of score_in_context_citations.
    Compute the cumulative weighted score for a system response such that the final score is between 0 and 1.
    :param question: The original question
    :param response: The system response
    :param citations: Dictionary of citation IDs to citation text
    :return: final weighted score between 0 and 1
    """
    if not citations:
        return 0.0

    def concatenate_citations(citation_ids: List[str], citations: Dict[str, str]) -> str:
        if len(citation_ids) == 0:
            return ""
        return "\n\n".join([citations[citation_id] for citation_id in citation_ids if citation_id in citations])

    claims = extract_claims_and_corresponding_citation_ids(response)

    citation_format_reward = score_citation_format(claims, citations)

    # Create async tasks for all F1 score calculations
    f1_tasks = []
    for claim_text, citation_ids in claims.items():
        concatenated_citations = concatenate_citations(citation_ids, citations)
        task = score_citation_f1_async(question, claim_text, concatenated_citations, response)
        f1_tasks.append(task)

    # Execute all F1 calculations concurrently
    if f1_tasks:
        f1_scores = await asyncio.gather(*f1_tasks)
        avg_f1 = sum(f1_scores) / len(f1_scores)
    else:
        avg_f1 = 0

    return 0.6 * avg_f1 + 0.4 * citation_format_reward


# ---------------------------------------------------------------------------
# Batch citation judge — single LLM call per response
# ---------------------------------------------------------------------------

_BATCH_CITATION_SYSTEM_PROMPT = """You are an expert citation evaluator. You will receive a question and a list of (claim, snippet) pairs extracted from an AI assistant's response. For each pair evaluate:
1. **Recall**: Does the snippet support the claim?
   - "fully" — most information in the claim is supported by or extracted from the snippet
   - "partially" — more than half of the claim is supported, but some part is not
   - "none" — the claim is largely unrelated to or contradicted by the snippet
2. **Precision**: Is the snippet relevant to the claim?
   - "relevant" — some key points of the claim are supported by or extracted from the snippet
   - "irrelevant" — the claim is almost unrelated to the snippet

Return ONLY a JSON array with one object per claim, in the same order as the input:
[{"claim_idx": 0, "recall": "fully", "precision": "relevant"}, ...]
No extra text outside the JSON array."""

_BATCH_CITATION_USER_TEMPLATE = """Question: {question}

Claims and their cited snippets:
{claims_block}

Return the JSON array now."""


def _build_batch_citation_prompt(
    question: str,
    cited_claims: List[Tuple[str, str]],  # [(claim_text, concatenated_snippets), ...]
) -> str:
    lines = []
    for i, (claim_text, snippet) in enumerate(cited_claims):
        lines.append(f"[{i}] Claim: {claim_text}")
        lines.append(f"    Snippet: {snippet[:1000]}")  # truncate very long snippets
        lines.append("")
    claims_block = "\n".join(lines).rstrip()
    return _BATCH_CITATION_USER_TEMPLATE.format(question=question, claims_block=claims_block)


def _parse_batch_citation_response(response: str, n: int) -> List[Tuple[float, float]]:
    """
    Parse the JSON array returned by the batch judge.
    Returns list of (recall_score, precision_score) for each claim.
    Falls back to (0.0, 0.0) for missing/malformed entries.
    """
    recall_map = {"fully": 1.0, "partially": 0.5, "none": 0.0}
    precision_map = {"relevant": 1.0, "irrelevant": 0.0}

    results = [(0.0, 0.0)] * n
    try:
        # Extract JSON array from response (model may wrap in markdown)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return results
        parsed = json.loads(json_match.group(0))
        for item in parsed:
            idx = int(item.get("claim_idx", -1))
            if 0 <= idx < n:
                recall = recall_map.get(str(item.get("recall", "")).lower(), 0.0)
                precision = precision_map.get(str(item.get("precision", "")).lower(), 0.0)
                results[idx] = (recall, precision)
    except Exception:
        pass
    return results


def score_in_context_citations_batch(
    question: str,
    response: str,
    citations: Dict[str, str],
) -> float:
    """
    Score citation quality with a single LLM call per response.
    Only evaluates claims that have <cite> tags (skips uncited text).
    Returns a float in [0, 1]: 0.6 * avg_f1 + 0.4 * citation_format_reward.
    """
    if not citations:
        return 0.0

    claims = extract_claims_and_corresponding_citation_ids(response)
    citation_format_reward = score_citation_format(claims, citations)

    # Only keep cited claims
    cited_claims = [
        (claim_text, citation_ids)
        for claim_text, citation_ids in claims.items()
        if citation_ids
    ]
    if not cited_claims:
        return 0.4 * citation_format_reward

    def concatenate_citations(citation_ids: List[str]) -> str:
        return "\n\n".join([citations[cid] for cid in citation_ids if cid in citations])

    pairs = [(ct, concatenate_citations(cids)) for ct, cids in cited_claims]
    user_prompt = _build_batch_citation_prompt(question, pairs)

    try:
        raw = get_judge_agent("citation_judge").single_turn(
            user_prompt, _BATCH_CITATION_SYSTEM_PROMPT, None, 50 * len(pairs) + 64,
        )
    except Exception as e:
        print(f"Error in citation batch judge: {e}")
        return None  # sentinel: caller can distinguish API failure from genuine score=0

    scores = _parse_batch_citation_response(raw, len(pairs))
    f1_scores = []
    for recall, precision in scores:
        if recall + precision == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * recall * precision / (recall + precision))

    avg_f1 = sum(f1_scores) / len(f1_scores)
    return 0.6 * avg_f1 + 0.4 * citation_format_reward


async def score_in_context_citations_batch_async(
    question: str,
    response: str,
    citations: Dict[str, str],
) -> float:
    """
    Async version of score_in_context_citations_batch.
    Single LLM call per response, only evaluates cited claims.
    """
    if not citations:
        return 0.0

    claims = extract_claims_and_corresponding_citation_ids(response)
    citation_format_reward = score_citation_format(claims, citations)

    cited_claims = [
        (claim_text, citation_ids)
        for claim_text, citation_ids in claims.items()
        if citation_ids
    ]
    if not cited_claims:
        return 0.4 * citation_format_reward

    def concatenate_citations(citation_ids: List[str]) -> str:
        return "\n\n".join([citations[cid] for cid in citation_ids if cid in citations])

    pairs = [(ct, concatenate_citations(cids)) for ct, cids in cited_claims]
    user_prompt = _build_batch_citation_prompt(question, pairs)

    try:
        raw = await asyncio.wait_for(
            get_judge_agent("citation_judge").async_single_turn(
                user_prompt, _BATCH_CITATION_SYSTEM_PROMPT, None, 50 * len(pairs) + 64,
            ),
            timeout=_CITATION_JUDGE_TIMEOUT_S,
        )
    except Exception as e:
        print(f"Error in batch citation judge: {e}")
        return 0.4 * citation_format_reward

    scores = _parse_batch_citation_response(raw, len(pairs))
    f1_scores = []
    for recall, precision in scores:
        if recall + precision == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * recall * precision / (recall + precision))

    avg_f1 = sum(f1_scores) / len(f1_scores)
    return 0.6 * avg_f1 + 0.4 * citation_format_reward
