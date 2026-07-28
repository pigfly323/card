import hashlib
import uuid
from typing import Dict, Any, Optional, List
import re
import logging
import os

LOGGER = logging.getLogger(__name__)


def generate_snippet_id() -> str:
    """
    Generate a unique random ID for snippets.
    Should be used when inserting snippets into the context.

    Returns:
        A unique string ID in format: '<hash>'
    """
    # Generate a random UUID and create a shorter hash
    random_uuid = str(uuid.uuid4())
    hash_object = hashlib.md5(random_uuid.encode())
    short_hash = hash_object.hexdigest()[:8]  # Take first 8 characters
    return f"{short_hash}"


def extract_answer_context_citations(response: str, use_full_response_as_answer: bool = False) -> tuple[str, str, Dict[str, str]]:
    """
    Extract the answer from the context.

    When multiple <answer>...</answer> blocks are present (e.g. one inside a
    verify_answer call and one as the final answer), the LAST match is used as
    the final answer. Context and citations are extracted from everything before
    the last <answer> tag.

    Returns:
        tuple: (extracted_context, extracted_answer, extracted_citations)
               Returns (None, None, None) if no answer tags found
    """
    if bool(os.environ.get("USE_FULL_RESPONSE_AS_ANSWER", use_full_response_as_answer)):
        return None, response, None

    # Find all <answer>...</answer> blocks, but only consider those that are
    # NOT nested inside a <call_tool> (verify_answer) block.
    # The final standalone <answer> is the agent's actual answer.
    pattern = r"<answer>(.*?)</answer>"
    all_matches = list(re.finditer(pattern, response, re.DOTALL))

    if not all_matches:
        LOGGER.warning(f"No <answer></answer> tags found in response with use_full_response_as_answer set to {use_full_response_as_answer}")
        return None, None, None

    # Build call_tool spans to filter out nested answers
    call_tool_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"<call_tool\b[^>]*>.*?(?:</call_tool>|</call>)", response, re.DOTALL)
    ]

    standalone_matches = [
        m for m in all_matches
        if not any(cs <= m.start() and m.end() <= ce for cs, ce in call_tool_spans)
    ]

    if standalone_matches:
        last_match = standalone_matches[-1]
    else:
        # Fallback: no standalone answer found — return None to signal missing answer
        LOGGER.warning(f"No standalone <answer></answer> found outside call_tool blocks")
        return None, None, None

    extracted_answer = last_match.group(1).strip()

    # Context is everything before the last <answer> tag
    context_text = response[: last_match.start()].strip()
    extracted_citations = extract_citations_from_context(context_text)
    extracted_context = context_text

    return extracted_context, extracted_answer, extracted_citations


def extract_citations_from_context(context: str) -> Dict[str, str]:
    """
    Extract citations from the context.

    Citations are expected to be in the format:
    <snippets id="a1b2c3d4" metadata='{"author": "smith", "source": "arxiv", "year": 2023}'>
        Search result content here
    </snippets>
    
    Other formats are also supported, such as:
    <snippet id="a1b2c3d4">Search result content here</snippet>
    <webpage id="a1b2c3d4">Search result content here</webpage>
    
    The id can be any string, including hash-like IDs (e.g., a1b2c3d4)

    Args:
        context: The context string containing citations

    Returns:
        Dictionary mapping id to search results content for all found citations
    """
    citations = {}

    # Pattern to match <snippets id="xxx" ...> content </snippets>
    # Updated to handle both quoted and unquoted attributes in HTML-like format
    pattern1 = r'<snippets?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippets?>'
    pattern2 = r'<snippet?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</snippet?>'
    pattern3 = r'<webpage?\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>(.*?)</webpage?>'

    matches1 = re.findall(pattern1, context, re.DOTALL)
    matches2 = re.findall(pattern2, context, re.DOTALL)
    matches3 = re.findall(pattern3, context, re.DOTALL)

    for quote_char, snippet_id, search_results in matches1 + matches2 + matches3:
        # Clean up the id and search results (remove extra whitespace)
        clean_id = snippet_id.strip()
        clean_search_results = search_results.strip()
        citations[clean_id] = clean_search_results

    return citations


def extract_search_tool_calls(context: str, mcp_parser_name: Optional[str] = None) -> List[str]:
    if not mcp_parser_name:
        matches = re.findall(r"<search>(.*?)</search>", context, re.DOTALL)
        # For the simple case, matches are strings, not tuples
        if matches:
            return [match.strip() for match in matches if match.strip()]
        else:
            return []
    elif mcp_parser_name == "unified":
        matches = re.findall(r"<tool name=(.*?)>(.*?)</tool>", context, re.DOTALL)
    elif mcp_parser_name == "v20250824":
        matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call_tool>", context, re.DOTALL)
        if not matches:
            matches = re.findall(r"<call_tool name=(.*?)>(.*?)</call>", context, re.DOTALL)
    else:
        raise ValueError(f"Unsupported MCP parser name: {mcp_parser_name}")
    
    # For MCP parser cases, matches are tuples, so we access match[1]
    if matches:
        return [match[1] for match in matches if match[1].strip()]
    else:
        return []
    
    
def compute_format_reward(response: str, mcp_parser_name: Optional[str] = None, use_full_response_as_answer: bool = False) -> float:
    if use_full_response_as_answer:
        return 1.0
    
    # Check if response contains a STANDALONE final <answer></answer> block,
    # i.e. one that is NOT nested inside a <call_tool> (verify_answer) call.
    # The agent must write the final answer independently after all tool calls.
    # An <answer> inside a verify_answer call does NOT count for format reward.
    #
    # Strategy: find all <answer>...</answer> matches, then check whether each
    # one is nested inside a <call_tool ...>...</call_tool> block.
    # Only the last match outside any call_tool block counts.
    answer_format_reward = 0.0
    answer_matches = list(re.finditer(r"<answer>.*?</answer>", response, re.DOTALL))
    if answer_matches:
        # Build set of (start, end) spans of all call_tool blocks
        call_tool_spans = [
            (m.start(), m.end())
            for m in re.finditer(r"<call_tool\b[^>]*>.*?(?:</call_tool>|</call>)", response, re.DOTALL)
        ]
        # Check if the LAST <answer> block is outside all call_tool spans
        last_answer = answer_matches[-1]
        inside_call_tool = any(
            cs <= last_answer.start() and last_answer.end() <= ce
            for cs, ce in call_tool_spans
        )
        if not inside_call_tool:
            answer_format_reward = 1.0
    
    # check if response contains citations between <cite></cite> tags
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    citation_match = re.search(citation_pattern, response, re.DOTALL)
    if citation_match:
        citation_format_reward = 1.0
    else:
        citation_format_reward = 0.0
    
    # check if response contains at least one valid query between <query></query> tags
    queries = extract_search_tool_calls(response, mcp_parser_name=mcp_parser_name)
    if queries:
        query_format_reward = 1.0
    else:
        query_format_reward = 0.0
    
    # compute weighted average of format rewards
    format_reward = 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward
    return format_reward