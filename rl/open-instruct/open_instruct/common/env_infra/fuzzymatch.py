"""
Fuzzy Match Module

Provides fuzzy string matching functionality for tool scripts to match
text fields (names, descriptions, locations) with tolerance for variations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from rapidfuzz import fuzz

# Minimum threshold to avoid returning completely unrelated matches
MIN_THRESHOLD = 50

IndexSelector = Optional[Union[int, str]]  # int index, "*" wildcard, None for direct access

_KEY_PATH_SEGMENT_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\[(\d+)\]|\[\*\]|\[\])?$"
)

_NL_TOKEN_RE = re.compile(r"[^a-z0-9]+")




def _normalize_text_simple(value: Any) -> str:
    """
    Simple normalization:
    - Convert to string
    - Lowercase
    - Replace non-alphanumeric with space
    - Collapse whitespace
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    
    value = value.strip().casefold()
    if not value:
        return ""
        
    value = _NL_TOKEN_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _compute_score(s1: str, s2: str) -> int:
    """
    Compute similarity score using token_sort_ratio on normalized strings.
    """
    n1 = _normalize_text_simple(s1)
    n2 = _normalize_text_simple(s2)
    
    if not n1 or not n2:
        return 0
        
    # hard rejection floor logic from legacy string_match is handled by callers if needed,
    # but strictly speaking fuzz.token_sort_ratio returns 0-100.
    return int(fuzz.token_sort_ratio(n1, n2))


def string_match(s1: Any, s2: Any, thresh: int = 80) -> bool:
    """
    Determine whether two strings match using RapidFuzz token_sort_ratio,
    with a hard floor: score < 80 always returns False.

    Args:
        s1: First string-like value.
        s2: Second string-like value.
        thresh: Matching threshold in [0, 100]. Defaults to 85.

    Returns:
        True if match, else False.
    """
    if s1 is None or s2 is None:
        return False
        
    # Clean inputs and compute score
    score = _compute_score(s1, s2)

    # Hard rejection floor (legacy behavior preservation)
    if score < 80:
        return False
    
    # Ensure safe threshold integer
    try:
        thresh_int = int(thresh)
    except Exception:
        thresh_int = 85
    thresh_int = max(0, min(100, thresh_int))

    return score >= thresh_int



