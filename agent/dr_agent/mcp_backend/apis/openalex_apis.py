"""
OpenAlex API — free, open scholarly search (no API key required).

Docs: https://docs.openalex.org/api-entities/works/search-works
"""

import logging
import os
import threading
import time
from typing import List, Optional

import requests

from ..cache import cached

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"
# Polite pool: include contact email so OpenAlex gives faster responses.
_USER_AGENT_EMAIL = os.environ.get("OPENALEX_EMAIL", "research-agent@example.com")
_TIMEOUT = int(os.environ.get("API_TIMEOUT", "10"))

# Module-level rate-limit flag: set when a 429 is received, auto-cleared after backoff
class _RateLimitFlag:
    def __init__(self):
        self._lock = threading.Lock()
        self._until: float = 0.0

    def set(self, backoff_seconds: float):
        with self._lock:
            self._until = time.monotonic() + backoff_seconds

    def is_set(self) -> bool:
        with self._lock:
            return time.monotonic() < self._until

_RATE_LIMITED = _RateLimitFlag()


def _headers() -> dict:
    return {"User-Agent": f"mailto:{_USER_AGENT_EMAIL}"}


def _normalize_year_filter(year: Optional[str]) -> Optional[str]:
    """Convert year string to OpenAlex publication_year filter.

    Supported formats:
      '2024'        → 'publication_year:2024'
      '2020-2024'   → 'publication_year:2020-2024'
      '2020-'       → 'publication_year:>2019'
      '-2024'       → 'publication_year:<2025'
    """
    if not year:
        return None
    year = year.strip()
    if "-" in year:
        parts = year.split("-", 1)
        start, end = parts[0].strip(), parts[1].strip()
        if start and end:
            return f"publication_year:{start}-{end}"
        elif start:
            return f"publication_year:>{int(start) - 1}"
        elif end:
            return f"publication_year:<{int(end) + 1}"
    return f"publication_year:{year}"


@cached()
def search_openalex_snippets(
    query: str,
    year: Optional[str] = None,
    limit: int = 10,
    sort: str = "relevance_score:desc",
    timeout: int = _TIMEOUT,
) -> dict:
    """
    Search OpenAlex works by keyword and return normalized snippet results.

    Returns:
        {"data": [{"snippet": {"text": ..., "snippetKind": "abstract"},
                   "paper": {"title": ..., "url": ...},
                   "score": float | None,
                   "source": "openalex",
                   "publish_time": str | None}, ...]}
    """
    if not query or not query.strip():
        return {"data": []}

    params: dict = {
        "search": query.strip(),
        "per_page": max(1, min(int(limit), 50)),
        "sort": sort,
        "select": "id,title,doi,publication_year,abstract_inverted_index,primary_location,open_access,relevance_score",
    }

    filters: List[str] = []
    year_filter = _normalize_year_filter(year)
    if year_filter:
        filters.append(year_filter)
    if filters:
        params["filter"] = ",".join(filters)

    import time as _time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{OPENALEX_BASE_URL}/works",
                params=params,
                headers=_headers(),
                timeout=timeout,
            )
            if resp.status_code == 429:
                raw_retry = resp.headers.get("Retry-After")
                if raw_retry is not None:
                    val = int(raw_retry)
                    # Retry-After may be an absolute Unix timestamp or a relative delta
                    now = int(_time.time())
                    retry_after = (val - now) if val > now + 1000 else val
                    retry_after = max(1, min(retry_after, 60))  # clamp to [1, 60]s
                else:
                    retry_after = 5 * (2 ** attempt)  # exponential backoff: 5, 10, 20s
                _RATE_LIMITED.set(retry_after)
                logger.warning("OpenAlex 429 rate limit, retrying after %ss (attempt %d/%d)", retry_after, attempt + 1, max_retries)
                _time.sleep(retry_after)
                continue
            resp.raise_for_status()
            raw = resp.json()
            break
        except requests.RequestException as e:
            logger.warning("OpenAlex API request failed: %s", e)
            return {"data": []}
    else:
        logger.warning("OpenAlex API request failed after %d retries (429 rate limit)", max_retries)
        return {"data": []}

    results = raw.get("results") or []
    normalized = []
    for item in results:
        title = (item.get("title") or "").strip()

        # Reconstruct abstract from inverted index
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

        # Best URL: DOI > OA URL > OpenAlex landing page
        url = ""
        doi = item.get("doi")
        if doi:
            url = doi if doi.startswith("http") else f"https://doi.org/{doi.lstrip('/')}"
        if not url:
            primary = item.get("primary_location") or {}
            url = primary.get("landing_page_url") or primary.get("pdf_url") or ""
        if not url:
            openalex_id = item.get("id", "")
            if openalex_id:
                url = openalex_id  # e.g. https://openalex.org/W123

        pub_year = item.get("publication_year")
        score = item.get("relevance_score")

        normalized.append({
            "snippet": {"text": abstract, "snippetKind": "abstract"},
            "paper": {"title": title, "url": url},
            "score": float(score) if score is not None else None,
            "source": "openalex",
            "publish_time": str(pub_year) if pub_year else None,
        })

    return {"data": normalized}


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    try:
        positions: dict[int, str] = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        return " ".join(positions[i] for i in sorted(positions))
    except Exception:
        return ""
