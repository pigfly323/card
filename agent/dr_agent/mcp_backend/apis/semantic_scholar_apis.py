import os
import warnings
import logging
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from requests.exceptions import RequestException

from ..cache import cached
from .data_model import (
    PaperSnippet,
    SemanticScholarAuthorData,
    SemanticScholarPaperData,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
logger = logging.getLogger(__name__)

S2_API_KEY = os.getenv("S2_API_KEY", "2004223713877516387")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
S2_SEARCH_LIMIT = 100
INTRANET_BING_SEARCH_URL = os.getenv(
    "UNIVERSAL_SEARCH_API_URL",
    os.getenv("SERPER_BASE_URL", "http://agi.sankuai.com/tools/universal-search/api/v1"),
)

S2_GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_RECOMMENDATIONS_API_URL = "https://api.semanticscholar.org/recommendations/v1"


def _s2_use_proxy() -> bool:
    """Whether Semantic Scholar requests should respect env proxy settings."""
    return os.getenv("S2_USE_PROXY", "0").lower() in {"1", "true", "yes"}


def _s2_request(method: str, url: str, **kwargs):
    """
    Wrapper for Semantic Scholar HTTP calls.
    By default disables env proxies to avoid corporate proxy 403 tunneling failures.
    """
    if _s2_use_proxy():
        return requests.request(method, url, **kwargs)
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)

# authors.authorId,authors.paperCount,authors.citationCount
S2_PAPER_SEARCH_FIELDS = "paperId,corpusId,url,title,abstract,authors,authors.name,year,venue,citationCount,openAccessPdf,externalIds,isOpenAccess"
S2_PAPER_CITATION_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_REFERENCE_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_RECOMMENDATION_FIELDS = "paperId,corpusId,title,abstract,year,venue"

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    total: int
    offset: int = Field(..., description="Offset for pagination")
    next: int = Field(..., description="Next page offset")
    data: List[T] = Field(..., description="Data items")


# This is because the Semantic Scholar API right now doesn't support
# pagination for snippets, so we will ignore the offset and limit.
class PaperSnippetApiResponse(BaseModel, Generic[T]):
    data: List[T] = Field(..., description="Data items")


# This is a subset of the supported query parameters for the Semantic Scholar API.
# We include the most important ones and remove some for clarity. (e.g., there are year and
# publicationDateOrYear, which can be confusing to the model).
class SemanticScholarSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Restrict results to the given range of publication year (inclusive).",
        examples=["2015-2020", "2015-", "-2015"],
    )
    minCitationCount: Optional[int] = Field(
        None,
        description="Restrict results to only include papers with the minimum number of citations, inclusive.",
        examples=[100, 1000],
    )
    sort: Optional[str] = Field(
        None,
        description="Sort results by publicationDate and citationCount in ascending or descending order.",
        examples=[
            "citationCount:asc",
            "publicationDate:desc",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )


################## NOT EXPLICITLY USED ##################
# fields: Optional[str] = Field(None, description="A comma-separated list of the fields to be returned.")
# publicationTypes: Optional[str] = Field(None, description="Restrict results by publication types, use a comma-separated list.")
# openAccessPdf: Optional[bool] = Field(None, description="Restrict results to only include papers with a public PDF.")
# publicationDateOrYear: Optional[str] = Field(None, description="""
# Restrict results to the given range of publication dates or years (inclusive).
# Accepts the format <startDate>:<endDate>. Prefixes supported for specific ranges.""")
# fieldsOfStudy: Optional[str] = Field(None, desc="Restrict results to given field-of-study, using the s2FieldsOfStudy paper field.")
# offset: Optional[int] = Field(0, description="Start with the element at this position in the list.")
# limit: Optional[int] = Field(100, description="The maximum number of results to return, must be <= 100.")


@cached()
def search_semantic_scholar_keywords(
    query_params: SemanticScholarSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 25,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/search",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()

    results = res.json()

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


class SemanticScholarSnippetSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Restrict results to the given range of publication year (inclusive).",
        examples=["2015-2020", "2015-", "-2015"],
    )
    paperIds: Optional[Union[str, List[str]]] = Field(
        None,
        description="Restricts results to snippets from specific papers. To specify papers, provide a comma-separated list of their IDs. You can provide up to approximately 100 IDs.",
        examples=[
            "649def34f8be52c8b66281af98ae884c09aef38b",
            "649def34f8be52c8b66281af98ae884c09aef38b,CorpusId:215416146",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )
    startDate: Optional[str] = Field(
        None,
        description="Bing search recency start date (YYYY-MM-DD).",
        examples=["2025-06-26"],
    )
    endDate: Optional[str] = Field(
        None,
        description="Bing search recency end date (YYYY-MM-DD).",
        examples=["2025-06-26"],
    )
    sites: Optional[List[str]] = Field(
        None,
        description="Bing site filters, e.g. ['https://www.csdn.net/'].",
    )
    searchEngine: Optional[str] = Field(
        None,
        description="Search engine type, e.g. search_pro_bing.",
        examples=["search_pro_bing"],
    )
    searchRecencyFilter: Optional[str] = Field(
        None,
        description="Recency filter: oneDay|oneWeek|oneMonth|oneYear|noLimit.",
        examples=["oneYear"],
    )
    insite: Optional[str] = Field(
        None,
        description='Single-domain filter, e.g. "zhihu.com".',
        examples=["zhihu.com"],
    )
    contentSize: Optional[str] = Field(
        None,
        description="Returned content size: medium|high.",
        examples=["medium"],
    )


@cached()
def search_semantic_scholar_snippets(
    query_params: SemanticScholarSnippetSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 10,
    timeout: int = TIMEOUT,
    fallbackEngine: str = None,
) -> PaperSnippetApiResponse[PaperSnippet]:
    if offset:
        warnings.warn("Intranet bing search does not support offset; offset is ignored.")

    params = query_params.model_dump(exclude_none=True)
    # For intranet web search, keep query clean to maximize recall.
    final_query = (params.get("query") or "").strip()
    if not final_query:
        return {"data": []}

    search_recency_filter = "noLimit"
    year_value = (params.get("year") or "").strip()
    start_date = params.get("startDate")
    end_date = params.get("endDate")

    if year_value:
        search_recency_filter = "oneYear"
        # Parse year into startDate / endDate for precise filtering
        if "-" in year_value:
            parts = year_value.split("-", 1)
            start_year, end_year = parts[0].strip(), parts[1].strip()
            if start_year and not start_date:
                start_date = f"{start_year}-01-01"
            if end_year and not end_date:
                end_date = f"{end_year}-12-31"
        else:
            # Single year: e.g. "2024" → 2024-01-01 ~ 2024-12-31
            if not start_date:
                start_date = f"{year_value}-01-01"
            if not end_date:
                end_date = f"{year_value}-12-31"

    import uuid

    sources = [s.strip() for s in os.getenv("UNIVERSAL_SEARCH_SOURCES", "serper-search").split(",") if s.strip()]
    ttl = int(os.getenv("UNIVERSAL_SEARCH_TTL", "3600"))

    # scholar search always uses type=scholar
    serper_search_param: dict = {"type": "scholar"}
    if start_date:
        serper_search_param["startDate"] = start_date
    if end_date:
        serper_search_param["endDate"] = end_date
    tbs = os.getenv("UNIVERSAL_SEARCH_TBS", "")
    if tbs:
        serper_search_param["tbs"] = tbs

    search_timeout_ms = int(os.getenv("UNIVERSAL_SEARCH_TIMEOUT_MS", "20000"))
    payload = {
        "query": final_query,
        "topK": max(1, min(int(limit), 50)),
        "timeout": search_timeout_ms,
        "sources": sources,
        "searchAccessAuth": {
            "userName": os.getenv("UNIVERSAL_SEARCH_USER_NAME", ""),
            "taskType": os.getenv("UNIVERSAL_SEARCH_TASK_TYPE", ""),
            "taskDesc": os.getenv("UNIVERSAL_SEARCH_TASK_DESC", ""),
        },
        "fallbackEngine": "bing",
    }
    if ttl > 0:
        payload["ttl"] = ttl
    if "serper-search" in sources:
        payload["serperSearchParam"] = serper_search_param

    fb = fallbackEngine or os.getenv("UNIVERSAL_SEARCH_FALLBACK_ENGINE", "")
    if fb:
        payload["fallbackEngine"] = fb
    if os.getenv("MCP_DEBUG_SNIPPET_SEARCH", "0").lower() in {"1", "true", "yes"}:
        logger.warning("snippet_search request payload=%s", payload)

    from .api_key_pool import get_s2_pool, is_google_serper_key
    pool = get_s2_pool()
    max_retries = pool.size()
    last_exc = None
    raw = None

    for attempt in range(max_retries):
        current_key = pool.get()

        # Route to serper-search (Google) when key is in GOOGLE_SERPENT_API_KEY_LIST;
        # otherwise use the default UNIVERSAL_SEARCH_SOURCES engine.
        if is_google_serper_key(current_key):
            request_payload = {**payload, "sources": ["serper-search"], "fallbackEngine": "bing"}
        else:
            request_payload = payload

        logger.info(
            "[snippet-search] key=%s... is_google=%s sources=%s fallback=%s attempt=%s/%s query=%s",
            (current_key or "")[:8],
            is_google_serper_key(current_key),
            request_payload.get("sources"),
            request_payload.get("fallbackEngine", ""),
            attempt + 1,
            max_retries,
            (request_payload.get("query", "") or "")[:180],
        )

        trace_id = str(uuid.uuid4()).replace("-", "")
        headers = {
            "Content-Type": "application/json",
            "m-traceid": trace_id,
        }
        if current_key:
            headers["Authorization"] = f"Bearer {current_key}"
        try:
            res = requests.post(
                INTRANET_BING_SEARCH_URL,
                json=request_payload,
                headers=headers,
                timeout=timeout,
            )
            if res.status_code == 429:
                pool.mark_rate_limited(current_key)
                last_exc = Exception(f"Rate limited (429) on S2 key {current_key[:8]}...")
                continue
            res.raise_for_status()
            raw = res.json()
            # code:1029 = RPM limit — treat as rate-limit, rotate key and retry
            inner = (raw.get("data") or {}) if isinstance(raw, dict) else {}
            if inner.get("code") == 1029:
                pool.mark_rate_limited(current_key)
                last_exc = Exception(f"RPM limit on S2 key {current_key[:8]}...: {inner.get('msg', '')}")
                raw = None
                continue
            break
        except RequestException as e:
            last_exc = e
            logger.warning("snippet_search request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            continue  # retry with next key
    else:
        raise last_exc or Exception("All S2 API keys exhausted or failed")

    if raw is None:
        logger.warning("snippet_search returned no data: %s", last_exc)
        return {"data": []}

    # Normalize intranet search format to snippet_search format expected by MCP tool interface.
    result_list = (((raw or {}).get("data") or {}).get("results")) or []
    normalized = []
    for item in result_list:
        if not isinstance(item, dict):
            continue
        snippet_text = (item.get("content") or item.get("snippet") or "").strip()
        title = (item.get("title") or "").strip()
        normalized.append(
            {
                "snippet": {"text": snippet_text, "snippetKind": "abstract"},
                "paper": {"title": title, "url": item.get("url", "").strip()},
                "score": None,
                "source": item.get("source"),
                "publish_time": item.get("publish_time"),
            }
        )

    return {"data": normalized}


def search_semantic_scholar_bulk_api(
    query_params: SemanticScholarSearchQueryParams,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/search/bulk",
        params={
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()

    results = res.json()

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


def download_paper_details(
    paper_id: str,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}",
        params={
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()
    results = res.json()
    return results


def download_paper_references(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_REFERENCE_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}/references",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


def download_paper_citations(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_CITATION_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}/citations",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


def download_paper_details_batch(
    paper_ids: List[str],
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.post(
        f"{S2_GRAPH_API_URL}/paper/batch",
        params={"fields": fields},
        json={"ids": paper_ids},
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


# Example usage:
if __name__ == "__main__":
    result1 = search_semantic_scholar_snippets(
        SemanticScholarSnippetSearchQueryParams(
            query="how to set learning rate in state space models?",
        ),
    )
