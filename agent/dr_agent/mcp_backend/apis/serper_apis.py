import json
import logging
import os
from typing import Dict, List, Optional, Union

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

# Load environment variables
dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
logger = logging.getLogger(__name__)


def _search_debug_enabled() -> bool:
    return (os.getenv("UNIVERSAL_SEARCH_DEBUG_LOG", "0").strip().lower() in ("1", "true", "yes"))


def _q_preview(q: str, max_len: int = 180) -> str:
    q = (q or "").replace("\n", " ").strip()
    if len(q) <= max_len:
        return q
    return q[:max_len] + f"...(len={len(q)})"


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",
    api_key: str = None,
    fallbackEngine: str = None,
) -> SearchResponse:
    """
    Search using the universal-search API (serper-search engine).

    Endpoint: SERPER_BASE_URL (default: http://agi.sankuai.com/tools/universal-search/api/v1)
    Auth:     Authorization: Bearer <appId>  (from SERPER_API_KEY / SERPER_API_KEY_LIST)

    Environment variables:
        UNIVERSAL_SEARCH_SOURCES      comma-separated sources (default: serper-search)
        UNIVERSAL_SEARCH_TTL          cache TTL in seconds, <=0 disables cache (default: 3600)
        UNIVERSAL_SEARCH_USER_NAME    searchAccessAuth.userName (default: dr_tulu)
        UNIVERSAL_SEARCH_TASK_TYPE    searchAccessAuth.taskType (default: rl_training)
        UNIVERSAL_SEARCH_TASK_DESC    searchAccessAuth.taskDesc (default: DR-Tulu RL training)
    """
    import uuid
    from .api_key_pool import get_serper_pool

    url = SERPER_BASE_URL.rstrip("/")
    sources = [s.strip() for s in os.getenv("UNIVERSAL_SEARCH_SOURCES", "serper-search").split(",") if s.strip()]
    ttl = int(os.getenv("UNIVERSAL_SEARCH_TTL", "3600"))

    search_timeout_ms = int(os.getenv("UNIVERSAL_SEARCH_TIMEOUT_MS", "20000"))
    payload_dict = {
        "query": query,
        "topK": num_results,
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
        payload_dict["ttl"] = ttl

    # serper-search supports pagination and time-range filtering via serperSearchParam
    serper_param = {}
    if os.getenv("UNIVERSAL_SEARCH_TBS"):
        serper_param["tbs"] = os.getenv("UNIVERSAL_SEARCH_TBS")
    if serper_param and "serper-search" in sources:
        payload_dict["serperSearchParam"] = serper_param

    fb = fallbackEngine or os.getenv("UNIVERSAL_SEARCH_FALLBACK_ENGINE", "")
    if fb:
        payload_dict["fallbackEngine"] = fb

    payload = json.dumps(payload_dict)

    # If caller provides explicit key, use it directly (no pool)
    if api_key:
        keys_to_try = [api_key]
        pool = None
    else:
        pool = get_serper_pool()
        keys_to_try = None  # will call pool.get() per attempt

    max_retries = pool.size() if pool else 1
    last_exc = None

    from .api_key_pool import is_google_serper_key

    for attempt in range(max_retries):
        current_key = keys_to_try[0] if keys_to_try else pool.get()

        # Route to serper-search (Google) when key is in GOOGLE_SERPENT_API_KEY_LIST;
        # otherwise use the default UNIVERSAL_SEARCH_SOURCES engine.
        if is_google_serper_key(current_key):
            payload_dict["sources"] = ["serper-search"]
            payload_dict["fallbackEngine"] = "bing"
        else:
            payload_dict["sources"] = sources
            payload_dict["fallbackEngine"] = fb if fb else "bing"
        payload = json.dumps(payload_dict)

        trace_id = str(uuid.uuid4()).replace("-", "")
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
            "m-traceid": trace_id,
        }
        logger.info(
            "[universal-search] key=%s... is_google=%s sources=%s fallback=%s attempt=%s/%s query=%s",
            current_key[:8],
            is_google_serper_key(current_key),
            payload_dict["sources"],
            payload_dict.get("fallbackEngine", ""),
            attempt + 1,
            max_retries,
            _q_preview(query),
        )
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=TIMEOUT)

            if response.status_code == 429:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search] 429 trace=%s key=%s... -> rotate",
                        trace_id,
                        current_key[:8],
                    )
                if pool:
                    pool.mark_rate_limited(current_key)
                last_exc = Exception(f"Rate limited (429) on key {current_key[:8]}...")
                continue

            if response.status_code != 200:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search] http_error trace=%s status=%s body=%s",
                        trace_id,
                        response.status_code,
                        response.text[:300],
                    )
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            if data.get("status") != 0:
                raise Exception(f"universal-search API error: {data.get('message', data)}")
            inner = data.get("data") or {}
            # code:1029 = RPM limit — treat as rate-limit, rotate key and retry
            inner_code = inner.get("code", 0)
            if inner_code == 1029:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search] rpm_limit trace=%s key=%s... msg=%s",
                        trace_id,
                        current_key[:8],
                        inner.get("msg", ""),
                    )
                if pool:
                    pool.mark_rate_limited(current_key)
                last_exc = Exception(f"RPM limit on key {current_key[:8]}...: {inner.get('msg', '')}")
                continue
            if inner_code != 0:
                if _search_debug_enabled():
                    logger.error(
                        "[universal-search] inner_error trace=%s code=%s msg=%s internal=%s",
                        trace_id,
                        inner_code,
                        inner.get("msg", ""),
                        inner.get("internalErrMap", {}),
                    )
                last_exc = Exception(
                    f"universal-search inner error code={inner_code}: "
                    f"{inner.get('msg', '')}, internal={inner.get('internalErrMap', {})}"
                )
                continue
            results = inner.get("results") or []
            if _search_debug_enabled():
                logger.info(
                    "[universal-search] response trace=%s code=%s result_count=%s msg=%s",
                    trace_id,
                    inner_code,
                    len(results),
                    inner.get("msg", ""),
                )
            return {
                "searchParameters": inner.get("searchParameters") or {"query": query, "topK": num_results},
                "organic": [
                    {
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                        "position": i + 1,
                    }
                    for i, r in enumerate(results)
                ],
            }

        except requests.exceptions.Timeout as e:
            if _search_debug_enabled():
                logger.warning(
                    "[universal-search] timeout trace=%s key=%s... err=%s",
                    trace_id,
                    current_key[:8],
                    str(e),
                )
            last_exc = Exception(f"Timeout on key {current_key[:8]}...: {str(e)}")
            continue  # retry with next key
        except requests.exceptions.RequestException as e:
            if _search_debug_enabled():
                logger.warning(
                    "[universal-search] request_exception trace=%s key=%s... err=%s",
                    trace_id,
                    current_key[:8],
                    str(e),
                )
            last_exc = Exception(f"Error performing Serper search: {str(e)}")
            continue  # retry with next key

    raise last_exc or Exception("All Serper API keys exhausted or failed")


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
    fallbackEngine: str = None,
) -> ScholarResponse:
    """
    Search academic papers using the universal-search API (serper-search engine).

    Environment variables:
        UNIVERSAL_SEARCH_SCHOLAR_SOURCES  comma-separated sources for scholar search
        UNIVERSAL_SEARCH_SOURCES          fallback sources (default: serper-search)
        UNIVERSAL_SEARCH_TTL              cache TTL in seconds (default: 3600)
    """
    import uuid
    from .api_key_pool import get_serper_pool

    url = SERPER_BASE_URL.rstrip("/")
    sources = (
        [s.strip() for s in os.getenv("UNIVERSAL_SEARCH_SCHOLAR_SOURCES", "").split(",") if s.strip()]
        or [s.strip() for s in os.getenv("UNIVERSAL_SEARCH_SOURCES", "serper-search").split(",") if s.strip()]
    )
    ttl = int(os.getenv("UNIVERSAL_SEARCH_TTL", "3600"))

    search_timeout_ms = int(os.getenv("UNIVERSAL_SEARCH_TIMEOUT_MS", "20000"))
    payload_dict = {
        "query": query,
        "topK": num_results,
        "timeout": search_timeout_ms,
        "sources": sources,
        "searchAccessAuth": {
            "userName": os.getenv("UNIVERSAL_SEARCH_USER_NAME", ""),
            "taskType": os.getenv("UNIVERSAL_SEARCH_TASK_TYPE", ""),
            "taskDesc": os.getenv("UNIVERSAL_SEARCH_TASK_DESC", ""),
        },
    }
    if ttl > 0:
        payload_dict["ttl"] = ttl

    fb = fallbackEngine or os.getenv("UNIVERSAL_SEARCH_FALLBACK_ENGINE", "")
    if fb:
        payload_dict["fallbackEngine"] = fb
    payload = json.dumps(payload_dict)

    if api_key:
        keys_to_try = [api_key]
        pool = None
    else:
        pool = get_serper_pool()
        keys_to_try = None

    max_retries = pool.size() if pool else 1
    last_exc = None

    from .api_key_pool import is_google_serper_key

    for attempt in range(max_retries):
        current_key = keys_to_try[0] if keys_to_try else pool.get()

        # Route to serper-search (Google) when key is in GOOGLE_SERPENT_API_KEY_LIST.
        if is_google_serper_key(current_key):
            payload_dict["sources"] = ["serper-search"]
            payload_dict["fallbackEngine"] = "bing"
        else:
            payload_dict["sources"] = sources
            payload_dict["fallbackEngine"] = fb if fb else "bing"
        payload = json.dumps(payload_dict)

        trace_id = str(uuid.uuid4()).replace("-", "")
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
            "m-traceid": trace_id,
        }
        logger.info(
            "[universal-search-scholar] key=%s... is_google=%s sources=%s fallback=%s attempt=%s/%s query=%s",
            current_key[:8],
            is_google_serper_key(current_key),
            payload_dict["sources"],
            payload_dict.get("fallbackEngine", ""),
            attempt + 1,
            max_retries,
            _q_preview(query),
        )
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=TIMEOUT)

            if response.status_code == 429:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search-scholar] 429 trace=%s key=%s... -> rotate",
                        trace_id,
                        current_key[:8],
                    )
                if pool:
                    pool.mark_rate_limited(current_key)
                last_exc = Exception(f"Rate limited (429) on key {current_key[:8]}...")
                continue

            if response.status_code != 200:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search-scholar] http_error trace=%s status=%s body=%s",
                        trace_id,
                        response.status_code,
                        response.text[:300],
                    )
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            if data.get("status") != 0:
                raise Exception(f"universal-search API error: {data.get('message', data)}")
            inner = data.get("data") or {}
            inner_code = inner.get("code", 0)
            if inner_code == 1029:
                if _search_debug_enabled():
                    logger.warning(
                        "[universal-search-scholar] rpm_limit trace=%s key=%s... msg=%s",
                        trace_id,
                        current_key[:8],
                        inner.get("msg", ""),
                    )
                if pool:
                    pool.mark_rate_limited(current_key)
                last_exc = Exception(f"RPM limit on key {current_key[:8]}...: {inner.get('msg', '')}")
                continue
            if inner_code != 0:
                if _search_debug_enabled():
                    logger.error(
                        "[universal-search-scholar] inner_error trace=%s code=%s msg=%s internal=%s",
                        trace_id,
                        inner_code,
                        inner.get("msg", ""),
                        inner.get("internalErrMap", {}),
                    )
                last_exc = Exception(
                    f"universal-search scholar inner error code={inner_code}: "
                    f"{inner.get('msg', '')}, internal={inner.get('internalErrMap', {})}"
                )
                continue
            results = inner.get("results") or []
            if _search_debug_enabled():
                logger.info(
                    "[universal-search-scholar] response trace=%s code=%s result_count=%s msg=%s",
                    trace_id,
                    inner_code,
                    len(results),
                    inner.get("msg", ""),
                )
            return {
                "searchParameters": inner.get("searchParameters") or {"query": query, "topK": num_results},
                "organic": [
                    {
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "publicationInfo": r.get("publicationInfo", ""),
                        "snippet": r.get("snippet", ""),
                        "year": r.get("year", ""),
                        "citedBy": int(r.get("citedBy", 0)) if r.get("citedBy") is not None else 0,
                    }
                    for r in results
                ],
            }

        except requests.exceptions.Timeout as e:
            if _search_debug_enabled():
                logger.warning(
                    "[universal-search-scholar] timeout trace=%s key=%s... err=%s",
                    trace_id,
                    current_key[:8],
                    str(e),
                )
            last_exc = Exception(f"Timeout on key {current_key[:8]}...: {str(e)}")
            continue  # retry with next key
        except requests.exceptions.RequestException as e:
            if _search_debug_enabled():
                logger.warning(
                    "[universal-search-scholar] request_exception trace=%s key=%s... err=%s",
                    trace_id,
                    current_key[:8],
                    str(e),
                )
            last_exc = Exception(f"Error performing Serper scholar search: {str(e)}")
            continue  # retry with next key

    raise last_exc or Exception("All Serper API keys exhausted or failed")


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        WebpageContentResponse containing:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
    """
    if not api_key:
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    scrape_url = "https://scrape.serper.dev"

    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(scrape_url, headers=headers, data=payload, timeout=TIMEOUT)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        data["url"] = url
        return data

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage content: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
