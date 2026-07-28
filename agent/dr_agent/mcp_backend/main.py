import argparse
import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Annotated, Dict, List, Optional

import aiohttp

import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .apis.data_model import Crawl4aiApiResult
from .apis.massive_serve_apis import parse_massive_serve_results, search_massive_serve
from .apis.pubmed_apis import search_pubmed
from .apis.reranker_apis import RerankerResult
from .apis.semantic_scholar_apis import (
    SemanticScholarSearchQueryParams,
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_keywords,
    search_semantic_scholar_snippets,
)
from .apis.serper_apis import (
    ScholarResponse,
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
    search_serper_scholar,
)
from .apis.jina_apis import JinaWebpageResponse, fetch_webpage_content_jina
from .apis.sankuai_searchparsing_api import fetch_webpage_content_sankuai_searchparsing
from .apis.draft_answer_feedback_apis import (
    DRAFT_ANSWER_SYSTEM_PROMPT,
    build_draft_answer_user_prompt,
    normalize_feedback_payload,
    call_openai_compatible_chat,
    resolve_rubrics_for_question,
)
from .apis.verify_answer_apis import verify_answer_handler
from .apis.verify_short_answer_apis import verify_short_answer_handler
from .apis.verify_longform_answer_apis import verify_longform_answer_handler
from .cache import set_cache_enabled
from .local.crawl4ai_fetcher import Crawl4AiResult
from .local.search import SearcherType
from .local.search.base import LocalSearchResponse

# Global instance for local search
local_searcher = None
snippet_tokenizer = None

# ---------------------------------------------------------------------------
# verify_answer backend selector
# Set to "short_answer" to use verify_short_answer_apis (JSON critique for
# multi-hop QA tasks).  Default "rubric" uses verify_answer_apis (rubric +
# citation feedback for long-form tasks).
# "long_form" uses verify_longform_answer_apis (reward-aligned long-form
# scoring: rubric + citation + format + search, weights match
# longform_rubric_rewards.py, output parseable by _lff_parse_r0_from_va_output).
# Overridden at startup by --verify-backend CLI argument.
# ---------------------------------------------------------------------------
_VERIFY_BACKEND: str = "rubric"  # "rubric" | "short_answer" | "long_form"

# ---------------------------------------------------------------------------
# Session-level citation store
# Keyed by question text. Populated by search/browse tools when they return
# snippets tagged with <snippets id="..."> in their results.
# verify_answer reads from this store to evaluate citation quality.
# ---------------------------------------------------------------------------
_citation_store: Dict[str, Dict[str, str]] = {}
snippet_max_tokens = 0

# Maximum characters for browse tool text responses (prevents SSE truncation).
# Set MCP_MAX_RESPONSE_CHARS=0 to disable. Default: 50000.
MCP_MAX_RESPONSE_CHARS = int(os.environ.get("MCP_MAX_RESPONSE_CHARS", "50000"))


def _truncate_text(text: str) -> str:
    if MCP_MAX_RESPONSE_CHARS > 0 and len(text) > MCP_MAX_RESPONSE_CHARS:
        return text[:MCP_MAX_RESPONSE_CHARS]
    return text


def _truncate_sankuai_result(result: dict) -> dict:
    """Truncate text fields inside a raw sankuai searchparsing API response."""
    if not isinstance(result, dict):
        return result
    data = result.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "text" in item:
                item["text"] = _truncate_text(item.get("text") or "")
    return result

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

mcp = FastMCP(
    "RL-RAG MCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse,rerank").split(","),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")


@mcp.tool(tags={"search", "necessary"})
def semantic_scholar_search(
    query: Annotated[str, "Search query string"],
    year: Annotated[
        Optional[str], "Year range filter (e.g., '2015-2020', '2015-', '-2015')"
    ] = None,
    min_citation_count: Annotated[Optional[int], "Minimum number of citations"] = None,
    sort: Annotated[
        Optional[str], "Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Maximum number of results to return (max: 100)"] = 25,
) -> dict:
    """
    Search for academic papers using Semantic Scholar API.

    Returns:
        Dictionary containing search results
    """
    query_params = SemanticScholarSearchQueryParams(
        query=query,
        year=year,
        minCitationCount=min_citation_count,
        sort=sort,
        venue=venue,
    )

    results = search_semantic_scholar_keywords(
        query_params=query_params,
        limit=min(limit, 100),  # Ensure limit doesn't exceed API maximum
    )

    return results


@mcp.tool(tags={"search"})
async def semantic_scholar_snippet_search(
    query: Annotated[str, "Search query string to find academic papers"],
    year: Annotated[
        Optional[str],
        "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
    ] = None,
    limit: Annotated[int, "Number of snippets to retrieve"] = 10,
    fieldsOfStudy: Annotated[
        Optional[str],
        "Comma-separated fields of study to filter results (e.g., 'Computer Science', 'Medicine', 'Computer Science, Biology')",
    ] = None,
) -> dict:
    """
    Search academic papers using Semantic Scholar (intranet Bing search).

    Purpose: Retrieve abstracts and metadata from scientific literature.
    Returns focused snippets from papers with title, abstract and source URL.

    Example:
        query="large language model retrieval evaluation", year="2023-2025", limit=8, fieldsOfStudy="Computer Science"
    """
    from .apis.semantic_scholar_apis import (
        search_semantic_scholar_snippets,
        SemanticScholarSnippetSearchQueryParams,
    )

    # Append fieldsOfStudy to query as a search hint since the backend
    # does not support native field filtering — it improves result relevance.
    effective_query = query
    if fieldsOfStudy:
        fields = [f.strip() for f in fieldsOfStudy.split(",") if f.strip()]
        if fields:
            effective_query = f"{query} {' '.join(fields)}"

    query_params = SemanticScholarSnippetSearchQueryParams(query=effective_query, year=year)
    _timeout = int(os.environ.get("SNIPPET_SEARCH_TIMEOUT", "20"))
    results = await asyncio.to_thread(
        search_semantic_scholar_snippets,
        query_params,
        limit=limit,
        timeout=_timeout,
    )
    return results


@mcp.tool(tags={"search"})
def pubmed_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search for medical and scientific papers using PubMed API.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        offset: Starting position for pagination (default: 0)

    Returns:
        Dictionary containing search results with the following fields:
        - total: Total number of results
        - offset: Current offset
        - next: Next offset for pagination
        - data: List of paper details including:
            - paperId: PubMed ID
            - title: Paper title
            - authors: List of authors
            - abstract: Paper abstract
            - year: Publication year
            - venue: Journal name
            - url: Link to PubMed page
            - citationCount: Number of citations (if available from Semantic Scholar)
    """
    results = search_pubmed(
        keywords=query,
        limit=limit,
        offset=offset,
    )

    return results


@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """
    Rerank a list of documents based on their relevance to the query using VLLM hosted reranker.

    Args:
        query: Search query string
        documents: List of document texts to rank
        top_n: Number of top documents to return
        model_name: Name of the reranker model (default: "BAAI/bge-reranker-v2-m3")
        api_url: Base URL for the VLLM reranker API (default: "http://localhost:30002")

    Returns:
        RerankerResult containing reranker results with method, model_name, and ranked results
    """
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_reranker

    results = vllm_hosted_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
    )

    return results


@mcp.tool(tags={"search", "necessary"})
async def draft_answer(
    draft_answer: Annotated[str, "Draft final answer to critique before producing the real final answer"],
    question: Annotated[Optional[str], "Original user question to evaluate against"] = None,
    rubrics: Annotated[
        Optional[str],
        "Optional rubric text; if empty, server will lookup instance-wise rubrics by question",
    ] = None,
) -> dict:
    """
    Critique a draft answer and return issue-focused rubric feedback.

    Purpose:
    - Help the agent self-correct before emitting the final answer.
    - Surface rubric misses, weak grounding, citation risks, and reasoning drift.

    Important:
    - This is a planning/feedback tool. Its internal feedback should be used to improve subsequent actions.
    - The final user-facing answer should not expose tool-internal instructions verbatim.
    """
    request_t0 = time.perf_counter()
    logger.info(
        "draft_answer request: has_question=%s has_rubrics_param=%s draft_chars=%d",
        bool(question and str(question).strip()),
        bool(rubrics and str(rubrics).strip()),
        len(draft_answer or ""),
    )

    if not draft_answer or not draft_answer.strip():
        logger.warning("draft_answer rejected: empty draft_answer")
        return {
            "feedback": {
                "global_issues": [
                    {
                        "issue": "Empty draft_answer.",
                        "why_it_matters": "No content can be evaluated against rubrics.",
                        "evidence": "",
                    }
                ],
                "rubric_issues": [],
                "strengths": [],
            }
        }

    # Hard constraint mode:
    # If question is still missing after any upstream fallback extraction,
    # reject this call and force a retry with explicit question to avoid rubric mismatch.
    if question is None or not str(question).strip():
        logger.warning("draft_answer rejected: missing question")
        return {
            "feedback": {
                "global_issues": [
                    {
                        "issue": "Missing required parameter: question.",
                        "why_it_matters": "Cannot reliably retrieve instance-wise rubrics; may cause rubric mismatch.",
                        "evidence": "draft_answer called without a usable question parameter.",
                    }
                ],
                "rubric_issues": [],
                "strengths": [],
            },
            "tool": "draft_answer",
            "retry_required": True,
            "retry_reason": "missing_question",
            "retry_hint": "Retry draft_answer and include question='original user question'.",
        }

    try:
        resolved_rubrics = resolve_rubrics_for_question(question=question, rubrics=rubrics)
        static_cnt = len(resolved_rubrics.get("static_rubrics", []) or [])
        positive_cnt = len(resolved_rubrics.get("positive_rubrics", []) or [])
        negative_cnt = len(resolved_rubrics.get("negative_rubrics", []) or [])
        rubric_source = resolved_rubrics.get("source", "unknown")
        logger.info(
            "draft_answer rubrics: source=%s static=%d positive=%d negative=%d",
            rubric_source,
            static_cnt,
            positive_cnt,
            negative_cnt,
        )
        user_prompt = build_draft_answer_user_prompt(
            draft_answer=draft_answer,
            question=question,
            rubrics=resolved_rubrics,
        )
        logger.debug(
            "draft_answer prompt stats: system_chars=%d user_chars=%d",
            len(DRAFT_ANSWER_SYSTEM_PROMPT),
            len(user_prompt),
        )
        model_t0 = time.perf_counter()
        raw_feedback = await call_openai_compatible_chat(
            system_prompt=DRAFT_ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        logger.info(
            "draft_answer model response: elapsed_ms=%.1f raw_chars=%d",
            (time.perf_counter() - model_t0) * 1000.0,
            len(raw_feedback or ""),
        )
        feedback = normalize_feedback_payload(raw_feedback)
        feedback["tool"] = "draft_answer"
        feedback["rubric_source"] = rubric_source
        parsed_global_issues = feedback.get("feedback", {}).get("global_issues", [])
        parsed_rubric_issues = feedback.get("feedback", {}).get("rubric_issues", [])
        logger.info(
            "draft_answer parsed feedback: global_issues=%d rubric_issues=%d total_elapsed_ms=%.1f",
            len(parsed_global_issues) if isinstance(parsed_global_issues, list) else -1,
            len(parsed_rubric_issues) if isinstance(parsed_rubric_issues, list) else -1,
            (time.perf_counter() - request_t0) * 1000.0,
        )
        return feedback
    except Exception as e:
        logger.exception("draft_answer failed: %s", str(e))
        return {
            "feedback": {
                "global_issues": [
                    {
                        "issue": f"draft_answer tool failed: {str(e)}",
                        "why_it_matters": "Cannot obtain rubric-guided feedback for this step.",
                        "evidence": "",
                    }
                ],
                "rubric_issues": [],
                "strengths": [],
            }
        }


@mcp.tool(tags={"search", "necessary"})
async def verify_answer(
    answer: Annotated[str, "Draft answer to verify (rubric mode) or predicted answer (short_answer mode)"],
    question: Annotated[Optional[str], "Original user question / problem"] = None,
    # --- rubric mode parameters (ignored in short_answer mode) ---
    rubrics: Annotated[
        Optional[str],
        "[rubric mode] Optional rubric text; if empty, server will look up instance-wise rubrics by question",
    ] = None,
    context: Annotated[
        Optional[str],
        "[rubric mode] Optional context containing <snippets id='...'> tags from prior search/browse tool calls, used for citation verification",
    ] = None,
    apply_adaptive_rubric_reward: Annotated[
        bool,
        "[rubric mode] If true, generate instance-wise adaptive rubrics from cached draft answers",
    ] = False,
    disable_rubrics_feedback: Annotated[
        bool,
        "[rubric mode] If true, skip rubric judge LLM calls and return rubric criteria directly",
    ] = False,
    # --- short_answer mode parameters (ignored in rubric mode) ---
    ground_truth: Annotated[
        Optional[str],
        "[short_answer mode] Reference answer used by the verify agent as oracle",
    ] = None,
    solution_path: Annotated[
        Optional[str],
        "[short_answer mode] Full reasoning trace (CoT + tool calls + tool outputs) to audit",
    ] = None,
) -> str:
    """
    Verify a draft answer and return feedback.

    Behaviour depends on the server's --verify-backend setting:

    rubric mode (default):
      Returns plain-text rubric and citation feedback.
      Required: answer, question
      Optional: rubrics, context, apply_adaptive_rubric_reward, disable_rubrics_feedback

    short_answer mode:
      Returns a JSON critique identifying errors in the solution path and
      suggesting corrective search actions. Designed for multi-hop QA tasks.
      Required: answer (predicted answer), question (problem),
                ground_truth (reference answer), solution_path (full trace)

    Important:
    - Do NOT repeat verify_answer calls after receiving feedback.
    - Use the feedback to improve and rewrite your final answer.
    """
    request_t0 = time.perf_counter()
    logger.info(
        "verify_answer request: backend=%s has_question=%s answer_chars=%d",
        _VERIFY_BACKEND,
        bool(question and str(question).strip()),
        len(answer or ""),
    )

    if _VERIFY_BACKEND == "short_answer":
        # verify_short_answer_handler 直接返回格式化字符串（含 Answer/Format/Search/Rubric reward 行），
        # 与 _verify_answer_native_short 输出格式一致，_lff_parse_r0_from_va_output 可直接解析。
        # short_answer 模式只用 context（snippets）做事实核验，不传 solution_path。
        result_str = await verify_short_answer_handler(
            problem=question or "",
            predicted_answer=answer,
            ground_truth=ground_truth or "",
            context=context or "",
        )
        print(f"result_str={result_str[:200]}...")  # debug
    elif _VERIFY_BACKEND == "long_form":
        # verify_longform_answer_handler 返回纯文本 feedback（含 Rubric reward / Claim N / Citation reward
        # 等可解析行），分数范围与权重对齐 longform_rubric_rewards.py，
        # _lff_parse_r0_from_va_output 可直接解析 r0。
        # long_form 模式用 context（snippets）做 citation 核验，solution_path 做 search-turn 计数，
        # ground_truth 提供 rubrics。
        result_str = await verify_longform_answer_handler(
            question=question or "",
            answer=answer,
            context=context or "",
            solution_path=solution_path or "",
            ground_truth=ground_truth or "",
        )
    else:
        from .apis.verify_answer_apis import _extract_citations_from_context_for_verify
        citation_store: Dict[str, str] = {}
        if context and str(context).strip():
            citation_store = _extract_citations_from_context_for_verify(context)

        _disable_rubrics = disable_rubrics_feedback or os.environ.get(
            "DISABLE_RUBRICS_FEEDBACK", ""
        ).lower() in ("1", "true", "yes")

        result_str = await verify_answer_handler(
            answer=answer,
            question=question,
            rubrics=rubrics,
            citation_store=citation_store,
            apply_adaptive_rubric_reward=apply_adaptive_rubric_reward,
            disable_rubrics_feedback=_disable_rubrics,
        )

    logger.info(
        "verify_answer completed: backend=%s elapsed_ms=%.1f result_preview=%s",
        _VERIFY_BACKEND,
        (time.perf_counter() - request_t0) * 1000.0,
        str(result_str)[:200],
    )
    return result_str


@mcp.tool(tags={"search", "necessary"})
def massive_serve_search(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: Optional[str] = None,
    nprobe: Optional[int] = None,
) -> dict:
    """
    Search for documents using massive-serve API for dense passage retrieval.

    This tool provides access to large-scale document collections using dense passage
    retrieval with various embedding models and indices.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10)
        domains: Domain/index to search in (default: "dpr_wiki_contriever_ivfpq")
        base_url: Base URL for the massive-serve API (optional, uses default if not provided)
        nprobe: Number of probes for search (optional, uses API default)

    Returns:
        Dictionary containing search results with the following fields:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - results: Dictionary with IDs, passages, and scores
        - data: Parsed list of search results with passage text, scores, and doc IDs
    """
    # Call the massive-serve API
    response = search_massive_serve(
        query=query,
        n_docs=n_docs,
        domains=domains,
        base_url=base_url,
        nprobe=nprobe,
    )

    # Parse the results for easier consumption
    parsed_results = parse_massive_serve_results(response)

    # Add parsed data to the response for convenience
    response["data"] = [
        {
            "passage": result.passage,
            "score": result.score,
            "doc_id": result.doc_id,
        }
        for result in parsed_results
    ]

    return response


@mcp.tool(tags={"search", "necessary"})
async def serper_google_webpage_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
    gl: Annotated[
        str,
        "Geolocation - country code to boost search results whose country of origin matches the parameter value",
    ] = "us",
    hl: Annotated[str, "Host language of user interface"] = "en",
):
    """
    General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources.

    Returns:
        Dictionary containing web search snippets with the following fields:
        - organic: List of organic search results with title, link, and snippet
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    try:
        results = await asyncio.to_thread(
            search_serper,
            query=query, num_results=num_results, search_type="search", gl=gl, hl=hl,
        )
    except Exception as e:
        return {"error": str(e), "organic": []}
    return results


@mcp.tool(tags={"browse", "necessary"})
async def serper_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    include_markdown: Annotated[
        bool, "Whether to include markdown formatting in the response"
    ] = True,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Returns:
        Dictionary containing the webpage content with the following fields:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - url: The original URL that was fetched
        - success: Boolean indicating if the fetch was successful
    """
    try:
        result = await asyncio.to_thread(
            fetch_webpage_content,
            url=webpage_url,
            include_markdown=include_markdown,
        )

        return {
            **result,
            "success": True,
        }
    except Exception as e:
        return {
            "text": "",
            "markdown": "",
            "metadata": {},
            "url": webpage_url,
            "success": False,
            "error": str(e),
        }


@mcp.tool(tags={"browse"})
def jina_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    timeout: Annotated[int, "Request timeout in seconds"] = 30,
) -> JinaWebpageResponse:
    """
    Fetch the content of a webpage using Jina Reader API with timeout support.

    Args:
        webpage_url: The URL of the webpage to fetch
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - url: The original URL that was fetched
        - title: Page title
        - content: The webpage content as clean text/markdown
        - description: Page description (if available)
        - publishedTime: Published time (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    result = fetch_webpage_content_jina(url=webpage_url, timeout=timeout)
    return result


@mcp.tool(tags={"search", "necessary"})
def serper_google_scholar_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> ScholarResponse:
    """
    Search for academic papers using google scholar (based on Serper.dev API).

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
    """
    results = search_serper_scholar(
        query=query,
        num_results=num_results,
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_fetch_webpage_content(
    url: Annotated[str, "URL to fetch and extract content from"],
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> dict:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4AiResult with extracted webpage content including markdown-formatted text
    """

    from dr_agent.mcp_backend.apis.sankuai_searchparsing_api import fetch_webpage_content_sankuai_searchparsing

    # `timeout_ms` is the tool-level per-page timeout.
    # The underlying Sankuai searchparsing API expects `timeout` in seconds.
    # Cap at 20s so aiohttp (timeout+5=25s) finishes before MCP client timeout (35s).
    timeout_seconds = min(20, max(1, int(timeout_ms) // 1000))

    try:
        result = await fetch_webpage_content_sankuai_searchparsing(
            urls=[url],
            timeout=timeout_seconds,
        )
    except Exception as e:
        return {"error": str(e), "success": False, "data": []}
    return _truncate_sankuai_result(result)


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_docker_fetch_webpage_content(
    url: Annotated[str, "Target URL to crawl and extract content from"],
    base_url: Annotated[
        Optional[str],
        "Base URL for the Crawl4AI Docker API (e.g., 'http://localhost:8000')",
    ] = None,
    api_key: Annotated[Optional[str], "API key for authentication"] = None,
    use_ai2_config: Annotated[
        bool,
        "If True, use AI2 bot configuration with blocklist (requires CRAWL4AI_BLOCKLIST_PATH env var)",
    ] = False,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> dict:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI Docker API.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4aiApiResult with url, success, markdown-formatted text, and optional fit_markdown/html/error fields
    """
    from dr_agent.mcp_backend.apis.sankuai_searchparsing_api import fetch_webpage_content_sankuai_searchparsing

    # `timeout_ms` is the tool-level per-page timeout.
    # The underlying Sankuai searchparsing API expects `timeout` in seconds.
    timeout_seconds = min(20, max(1, int(timeout_ms) // 1000))

    try:
        result = await fetch_webpage_content_sankuai_searchparsing(
            urls=[url],
            timeout=timeout_seconds,
        )
    except Exception as e:
        return {"error": str(e), "success": False, "data": []}
    return _truncate_sankuai_result(result)


@mcp.tool(tags={"browse", "necessary"})
async def sankuai_searchparsing_fetch_webpage_content(
    url: Annotated[str, "Target URL to crawl and extract content from"],
    user_id: Annotated[
        Optional[str],
        "User identifier for analytics (optional). If not provided, uses env SANKUAI_USER_ID if set.",
    ] = None,
    parser_method: Annotated[
        str, "Parsing method: online (default) | tra"
    ] = "online",
    timeout: Annotated[int, "Request timeout in seconds (default: 20)"] = 20,
    cache: Annotated[bool, "Whether to enable remote caching"] = False,
    include_tables: Annotated[bool, "Whether to parse tables"] = False,
    include_images: Annotated[bool, "Whether to parse images"] = False,
    include_formatting: Annotated[
        bool, "Whether to include markdown formatting"
    ] = True,
    include_links: Annotated[bool, "Whether to include links"] = False,
    expiration_time: Annotated[
        Optional[int],
        "Cache expiration time in seconds (effective only if cache=True). None means no expiration.",
    ] = None,
    endpoint_url: Annotated[
        Optional[str],
        "Override API endpoint URL (default from env SANKUAI_CRAWL_URL).",
    ] = None,
) -> dict:
    """
    Fetch webpage content using Meituan (Sankuai) searchparsing API.

    Normalizes the response to a simple structure for downstream browse tools:
    - success: bool
    - url: str
    - title: str
    - text: str
    - error: str (optional)
    - raw: original JSON response
    """
    try:
        effective_user_id = user_id or os.getenv("SANKUAI_USER_ID")
        raw = await fetch_webpage_content_sankuai_searchparsing(
            urls=[url],
            user_id=effective_user_id,
            parser_method=parser_method,
            timeout=timeout,
            cache=cache,
            include_tables=include_tables,
            include_images=include_images,
            include_formatting=include_formatting,
            include_links=include_links,
            expiration_time=expiration_time,
            endpoint_url=endpoint_url,
        )

        data = raw.get("data") if isinstance(raw, dict) else None
        item = data[0] if isinstance(data, list) and data else {}

        crawl_status = item.get("crawlStatus")
        ok = crawl_status == 0 and bool(item.get("text"))
        title = (item.get("title") or "").strip()
        text = item.get("text") or ""

        if ok:
            return {
                "success": True,
                "url": item.get("url") or url,
                "title": title,
                "text": _truncate_text(text),
                "raw": raw,
            }

        msg = item.get("message") or raw.get("msg") if isinstance(raw, dict) else None
        error = msg or f"crawlStatus={crawl_status}"
        return {
            "success": False,
            "url": item.get("url") or url,
            "title": title,
            "text": _truncate_text(text or ""),
            "error": error,
            "raw": raw,
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "title": "",
            "text": "",
            "error": str(e),
        }


@mcp.tool(tags={"browse"})
def webthinker_fetch_webpage_content(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import extract_text_from_url

    text = extract_text_from_url(
        url=url,
        snippet=snippet,
        keep_links=keep_links,
    )

    return {"url": url, "text": text}


@mcp.tool(tags={"browse"})
async def webthinker_fetch_webpage_content_async(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import (
        extract_text_from_url_async,
    )

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=240, connect=10)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        text = await extract_text_from_url_async(
            url=url,
            session=session,
            snippet=snippet,
            keep_links=keep_links,
        )

    return {"url": url, "text": text}


@mcp.tool(tags={"search", "local"})
def local_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> LocalSearchResponse:
    """
    Perform a search on a local knowledge source.
    Useful for retrieving relevant passages from specific datasets or local indices.
    """
    if local_searcher is None:
        return {
            "results": [],
            "error": "Local searcher not initialized."
        }
    
    try:
        response = local_searcher.search(query, k=num_results)
        
        if snippet_max_tokens > 0 and snippet_tokenizer:
            for cand in response.get("results", []):
                snippet_text = cand["snippet"]
                tokens = snippet_tokenizer.encode(snippet_text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    cand["snippet"] = snippet_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return response
    except Exception as e:
        logger.error(f"Error during local search: {e}")
        return {"results": [], "error": str(e)}


@mcp.tool(tags={"browse", "local"})
def local_browse(
    url: Annotated[str, "URL to fetch and extract content from"],
) -> dict:
    """
    Retrieve full text content for a given URL from the local knowledge source.
    Useful for reading the full content of documents found via local_search.
    """
    if local_searcher is None:
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": "Local index not initialized."
        }
    
    try:
        text = local_searcher.get_text_by_url(url)
        if text is None:
            return {
                "url": url,
                "success": False,
                "markdown": "",
                "error": f"URL not found: {url}"
            }
            
        return {
            "url": url,
            "success": True,
            "markdown": text
        }
    except Exception as e:
        logger.error(f"Error during local browse: {e}")
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Run in stateless HTTP mode (no session initialization handshake). "
             "Recommended when called from scripts that create a new client per request.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1). "
             "Set >1 to handle concurrent requests in parallel. "
             "Only effective for HTTP transports.",
    )
    parser.add_argument(
        "--local-searcher-type",
        type=str,
        default=None,
        choices=SearcherType.get_choices(),
        help="Type of local searcher to use (default: None, no local search)",
    )
    parser.add_argument(
        "--local-search-max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens for local search snippets (default: 100, set to 0 to disable truncation)",
    )
    parser.add_argument(
        "--verify-backend",
        type=str,
        default="rubric",
        choices=["rubric", "short_answer", "long_form"],
        help=(
            "Backend for the verify_answer tool. "
            "'rubric' (default): rubric + citation feedback for long-form tasks. "
            "'short_answer': JSON critique of solution path for multi-hop QA tasks. "
            "'long_form': reward-aligned long-form scoring (rubric+citation+format+search, "
            "weights match longform_rubric_rewards.py, output parseable by _lff_parse_r0_from_va_output)."
        ),
    )

    # We need a first pass to get the searcher type to add its specific args
    temp_args, _ = parser.parse_known_args()
    if temp_args.local_searcher_type:
        searcher_cls = SearcherType.get_searcher_class(temp_args.local_searcher_type)
        searcher_cls.parse_args(parser)

    args = parser.parse_args()

    # Apply verify_answer backend selection (module-level, no global needed)
    _VERIFY_BACKEND = args.verify_backend
    logger.info("verify_answer backend: %s", _VERIFY_BACKEND)

    # Initialize local searcher if specified
    if args.local_searcher_type:
        try:
            searcher_cls = SearcherType.get_searcher_class(args.local_searcher_type)
            local_searcher = searcher_cls(args)
            logger.info(f"Initialized local searcher: {args.local_searcher_type}")
            
            snippet_max_tokens = args.local_search_max_tokens
            if snippet_max_tokens > 0:
                # Reuse the encoder's tokenizer (already loaded locally) instead of
                # fetching a separate model from HuggingFace Hub.
                if hasattr(local_searcher, "tokenizer") and local_searcher.tokenizer is not None:
                    snippet_tokenizer = local_searcher.tokenizer
                    logger.info(
                        "Using encoder tokenizer for snippet truncation (max %d tokens).",
                        snippet_max_tokens,
                    )
                else:
                    logger.warning(
                        "Searcher has no tokenizer; snippet truncation disabled."
                    )
                    snippet_max_tokens = 0
        except Exception as e:
            logger.error(f"Failed to initialize local searcher: {e}")

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
            stateless_http=args.stateless,
        )
