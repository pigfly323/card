import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
import dotenv

from ..cache import cached

dotenv.load_dotenv()


DEFAULT_SANKUAI_CRAWL_URL = os.getenv(
    "SANKUAI_CRAWL_URL", "https://model.sankuai.com/crawl-webpage/searchparsing"
)


@cached()
async def fetch_webpage_content_sankuai_searchparsing(
    *,
    urls: List[str],
    user_id: Optional[str] = None,
    parser_method: str = "online",
    timeout: int = 20,
    cache: bool = False,
    include_tables: bool = False,
    include_images: bool = False,
    include_formatting: bool = True,
    include_links: bool = False,
    expiration_time: Optional[int] = None,
    endpoint_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Meituan (Sankuai) searchparsing API to crawl webpages.

    API: https://model.sankuai.com/crawl-webpage/searchparsing

    Returns the raw JSON response as a dict.
    """
    if not urls:
        raise ValueError("urls must be a non-empty list")

    endpoint = (endpoint_url or DEFAULT_SANKUAI_CRAWL_URL).strip()
    if not endpoint:
        raise ValueError("Searchparsing endpoint URL is empty")

    payload: Dict[str, Any] = {
        "urls": urls,
        "parser_method": parser_method or "online",
        "timeout": int(timeout) if timeout is not None else 20,
        "cache": bool(cache),
        "include_tables": bool(include_tables),
        "include_images": bool(include_images),
        "include_formatting": bool(include_formatting),
        "include_links": bool(include_links),
    }
    if user_id:
        payload["user_id"] = user_id
    if expiration_time is not None:
        payload["expiration_time"] = int(expiration_time)

    # Use both connect and total timeout to prevent hanging when the server
    # is unreachable (connect timeout fires even before data transfer starts).
    request_timeout = aiohttp.ClientTimeout(
        total=payload["timeout"] + 5,
        connect=10,
    )
    try:
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.post(endpoint, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"Searchparsing API request failed: status={resp.status}, body={text}"
                    )
                try:
                    return await resp.json()
                except Exception as e:
                    raise RuntimeError(
                        f"Searchparsing API returned non-JSON response: {text}"
                    ) from e
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
        raise RuntimeError(
            f"Searchparsing API timed out after {payload['timeout']} seconds"
        ) from e

