import asyncio
import functools
import hashlib
import inspect
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from diskcache import Cache

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global cache control
_cache_enabled = True


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable caching globally."""
    global _cache_enabled
    _cache_enabled = enabled


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _cache_enabled


class ApiCache:
    """Cache for API responses using diskcache."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = None,
        db_file: str = "api_cache.json",  # Kept for compatibility, not used
        cache_ttl: int = 86400,  # Default: 1 day in seconds
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store the cache file. If None, uses MCP_CACHE_DIR env var or defaults to ".cache"
            db_file: Kept for compatibility, not used with diskcache
            cache_ttl: Time-to-live for cache entries in seconds
        """
        if cache_dir is None:
            cache_dir = os.getenv("MCP_CACHE_DIR", ".cache")

        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl

        # Initialize diskcache - handles threading/locking automatically
        self.cache = self._init_cache_with_recovery()

    def _init_cache_with_recovery(self) -> Cache:
        """Initialize cache and recover automatically from malformed sqlite db."""
        cache_path = self.cache_dir / "diskcache"
        try:
            return Cache(str(cache_path))
        except Exception as e:
            if "database disk image is malformed" not in str(e).lower():
                raise
            logger.error(f"Detected malformed cache database at {cache_path}: {e}")
            self._backup_and_reset_cache_dir(cache_path)
            return Cache(str(cache_path))

    def _backup_and_reset_cache_dir(self, cache_path: Path) -> None:
        """Move corrupted cache directory away and create a clean one."""
        if cache_path.exists():
            ts = int(time.time())
            backup_path = cache_path.parent / f"{cache_path.name}.corrupted.{ts}"
            try:
                shutil.move(str(cache_path), str(backup_path))
                logger.warning(f"Moved corrupted cache to {backup_path}")
            except Exception as move_err:
                logger.error(f"Failed to move corrupted cache dir: {move_err}")
                # Fall back to best-effort cleanup
                shutil.rmtree(cache_path, ignore_errors=True)
        cache_path.mkdir(parents=True, exist_ok=True)

    def _is_malformed_db_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        return (
            "database disk image is malformed" in msg
            or isinstance(e, sqlite3.DatabaseError)
        )

    def _recover_cache_if_needed(self, e: Exception) -> None:
        if self._is_malformed_db_error(e):
            logger.error(f"Cache database malformed, rebuilding cache: {e}")
            self._backup_and_reset_cache_dir(self.cache_dir / "diskcache")
            self.cache = Cache(str(self.cache_dir / "diskcache"))

    def _get_cache_key(
        self, func: Callable, args: tuple, kwargs: Dict[str, Any]
    ) -> str:
        """Generate a unique cache key based on function signature and normalized arguments."""
        if kwargs and "timeout" in kwargs:
            kwargs = kwargs.copy()
            kwargs.pop("timeout")

        # Use function signature to normalize arguments
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Apply default values for missing arguments
            normalized_kwargs = dict(bound_args.arguments)
        except (TypeError, ValueError):
            # Fallback to original approach if signature binding fails
            normalized_kwargs = {}
            for i, arg in enumerate(args):
                normalized_kwargs[f"__pos_arg_{i}__"] = arg
            normalized_kwargs.update(kwargs)

        # Process the normalized arguments
        processed_kwargs = {}
        for key, value in normalized_kwargs.items():
            if hasattr(value, "model_dump"):  # Handle Pydantic models
                processed_kwargs[key] = value.model_dump()
            elif hasattr(value, "__dict__"):  # Handle other custom objects
                processed_kwargs[key] = str(value.__dict__)
            else:
                processed_kwargs[key] = value

        try:
            kwargs_str = json.dumps(processed_kwargs, sort_keys=True)
        except TypeError:
            kwargs_str = str(sorted(processed_kwargs.items()))

        cache_key = hashlib.md5(f"{func.__name__}:{kwargs_str}".encode()).hexdigest()
        return cache_key

    def get(self, cache_key: str, cache_ttl: int = None) -> Optional[Dict[str, Any]]:
        """Get a value from the cache if it exists and is not expired."""
        try:
            result = self.cache.get(cache_key)
        except Exception as e:
            self._recover_cache_if_needed(e)
            return None
        if result is not None:
            logger.info(f"Cache hit for key: {cache_key}")
        return result

    def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store a value in the cache with the current timestamp."""
        try:
            self.cache.set(cache_key, data, expire=self.cache_ttl)
            logger.info(f"Cache set for key: {cache_key}")
        except Exception as e:
            self._recover_cache_if_needed(e)
            logger.error(f"Failed to set cache for key {cache_key}: {e}")
            raise

    def clear_expired(self) -> int:
        """Clear expired cache entries and return the count of removed entries."""
        # diskcache handles expiration automatically
        return self.cache.expire()

    def clear_all(self) -> int:
        """Clear all cache entries and return the count of removed entries."""
        try:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {count} cache entries")
            return count
        except Exception as e:
            logger.error(f"Failed to clear all cache entries: {e}")
            return 0


# Create a global cache instance
DEFAULT_CACHE = ApiCache(
    cache_dir=os.getenv(
        "MCP_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".cache")
    ),
    cache_ttl=86400 * 10,  # Default: 10 days cache
)


def _is_empty_result(result: Any) -> bool:
    """
    Return True if the result looks like an empty/failed API response that
    should NOT be cached (so the next request retries the real API).

    Covers the common patterns returned by search/browse tools:
      {"data": []}
      {"results": []}
      {"organic": []}
      {"success": False, ...}
      {}  (empty dict)
      []  (empty list)
    """
    if result is None:
        return True
    if isinstance(result, list):
        return len(result) == 0
    if isinstance(result, dict):
        if not result:
            return True
        # Explicit failure flag
        if result.get("success") is False:
            return True
        # Common empty-data patterns
        for key in ("data", "results", "organic", "items", "hits"):
            if key in result:
                val = result[key]
                if isinstance(val, list) and len(val) == 0:
                    return True
    return False


def cached(
    cache: Optional[ApiCache] = None,
    ttl: Optional[int] = None,
) -> Callable:
    """
    Decorator to cache function results for both sync and async functions.

    Empty or failed results (e.g. {"data": []}, {"success": False}) are NOT
    cached so that the next call retries the real API instead of serving a
    stale empty response.

    Args:
        cache: ApiCache instance
        ttl: Override the default cache TTL for this function

    Returns:
        Decorated function with caching capability
    """
    if cache is None:
        cache = DEFAULT_CACHE

    def decorator(func: Callable) -> Callable:
        # If caching is disabled, return the original function
        if not is_cache_enabled():
            return func

        if asyncio.iscoroutinefunction(func):
            # Async function wrapper
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = cache._get_cache_key(func, args, kwargs)

                # Try to get from cache
                cached_result = cache.get(cache_key, cache_ttl=ttl)
                if cached_result is not None:
                    return cached_result

                # Call the async function and await the result
                result = await func(*args, **kwargs)
                # Do not cache empty/failed results — let the next call retry
                if _is_empty_result(result):
                    logger.info(f"Skipping cache for empty result: {func.__name__}")
                    return result
                try:
                    if ttl:
                        cache.cache.set(cache_key, result, expire=ttl)
                    else:
                        cache.set(cache_key, result)
                except Exception as e:
                    logger.error(f"Failed to cache result: {e}")
                return result

            return async_wrapper
        else:
            # Sync function wrapper
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = cache._get_cache_key(func, args, kwargs)

                # Try to get from cache
                cached_result = cache.get(cache_key, cache_ttl=ttl)
                if cached_result is not None:
                    return cached_result

                # Call the function and cache the result
                result = func(*args, **kwargs)
                # Do not cache empty/failed results — let the next call retry
                if _is_empty_result(result):
                    logger.info(f"Skipping cache for empty result: {func.__name__}")
                    return result
                try:
                    if ttl:
                        cache.cache.set(cache_key, result, expire=ttl)
                    else:
                        cache.set(cache_key, result)
                except Exception as e:
                    logger.error(f"Failed to cache result: {e}")
                return result

            return sync_wrapper

    return decorator
