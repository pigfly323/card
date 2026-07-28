"""
API Key Pool Module

Provides load-balanced API key management for distributed LLM requests.
Supports:
- Multiple API keys with RPM (requests per minute) limits
- Weighted random selection based on RPM capacity
- Thread-safe key allocation
- Request tracking and rate limiting
"""

import time
import random
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta


@dataclass
class APIKeyConfig:
    """Configuration for a single API key."""
    api_key: str
    base_url: str
    rpm: int = 50  # Requests per minute
    weight: float = 1.0  # Weight for load balancing (auto-calculated from RPM)

    # Runtime tracking
    request_count: int = field(default=0, init=False)
    last_request_time: float = field(default=0.0, init=False)
    request_timestamps: List[float] = field(default_factory=list, init=False)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    default_base_url: str
    temperature: float = 0.7
    max_context_tokens: int = 32000
    max_response_tokens: int = 8192
    timeout_seconds: int = 180


class APIKeyPool:
    """
    Thread-safe API Key pool with load balancing.

    Features:
    - Weighted random selection based on RPM limits
    - Request rate tracking per key
    - Automatic rate limiting
    - Thread-safe operations
    """

    def __init__(
        self,
        api_keys: List[Dict[str, Any]],
        model_config: ModelConfig,
    ):
        """
        Initialize API Key pool.

        Args:
            api_keys: List of API key configurations, each with:
                - api_key: The API key string
                - base_url: API endpoint URL (optional, uses model default if not provided)
                - rpm: Requests per minute limit (default: 50)
            model_config: Model configuration (model name, default base_url, etc.)
        """
        self.model_config = model_config
        self._lock = threading.Lock()

        # Initialize API key configs
        self.api_keys: List[APIKeyConfig] = []
        for key_data in api_keys:
            config = APIKeyConfig(
                api_key=key_data["api_key"],
                base_url=key_data.get("base_url", model_config.default_base_url),
                rpm=key_data.get("rpm", 50),
            )
            # Calculate weight from RPM
            config.weight = config.rpm
            self.api_keys.append(config)

        if not self.api_keys:
            raise ValueError("At least one API key must be provided")

        # Calculate total capacity
        self.total_rpm = sum(key.rpm for key in self.api_keys)
        self.total_weight = sum(key.weight for key in self.api_keys)

        # Statistics
        self.total_requests = 0
        self.key_usage_stats = defaultdict(int)

    def get_api_key(self) -> APIKeyConfig:
        """
        Get an available API key using weighted random selection.

        Returns:
            APIKeyConfig object with api_key and base_url

        Raises:
            RuntimeError: If all keys are rate-limited
        """
        with self._lock:
            # Filter out rate-limited keys
            available_keys = self._get_available_keys()

            if not available_keys:
                # All keys are rate-limited, wait for the earliest available
                wait_time = self._get_min_wait_time()
                raise RuntimeError(
                    f"All API keys are rate-limited. "
                    f"Please wait {wait_time:.1f} seconds or add more keys."
                )

            # Weighted random selection
            weights = [key.weight for key in available_keys]
            selected_key = random.choices(available_keys, weights=weights, k=1)[0]

            # Record request
            self._record_request(selected_key)

            return selected_key

    def _get_available_keys(self) -> List[APIKeyConfig]:
        """
        Get list of API keys that are not rate-limited.

        Returns:
            List of available APIKeyConfig objects
        """
        current_time = time.time()
        available = []

        for key in self.api_keys:
            # Clean up old timestamps (older than 1 minute)
            key.request_timestamps = [
                ts for ts in key.request_timestamps
                if current_time - ts < 60
            ]

            # Check if key is under RPM limit
            if len(key.request_timestamps) < key.rpm:
                available.append(key)

        return available

    def _record_request(self, key: APIKeyConfig) -> None:
        """
        Record a request for the given API key.

        Args:
            key: The API key config to record request for
        """
        current_time = time.time()
        key.request_count += 1
        key.last_request_time = current_time
        key.request_timestamps.append(current_time)

        self.total_requests += 1
        self.key_usage_stats[key.api_key] += 1

    def _get_min_wait_time(self) -> float:
        """
        Get minimum wait time until any key becomes available.

        Returns:
            Wait time in seconds
        """
        current_time = time.time()
        min_wait = float('inf')

        for key in self.api_keys:
            if key.request_timestamps:
                # Time until oldest request expires (60 seconds window)
                oldest_ts = min(key.request_timestamps)
                wait_time = 60 - (current_time - oldest_ts)
                min_wait = min(min_wait, max(0, wait_time))

        return min_wait if min_wait != float('inf') else 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the pool.

        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_rpm_capacity": self.total_rpm,
                "num_keys": len(self.api_keys),
                "key_usage": dict(self.key_usage_stats),
                "current_load": {
                    key.api_key[:10] + "...": len(key.request_timestamps)
                    for key in self.api_keys
                }
            }

    def get_optimal_max_workers(self, safety_factor: float = 0.8) -> int:
        """
        Calculate optimal max_workers based on total RPM capacity.

        Args:
            safety_factor: Safety factor to avoid hitting rate limits (0.0-1.0)

        Returns:
            Recommended max_workers value
        """
        # Assume average request takes 3-5 seconds
        avg_request_duration = 4.0  # seconds

        # Calculate how many concurrent requests we can sustain
        # RPM / 60 = requests per second
        # requests_per_second * avg_duration = concurrent capacity
        concurrent_capacity = (self.total_rpm / 60) * avg_request_duration

        # Apply safety factor
        optimal_workers = int(concurrent_capacity * safety_factor)

        # Ensure at least 1 worker
        return max(1, optimal_workers)


class APIKeyPoolManager:
    """
    Manager for multiple API key pools (one per model).

    Allows different models to have separate key pools while sharing
    the same keys across models if needed.
    """

    def __init__(self):
        """Initialize the pool manager."""
        self._pools: Dict[str, APIKeyPool] = {}
        self._lock = threading.Lock()

    def create_pool(
        self,
        model_name: str,
        api_keys: List[Dict[str, Any]],
        model_config: ModelConfig,
    ) -> APIKeyPool:
        """
        Create or get an API key pool for a specific model.

        Args:
            model_name: Name of the model
            api_keys: List of API key configurations
            model_config: Model configuration

        Returns:
            APIKeyPool instance
        """
        with self._lock:
            if model_name not in self._pools:
                self._pools[model_name] = APIKeyPool(api_keys, model_config)
            return self._pools[model_name]

    def get_pool(self, model_name: str) -> Optional[APIKeyPool]:
        """
        Get existing pool for a model.

        Args:
            model_name: Name of the model

        Returns:
            APIKeyPool instance or None if not found
        """
        return self._pools.get(model_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all pools.

        Returns:
            Dictionary mapping model names to their pool stats
        """
        return {
            model: pool.get_stats()
            for model, pool in self._pools.items()
        }


# Global pool manager instance
_global_pool_manager = APIKeyPoolManager()


def get_pool_manager() -> APIKeyPoolManager:
    """Get the global API key pool manager."""
    return _global_pool_manager
