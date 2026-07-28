"""
Async Executor Module

Provides utilities for parallel execution of API calls with concurrency control.
Supports both synchronous and asynchronous execution patterns.
"""

import asyncio
import time
from typing import Any, Callable, List, Optional, TypeVar, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

T = TypeVar('T')


@dataclass
class ExecutorConfig:
    """Configuration for async executor."""
    max_concurrent: int = 20
    max_retries: int = 10
    retry_delay: float = 10.0  # seconds


class AsyncExecutor:
    """
    Executor for parallel API calls with concurrency control.
    
    Supports:
    - Thread-based parallel execution for sync functions
    - Async/await based execution for async functions
    - Automatic retry with exponential backoff
    - Progress tracking
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        """
        Initialize the executor.
        
        Args:
            config: Executor configuration (uses defaults if None)
        """
        self.config = config or ExecutorConfig()
    
    def execute_parallel(
        self,
        func: Callable[..., T],
        items: List[Any],
        desc: str = "Processing",
        show_progress: bool = True,
    ) -> List[T]:
        """
        Execute a function in parallel over a list of items using threads.
        
        Args:
            func: Function to execute (should accept one item as argument)
            items: List of items to process
            desc: Description for progress display
            show_progress: Whether to show progress bar
            
        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []
        
        results: List[Any] = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._execute_with_retry, func, item): idx
                for idx, item in enumerate(items)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Task {idx} failed after retries: {e}")
                
                completed += 1
                if show_progress:
                    print(f"\r{desc}: {completed}/{len(items)}", end="", flush=True)
            
            if show_progress:
                print()  # New line after progress
        
        return results
    
    def _execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = min(
                        self.config.retry_delay * (2 ** attempt),
                        60.0  # Max 60 seconds
                    )
                    time.sleep(wait_time)
        
        raise last_error or RuntimeError("All retries failed")
    
    async def execute_async_parallel(
        self,
        async_func: Callable[..., Coroutine[Any, Any, T]],
        items: List[Any],
        desc: str = "Processing",
        show_progress: bool = True,
    ) -> List[T]:
        """
        Execute an async function in parallel over a list of items.
        
        Args:
            async_func: Async function to execute
            items: List of items to process
            desc: Description for progress display
            show_progress: Whether to show progress bar
            
        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def bounded_task(item: Any, idx: int) -> tuple[int, T]:
            """Execute task with semaphore."""
            async with semaphore:
                result = await self._execute_async_with_retry(async_func, item)
                return idx, result
        
        # Execute all tasks
        tasks = [bounded_task(item, idx) for idx, item in enumerate(items)]
        
        results: List[Any] = [None] * len(items)
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            completed += 1
            
            if show_progress:
                print(f"\r{desc}: {completed}/{len(items)}", end="", flush=True)
        
        if show_progress:
            print()  # New line after progress
        
        return results
    
    async def _execute_async_with_retry(
        self,
        async_func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute an async function with retry logic.
        
        Args:
            async_func: Async function to execute
            *args: Positional arguments for async_func
            **kwargs: Keyword arguments for async_func
            
        Returns:
            Result from async_func
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await async_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = min(
                        self.config.retry_delay * (2 ** attempt),
                        60.0  # Max 60 seconds
                    )
                    await asyncio.sleep(wait_time)
        
        raise last_error or RuntimeError("All retries failed")


def create_executor_from_config(config: Any) -> AsyncExecutor:
    """
    Factory function to create an AsyncExecutor from application config.
    
    Args:
        config: Application Config object
        
    Returns:
        Configured AsyncExecutor instance
    """
    executor_config = ExecutorConfig(
        max_concurrent=config.execution.max_concurrent,
        max_retries=config.execution.max_retries,
        retry_delay=config.execution.retry_delay,
    )
    return AsyncExecutor(executor_config)
