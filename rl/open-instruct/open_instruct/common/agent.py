"""
Agent Module

Provides a unified Agent class for managing LLM multi-turn conversations.
Used across all three roles: Generator, Simulator, Judge.

Features:
- Message history management
- Token counting and context truncation
- JSON-mode output support
- System prompt configuration
- Batch parallel execution support
"""

import json
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .clients import LLMClient, LLMClientError, JSONParseError, ContentPolicyError
from .config import Config


class AgentError(Exception):
    """Exception raised for Agent errors."""
    pass


@dataclass
class AgentConfig:
    """Configuration for Agent."""
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    max_context_tokens: int = 32000
    max_response_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60


class Agent:
    """
    Unified LLM context management for all three roles.

    Supports:
    - Multi-turn conversations with message history
    - Single-turn stateless calls
    - JSON-mode structured output
    - Context management and truncation
    """

    def __init__(
        self,
        config: Union[AgentConfig, Dict[str, Any]],
        system_prompt: Optional[str] = None,
        role_name: str = "agent",
    ):
        """
        Initialize the Agent.

        Args:
            config: AgentConfig object or dict with LLM configuration
            system_prompt: Optional system prompt for the agent
            role_name: Name of the agent role (for logging)
        """
        # Store original config dict for LLMClient (includes max_retries, retry_delay)
        config_dict = config if isinstance(config, dict) else config.__dict__
        
        if isinstance(config, dict):
            self._config = AgentConfig(
                model=config.get("model", "deepseek-chat"),
                base_url=config.get("base_url", "https://api.deepseek.com/v1"),
                api_key=config.get("api_key", ""),
                max_context_tokens=config.get("max_context_tokens", 32000),
                max_response_tokens=config.get("max_response_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                timeout_seconds=config.get("timeout_seconds", 60),
            )
        else:
            self._config = config

        # Pass the full config dict to LLMClient (includes retry settings)
        self._client = LLMClient(config_dict)
        self._system_prompt = system_prompt
        self._role_name = role_name
        self._messages: List[Dict[str, str]] = []

    @property
    def role_name(self) -> str:
        """Get the agent's role name."""
        return self._role_name

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the current system prompt."""
        return self._system_prompt

    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get the current message history (excluding system prompt)."""
        return self._messages.copy()

    @property
    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self._messages)

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: New system prompt
        """
        self._system_prompt = prompt

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to history.

        Args:
            content: Message content
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to history.

        Args:
            content: Message content
        """
        self._messages.append({"role": "assistant", "content": content})

    def chat(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Multi-turn chat: send a message and get a response.

        Maintains conversation history across calls.

        Args:
            user_message: The user's message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, parse response as JSON and return dict

        Returns:
            Assistant's response (str or dict if json_mode=True)

        Raises:
            AgentError: If LLM call fails or JSON parsing fails
        """
        # Add user message to history
        self.add_user_message(user_message)

        # Build full message list with system prompt
        messages = self._build_messages()

        try:
            if json_mode:
                response = self._client.complete_json(
                    prompt=user_message,
                    system_prompt=self._system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Store raw JSON string in history
                self.add_assistant_message(json.dumps(response, ensure_ascii=False))
                return response
            else:
                response = self._client.complete_chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.add_assistant_message(response)
                return response
        except (ContentPolicyError, LLMClientError, JSONParseError) as e:
            # All unrecoverable LLM errors are absorbed here so callers never
            # need their own try-except for API failures.  An empty response
            # naturally scores 0 reward; training continues uninterrupted.
            _etype = "ContentPolicyError(450)" if isinstance(e, ContentPolicyError) else type(e).__name__
            print(f"[Agent.chat] {_etype}, returning empty response: {e}")
            return {} if json_mode else ""

    def single_turn(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Single-turn stateless call: send a message and get a response.

        Does NOT modify conversation history.

        Args:
            user_message: The user's message
            system_prompt: Override system prompt for this call
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, parse response as JSON and return dict

        Returns:
            Assistant's response (str or dict if json_mode=True).
            Returns "" (or {}) on any LLM/parse error so callers never crash.
        """
        # Use provided system prompt or default
        sys_prompt = system_prompt or self._system_prompt

        try:
            if json_mode:
                return self._client.complete_json(
                    prompt=user_message,
                    system_prompt=sys_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": user_message})

                return self._client.complete_chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except (ContentPolicyError, LLMClientError, JSONParseError) as e:
            _etype = "ContentPolicyError(450)" if isinstance(e, ContentPolicyError) else type(e).__name__
            print(f"[Agent.single_turn] {_etype}, returning empty response: {e}")
            return {} if json_mode else ""

    async def async_single_turn(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Async single-turn call using aiohttp — does NOT block the event loop.
        Use this in async contexts (asyncio.gather) instead of asyncio.to_thread.
        Does NOT modify conversation history.
        """
        sys_prompt = system_prompt or self._system_prompt
        try:
            if json_mode:
                return await self._client.complete_json_async(
                    prompt=user_message,
                    system_prompt=sys_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": user_message})
                return await self._client.complete_chat_async(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except (ContentPolicyError, LLMClientError, JSONParseError) as e:
            _etype = "ContentPolicyError(450)" if isinstance(e, ContentPolicyError) else type(e).__name__
            print(f"[Agent.async_single_turn] {_etype}, returning empty response: {e}")
            return {} if json_mode else ""

    def reset_context(self) -> None:
        """Clear all message history."""
        self._messages = []

    def truncate_context(self, target_tokens: int) -> int:
        """
        Truncate message history to fit within target token count.

        Removes oldest messages (except system prompt) until within limit.

        Args:
            target_tokens: Target maximum token count

        Returns:
            Number of messages removed
        """
        removed = 0
        current_tokens = self.estimate_context_tokens()

        while current_tokens > target_tokens and len(self._messages) > 1:
            # Remove oldest message (keep at least one)
            self._messages.pop(0)
            removed += 1
            current_tokens = self.estimate_context_tokens()

        return removed

    def estimate_context_tokens(self) -> int:
        """
        Estimate total token count for current context.

        Returns:
            Estimated token count (system prompt + all messages)
        """
        total = 0

        if self._system_prompt:
            total += self._client.estimate_tokens(self._system_prompt)

        for msg in self._messages:
            total += self._client.estimate_tokens(msg["content"])
            total += 4  # Overhead for role and formatting

        return total

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        Build the full message list including system prompt.

        Returns:
            List of message dicts ready for API call
        """
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.extend(self._messages)

        return messages

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.

        Returns:
            Dict with context statistics
        """
        return {
            "role_name": self._role_name,
            "message_count": len(self._messages),
            "estimated_tokens": self.estimate_context_tokens(),
            "has_system_prompt": self._system_prompt is not None,
            "max_context_tokens": self._config.max_context_tokens,
        }
    
    def batch_single_turn(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        max_workers: int = 20,
        desc: str = "Processing",
        show_progress: bool = True,
        pbar: Optional[Any] = None,
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Execute multiple single-turn calls in parallel.
        
        Does NOT modify conversation history.
        
        Args:
            prompts: List of user messages to process
            system_prompt: Override system prompt for all calls
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, parse responses as JSON and return dicts
            max_workers: Maximum number of parallel workers
            desc: Description for progress bar
            show_progress: Whether to show progress bar
            pbar: Optional existing tqdm progress bar to use
            
        Returns:
            List of responses in the same order as input prompts.
            Failed calls yield "" (or {}) rather than raising.
        """
        if not prompts:
            return []
        
        results: List[Any] = [None] * len(prompts)
        
        def process_prompt(idx: int, prompt: str) -> Tuple[int, Union[str, Dict[str, Any]]]:
            """Process a single prompt and return (index, result)."""
            result = self.single_turn(
                user_message=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            return idx, result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_prompt, idx, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results as they complete with progress bar
            if pbar is not None:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, result = future.result()
                        results[idx] = result
                    except Exception as e:
                        # Log error but don't stop the whole batch
                        from tqdm import tqdm
                        tqdm.write(f"Task {idx} failed in batch: {e}")
                        results[idx] = None
                    finally:
                        pbar.update(1)
            elif show_progress:
                from tqdm import tqdm
                with tqdm(total=len(prompts), desc=desc, leave=False) as internal_pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            _, result = future.result()
                            results[idx] = result
                        except Exception as e:
                            tqdm.write(f"Task {idx} failed in batch: {e}")
                            results[idx] = None
                        finally:
                            internal_pbar.update(1)
            else:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, result = future.result()
                        results[idx] = result
                    except Exception as e:
                        results[idx] = None
        
        return results


def create_agent_from_config(
    config: Config,
    role: str,
    llm_backend: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Agent:
    """
    Factory function to create an Agent from application config.

    Args:
        config: Application Config object
        role: Role name (generator, simulator, judge)
        llm_backend: LLM backend name from config
        system_prompt: Optional system prompt
        temperature: Optional temperature override (step-level)

    Returns:
        Configured Agent instance
    """
    llm_config = config.get_llm_config(llm_backend)

    # Override temperature if step-level temperature is provided
    if temperature is not None:
        llm_config["temperature"] = temperature

    # Merge execution config (retry settings) into llm_config
    llm_config["max_retries"] = config.execution.max_retries
    llm_config["retry_delay"] = config.execution.retry_delay
    llm_config["json_parse_retries"] = config.execution.json_parse_retries

    return Agent(
        config=llm_config,
        system_prompt=system_prompt,
        role_name=role,
    )