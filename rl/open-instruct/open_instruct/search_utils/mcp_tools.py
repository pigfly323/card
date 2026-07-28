"""
A wrapper and registry for tools in rl-rag-mcp.
"""
from typing import List
import inspect
import asyncio
import os
import time
import re
import httpx
import httpcore


try:
    from dr_agent.tool_interface.mcp_tools import MassiveServeSearchTool, SemanticScholarSnippetSearchTool, SerperSearchTool, Crawl4AIBrowseTool, SerperBrowseTool, LocalSearchTool, LocalBrowseTool, DraftAnswerFeedbackTool, VerifyAnswerTool
except ImportError as e:
    print(f"Failed to import dr_agent. Please install it:\n{e}")
    raise e

from open_instruct.search_rewards.utils.format_utils import generate_snippet_id
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput

USE_LOCAL_SEARCH = os.environ.get("USE_LOCAL_SEARCH", "false").lower() == "true"

MCP_TOOL_REGISTRY = {
    "snippet_search": LocalSearchTool if USE_LOCAL_SEARCH else SemanticScholarSnippetSearchTool,
    "google_search": LocalSearchTool if USE_LOCAL_SEARCH else SerperSearchTool,
    "massive_serve": MassiveServeSearchTool,
    "browse_webpage": LocalBrowseTool if USE_LOCAL_SEARCH else Crawl4AIBrowseTool,
    "draft_answer": DraftAnswerFeedbackTool,
    "verify_answer": VerifyAnswerTool,
    # "browse_webpage": SerperBrowseTool
}

def truncate_at_second_last_stop(text: str, stops: list[str]) -> str:
    # dedup stop strings
    stops = list(set(stops))
    # Collect all stop occurrences (position, stopstring)
    positions = []
    for stop in stops:
        start = 0
        while True:
            idx = text.find(stop, start)
            if idx == -1:
                break
            positions.append((idx, stop))
            start = idx + len(stop)

    # If fewer than 2 stops, return unchanged
    if len(positions) < 2:
        return text

    # Sort by position in the string
    positions.sort(key=lambda x: x[0])

    # Take the second-last occurrence
    idx, stop = positions[-2]

    # Remove everything up to and including this occurrence
    return text[idx + len(stop):]


def _extract_question_from_full_prompt(prompt: str) -> str | None:
    """从完整 prompt 文本中提取原始用户问题。

    策略（按优先级）：
    1. 解析 chat template 格式，提取第一个 user turn 的内容。
       覆盖常见格式：
         - Qwen/ChatML: <|im_start|>user\\n...\\n<|im_end|>
         - Tulu/generic: <|user|>\\n...\\n<|assistant|>  或  [INST]...[/INST]
    2. 兜底：寻找 <question>...</question> 标签（旧格式兼容）。

    注意：不依赖模型生成内容（question 参数），只解析 prompt 结构，
    因此不存在模型幻觉导致 question 不准确的风险。
    """
    if not prompt:
        return None

    # additional_question_instructions are appended to the user turn via "\n\n".
    # Strip them by cutting at the last "\n\nFor the given question," prefix.
    def _strip_additional_instructions(text: str) -> str:
        cut = re.search(r"\n\nFor the given question[,\s]", text)
        if cut:
            return text[: cut.start()].strip()
        return text.strip()

    # --- 策略1：解析 chat template 格式，提取第一个 user turn ---

    # Qwen/ChatML 格式：<|im_start|>user\n内容\n<|im_end|>
    m = re.search(
        r"<\|im_start\|>user\s*\n(.*?)(?:\n?<\|im_end\|>|<\|im_start\|>)",
        prompt,
        re.DOTALL,
    )
    if m:
        q = _strip_additional_instructions(m.group(1))
        if q:
            return q

    # Tulu 格式：<|user|>\n内容\n<|assistant|>
    m = re.search(
        r"<\|user\|>\s*\n(.*?)(?:\n?<\|assistant\|>|\Z)",
        prompt,
        re.DOTALL,
    )
    if m:
        q = _strip_additional_instructions(m.group(1))
        if q:
            return q

    # Llama/Mistral 格式：[INST]内容[/INST]
    m = re.search(r"\[INST\](.*?)\[/INST\]", prompt, re.DOTALL)
    if m:
        q = _strip_additional_instructions(m.group(1))
        if q:
            return q

    # --- 策略2：兜底，兼容旧格式 <question> 标签 ---
    matches = re.findall(r"<question>(.*?)</question>", prompt, re.DOTALL | re.IGNORECASE)
    if matches:
        q = matches[-1].strip()
        if q:
            return q

    return None


def _strip_system_prompt(text: str) -> str:
    """剔除 system prompt 块（ChatML/Qwen、Tulu、Llama 三种 chat template 格式）。

    system prompt 含大量 call_tool / snippet few-shot 示例，不剔除会污染
    solution_path 和 snippets_context 的提取。被 _extract_solution_path 和
    _extract_snippets_context 共用。
    """
    if not text:
        return ""
    system_re = re.compile(
        # ChatML/Qwen: <|im_start|>system\n ... <|im_end|>
        r'<\|im_start\|>system\s*\n.*?<\|im_end\|>'
        # Tulu: <|system|>\n ... <|user|>
        r'|<\|system\|>\s*\n.*?(?:<\|user\|>|<\|assistant\|>)'
        # Llama/Mistral: [SYS] ... [/SYS]
        r'|\[SYS\].*?\[/SYS\]',
        re.DOTALL | re.IGNORECASE,
    )
    return system_re.sub("", text)


def _truncate_before_last_verify_answer(text: str) -> tuple[str, bool]:
    """截断到最后一个 verify_answer <call_tool> 调用之前（不含调用本身）。

    用最后一个匹配而非第一个 —— 残留文本里可能仍有 verify_answer 示例，
    re.search 会命中第一个导致真实交互被截掉。

    Returns:
        (截断后的文本, 是否发生了截断)
    """
    if not text:
        return "", False
    verify_matches = list(re.finditer(
        r'<call_tool\s+[^>]*name=["\']?verify_answer["\']?',
        text,
        re.IGNORECASE,
    ))
    if not verify_matches:
        return text, False
    return text[:verify_matches[-1].start()], True
    """将 question="..." 属性注入到最近一次 draft_answer 工具调用的开始标签中。

    question 来自训练侧的 raw_user_query（数据集原始用户问题），不依赖模型生成。
    若模型已自行填写了 question 属性（旧 checkpoint 的幻觉行为），强制覆盖，
    确保 MCP Backend 始终使用数据集中的真实 query 进行 rubric 匹配。
    """
    if not question:
        return trunc_prompt
    escaped = question.replace('"', "&quot;")

    patterns = [
        r'(<tool\s+[^>]*name="draft_answer"[^>]*)(>)(.*?)(</tool>)',
        r'(<call_tool\s+[^>]*name="draft_answer"[^>]*)(>)(.*?)(</call_tool>|</call>)',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, trunc_prompt, re.DOTALL | re.IGNORECASE))
        if not matches:
            continue
        m = matches[-1]
        opening = m.group(1)
        # 若模型已填写 question 属性，强制替换为数据集的真实 query（覆盖幻觉值）
        opening = re.sub(r'\s+question="[^"]*"', "", opening)
        new_opening = f'{opening} question="{escaped}"'
        return trunc_prompt[: m.start(1)] + new_opening + trunc_prompt[m.end(1) :]
    return trunc_prompt


def _extract_snippets_context_from_full_text(full_text: str) -> str:
    """从完整对话文本中提取所有真实检索的 <snippet>/<webpage> 块。

    这些块是搜索/浏览工具调用返回的 tool_output 内容，含 citation ID，
    verify_answer 据此验证 draft answer 中 <cite> 引用的质量。

    复用 _extract_solution_path 的清洗逻辑（剔除 system prompt + 截断 verify_answer），
    避免误提取 system prompt 里的 few-shot snippet 示例（如 S_a1B9xQ2）。
    再从清洗后的交互轨迹中提取 snippet/webpage 块。
    """
    if not full_text:
        return ""
    # 复用 solution_path 的清洗：剔除 system prompt + 截断到最后一个 verify_answer
    cleaned = _strip_system_prompt(full_text)
    cleaned, _ = _truncate_before_last_verify_answer(cleaned)

    patterns = [
        r'<snippets?\s+id=["\']?[^"\'>\s]+["\']?[^>]*>.*?</snippets?>',
        r'<snippet\s+id=["\']?[^"\'>\s]+["\']?[^>]*>.*?</snippet>',
        r'<webpage\s+id=["\']?[^"\'>\s]+["\']?[^>]*>.*?</webpage>',
    ]
    blocks = []
    for pattern in patterns:
        blocks.extend(re.findall(pattern, cleaned, re.DOTALL))
    result = "\n".join(blocks)
    print(f"[verify_answer][context_extract] found {len(blocks)} snippet/webpage blocks, total_chars={len(result)}")
    return result


def _extract_solution_path_from_full_text(full_text: str) -> str:
    """从完整对话文本中提取 agent 与 tool 的交互轨迹作为 solution path。

    只保留 <call_tool>...</call_tool> 和 <tool_output>...</tool_output> 块，
    剔除 <think>...</think> 推理内容，供 judge model 判断答案是否有证据支撑。

    一个完整的交互轨迹形如：
        <call_tool name="google_search">query</call_tool>
        <tool_output><snippet id=...>...</snippet></tool_output>
        <call_tool name="browse_webpage">url</call_tool>
        <tool_output><webpage id=...>...</webpage></tool_output>
        ...

    提取步骤：
      1. 定位并剔除 system prompt（<|im_start|>system ... <|im_end|>），
         避免 system prompt 里的 google_search/snippet_search/verify_answer
         few-shot 示例被误提取进 solution_path。
      2. 截断到最后一个 verify_answer 调用之前（不含 verify_answer 本身）。
         用最后一个匹配而非第一个 —— 残留的 verify_answer 示例可能仍在。
      3. 从剩余文本中提取所有 <call_tool> 和 <tool_output> 块（保持原顺序）
         - 剔除 verify_answer 调用块（name="verify_answer"）
         - 剔除 <think>...</think> 推理块

    Returns:
        拼接后的交互轨迹字符串；提取失败时返回空字符串。
    """
    if not full_text:
        return ""

    # Step 1: 剔除 system prompt（复用公共函数，与 _extract_snippets_context 一致）
    text = _strip_system_prompt(full_text)

    # Step 2: 截断到最后一个 verify_answer 调用之前（复用公共函数）
    text, verify_truncated = _truncate_before_last_verify_answer(text)

    # Step 3: 只提取 <call_tool> 和 <tool_output> 块，保持原顺序，剔除 <think>。
    # 匹配 <call_tool ...>...</call_tool>（含 </call> 短闭合兜底）和
    # <tool_output>...</tool_output>（含 <output> 旧标签兜底）。
    block_re = re.compile(
        r'<call_tool\s+[^>]*>.*?(?:</call_tool>|</call>)'
        r'|<tool_output\s*[^>]*>.*?</tool_output>'
        r'|<output\s*[^>]*>.*?</output>',
        re.DOTALL | re.IGNORECASE,
    )

    blocks = []
    for m in block_re.finditer(text):
        block = m.group(0)
        # 剔除 verify_answer 调用块（不应出现在 solution_path 中）
        if re.search(r'name=["\']?verify_answer["\']?', block, re.IGNORECASE):
            continue
        blocks.append(block.strip())

    solution_path = "\n\n".join(blocks)

    # 诊断：n_blocks=0 时打印截断后文本的头部/尾部，便于定位格式不匹配原因
    _head = text[:300].replace("\n", "\\n") if text else ""
    _tail = text[-300:].replace("\n", "\\n") if text else ""
    print(
        f"[solution_path_extract] chars={len(solution_path)} "
        f"n_blocks={len(blocks)} "
        f"verify_call_truncated={verify_truncated} "
        f"text_len={len(text)} "
        f"head={_head!r} "
        f"tail={_tail!r}"
    )
    return solution_path


class GTLookup:
    """Ground truth lookup table built from a dataset JSON file.

    Supports question → answer lookup so MCPTool can inject the correct
    ground truth into verify_answer calls without the model knowing it.

    Expected JSON format (list of objects):
      [{"Question": "...", "answer": ["..."]}]  # HotpotQA style
      [{"question": "...", "answers": ["..."]}]  # generic style

    Field priority for question : "Question" > "question" > "problem"
    Field priority for answer   : "answer"   > "answers"  > "Answer"
    """

    def __init__(self, gt_file: str) -> None:
        import json as _json
        with open(gt_file, encoding="utf-8") as f:
            data = _json.load(f)
        self._lookup: dict[str, str] = {}
        for item in (data if isinstance(data, list) else [data]):
            q = str(
                item.get("Question") or item.get("question") or item.get("problem") or ""
            ).strip()
            raw_ans = (
                item.get("answer") or item.get("answers") or item.get("Answer") or ""
            )
            if isinstance(raw_ans, list):
                raw_ans = raw_ans[0] if raw_ans else ""
            ans = str(raw_ans).strip()
            if q and ans:
                self._lookup[q] = ans
        print(f"[GTLookup] loaded {len(self._lookup)} entries from {gt_file}")

    def get(self, question: str) -> str | None:
        if not question:
            return None
        q = question.strip()
        # 1. Exact match
        if q in self._lookup:
            return self._lookup[q]
        # 2. Case-insensitive fallback
        q_lower = q.lower()
        for k, v in self._lookup.items():
            if k.lower() == q_lower:
                return v
        return None


def _inject_question_and_context_for_verify_answer_call(
    trunc_prompt: str,
    question: str | None,
    context: str | None,
    apply_adaptive_rubric_reward: bool = False,
    solution_path: str | None = None,
    ground_truth: str | None = None,
) -> str:
    """将 question="..." 属性注入到最近一次 verify_answer 工具调用的开始标签中，
    并在工具调用内容中追加 <context>/<solution_path>/<ground_truth> 块。

    question:      从完整 prompt 提取的原始用户问题，用于 rubric 匹配。
    context:       从完整对话历史提取的所有 snippet/webpage 块，用于 citation 验证。
    solution_path: 从完整对话历史提取的 CoT + 工具调用 + 工具返回，
                   供 short_answer verify agent 使用（rubric 模式下忽略）。
    ground_truth:  从数据集 GT lookup 获取的参考答案，作为 oracle 私有提示，
                   仅在 short_answer 模式下使用（rubric 模式下忽略）。
    apply_adaptive_rubric_reward: 若为 True，注入 apply_adaptive_rubric_reward="true"
        属性，MCP server 据此生成 instance-wise adaptive rubrics。
    """
    patterns = [
        r'(<tool\s+[^>]*name="verify_answer"[^>]*)(>)(.*?)(</tool>)',
        r'(<call_tool\s+[^>]*name="verify_answer"[^>]*)(>)(.*?)(</call_tool>|</call>)',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, trunc_prompt, re.DOTALL | re.IGNORECASE))
        if not matches:
            continue
        m = matches[-1]
        opening = m.group(1)
        inner = m.group(3)
        closing = m.group(4)

        # Inject question attribute into opening tag
        if question:
            escaped_q = question.replace('"', "&quot;")
            opening = re.sub(r'\s+question="[^"]*"', "", opening)
            opening = f'{opening} question="{escaped_q}"'

        # Inject apply_adaptive_rubric_reward attribute
        opening = re.sub(r'\s+apply_adaptive_rubric_reward="[^"]*"', "", opening)
        opening = f'{opening} apply_adaptive_rubric_reward="{"true" if apply_adaptive_rubric_reward else "false"}"'

        # Inject <context> block into inner content if not already present
        if context and "<context>" not in inner:
            inner = inner.rstrip() + f"\n<context>{context}</context>"

        # Inject <solution_path> block into inner content if not already present
        if solution_path and "<solution_path>" not in inner:
            # 截断超长 solution_path，避免注入后 inner 过长导致 </solution_path>
            # 被下游长度限制（MCP 传输 / max_tokens）截掉，使 dr_agent 侧解析失败。
            _max_sp = int(os.environ.get("VERIFY_SHORT_ANSWER_MAX_SP_CHARS", "12000"))
            if len(solution_path) > _max_sp:
                solution_path = solution_path[:_max_sp] + "\n... [truncated]"
            inner = inner.rstrip() + f"\n<solution_path>{solution_path}</solution_path>"

        # Inject <ground_truth> block into inner content if not already present
        if ground_truth and "<ground_truth>" not in inner:
            inner = inner.rstrip() + f"\n<ground_truth>{ground_truth}</ground_truth>"

        rebuilt = opening + ">" + inner + closing
        # 诊断：打印 inner 末尾 + closing，确认 </solution_path> 是否在 inner 里
        print(
            f"[verify_answer][inject] question_chars={len(question or '')} "
            f"context_chars={repr((context or '')[:40])} "
            f"solution_path_chars={repr((solution_path or '')[:40])} "
            f"ground_truth={repr((ground_truth or '')[:40])} "
            f"answer_chars={repr(inner[:40])} "
            f"apply_adaptive_rubric_reward={apply_adaptive_rubric_reward} "
            f"inner_tail={repr(inner[-120:])} "
            f"closing={repr(closing)} "
            f"inner_has_close_sp={'</solution_path>' in inner}"
        )
        return trunc_prompt[: m.start()] + rebuilt + trunc_prompt[m.end() :]
    print("[verify_answer][inject] WARNING: no verify_answer call pattern found in trunc_prompt")
    return trunc_prompt


class MCPTool(Tool):
    """
    Unlike other tools, this guy handles *all mcp tools*. Why?
    because they share the same end string (</tool>). Hence, we need the parsers
    to work out how to route them. Ideally, this would be more tightly integrated into vllm,
    but for now, this is a bit cleaner.
    """
    def __init__(
        self,
        mcp_tool_names: List[str] | str,
        mcp_parser_name: str = "unified",
        transport_type: str | None = None,
        mcp_host: str | None = None,
        mcp_port: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        search_api_endpoint: str | None = None,
        start_str: str = "",
        end_str: str | None = None,
        mcp_timeout: int = 180,
        base_url: str | None = None,
        number_documents_to_search: int = 10,
        use_localized_snippets: bool = False,
        context_chars: int = 6000,
        apply_adaptive_rubric_reward: bool = False,
        ground_truth_file: str | None = None,
        enable_native_verify_answer: bool = False,
        native_verify_use_general_rubric: bool = True,
        native_verify_mcp_parser_name: str | None = None,
        *args,
        **kwargs,
    ):
        self.mcp_tools = []
        self.stop_strings = []
        # verify_answer-specific: whether to generate adaptive rubrics per call
        self.apply_adaptive_rubric_reward = bool(apply_adaptive_rubric_reward)
        # Ground truth lookup for verify_answer (short_answer backend)
        self._gt_lookup: GTLookup | None = None
        if ground_truth_file:
            try:
                self._gt_lookup = GTLookup(ground_truth_file)
            except Exception as e:
                print(f"[MCPTool] WARNING: failed to load ground_truth_file '{ground_truth_file}': {e}")
        # Native verify_answer: compute reward locally without MCP Server round-trip
        self.enable_native_verify_answer = bool(enable_native_verify_answer)
        self.native_verify_use_general_rubric = bool(native_verify_use_general_rubric)
        self.native_verify_mcp_parser_name = (
            native_verify_mcp_parser_name or mcp_parser_name
        )
        # Allow selecting transport via arg or env; default to StreamableHttpTransport
        self.transport_type = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        # 0.0.0.0 is only for server bind, not for client requests.
        self.mcp_host = mcp_host or os.environ.get("MCP_TRANSPORT_HOST", "127.0.0.1")
        if self.mcp_host == "0.0.0.0":
            self.mcp_host = "127.0.0.1"
        if self.mcp_host is not None:
            os.environ["MCP_TRANSPORT_HOST"] = str(self.mcp_host)
        self.mcp_port = mcp_port or os.environ.get("MCP_TRANSPORT_PORT", 8000)
        if self.mcp_port is not None:
            os.environ["MCP_TRANSPORT_PORT"] = str(self.mcp_port)
        # Transient error retry settings
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Support comma-separated string for mcp_tool_names
        if isinstance(mcp_tool_names, str):
            mcp_tool_names = [n.strip() for n in mcp_tool_names.split(",") if n.strip()]
        for mcp_tool_name in mcp_tool_names:
            # filter kwargs so we only pass ones the constructor understands
            mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in valid_params
            }
            if "base_url" in valid_params:
                filtered_kwargs["base_url"] = base_url
            if "number_documents_to_search" in valid_params:
                filtered_kwargs["number_documents_to_search"] = number_documents_to_search
            if "use_localized_snippets" in valid_params:
                filtered_kwargs["use_localized_snippets"] = use_localized_snippets
            if "context_chars" in valid_params:
                filtered_kwargs["context_chars"] = context_chars
            # special case for crawl4ai
            if mcp_tool_name == "browse_webpage":
                if "use_docker_version" in valid_params:
                    filtered_kwargs["use_docker_version"] = True
                if "use_ai2_config" in valid_params:
                    filtered_kwargs["use_ai2_config"] = True
            # basically, we want to defer as much as possible to the mcp tool.
            # this 'tool' actually just passes everything down to the mcp tool.
            self.mcp_tools.append(mcp_tool_cls(
                timeout=mcp_timeout,
                name=mcp_tool_name,
                tool_parser=mcp_parser_name,
                transport_type=self.transport_type,
                **filtered_kwargs,
            ))
            # assign the stop strings from the parser itself.
            self.stop_strings += self.mcp_tools[-1].tool_parser.stop_sequences
        # MCP tool handles its own start and end strings.
        super().__init__(start_str=start_str, end_str=end_str or self.stop_strings[-1])

    def get_stop_strings(self) -> List[str]:
        return self.stop_strings

    def __call__(self, prompt) -> ToolOutput:
        # prompt 可能是 str 或 (assistant_text, full_text) 元组。
        # assistant_text: 用于工具调用检测和解析（不含 system prompt，避免误匹配示例调用）
        # full_text: 仅用于 draft_answer 的 question 提取（需要 user turn 内容）
        if isinstance(prompt, tuple):
            assistant_text, full_text_for_question = prompt
        else:
            assistant_text, full_text_for_question = prompt, prompt

        # the one thing open-instruct needs to do: remove older tool calls.
        trunc_prompt = truncate_at_second_last_stop(assistant_text, self.stop_strings)
        # work out which mcp tool to call.
        document_tool_output = None
        error = None
        found_tool = False
        text_output = ""
        tool_used_name = None
        try:
            for mcp_tool in self.mcp_tools:
                if mcp_tool.tool_parser.has_calls(trunc_prompt, mcp_tool.name):
                    call_prompt = trunc_prompt
                    if mcp_tool.name == "draft_answer":
                        hidden_question = _extract_question_from_full_prompt(full_text_for_question)
                        call_prompt = _inject_question_for_draft_answer_call(trunc_prompt, hidden_question)
                    elif mcp_tool.name == "verify_answer":
                        hidden_question = _extract_question_from_full_prompt(full_text_for_question)

                        # ── Native path: compute reward locally, no MCP Server call ──
                        if self.enable_native_verify_answer:
                            return self._verify_answer_native(
                                full_text_for_question, hidden_question
                            )

                        # ── Fallback: route to MCP Server (original path) ──────────
                        snippets_context = _extract_snippets_context_from_full_text(full_text_for_question)
                        solution_path = _extract_solution_path_from_full_text(full_text_for_question)
                        ground_truth = (
                            self._gt_lookup.get(hidden_question)
                            if self._gt_lookup and hidden_question
                            else None
                        )
                        print(
                            f"[verify_answer][mcp_fallback] question={repr((hidden_question or '')[:80])} "
                            f"full_text_chars={len(full_text_for_question)} "
                            f"context_chars={len(snippets_context)} "
                            f"solution_path_chars={len(solution_path)} "
                            f"ground_truth={repr((ground_truth or '')[:40])}"
                        )
                        call_prompt = _inject_question_and_context_for_verify_answer_call(
                            trunc_prompt, hidden_question, snippets_context,
                            apply_adaptive_rubric_reward=self.apply_adaptive_rubric_reward,
                            solution_path=solution_path,
                            ground_truth=ground_truth,
                        )
                    # Retry on transient stream/network errors
                    last_exc: Exception | None = None
                    for attempt in range(self.max_retries):
                        try:
                            # asyncio.run() creates a new event loop each time.
                            # Reset all event-loop-bound state (Lock, persistent client)
                            # before entering the new loop to avoid 'bound to a different
                            # event loop' errors.
                            try:
                                from dr_agent.tool_interface.mcp_tools import MCPMixin
                                MCPMixin._reset_event_loop_state()
                            except Exception:
                                pass
                            document_tool_output = asyncio.run(mcp_tool(call_prompt))
                            print("Using MCP tool: ", mcp_tool.name)
                            break
                        except Exception as e:
                            last_exc = e
                            print(f"{mcp_tool.name} Error [{type(e).__name__}]: {e}, retrying (attempt {attempt+1}/{self.max_retries})...")
                            if attempt + 1 >= self.max_retries:
                                raise
                            time.sleep(self.retry_backoff * (2 ** attempt))
                    # first format the output
                    text_output = mcp_tool._format_output(document_tool_output)
                    # wrap in the tags
                    text_output = mcp_tool.tool_parser.format_result(text_output, document_tool_output)
                    found_tool = True
                    tool_used_name = mcp_tool.name
                    break
        except Exception as e:
            import traceback as _tb
            error = str(e)
            print(f"MCP {tool_used_name} Exception [{type(e).__name__}]: {error}\n{_tb.format_exc()}")
        if document_tool_output is None:
            if error is None and not found_tool:
                error = "No valid tool calls found."
                registered_tools = [t.name for t in self.mcp_tools]
                # 打印 agent 输出的末尾 500 字符，方便看实际调用格式
                tail = trunc_prompt[-500:] if len(trunc_prompt) > 500 else trunc_prompt
                print(f"MCP Tool Error: {error} | registered_tools={registered_tools} | prompt_tail={repr(tail)}")
                return ToolOutput(
                    output=error,
                    called=False,
                    error=error,
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )
            elif error is not None:
                print(f"MCP {tool_used_name} Tool Error: {error}")
                return ToolOutput(
                    output=error,
                    called=False,
                    error=error,
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )
            else:
                print(f"MCP {tool_used_name} Tool Error: Unknown error, no MCP response and no error found.")
                return ToolOutput(
                    output="Unknown error, no MCP response and no error found.",
                    called=False,
                    error="Unknown error, no MCP response and no error found.",
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )

        if document_tool_output.error:
            print(f"MCP {tool_used_name} Tool Error: {document_tool_output.error} | "
                  f"timeout={document_tool_output.timeout} | "
                  f"output_len={len(text_output)} | "
                  f"raw={str(document_tool_output.raw_output)[:200] if hasattr(document_tool_output, 'raw_output') else 'N/A'}")
            print("Returning error output anyway.")
        # munge into format that open-instruct likes.
        return ToolOutput(
            output=text_output,
            called=True,
            error=document_tool_output.error,
            timeout=document_tool_output.timeout,
            runtime=document_tool_output.runtime,
            start_str="\n",
            end_str="\n\n",
        )


    # ------------------------------------------------------------------
    def _verify_answer_native(
        self,
        full_text: str,
        question: str | None,
    ) -> ToolOutput:
        """
        Compute r0 reward for the draft answer inside the verify_answer call.
        Uses the internal reward system (rubric + citation + format + search turns).
        Returns a ToolOutput whose ``output`` is human-readable feedback that the
        model reads as the tool's response.

        No MCP Server HTTP round-trip.
        """
        from open_instruct.search_rewards.verify_answer_local import (
            compute_verify_answer_feedback,
            format_verify_answer_output,
        )
        from open_instruct.search_rewards.utils.format_utils import (
            extract_citations_from_context,
        )
        import time as _time

        t0 = _time.perf_counter()
        q = question or ""

        # Extract citations from all prior retrieved snippets in full_text
        # (before the verify_answer call, so citations are actually available)
        prior_snippets_raw = _extract_snippets_context_from_full_text(full_text)

        try:
            feedback = asyncio.run(
                compute_verify_answer_feedback(
                    full_response=full_text,
                    question=q,
                    mcp_parser_name=self.native_verify_mcp_parser_name,
                    use_general_rubric=self.native_verify_use_general_rubric,
                    pre_extracted_citations=(
                        extract_citations_from_context(prior_snippets_raw)
                        if prior_snippets_raw else None
                    ),
                )
            )
            text_output = format_verify_answer_output(feedback)
            elapsed = _time.perf_counter() - t0
            print(
                f"[verify_answer][native] question={repr(q[:80])} "
                f"r0={feedback.get('r0', 0.0):.4f} "
                f"rubric={feedback.get('rubric_reward', 0.0):.3f} "
                f"citation={feedback.get('citation_reward', 0.0):.3f} "
                f"format={feedback.get('format_reward', 0.0):.3f} "
                f"search={feedback.get('search_turns_reward', 0.0):.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            return ToolOutput(
                output=text_output,
                called=True,
                error=None,
                timeout=False,
                runtime=elapsed,
                start_str="\n",
                end_str="\n\n",
            )
        except Exception as exc:
            import traceback as _tb
            err_msg = f"[verify_answer][native] ERROR: {exc}\n{_tb.format_exc()}"
            print(err_msg, flush=True)
            return ToolOutput(
                output=f"[Verification error: {exc}]",
                called=False,
                error=str(exc),
                timeout=False,
                runtime=_time.perf_counter() - t0,
                start_str="\n",
                end_str="\n\n",
            )


if __name__ == "__main__":
    # example usage.
    from open_instruct.grpo_fast import launch_mcp_subprocess
    import time
    # need to launch mcp server first.
    launch_mcp_subprocess("python -m dr_agent.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp", "./mcp_logs")
    # wait for it to launch.
    time.sleep(10)

    # ── verify_answer 调用测试 ──────────────────────────────────────────────
    # 构造一个含完整交互轨迹的 full_text（system + user + assistant + 工具调用），
    # 验证 _extract_question / _extract_solution_path / _extract_snippets_context
    # 以及 verify_answer 的 MCP server 调用是否正常工作。
    #
    # 前置条件：
    #   1. MCP server 已启动（--verify-backend short_answer）
    #   2. GT_DATASET_PATH 指向转换后的 GT 文件（train_gt.jsonl）
    #   3. ground_truth_file 参数指向同一份 GT 文件（供 MCPTool 本地 GTLookup）

    _SYSTEM = """<|im_start|>system
You are a search agent. Use tools to find information.
Available tools:
  - google_search: <call_tool name="google_search">query</call_tool>
  - verify_answer: <call_tool name="verify_answer"><answer>draft</answer></call_tool>
Example: <call_tool name="google_search">2024 renewable energy</call_tool>
<|im_end|>"""
    _USER = "<|im_start|>user\nwho got the first nobel prize in physics?\n<|im_end|>"
    _ASSISTANT = """<|im_start|>assistant
<think>I need to search for the first Nobel Prize in Physics winner.</think>

<call_tool name="google_search">who got the first nobel prize in physics</call_tool>

<tool_output><snippet id="s1">
Title: Nobel Prize in Physics - Wikipedia
URL: https://en.wikipedia.org/wiki/Nobel_Prize_in_Physics
Snippet: The first Nobel Prize in Physics was awarded to Wilhelm Conrad Röntgen in 1901 for his discovery of X-rays.
</snippet></tool_output>

<call_tool name="verify_answer"><answer>The first Nobel Prize in Physics was awarded to Wilhelm Conrad Röntgen in 1901.</answer></call_tool>"""

    full_text = f"{_SYSTEM}\n{_USER}\n{_ASSISTANT}"

    # 先单独验证三个提取函数
    print("=" * 60)
    print(" 提取函数验证")
    print("=" * 60)
    q = _extract_question_from_full_prompt(full_text)
    print(f"question: {q!r}\n")
    sp = _extract_solution_path_from_full_text(full_text)
    print(f"solution_path ({len(sp)} chars):\n{sp}\n")
    ctx = _extract_snippets_context_from_full_text(full_text)
    print(f"snippets_context ({len(ctx)} chars):\n{ctx}\n")
    print("=" * 60)

    # 构造 MCPTool（含 verify_answer + google_search，并提供 GT 文件）
    # 将 GT_FILE 替换为实际转换后的 train_gt.jsonl 路径
    import os as _os
    _GT_FILE = _os.environ.get("GT_DATASET_PATH", "/tmp/train_gt.jsonl")
    mcp_tool = MCPTool(
        ["google_search", "browse_webpage", "verify_answer"],
        number_documents_to_search=10,
        mcp_host="127.0.0.1",
        mcp_port=8000,
        ground_truth_file=_GT_FILE if _os.path.exists(_GT_FILE) else None,
    )

    # MCPTool.__call__ 接收 (assistant_text, full_text) 元组：
    #   assistant_text 用于工具调用检测（truncated at second-last stop）
    #   full_text 用于 question / solution_path / snippets 提取
    result = mcp_tool((_ASSISTANT, full_text))
    print("\n" + "=" * 60)
    print(" verify_answer 返回结果")
    print("=" * 60)
    print(f"called: {result.called}")
    print(f"error: {result.error}")
    print(f"timeout: {result.timeout}")
    print(f"runtime: {result.runtime:.2f}s")
    print(f"output:\n{result.output}")
    print("=" * 60)

