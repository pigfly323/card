"""
Utility Functions Module

Provides common utilities for:
- File I/O (JSON loading/saving)
- Directory management
- DAG validation
- Prompt template loading
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


# =============================================================================
# File I/O Utilities
# =============================================================================

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON content

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Target file path
        indent: JSON indentation level (default: 2)
    """
    # Ensure parent directory exists
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    if _should_strip_schema_for_export(file_path):
        data = _strip_json_key_recursive(data, key="$schema")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _should_strip_schema_for_export(path: Path) -> bool:
    """
    Return True if `path` looks like an exported artifact under:
      - Data/data_synthesis_stage
      - Data/env_synthesis_stage
    """
    parts = path.parts
    markers = (("Data", "data_synthesis_stage"), ("Data", "env_synthesis_stage"))
    for marker in markers:
        for i in range(len(parts) - len(marker) + 1):
            if parts[i : i + len(marker)] == marker:
                return True
    return False


def _strip_json_key_recursive(obj: Any, *, key: str) -> Any:
    """
    Return a deep-copied object with all dict entries whose key == `key` removed.

    Notes:
    - Does NOT mutate the input object.
    - Preserves non-dict/list values as-is.
    """
    if isinstance(obj, dict):
        return {k: _strip_json_key_recursive(v, key=key) for k, v in obj.items() if k != key}
    if isinstance(obj, list):
        return [_strip_json_key_recursive(v, key=key) for v in obj]
    return obj


def ensure_dir(dir_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Directory path to ensure exists
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def list_json_files(dir_path: Union[str, Path]) -> List[Path]:
    """
    List all JSON files in a directory.

    Args:
        dir_path: Directory to search

    Returns:
        List of Path objects for JSON files, sorted by name
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []

    return sorted(dir_path.glob("*.json"))


def list_directories(dir_path: Union[str, Path]) -> List[Path]:
    """
    List all subdirectories in a directory.

    Args:
        dir_path: Directory to search

    Returns:
        List of Path objects for subdirectories, sorted by name
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []

    return sorted([p for p in dir_path.iterdir() if p.is_dir()])


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


# =============================================================================
# Topic Management Utilities
# =============================================================================

def list_topic_files(domain_dir: Union[str, Path]) -> List[Path]:
    """
    List all topic JSON files in a domain directory.

    Args:
        domain_dir: Directory containing topic files (Data/domains/{domain_id}/)

    Returns:
        List of Path objects for topic JSON files, sorted by name
    """
    domain_dir = Path(domain_dir)
    if not domain_dir.exists():
        return []

    # Return all JSON files in the domain directory
    return sorted(domain_dir.glob("*.json"))


def load_topic(topic_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate a topic file.

    Args:
        topic_path: Path to the topic JSON file

    Returns:
        Topic dict with keys: domain_id, domain, topic_id, topic, topic_description

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If topic file is missing required fields
    """
    topic = load_json(topic_path)

    required_fields = ["domain_id", "domain", "topic_id", "topic", "topic_description"]
    missing = [f for f in required_fields if f not in topic]

    if missing:
        raise ValueError(
            f"Topic file {topic_path} missing required fields: {missing}"
        )

    return topic


def get_domain_info_from_topics(domain_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Get domain-level info by reading any topic file in the domain directory.

    Args:
        domain_dir: Directory containing topic files

    Returns:
        Dict with domain_id and domain name

    Raises:
        ValueError: If no topic files found or domain info inconsistent
    """
    topic_files = list_topic_files(domain_dir)

    if not topic_files:
        raise ValueError(f"No topic files found in {domain_dir}")

    # Read the first topic file to get domain info
    first_topic = load_topic(topic_files[0])

    return {
        "domain_id": first_topic["domain_id"],
        "domain": first_topic["domain"],
    }


def get_topic_id_from_path(topic_path: Union[str, Path]) -> str:
    """
    Extract topic_id from a topic file path.

    Args:
        topic_path: Path to topic file (e.g., Data/domains/travel_maps/travel_planning.json)

    Returns:
        Topic ID (filename without extension, e.g., "travel_planning")
    """
    return Path(topic_path).stem


def collect_all_topic_ids(domain_dir: Union[str, Path]) -> List[str]:
    """
    Collect all topic IDs from a domain directory.

    Args:
        domain_dir: Directory containing topic files

    Returns:
        List of topic IDs sorted alphabetically
    """
    topic_files = list_topic_files(domain_dir)
    return [get_topic_id_from_path(f) for f in topic_files]


# =============================================================================
# Prompt Loading Utilities
# =============================================================================

def load_prompt(prompt_path: Union[str, Path]) -> str:
    """
    Load a prompt template from a file.

    Args:
        prompt_path: Path to the prompt template file

    Returns:
        Prompt template content as string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with variable substitution.

    Uses regex-based substitution to only replace named placeholders,
    leaving other curly braces (like in JSON examples) untouched.

    Args:
        template: Prompt template string with {variable_name} placeholders
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string
    """
    import re

    def replace_var(match):
        var_name = match.group(1)
        if var_name in kwargs:
            return str(kwargs[var_name])
        return match.group(0)  # Keep original if not found

    # Only match {word_chars} patterns that are known variables
    pattern = r"\{(" + "|".join(re.escape(k) for k in kwargs.keys()) + r")\}"
    return re.sub(pattern, replace_var, template)


# =============================================================================
# DAG Validation Utilities
# =============================================================================


def validate_dag(
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]]
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a graph is a valid DAG (Directed Acyclic Graph).

    Args:
        nodes: List of node dicts with "node_id" key
        edges: List of edge dicts with "from" and "to" keys

    Returns:
        Tuple of (is_valid, error_message)
        error_message is None if valid
    """
    # Extract node IDs
    node_ids = {node["node_id"] for node in nodes}

    # Check all edge endpoints exist
    for edge in edges:
        if edge["from"] not in node_ids:
            return False, f"Edge source not found: {edge['from']}"
        if edge["to"] not in node_ids:
            return False, f"Edge target not found: {edge['to']}"

    # Check for cycles using DFS
    has_cycle, cycle_path = detect_cycle(node_ids, edges)
    if has_cycle:
        return False, f"Cycle detected: {' -> '.join(cycle_path)}"

    return True, None


def detect_cycle(
    node_ids: Set[str],
    edges: List[Dict[str, str]]
) -> Tuple[bool, List[str]]:
    """
    Detect if a graph contains a cycle.

    Args:
        node_ids: Set of node identifiers
        edges: List of edge dicts

    Returns:
        Tuple of (has_cycle, cycle_path)
        cycle_path is empty if no cycle
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        adj[edge["from"]].append(edge["to"])

    # DFS state: 0 = unvisited, 1 = visiting, 2 = visited
    state: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    parent: Dict[str, Optional[str]] = {node_id: None for node_id in node_ids}

    def dfs(node: str, path: List[str]) -> Tuple[bool, List[str]]:
        state[node] = 1  # Mark as visiting
        path.append(node)

        for neighbor in adj[node]:
            if state[neighbor] == 1:  # Found cycle
                # Extract cycle path
                cycle_start = path.index(neighbor)
                return True, path[cycle_start:] + [neighbor]
            elif state[neighbor] == 0:
                has_cycle, cycle_path = dfs(neighbor, path)
                if has_cycle:
                    return True, cycle_path

        path.pop()
        state[node] = 2  # Mark as visited
        return False, []

    for node_id in node_ids:
        if state[node_id] == 0:
            has_cycle, cycle_path = dfs(node_id, [])
            if has_cycle:
                return True, cycle_path

    return False, []


def find_sink_nodes(
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]]
) -> List[str]:
    """
    Find all sink nodes (nodes with no outgoing edges).

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        List of sink node IDs
    """
    node_ids = {node["node_id"] for node in nodes}
    sources = {edge["from"] for edge in edges}

    return [node_id for node_id in node_ids if node_id not in sources]


def find_source_nodes(
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]]
) -> List[str]:
    """
    Find all source nodes (nodes with no incoming edges).

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        List of source node IDs
    """
    node_ids = {node["node_id"] for node in nodes}
    targets = {edge["to"] for edge in edges}

    return [node_id for node_id in node_ids if node_id not in targets]


def topological_sort(
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Perform topological sort on a DAG.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        List of node dicts in topological order

    Raises:
        ValueError: If graph contains a cycle
    """
    is_valid, error = validate_dag(nodes, edges)
    if not is_valid:
        raise ValueError(error)

    # Build adjacency list and in-degree count
    node_ids = [node["node_id"] for node in nodes]
    adj: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        adj[edge["from"]].append(edge["to"])
        in_degree[edge["to"]] += 1

    # Kahn's algorithm
    node_by_id = {node["node_id"]: node for node in nodes}
    queue = [node for node in nodes if in_degree[node["node_id"]] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        for neighbor in adj[node["node_id"]]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(node_by_id[neighbor])

    return result


def build_dependencies(
    to: str,
    edges: List[Dict[str, str]]
) -> List[str]:
    dependencies = []
    for edge in edges:
        if edge["to"] == to:
            dependencies.append(edge["from"])
    return dependencies

    

def can_reach(
    source: str,
    target: str,
    edges: List[Dict[str, str]]
) -> bool:
    """
    Check if target is reachable from source.

    Args:
        source: Source node ID
        target: Target node ID
        edges: List of edge dicts

    Returns:
        True if target is reachable from source
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        adj[edge["from"]].append(edge["to"])

    # BFS
    visited = set()
    queue = [source]

    while queue:
        node = queue.pop(0)
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node])

    return False


# Miscellaneous Utilities

def generate_id(prefix: str, index: int, width: int = 3) -> str:
    """
    Generate a formatted ID string.

    Args:
        prefix: ID prefix (e.g., "travel_task")
        index: Index number
        width: Zero-padding width (default: 3)

    Returns:
        Formatted ID string (e.g., "travel_task_001")
    """
    return f"{prefix}_{str(index).zfill(width)}"


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# Task ID Utilities
# =============================================================================


def generate_graph_id(topic_id: str, toolset_index: int, num_nodes: int) -> str:
    """
    Generate graph-level task_id (toolset level).

    Format: {topic_id}_nodes{YY}_toolset{XX}

    Args:
        topic_id: Topic identifier
        toolset_index: Toolset index (increments per topic, starting from 1)
        num_nodes: Number of nodes in the graph

    Returns:
        Graph-level task_id string
    """
    return f"{topic_id}_nodes{num_nodes:02d}_toolset{toolset_index:02d}"


def generate_task_id(graph_id: str, task_index: int) -> str:
    """
    Generate task-level task_id from graph_id.

    Format: {graph_id}_task{ZZ}

    Args:
        graph_id: Graph-level task_id
        task_index: Task index (starting from 1)

    Returns:
        Task-level task_id string
    """
    return f"{graph_id}_task{task_index:02d}"


def extract_graph_id(task_id: str) -> str:
    """
    Extract graph_id from task_id (remove _taskZ suffix).

    Args:
        task_id: Full task_id (may include _taskZ suffix)

    Returns:
        Graph-level task_id without task suffix
    """
    match = re.match(r"^(.+_nodes\d+_toolset\d+)(?:_task\d+)?$", task_id)
    if match:
        return match.group(1)
    return task_id


def extract_topic_id(task_id: str) -> str:
    """
    Extract topic_id from task_id.

    Args:
        task_id: Full task_id

    Returns:
        Topic identifier
    """
    parts = task_id.split("_nodes", 1)
    return parts[0] if parts else task_id

