"""
TaskEnvironment - Runtime environment for task execution.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .validation import validate_params as validate_tool_params

from ..utils import load_json, extract_graph_id
from .response_format import env_error, env_error_message, format_message_value
from .tool_subprocess import run_tool_script


@dataclass
class State:
    """
    Represents the state after a tool call execution.

    This class is used by both TaskEnvironment (runtime state tracking)
    and GroundTruthGraph (expected state for validation).
    """
    db: Dict[str, Any]                              # Full db snapshot
    call: Optional[Dict[str, Any]] = None           # {"name": ..., "arguments": ...}
    response: Optional[Dict[str, Any]] = None       # {"status": bool, "message": <string>}
    terminal_output: str = ""                       # Usually empty string
    db_change: List[Dict[str, Any]] = field(default_factory=list)  # [{"type": "delete/add", "subdb_name": ..., "content": ...}]


class TaskEnvironmentError(Exception):
    """Exception raised for TaskEnvironment errors."""
    pass


def _get_record_id(record: Dict[str, Any]) -> Optional[str]:
    """
    Extract a unique identifier from a record for identity matching.
    Simplified version: only looks for common ID fields.
    """
    if not isinstance(record, dict):
        return None

    # 1. Try generic ID fields
    for id_field in ["id", "record_id"]:
        if id_field in record and record[id_field] is not None:
            return str(record[id_field])

    # 2. Try record-type-specific ID field
    record_type = record.get("record_type")
    if isinstance(record_type, str):
        type_specific_id = f"{record_type}_id"
        if type_specific_id in record and record[type_specific_id] is not None:
            return str(record[type_specific_id])

    return None


def compute_db_change(
    db_before: Dict[str, Any],
    db_after: Dict[str, Any],
    subdb_name: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Compute db_change as add/delete/modify list for a single subdb.

    Args:
        db_before: Full db snapshot before the change
        db_after: Full db snapshot after the change
        subdb_name: Name of the subdb to compute changes for

    Returns:
        List of change dicts: [
            {"type": "add", "subdb_name": ..., "content": ...},
            {"type": "delete", "subdb_name": ..., "content": ...},
            {"type": "modify", "subdb_name": ..., "content": ..., "before": ...}
        ]
    """
    if not subdb_name:
        return []

    changes = []

    # Get records before and after
    records_before_list = db_before.get(subdb_name, {}).get("records", [])
    records_after_list = db_after.get(subdb_name, {}).get("records", [])

    # Index records by their identity (ID)
    def get_indexed_records(records_list):
        by_id = {}
        others = []
        for r in records_list:
            rid = _get_record_id(r)
            if rid:
                if rid not in by_id:
                    by_id[rid] = []
                by_id[rid].append(r)
            else:
                others.append(r)
        return by_id, others

    init_by_id, init_others = get_indexed_records(records_before_list)
    final_by_id, final_others = get_indexed_records(records_after_list)

    # 1. Process records with IDs
    all_ids = set(init_by_id.keys()) | set(final_by_id.keys())
    for rid in all_ids:
        in_init = init_by_id.get(rid, [])
        in_final = final_by_id.get(rid, [])

        # If exactly one record per ID, we can check for modification
        if len(in_init) == 1 and len(in_final) == 1:
            r_init = in_init[0]
            r_final = in_final[0]
            if json.dumps(r_init, sort_keys=True) != json.dumps(r_final, sort_keys=True):
                changes.append({
                    "type": "modify",
                    "subdb_name": subdb_name,
                    "content": r_final,
                    "before": r_init
                })
        else:
            # For multiple records with same ID, treat as simple add/delete
            init_set = {json.dumps(r, sort_keys=True): r for r in in_init}
            final_set = {json.dumps(r, sort_keys=True): r for r in in_final}
            for k, r in init_set.items():
                if k not in final_set:
                    changes.append({"type": "delete", "subdb_name": subdb_name, "content": r})
            for k, r in final_set.items():
                if k not in init_set:
                    changes.append({"type": "add", "subdb_name": subdb_name, "content": r})

    # 2. Process records without IDs (simple set diff)
    init_others_set = {json.dumps(r, sort_keys=True): r for r in init_others}
    final_others_set = {json.dumps(r, sort_keys=True): r for r in final_others}
    for k, r in init_others_set.items():
        if k not in final_others_set:
            changes.append({"type": "delete", "subdb_name": subdb_name, "content": r})
    for k, r in final_others_set.items():
        if k not in init_others_set:
            changes.append({"type": "add", "subdb_name": subdb_name, "content": r})

    return changes

    return changes


def _find_task_location(task_id: str, project_root: Path, env_synthesis_root: str) -> Tuple[str, str]:
    """Find domain_id and topic_id for a task by searching json_dbs directory."""
    dbs_dir = project_root / env_synthesis_root / "json_dbs"
    if not dbs_dir.exists():
        raise TaskEnvironmentError(f"JSON DBs directory not found: {dbs_dir}")

    # Search for {task_id}_db.json
    for domain_dir in dbs_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        for topic_dir in domain_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            db_file = topic_dir / f"{task_id}_db.json"
            if db_file.exists():
                return domain_dir.name, topic_dir.name

    raise TaskEnvironmentError(f"Task not found: {task_id}")


class TaskEnvironment:
    """Runtime environment for a single task execution."""

    @staticmethod
    def _env_error(code: str, detail: Optional[str] = None) -> str:
        return env_error(code, detail)

    @classmethod
    def _env_error_message(cls, code: str, detail: Optional[str] = None) -> str:
        # Backward-compatible name: the ENV's public response now uses a string message.
        return env_error_message(code, detail)

    @staticmethod
    def _format_message_value(data: Any) -> str:
        return format_message_value(data)

    @classmethod
    def from_task_id(
        cls,
        task_id: str,
        project_root: Optional[str] = None,
        env_synthesis_root: str = "Data/env_synthesis_stage",
    ) -> "TaskEnvironment":
        """Create TaskEnvironment by auto-detecting domain_id and topic_id from task_id."""
        root = Path(project_root) if project_root else Path.cwd()
        env_root = str(env_synthesis_root or "Data/env_synthesis_stage").strip() or "Data/env_synthesis_stage"
        domain_id, topic_id = _find_task_location(task_id, root, env_root)
        return cls(task_id, domain_id, topic_id, project_root, env_synthesis_root=env_root)

    def __init__(
        self,
        task_id: str,
        domain_id: str,
        topic_id: str,
        project_root: Optional[str] = None,
        *,
        env_synthesis_root: str = "Data/env_synthesis_stage",
    ):
        self._task_id = task_id
        self._domain_id = domain_id
        self._topic_id = topic_id
        self._graph_id = extract_graph_id(task_id)
        self._root = Path(project_root) if project_root else Path.cwd()
        self._env_synthesis_root = str(env_synthesis_root or "Data/env_synthesis_stage").strip() or "Data/env_synthesis_stage"
        base = self._root / self._env_synthesis_root

        # Paths
        self._db_path = base / f"json_dbs/{domain_id}/{topic_id}/{task_id}_db.json"
        # Graph/toolset-level scripts directory (shared across tasks)
        self._tools_dir = base / f"tool_scripts/{domain_id}/{topic_id}/{self._graph_id}"
        # Graph/toolset-level tooldocs directory (shared across tasks)
        self._tooldocs_dir = base / f"tooldocs/{domain_id}/{topic_id}/{self._graph_id}"

        # Load DB and store initial state
        self._db: Dict[str, Any] = load_json(self._db_path) if self._db_path.exists() else {}
        if not isinstance(self._db, dict):
            self._db = {}
        self._initial_db_state = copy.deepcopy(self._db)

        # State tracking
        self._state: Optional[State] = None
        self._state_history: List[State] = []

        self._tool_docs: Dict[str, Dict] = {}

    def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool with given parameters and update state.

        Returns:
            {"status": bool, "message": str}
        """
        if params is None:
            params = {}
        if not isinstance(params, dict):
            params = {}

        # Capture db state before execution
        db_before = self.get_db_state()

        try:
            tool_doc = self._get_tool_doc(tool_name)
        except Exception as e:
            response = {"status": False, "message": self._env_error_message("TOOL_NOT_FOUND", str(e))}
            self._update_state(tool_name, params, response, "", db_before)
            return response

        # Validate params
        try:
            error_code, error_data = self._validate_params(tool_doc, params)
        except Exception as e:
            response = {
                "status": False,
                "message": self._env_error_message("INPUT_SCHEMA_VALIDATION_EXCEPTION", str(e)),
            }
            self._update_state(tool_name, params, response, "", db_before)
            return response
        if error_code:
            response = {
                "status": False,
                "message": self._env_error_message(
                    "INPUT_SCHEMA_VALIDATION_FAILED",
                    f"tool={tool_name!r}, error_code={error_code}, error_data={json.dumps(error_data, ensure_ascii=False)}",
                ),
            }
            self._update_state(tool_name, params, response, "", db_before)
            return response

        tool_mapping = self._db.get("tool_mapping", {})
        if not isinstance(tool_mapping, dict):
            tool_mapping = {}

        subdb_name = tool_mapping.get(tool_name)
        subdb_obj: Dict[str, Any]
        if isinstance(subdb_name, str) and subdb_name:
            raw_subdb = self._db.get(subdb_name, {})
            subdb_obj = raw_subdb if isinstance(raw_subdb, dict) else {}
        else:
            subdb_name = None
            subdb_obj = {}

        script_path = self._tools_dir / f"{tool_name}.py"
        if not script_path.exists():
            injected_script_path = self._tools_dir / f"{tool_name}_injected.py"
            if injected_script_path.exists():
                script_path = injected_script_path
        result = run_tool_script(script_path=script_path, subdb=subdb_obj, input_dict=params)

        terminal_output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

        if not result.ok or result.status is None or result.new_subdb is None:
            response = {
                "status": False,
                "message": self._env_error_message(
                    "TOOL_SUBPROCESS_FAILED",
                    result.error or "unknown_error",
                ),
            }
            self._update_state(tool_name, params, response, terminal_output, db_before)
            return response

        # Validate ToolScript payload shape:
        # - status==true: data MUST satisfy simu_doc.output_schema
        # - status==false: data MUST be {"message": "<non-empty string>"}
        if bool(result.status) is False:
            if not isinstance(result.data, dict):
                response = {
                    "status": False,
                    "message": self._env_error_message(
                        "TOOL_FAILURE_DATA_INVALID",
                        f"expected object {{\"message\": string}}, got {type(result.data).__name__}",
                    ),
                }
                self._update_state(tool_name, params, response, terminal_output, db_before)
                return response
            msg = result.data.get("message")
            if not isinstance(msg, str) or not msg.strip():
                response = {
                    "status": False,
                    "message": self._env_error_message(
                        "TOOL_FAILURE_DATA_INVALID",
                        "missing or empty 'message' field",
                    ),
                }
                self._update_state(tool_name, params, response, terminal_output, db_before)
                return response
            extra_keys = sorted([k for k in result.data.keys() if k != "message"])
            if extra_keys:
                response = {
                    "status": False,
                    "message": self._env_error_message(
                        "TOOL_FAILURE_DATA_INVALID",
                        f"unexpected keys in failure data: {extra_keys}",
                    ),
                }
                self._update_state(tool_name, params, response, terminal_output, db_before)
                return response
        else:
            output_schema = ((tool_doc.get("simu_doc") or {}).get("output_schema") or {})
            if not isinstance(output_schema, dict) or not output_schema:
                response = {
                    "status": False,
                    "message": self._env_error_message("OUTPUT_SCHEMA_MISSING", f"tool={tool_name!r}"),
                }
                self._update_state(tool_name, params, response, terminal_output, db_before)
                return response
            if not isinstance(result.data, dict):
                response = {
                    "status": False,
                    "message": self._env_error_message(
                        "TOOL_SUCCESS_DATA_INVALID",
                        f"expected object payload, got {type(result.data).__name__}",
                    ),
                }
                self._update_state(tool_name, params, response, terminal_output, db_before)
                return response
            # try:
            #     Draft202012Validator(output_schema, format_checker=FORMAT_CHECKER).validate(result.data)
            # except ValidationError as e:
            #     response = {
            #         "status": False,
            #         "message": self._env_error_message("OUTPUT_SCHEMA_VALIDATION_FAILED", e.message),
            #     }
            #     self._update_state(tool_name, params, response, terminal_output, db_before)
            #     return response
            # except SchemaError as e:
            #     response = {
            #         "status": False,
            #         "message": self._env_error_message("OUTPUT_SCHEMA_INVALID", str(e)),
            #     }
            #     self._update_state(tool_name, params, response, terminal_output, db_before)
            #     return response

        # Persist DB state update (always) when the tool is mapped to a subdb.
        if isinstance(subdb_name, str) and subdb_name:
            self._db[subdb_name] = result.new_subdb

        response = {"status": bool(result.status), "message": self._format_message_value(result.data)}
        self._update_state(tool_name, params, response, terminal_output, db_before)
        return response

    def _update_state(
        self,
        tool_name: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        terminal_output: str,
        db_before: Dict[str, Any]
    ) -> None:
        """Update state after a tool call."""
        db_after = self.get_db_state()
        tool_mapping = self._db.get("tool_mapping", {})
        subdb_name = tool_mapping.get(tool_name) if isinstance(tool_mapping, dict) else None
        db_change = self._compute_db_change(db_before, db_after, subdb_name)

        self._state = State(
            db=db_after,
            call={"name": tool_name, "arguments": params},
            response=response,
            terminal_output=terminal_output,
            db_change=db_change
        )
        self._state_history.append(self._state)

    def _compute_db_change(
        self,
        db_before: Dict[str, Any],
        db_after: Dict[str, Any],
        subdb_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Compute db_change - delegates to module-level compute_db_change."""
        return compute_db_change(db_before, db_after, subdb_name)

    def reset(self) -> None:
        """Reset DB to initial state and clear state history."""
        self._db = copy.deepcopy(self._initial_db_state)
        self._state = None
        self._state_history = []

    @property
    def state(self) -> Optional[State]:
        """Get current state (after last tool call)."""
        return self._state

    @property
    def state_history(self) -> List[State]:
        """Get copy of state history."""
        return self._state_history.copy()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def domain_id(self) -> str:
        return self._domain_id

    @property
    def topic_id(self) -> str:
        return self._topic_id

    @property
    def db(self) -> Dict[str, Any]:
        return self._db

    def get_task_info(self) -> Dict[str, Any]:
        """Get task information including user_query from TSCG file."""
        info = {
            "task_id": self._task_id,
            "domain_id": self._domain_id,
            "topic_id": self._topic_id,
            "user_query": ""
        }
        # Try to load user_query from TSCG file
        tscg_path = self._root / f"Data/data_synthesis_stage/tscgs/{self._domain_id}/{self._topic_id}/{self._task_id}_tscg.json"
        if tscg_path.exists():
            tscg = load_json(tscg_path)
            info["user_query"] = tscg.get("task", {}).get("user_query", "")
        return info

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tool docs for this task."""
        tools = []
        if not self._tooldocs_dir.exists() or not self._tools_dir.exists():
            return tools

        # Only return tools that have scripts in the task directory
        available_scripts = {p.stem for p in self._tools_dir.glob("*.py")}

        for path in self._tooldocs_dir.glob("*.json"):
            # Skip backup files
            if path.stem.endswith("_BACKUP"):
                continue
            if path.stem in available_scripts:
                tools.append(load_json(path))
        return tools

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available for this task."""
        script_exists = (self._tools_dir / f"{tool_name}.py").exists() or (
            self._tools_dir / f"{tool_name}_injected.py"
        ).exists()
        doc_exists = (self._tooldocs_dir / f"{tool_name}.json").exists() or (
            self._tooldocs_dir / f"{tool_name}_injected.json"
        ).exists()
        return script_exists and doc_exists

    def get_db_state(self) -> Dict[str, Any]:
        """
        Get current database state snapshot (no metadata).

        Returns:
            Snapshot dict without task_id/domain_id/topic_id,
            for use in State objects and state comparison.
        """
        snapshot = copy.deepcopy(self._db)
        if isinstance(snapshot, dict):
            snapshot.pop("task_id", None)
            snapshot.pop("domain_id", None)
            snapshot.pop("topic_id", None)
        return snapshot

    def _validate_params(self, tool_doc: Dict, params: Dict) -> Tuple[Optional[str], Dict]:
        """Validate params against tool_doc schema. Returns (error_code, error_data) or (None, {})."""
        return validate_tool_params(tool_doc, params)

    def _get_tool_doc(self, tool_name: str) -> Dict:
        """Load and cache tool doc."""
        if tool_name not in self._tool_docs:
            path = self._tooldocs_dir / f"{tool_name}.json"
            if not path.exists():
                injected_path = self._tooldocs_dir / f"{tool_name}_injected.json"
                if injected_path.exists():
                    path = injected_path
                else:
                    raise TaskEnvironmentError(
                        f"ToolDoc not found: {path} (or {injected_path})"
                    )
            self._tool_docs[tool_name] = load_json(path)
        return self._tool_docs[tool_name]

    def __repr__(self) -> str:
        return f"TaskEnvironment(task_id='{self._task_id}', domain_id='{self._domain_id}', topic_id='{self._topic_id}')"
