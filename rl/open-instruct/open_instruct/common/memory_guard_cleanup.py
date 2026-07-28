import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.utils import load_json, save_json

logger = logging.getLogger(__name__)


def delete_toolset(
    config: Any,
    domain_id: str,
    topic_id: str,
    graph_id: str,
    reason: str = "OOM_FAILURE",
    diagnostics: Optional[Dict] = None,
):
    """
    Deletes all artifacts associated with a toolset/graph_id due to memory safety violation.

    Args:
        config: Config object (used to resolve base paths if possible, else defaults)
        domain_id: Domain ID
        topic_id: Topic ID
        graph_id: Graph ID (toolset ID)
        reason: Reason for deletion
        diagnostics: Optional diagnostic info
    """
    if not graph_id or graph_id == "unknown":
        logger.error(f"Cannot delete toolset with invalid graph_id: {graph_id}")
        return

    # Determine env_synthesis_root
    # Try to get from config.paths.env_synthesis, default to Data/env_synthesis_stage
    env_root = Path("Data/env_synthesis_stage")
    try:
        if hasattr(config, "paths") and hasattr(config.paths, "env_synthesis"):
            env_root = Path(str(config.paths.env_synthesis))
    except Exception:
        pass

    paths_to_delete = []

    # 1. Tool Scripts Directory
    # Data/env_synthesis_stage/tool_scripts/{domain}/{topic}/{graph_id}
    paths_to_delete.append(env_root / "tool_scripts" / domain_id / topic_id / graph_id)

    # 2. Tool Docs Directory
    # Data/env_synthesis_stage/tooldocs/{domain}/{topic}/{graph_id}
    paths_to_delete.append(env_root / "tooldocs" / domain_id / topic_id / graph_id)

    # 3. DB Plans & Schema
    # Data/env_synthesis_stage/db_plans/{domain}/{topic}/{graph_id}_DBPLAN.json
    paths_to_delete.append(
        env_root / "db_plans" / domain_id / topic_id / f"{graph_id}_DBPLAN.json"
    )
    paths_to_delete.append(
        env_root / "db_plans" / domain_id / topic_id / f"{graph_id}_DB_SCHEMA.json"
    )

    # 4. Verifiers (if enabled)
    # Data/env_synthesis_stage/verifiers/{domain}/{topic}/{graph_id}
    paths_to_delete.append(env_root / "verifiers" / domain_id / topic_id / graph_id)

    # 5. GroundTruth Graphs for this toolset (task-level artifacts)
    # Data/env_synthesis_stage/groundtruth_graphs/{domain}/{topic}/*{graph_id}*_gtgraph.json
    gtgraphs_dir = env_root / "groundtruth_graphs" / domain_id / topic_id
    if gtgraphs_dir.exists():
        for p in gtgraphs_dir.glob(f"*{graph_id}*_gtgraph.json"):
            paths_to_delete.append(p)

    # 6. Step9 Checklists for this toolset (task-level artifacts)
    # Data/env_synthesis_stage/step9_checklists/{domain}/{topic}/*{graph_id}*_checklist.json
    checklists_dir = env_root / "step9_checklists" / domain_id / topic_id
    if checklists_dir.exists():
        for p in checklists_dir.glob(f"*{graph_id}*_checklist.json"):
            paths_to_delete.append(p)

    # 7. User prompts for this toolset (task-level artifacts, if naming includes graph_id)
    # Data/env_synthesis_stage/user_prompts/{domain}/{topic}/*{graph_id}*
    user_prompts_dir = env_root / "user_prompts" / domain_id / topic_id
    if user_prompts_dir.exists():
        for p in user_prompts_dir.glob(f"*{graph_id}*"):
            paths_to_delete.append(p)

    # 8. JSON DB artifacts (best-effort, graph folder naming may vary)
    # Data/env_synthesis_stage/json_dbs/{domain}/{topic}/{graph_id}
    paths_to_delete.append(env_root / "json_dbs" / domain_id / topic_id / graph_id)

    deleted_items = []
    failed_items = []

    for path in paths_to_delete:
        if not path.exists():
            continue

        # Safety Check: path must contain the graph_id to prevent accidental deletion of parent dirs
        if graph_id not in str(path):
            logger.warning(
                f"Safety check failed: graph_id {graph_id} not in path {path}. Skipping."
            )
            failed_items.append(str(path))
            continue

        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            deleted_items.append(str(path))
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            failed_items.append(str(path))

    if not deleted_items and not failed_items:
        return

    # Log deletion
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain_id": domain_id,
        "topic_id": topic_id,
        "graph_id": graph_id,
        "reason": reason,
        "deleted_paths": deleted_items,
        "failed_paths": failed_items,
        "diagnostics": diagnostics,
    }

    _append_log(log_entry)
    logger.info(
        f"Deleted toolset {graph_id}: {len(deleted_items)} paths removed due to {reason}."
    )


def _append_log(entry: Dict):
    log_path = Path("Data/memory_cleanup_log.json")
    try:
        logs = []
        if log_path.exists():
            try:
                content = load_json(log_path)
                if isinstance(content, list):
                    logs = content
            except Exception:
                pass
        logs.append(entry)
        save_json(logs, log_path)
    except Exception as e:
        logger.error(f"Failed to write cleanup log: {e}")
