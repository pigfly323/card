#!/usr/bin/env python3
"""
Utilities for checkpoint/resume functionality across synthesis stages.

This module provides helpers to:
1. Detect which step to resume from
2. Filter already-completed items
3. Integrate seamlessly with existing pipeline logic
"""

from pathlib import Path
from typing import List, Optional, Callable, Any
import json


def detect_completed_files(
    output_dir: str,
    file_pattern: str,
    required_fields: Optional[List[str]] = None,
    validation_func: Optional[Callable[[dict], bool]] = None,
) -> set:
    """
    Detect files that have been successfully completed in an output directory.

    Args:
        output_dir: Directory to scan for completed files
        file_pattern: Glob pattern to match files (e.g., "*_gtgraph.json")
        required_fields: List of required fields that must exist in JSON (optional)
        validation_func: Custom validation function that returns True if file is valid (optional)

    Returns:
        Set of identifiers (stems without suffix) that are completed
    """
    completed = set()
    output_path = Path(output_dir)

    if not output_path.exists():
        return completed

    for file_path in output_path.glob(file_pattern):
        try:
            # For JSON files, optionally validate contents
            if file_pattern.endswith('.json'):
                data = json.loads(file_path.read_text(encoding='utf-8'))
                
                # Check required fields
                if required_fields:
                    if not all(field in data for field in required_fields):
                        continue
                
                # Custom validation
                if validation_func and not validation_func(data):
                    continue
            
            # Extract identifier from filename
            # Remove suffix from pattern to get the identifier
            suffix = file_pattern.replace('*', '')
            identifier = file_path.stem
            if identifier.endswith(suffix.replace('.json', '')):
                identifier = identifier[:-(len(suffix.replace('.json', '')))]
            
            completed.add(identifier)
        except Exception:
            # Skip invalid files
            continue

    return completed


def filter_completed_tasks(
    all_tasks: List[dict],
    task_id_field: str,
    output_dir: str,
    output_pattern: str,
    validation_func: Optional[Callable[[dict], bool]] = None,
) -> List[dict]:
    """
    Filter out tasks that have already been completed.

    Args:
        all_tasks: List of task dictionaries
        task_id_field: Field name in task dict that contains the task ID
        output_dir: Directory where output files are stored
        output_pattern: Pattern for output files (must contain {task_id} placeholder)
        validation_func: Optional validation function for output files

    Returns:
        List of tasks that need to be processed
    """
    if not all_tasks:
        return []

    output_path = Path(output_dir)
    if not output_path.exists():
        return all_tasks

    pending_tasks = []
    
    for task in all_tasks:
        task_id = task.get(task_id_field)
        if not task_id:
            # If no task_id, always process
            pending_tasks.append(task)
            continue
        
        # Check if output file exists
        output_file = output_path / output_pattern.format(task_id=task_id)
        
        if not output_file.exists():
            pending_tasks.append(task)
            continue
        
        # Validate output file if validation function provided
        if validation_func:
            try:
                data = json.loads(output_file.read_text(encoding='utf-8'))
                if not validation_func(data):
                    pending_tasks.append(task)
                    continue
            except Exception:
                # Invalid file, need to reprocess
                pending_tasks.append(task)
                continue
        
        # File exists and is valid, skip this task

    return pending_tasks


def detect_last_completed_step(
    domain_id: str,
    topic_id: Optional[str],
    config,
    stage_config,
    step_checks: List[dict],
) -> Optional[int]:
    """
    Detect the last successfully completed step in a synthesis stage.

    Args:
        domain_id: Domain identifier
        topic_id: Topic identifier (optional)
        config: Config object
        stage_config: Stage configuration (e.g., config.env_synthesis or config.data_synthesis)
        step_checks: List of dicts with keys:
            - step_num: int
            - output_dir_template: str (template with {domain_id}, {topic_id} placeholders)
            - file_pattern: str (glob pattern for output files)
            - min_files: int (minimum number of files required)

    Returns:
        Last completed step number, or None if no steps completed
    """
    last_completed = None

    for check in sorted(step_checks, key=lambda x: x['step_num']):
        step_num = check['step_num']
        output_dir_template = check['output_dir_template']
        file_pattern = check['file_pattern']
        min_files = check.get('min_files', 1)

        # Format output directory
        output_dir = config.format_path(
            output_dir_template,
            domain_id=domain_id,
            topic_id=topic_id or "",
        )

        output_path = Path(output_dir)
        if not output_path.exists():
            break

        # Count matching files
        matching_files = list(output_path.glob(file_pattern))
        if len(matching_files) < min_files:
            break

        # This step appears to be completed
        last_completed = step_num

    return last_completed


def validate_gtgraph(gtgraph_data: dict) -> bool:
    """Validate that a GTGraph file is successfully completed."""
    return gtgraph_data.get("completed", "failure") == "success"


def validate_tscg(tscg_data: dict) -> bool:
    """Validate that a TSCG file is valid."""
    return bool(tscg_data.get("task_id") and tscg_data.get("nodes"))


def validate_rcsg(rcsg_data: dict) -> bool:
    """Validate that an RCSG file is valid and not marked as failed."""
    if rcsg_data.get("generation_status", {}).get("failed"):
        return False
    return bool(rcsg_data.get("task_id") and rcsg_data.get("nodes"))
