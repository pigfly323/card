"""
Parameter validation for tool calls.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from src.common.jsonschema_helpers import FORMAT_CHECKER

from .errors import MISSING_PARAM, INVALID_PARAMS, PARAM_SCHEMA_VIOLATION, PARAM_TYPE_MISMATCH, PYTHON_ERROR


def _select_schema_branch_for_defaults(schema: Dict[str, Any], instance: Any) -> Dict[str, Any]:
    """
    Best-effort selection of a branch for default injection when the schema uses combiners.

    Notes:
    - JSON Schema defaults are annotations; jsonschema does not apply them automatically.
    - Our runtime injects defaults before validation. For `anyOf`/`oneOf`/`allOf`, we pick a
      compatible-looking branch based on the current instance shape, if possible.
    """
    for combiner in ("anyOf", "oneOf", "allOf"):
        alts = schema.get(combiner)
        if not isinstance(alts, list) or not alts:
            continue

        def _is_object_schema(s: Any) -> bool:
            return isinstance(s, dict) and (s.get("type") == "object" or "properties" in s)

        def _is_array_schema(s: Any) -> bool:
            return isinstance(s, dict) and (s.get("type") == "array" or "items" in s)

        def _is_string_schema(s: Any) -> bool:
            return isinstance(s, dict) and (s.get("type") == "string" or "enum" in s or "const" in s)

        def _is_bool_schema(s: Any) -> bool:
            return isinstance(s, dict) and s.get("type") == "boolean"

        def _is_int_schema(s: Any) -> bool:
            return isinstance(s, dict) and s.get("type") == "integer"

        def _is_num_schema(s: Any) -> bool:
            return isinstance(s, dict) and s.get("type") == "number"

        preferred = None
        if isinstance(instance, dict):
            preferred = next((alt for alt in alts if _is_object_schema(alt)), None)
        elif isinstance(instance, list):
            preferred = next((alt for alt in alts if _is_array_schema(alt)), None)
        elif isinstance(instance, bool):
            preferred = next((alt for alt in alts if _is_bool_schema(alt)), None)
        elif isinstance(instance, int) and not isinstance(instance, bool):
            preferred = next((alt for alt in alts if _is_int_schema(alt) or _is_num_schema(alt)), None)
        elif isinstance(instance, float):
            preferred = next((alt for alt in alts if _is_num_schema(alt)), None)
        elif isinstance(instance, str):
            preferred = next((alt for alt in alts if _is_string_schema(alt)), None)

        if isinstance(preferred, dict):
            return preferred

        # Fallback: first dict branch.
        for alt in alts:
            if isinstance(alt, dict):
                return alt

    return schema


def _inject_defaults_recursive(instance: Any, schema: Any) -> None:
    """
    Recursively inject schema `default` values into `instance` in-place.

    Behavior:
    - If `instance` is a dict and `schema` is an object schema, for each missing property `k`:
      - if `schema.properties[k].default` exists and is non-null, set `instance[k] = deepcopy(default)`.
      - then recurse into `instance[k]` using the property schema.
    - If `instance` is a list and `schema` is an array schema, recurse into each element using `items`.
    - Does NOT create new list elements (defaults only apply via existing elements or via an array default at the parent).
    """
    if not isinstance(schema, dict):
        return

    schema = _select_schema_branch_for_defaults(schema, instance)

    is_object_schema = schema.get("type") == "object" or "properties" in schema or "additionalProperties" in schema
    if is_object_schema and isinstance(instance, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return
        for key, prop_schema in props.items():
            if not isinstance(key, str) or not key:
                continue
            if not isinstance(prop_schema, dict):
                continue
            if key not in instance:
                if "default" in prop_schema and prop_schema.get("default") is not None:
                    instance[key] = copy.deepcopy(prop_schema.get("default"))
            if key in instance:
                _inject_defaults_recursive(instance.get(key), prop_schema)
        return

    is_array_schema = schema.get("type") == "array" or "items" in schema
    if is_array_schema and isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for item in instance:
                _inject_defaults_recursive(item, items)
        elif isinstance(items, list):
            # Tuple validation style: apply element-wise when possible.
            for idx, item_schema in enumerate(items):
                if idx >= len(instance):
                    break
                if isinstance(item_schema, dict):
                    _inject_defaults_recursive(instance[idx], item_schema)
        return

def validate_params(
    tool_doc: Dict[str, Any],
    params: Dict[str, Any]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Validate parameters against tool_doc schema.

    Args:
        tool_doc: Tool document containing function schema
        params: Parameters to validate

    Returns:
        Tuple of (error_code, error_data) if validation fails,
        or (None, {}) if validation passes.

    Error codes:
        - MISSING_PARAM: Required parameter is missing
        - INVALID_PARAMS: Extra parameters not in schema
        - PARAM_TYPE_MISMATCH: Parameter type doesn't match schema
        - PARAM_SCHEMA_VIOLATION: Parameter violates non-type schema constraints (enum/pattern/format/range/etc)
    """
    schema = tool_doc.get("function", {}).get("parameters", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if not isinstance(params, dict):
        return (
            PARAM_TYPE_MISMATCH,
            {
                "param_name": "<root>",
                "expected_type": "object",
                "actual_type": _get_json_type(params),
                "actual_value": _safe_repr(params),
            },
        )

    # Check extra params not in schema
    extra = [p for p in params if p not in properties]
    if extra:
        return (INVALID_PARAMS, {"invalid": extra})

    # Apply schema defaults for missing params (project convention).
    # NOTE: This mutates `params` in-place so downstream tool scripts see the defaults.
    #
    # Default injection is recursive: for each object schema level, if a property is missing
    # and its schema defines a non-null `default`, inject it (deep-copy), then continue
    # descending into nested objects/arrays.
    _inject_defaults_recursive(params, schema)

    # Check missing required params AFTER default injection
    missing = [p for p in required if p not in params]
    if missing:
        return (MISSING_PARAM, {"missing": missing})

    try:
        validator = jsonschema.Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    except Exception as e:
        # Tool schema itself is invalid; treat as internal error.
        return (PYTHON_ERROR, {"message": f"Invalid tool parameter schema: {e}"})

    error = next(validator.iter_errors(params), None)
    if error is not None:
        return _format_schema_error(error)

    return (None, {})


def _check_type(
    param_name: str,
    value: Any,
    schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if a value matches the expected type from schema.

    Args:
        param_name: Name of the parameter (for error reporting)
        value: Value to check
        schema: JSON schema for the parameter

    Returns:
        Error dict if type mismatch, None if OK
    """
    expected_type = schema.get("type")
    if expected_type is None:
        return None

    actual_type = _get_json_type(value)

    # Handle type matching
    if not _types_match(expected_type, actual_type, value):
        return {
            "param_name": param_name,
            "expected_type": expected_type,
            "actual_type": actual_type,
            "actual_value": _safe_repr(value),
        }

    # Recursive validation for arrays
    if expected_type == "array" and isinstance(value, list):
        items_schema = schema.get("items", {})
        if items_schema:
            for i, item in enumerate(value):
                item_error = _check_type(f"{param_name}[{i}]", item, items_schema)
                if item_error:
                    return item_error

    # Recursive validation for objects
    if expected_type == "object" and isinstance(value, dict):
        obj_properties = schema.get("properties", {})
        obj_required = schema.get("required", [])

        # Check required fields
        for req_field in obj_required:
            if req_field not in value:
                return {
                    "param_name": f"{param_name}.{req_field}",
                    "expected_type": obj_properties.get(req_field, {}).get("type", "unknown"),
                    "actual_type": "missing",
                    "actual_value": None,
                }

        # Check field types
        for field_name, field_value in value.items():
            if field_name in obj_properties:
                field_error = _check_type(
                    f"{param_name}.{field_name}",
                    field_value,
                    obj_properties[field_name]
                )
                if field_error:
                    return field_error

    return None


def _format_schema_error(err: jsonschema.ValidationError) -> Tuple[str, Dict[str, Any]]:
    """
    Map a jsonschema.ValidationError into the project's public error codes/data shapes.
    """
    validator = getattr(err, "validator", None)
    path = _format_json_path(getattr(err, "path", []))

    if validator == "required" and isinstance(err.schema, dict):
        missing = []
        req = err.schema.get("required")
        inst = err.instance if isinstance(err.instance, dict) else {}
        if isinstance(req, list) and isinstance(inst, dict):
            missing = [k for k in req if isinstance(k, str) and k not in inst]
        if path:
            missing = [f"{path}.{k}" for k in missing]
        return (MISSING_PARAM, {"missing": missing})

    if validator == "additionalProperties" and isinstance(err.schema, dict):
        props = err.schema.get("properties")
        inst = err.instance if isinstance(err.instance, dict) else {}
        invalid: List[str] = []
        if isinstance(props, dict) and isinstance(inst, dict):
            bad = sorted(set(inst.keys()) - set(props.keys()))
            invalid = [f"{path}.{k}" if path else str(k) for k in bad]
        return (INVALID_PARAMS, {"invalid": invalid})

    if validator == "type" and isinstance(err.schema, dict):
        expected_type = err.schema.get("type")
        return (
            PARAM_TYPE_MISMATCH,
            {
                "param_name": path or "<root>",
                "expected_type": expected_type,
                "actual_type": _get_json_type(err.instance),
                "actual_value": _safe_repr(err.instance),
            },
        )

    # Everything else is a schema constraint violation (enum/pattern/format/range/etc).
    return (
        PARAM_SCHEMA_VIOLATION,
        {
            "param_name": path or "<root>",
            "validator": str(validator) if validator is not None else "unknown",
            "message": err.message,
            "actual_value": _safe_repr(err.instance),
        },
    )


def _format_json_path(path: Any) -> str:
    """
    Convert jsonschema error path (deque/list of keys/indices) into a dotted/bracket string.
    """
    parts: List[str] = []
    for p in list(path) if isinstance(path, (list, tuple)) else list(path):
        if isinstance(p, int):
            if not parts:
                parts.append(f"[{p}]")
            else:
                parts[-1] = f"{parts[-1]}[{p}]"
            continue
        if isinstance(p, str):
            parts.append(p)
            continue
        parts.append(str(p))
    return ".".join(parts)


def _get_json_type(value: Any) -> str:
    """
    Get the JSON type name for a Python value.

    Args:
        value: Python value

    Returns:
        JSON type name: "string", "number", "integer", "boolean", "array", "object", "null"
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _types_match(expected: str, actual: str, value: Any) -> bool:
    """
    Check if actual type matches expected type.

    Handles special cases:
    - "number" accepts both "integer" and "number" (float)
    - "integer" only accepts "integer"

    Args:
        expected: Expected JSON type
        actual: Actual JSON type
        value: The actual value (for additional checks)

    Returns:
        True if types match, False otherwise
    """
    if expected == actual:
        return True

    # "number" accepts both int and float
    if expected == "number" and actual == "integer":
        return True

    return False


def _safe_repr(value: Any, max_length: int = 100) -> str:
    """
    Get a safe string representation of a value for error messages.

    Args:
        value: Value to represent
        max_length: Maximum length of the representation

    Returns:
        String representation, truncated if necessary
    """
    try:
        repr_str = repr(value)
        if len(repr_str) > max_length:
            return repr_str[:max_length - 3] + "..."
        return repr_str
    except Exception:
        return "<unrepresentable>"
