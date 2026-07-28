"""
Shared helpers for JSON Schema usage across the project.

Goals:
- Centralize the JSON Schema draft choice for validation (Draft 2020-12).
- Provide a shared FormatChecker (with the project's stricter format rules).
- Provide utilities for schema validation and instance validation.
- Provide canonical non-null default strings for common string formats.

Note:
- This project standardizes on Draft 2020-12. Runtime validation always uses
  Draft202012Validator (plus the shared FormatChecker) regardless of whether a
  schema declares `$schema`.
- For deterministic prompting and artifact consistency, schema-producing
  pipelines may inject `$schema` at the schema root (see Section 11 in
  `documents/dataprotocol.md`).
"""

from __future__ import annotations

import datetime as _dt
import re
from ipaddress import ip_address as _ip_address
from typing import Any, Dict, List, Optional

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


JSON_SCHEMA_DRAFT = "2020-12"
JSON_SCHEMA_DRAFT_URI_2020_12 = "https://json-schema.org/draft/2020-12/schema"

STRING_FORMATS = (
    "date-time",
    "date",
    "time",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
)

ALLOWED_STRING_FORMATS = set(STRING_FORMATS)

DEFAULT_STRING_BY_FORMAT: Dict[str, str] = {
    "date-time": "1970-01-01T00:00:00Z",
    "date": "1970-01-01",
    "time": "00:00:00",
    "email": "any@example.com",
    "hostname": "example.com",
    "ipv4": "0.0.0.0",
    "ipv6": "::1",
}

# Simplified RFC1123-ish hostname (labels 1..63, total <=253, letters/digits/hyphen, no leading/trailing hyphen).
_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


FORMAT_CHECKER = jsonschema.FormatChecker()


@FORMAT_CHECKER.checks("date-time")
def _check_format_date_time(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    if "T" not in value:
        return False
    try:
        s = value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(s)
        return dt.tzinfo is not None
    except Exception:
        return False


@FORMAT_CHECKER.checks("time")
def _check_format_time(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    # Strict HH:MM:SS (no timezone suffix).
    return re.fullmatch(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$", value) is not None

@FORMAT_CHECKER.checks("hostname")
def _check_format_hostname(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    if not value or len(value) > 253:
        return False
    labels = value.split(".")
    if not labels:
        return False
    return all(_HOST_LABEL_RE.fullmatch(label or "") is not None for label in labels)


@FORMAT_CHECKER.checks("ipv4")
def _check_format_ipv4(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    try:
        ip = _ip_address(value)
        return ip.version == 4
    except Exception:
        return False


@FORMAT_CHECKER.checks("ipv6")
def _check_format_ipv6(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    try:
        ip = _ip_address(value)
        return ip.version == 6
    except Exception:
        return False


def check_schema(schema: Any) -> None:
    """Raise SchemaError if `schema` is not a valid Draft 2020-12 JSON Schema."""
    Draft202012Validator.check_schema(schema)


def schema_error(schema: Any) -> Optional[str]:
    """Return an error string if schema is invalid, else None."""
    try:
        check_schema(schema)
        return None
    except Exception as e:
        return str(e)


# =============================================================================
# PLACEHOLDER-AWARE SCHEMA AUGMENTATION
# =============================================================================
# Used to support placeholder values during data validation.
# When the data contains a placeholder, the placeholder will be automatically 
# added to the enum in the schema to avoid validation failure.

def _get_placeholder_for_schema_type(schema: Dict[str, Any]) -> Optional[Any]:
    """
    Get the corresponding placeholder value based on the schema's type and format.
    
    Supports all JSON Schema types:
    - string (with optional format)
    - integer
    - number
    - boolean
    - array
    - object
    
    Args:
        schema: JSON Schema definition.
        
    Returns:
        The corresponding placeholder value, or None if it cannot be determined.
    """
    # Lazy import to avoid circular dependencies
    from src.common.constants import (
        PLACEHOLDER_STRING,
        PLACEHOLDER_INTEGER,
        PLACEHOLDER_NUMBER,
        PLACEHOLDER_BOOLEAN,
        PLACEHOLDER_ARRAY,
        PLACEHOLDER_OBJECT,
        PLACEHOLDER_BY_FORMAT,
    )
    
    if not isinstance(schema, dict):
        return None
    
    schema_type = schema.get("type")
    schema_format = schema.get("format")
    
    # If format exists, prioritize the placeholder corresponding to the format
    if schema_format and schema_format in PLACEHOLDER_BY_FORMAT:
        return PLACEHOLDER_BY_FORMAT[schema_format]
    
    # Map type to placeholder
    if schema_type == "string":
        return PLACEHOLDER_STRING
    elif schema_type == "integer":
        return PLACEHOLDER_INTEGER
    elif schema_type == "number":
        return PLACEHOLDER_NUMBER
    elif schema_type == "boolean":
        return PLACEHOLDER_BOOLEAN
    elif schema_type == "array":
        return PLACEHOLDER_ARRAY
    elif schema_type == "object":
        return PLACEHOLDER_OBJECT
    
    return None


def augment_schema_with_placeholders(schema: Any) -> Any:
    """
    Recursively process the schema and add placeholder values to the enum.
    
    This function deep copies the schema and modifies it to ensure that all 
    fields with enum constraints include their corresponding placeholder value. 
    This prevents validation failure when the data contains a placeholder.
    
    Args:
        schema: Original JSON Schema.
        
    Returns:
        Augmented JSON Schema (deep copy).
    """
    import copy
    from src.common.constants import (
        PLACEHOLDER_STRING,
        PLACEHOLDER_INTEGER,
        PLACEHOLDER_NUMBER,
        get_placeholder_for_enum,
    )
    
    if not isinstance(schema, dict):
        return schema
    
    # Deep copy to avoid modifying the original schema
    schema = copy.deepcopy(schema)
    
    def _augment_recursive(node: Any) -> None:
        if not isinstance(node, dict):
            return
        
        # Handle enum
        if "enum" in node and isinstance(node["enum"], list):
            placeholder = get_placeholder_for_enum(node)
            if placeholder is not None and placeholder not in node["enum"]:
                node["enum"] = node["enum"] + [placeholder]
        
        # Recursively process properties
        if "properties" in node and isinstance(node["properties"], dict):
            for prop_schema in node["properties"].values():
                _augment_recursive(prop_schema)
        
        # Recursively process items
        if "items" in node:
            _augment_recursive(node["items"])
        
        # Recursively process anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node and isinstance(node[key], list):
                for item in node[key]:
                    _augment_recursive(item)
        
        # Recursively process $defs
        if "$defs" in node and isinstance(node["$defs"], dict):
            for def_schema in node["$defs"].values():
                _augment_recursive(def_schema)
    
    _augment_recursive(schema)
    return schema


def validate_instance(instance: Any, schema: Dict[str, Any]) -> None:
    """
    Validate `instance` against `schema` with the project's shared FormatChecker.
    
    This function automatically augments the schema to allow placeholder values
    in enum constraints. Placeholder values are predefined values used to fill
    missing fields in CREATE operations.

    Raises:
        SchemaError: invalid schema
        ValidationError: instance does not satisfy schema
    """
    # Automatically augment schema to support placeholder values
    augmented_schema = augment_schema_with_placeholders(schema)
    Draft202012Validator(augmented_schema, format_checker=FORMAT_CHECKER).validate(instance)


def iter_instance_errors(instance: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Collect all validation error messages (best-effort).
    
    This function automatically augments the schema to allow placeholder values
    in enum constraints. Placeholder values are predefined values used to fill
    missing fields in CREATE operations.
    """
    # Automatically augment schema to support placeholder values
    augmented_schema = augment_schema_with_placeholders(schema)
    try:
        validator = Draft202012Validator(augmented_schema, format_checker=FORMAT_CHECKER)
    except SchemaError as e:
        return [f"SchemaError: {e}"]

    errors: List[str] = []
    for err in validator.iter_errors(instance):
        errors.append(err.message)
    return errors


def string_default_conforms_format(value: Any, fmt: str) -> bool:
    """
    Strict check for a *default* value of a `type: string, format: ...` schema.

    Unlike JSON Schema validation, defaults are annotations; we enforce:
    - default must be a non-empty string
    - and must satisfy the format checker (if known)
    """
    if not isinstance(value, str) or value == "":
        return False
    if fmt not in ALLOWED_STRING_FORMATS:
        return False
    return FORMAT_CHECKER.conforms(value, fmt)
