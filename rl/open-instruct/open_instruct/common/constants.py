"""
Constants for MCP-Factory pipeline.
Pre-defined specifications that don't require LLM generation.
"""

import copy
from typing import Any, Dict, Set


# =============================================================================
# PLACEHOLDER VALUES
# =============================================================================
# Predefined placeholder values used to fill missing keys when creating 
# new records in MANIPULATION operations.
# These values are fixed and identifiable to avoid interference caused by 
# hardcoded random values.
#
# Usage scenarios:
# - CREATE operations: When a new record requires all fields but some are 
#   not provided, use placeholders to fill them.
# - UPDATE operations: Inherit values from the original record, no 
#   placeholders needed.
# - Testing and verification: These placeholder values should be 
#   recognized and handled specially.

# =============================================================================
# PRIMITIVE TYPE PLACEHOLDERS
# =============================================================================

# String type placeholder
PLACEHOLDER_STRING = "__PLACEHOLDER__"

# Integer type placeholder (using an unlikely negative number)
PLACEHOLDER_INTEGER = -999999

# Float/Number type placeholder
PLACEHOLDER_NUMBER = -999999.0

# Boolean type placeholder (False as the "empty/default" state)
PLACEHOLDER_BOOLEAN = False

# =============================================================================
# COMPOSITE TYPE PLACEHOLDERS
# =============================================================================

# Array type placeholder (empty list)
PLACEHOLDER_ARRAY: list = []

# Object type placeholder (empty dict)
PLACEHOLDER_OBJECT: dict = {}

# =============================================================================
# FORMAT-SPECIFIC PLACEHOLDERS (for string type with format)
# =============================================================================
# These are the ONLY allowed format values per dataprotocol.md Section 11.1:
#   date-time, time, date, email, hostname, ipv4, ipv6
#
# Placeholder values are aligned with:
# 1. dataprotocol.md format examples (Section 11.1, lines 659-665)
# 2. jsonschema_helpers.py DEFAULT_STRING_BY_FORMAT
#
# Values must pass format validation and be recognizable as placeholders.

# date-time: ISO 8601 with timezone "YYYY-MM-DDTHH:MM:SSZ"
# Example from dataprotocol: "1970-01-01T00:00:00Z"
PLACEHOLDER_DATETIME = "1970-01-01T00:00:00Z"

# date: "YYYY-MM-DD"
# Example from dataprotocol: "1970-01-01"
PLACEHOLDER_DATE = "1970-01-01"

# time: "HH:MM:SS" 24-hour with seconds
# Example from dataprotocol: "00:00:00"
PLACEHOLDER_TIME = "00:00:00"

# email: standard email format
# Example from dataprotocol: "any@example.com"
PLACEHOLDER_EMAIL = "any@example.com"

# hostname: domain name
# Example from dataprotocol: "example.com"
PLACEHOLDER_HOSTNAME = "example.com"

# ipv4: dotted decimal
# Example from dataprotocol: "0.0.0.0"
PLACEHOLDER_IPV4 = "0.0.0.0"

# ipv6: colon-separated hex
# Example from dataprotocol: "::1"
PLACEHOLDER_IPV6 = "::1"

# =============================================================================
# PLACEHOLDER VALUE SETS AND MAPPINGS
# =============================================================================

# Set of hashable placeholder values for quick checking
# Note: PLACEHOLDER_ARRAY and PLACEHOLDER_OBJECT are not hashable, 
# they are checked separately in is_placeholder_value()
_HASHABLE_PLACEHOLDER_VALUES: Set[Any] = {
    PLACEHOLDER_STRING,
    PLACEHOLDER_INTEGER,
    PLACEHOLDER_NUMBER,
    PLACEHOLDER_BOOLEAN,
    PLACEHOLDER_DATETIME,
    PLACEHOLDER_DATE,
    PLACEHOLDER_TIME,
    PLACEHOLDER_EMAIL,
    PLACEHOLDER_HOSTNAME,
    PLACEHOLDER_IPV4,
    PLACEHOLDER_IPV6,
}

# Mapping of JSON Schema types to their corresponding placeholder values
PLACEHOLDER_BY_TYPE: Dict[str, Any] = {
    "string": PLACEHOLDER_STRING,
    "integer": PLACEHOLDER_INTEGER,
    "number": PLACEHOLDER_NUMBER,
    "boolean": PLACEHOLDER_BOOLEAN,
    "array": PLACEHOLDER_ARRAY,
    "object": PLACEHOLDER_OBJECT,
}

# Mapping of JSON Schema formats to their corresponding placeholder values
PLACEHOLDER_BY_FORMAT: Dict[str, str] = {
    "date-time": PLACEHOLDER_DATETIME,
    "date": PLACEHOLDER_DATE,
    "time": PLACEHOLDER_TIME,
    "email": PLACEHOLDER_EMAIL,
    "hostname": PLACEHOLDER_HOSTNAME,
    "ipv4": PLACEHOLDER_IPV4,
    "ipv6": PLACEHOLDER_IPV6,
}

# =============================================================================
# ENUM PLACEHOLDERS
# =============================================================================
# Specific placeholders allowed for fields with 'enum' constraints.
# They follow the base data type of the enum.
# Note: Enum values are typically string, integer, number, or boolean.

PLACEHOLDER_ENUM_STRING = PLACEHOLDER_STRING
PLACEHOLDER_ENUM_INTEGER = PLACEHOLDER_INTEGER
PLACEHOLDER_ENUM_NUMBER = PLACEHOLDER_NUMBER
PLACEHOLDER_ENUM_BOOLEAN = PLACEHOLDER_BOOLEAN

# Set of placeholders specifically used for enums
ENUM_PLACEHOLDERS: Set[Any] = {
    PLACEHOLDER_ENUM_STRING,
    PLACEHOLDER_ENUM_INTEGER,
    PLACEHOLDER_ENUM_NUMBER,
    PLACEHOLDER_ENUM_BOOLEAN,
}


def is_enum_placeholder(value: Any) -> bool:
    """
    Check if a value is an allowed placeholder for enum fields.
    """
    return value in ENUM_PLACEHOLDERS


def get_placeholder_for_enum(schema: Dict[str, Any]) -> Any:
    """
    Get the appropriate placeholder for an enum field based on its type.
    
    Supports: string, integer, number, boolean
    """
    if not isinstance(schema, dict):
        return PLACEHOLDER_ENUM_STRING
    schema_type = schema.get("type")
    if schema_type == "integer":
        return PLACEHOLDER_ENUM_INTEGER
    if schema_type == "number":
        return PLACEHOLDER_ENUM_NUMBER
    if schema_type == "boolean":
        return PLACEHOLDER_ENUM_BOOLEAN
    return PLACEHOLDER_ENUM_STRING


def is_placeholder_value(value: Any) -> bool:
    """
    Check if a value is a predefined placeholder value.
    
    Args:
        value: The value to check.
        
    Returns:
        True if it is a placeholder value, False otherwise.
    """
    # Check hashable values first
    try:
        if value in _HASHABLE_PLACEHOLDER_VALUES:
            return True
    except TypeError:
        # Value is not hashable (e.g., list or dict)
        pass
    
    # Check for empty list placeholder
    if isinstance(value, list) and value == PLACEHOLDER_ARRAY:
        return True
    
    # Check for empty dict placeholder
    if isinstance(value, dict) and value == PLACEHOLDER_OBJECT:
        return True
    
    return False


def get_placeholder_for_schema(schema: Dict[str, Any]) -> Any:
    """
    Get the appropriate placeholder value based on the JSON Schema definition.
    
    Args:
        schema: JSON Schema definition, containing type and optional format fields.
        
    Returns:
        The corresponding placeholder value, defaults to PLACEHOLDER_STRING if uncertain.
    """
    if not isinstance(schema, dict):
        return PLACEHOLDER_STRING
    
    schema_type = schema.get("type")
    schema_format = schema.get("format")
    
    # If format exists, prioritize the placeholder corresponding to the format
    if schema_format and schema_format in PLACEHOLDER_BY_FORMAT:
        return PLACEHOLDER_BY_FORMAT[schema_format]
    
    # Otherwise use the placeholder corresponding to the type
    if schema_type and schema_type in PLACEHOLDER_BY_TYPE:
        return PLACEHOLDER_BY_TYPE[schema_type]
    
    # Default to string placeholder
    return PLACEHOLDER_STRING


def get_all_placeholder_values() -> Set[Any]:
    """
    Get a set of all hashable placeholder values.
    
    Note: PLACEHOLDER_ARRAY ([]) and PLACEHOLDER_OBJECT ({}) are not included
    as they are not hashable. Use is_placeholder_value() for complete checking.
    
    Returns:
        A set containing all hashable placeholder values.
    """
    return _HASHABLE_PLACEHOLDER_VALUES.copy()


# =============================================================================
# SUBMIT_ANSWER CONSTANTS
# =============================================================================

# Pre-defined subtask text for SUBMIT_ANSWER nodes
SUBMIT_ANSWER_SUBTASK_TEXT = "Summarize and output the final response to the user"

# Pre-defined Call Doc for SUBMIT_ANSWER nodes (clean format, no metadata)
SUBMIT_ANSWER_CALL_DOC = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": 'Submit the final answer to the user query. The answer must be a JSON-formatted string that contains only the information required to precisely and completely address the user’s request, extracted from the previous tool calls. The output must not include any meta-level fields (e.g., "summary", "reason", "analysis").',
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A JSON-formatted string containing the precise and complete response to the user's query.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

# Pre-defined ToolDoc for SUBMIT_ANSWER (Call Doc + SimuDoc)
# NOTE: No failure_modes - param validation handled by TaskEnvironment, no DB operations
SUBMIT_ANSWER_TOOL_DOC = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": 'Submit the final answer to the user query. The answer must be a JSON-formatted string that contains only the information required to precisely and completely address the user’s request, extracted from the previous tool calls. The output must not include any meta-level fields (e.g., "summary", "reason", "analysis").',
        "parameters": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A JSON-formatted string containing the precise andcomplete response to the user's query.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
    "simu_doc": {
        "node_type": "SUBMIT_ANSWER",
        "output_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }
}

# Pre-defined Script for SUBMIT_ANSWER (no DB operations, no param validation)
SUBMIT_ANSWER_SCRIPT = '''"""
Tool: submit_answer
Pre-defined tool for final response output.
"""
import argparse
import json
from typing import Any, Dict, Tuple

SUBDB: Dict[str, Any] = {}

def submit_answer(answer: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    global SUBDB
    new_subdb = dict(SUBDB)
    return True, {"answer": answer}, new_subdb

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdbdict", required=True, help="JSON string of subdb dict")
    parser.add_argument("--inputdict", required=True, help="JSON string of tool input dict")
    args = parser.parse_args()

    global SUBDB
    SUBDB = json.loads(args.subdbdict)
    tool_function_input = json.loads(args.inputdict)

    status, data, new_subdb = submit_answer(**tool_function_input)
    print(
        json.dumps(
            {
                "status": status,
                "data": data,
                "new_subdb": new_subdb,
            },
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()
'''


def get_submit_answer_call_doc() -> dict:
    """
    Get a copy of the SUBMIT_ANSWER Call Doc (clean format, no metadata).

    Returns:
        A deep copy of the SUBMIT_ANSWER_CALL_DOC
    """
    return copy.deepcopy(SUBMIT_ANSWER_CALL_DOC)


def get_submit_answer_tool_doc() -> dict:
    """
    Get a copy of the SUBMIT_ANSWER ToolDoc (Call Doc + SimuDoc).

    Returns:
        A deep copy of the SUBMIT_ANSWER_TOOL_DOC
    """
    return copy.deepcopy(SUBMIT_ANSWER_TOOL_DOC)


def get_submit_answer_script() -> str:
    """
    Get the SUBMIT_ANSWER script.

    Returns:
        The pre-defined script as a string
    """
    return SUBMIT_ANSWER_SCRIPT
