"""
CallDoc + JSON Schema normalization utilities.

Targets OpenAI "function" tool format:
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": { ...JSON Schema... }
  }
}

Normalization goals (project conventions):
- Object schemas must set additionalProperties=false
- Nested object schemas may use optional keys (required ⊆ properties.keys())
- Root tool parameters may use top-level optional keys (required ⊆ properties.keys()) and MUST set defaults for optional keys
- Defaults are used to make parameters omittable (platform injects defaults before validation)
- Optional keys MUST have defaults (at all object levels). Defaults are not a no-op sentinel.
- Enum-like params MUST NOT default to neutral/no-op tokens (e.g., "all", "").
- Never emit null defaults (default:null is forbidden)
- Never emit null (no default:null, enum:null, or type unions containing "null")
- Strip legacy "Defaults to ..." phrases from descriptions
- Ban projection/field-selection parameters (e.g., "fields", "return_fields")
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from .jsonschema_helpers import (
    ALLOWED_STRING_FORMATS,
    DEFAULT_STRING_BY_FORMAT,
    JSON_SCHEMA_DRAFT_URI_2020_12,
    string_default_conforms_format,
)

_DEFAULT_EMPTY_STRING = ""

_PROJECTION_PARAM_NAMES = {
    "fields",
    "field_names",
    "return_fields",
    "include_fields",
    "exclude_fields",
    "include_metadata",
    "include_details",
    "include_detail",
    "with_details",
    "show_details",
    "return_details",
    "select_fields",
    "projection",
    "columns",
    "column_names",
}

_PROJECTION_PARAM_SUFFIXES = (
    "_fields",
    "_field_names",
    "_columns",
    "_column_names",
    "_projection",
)

_CONTROL_PARAM_NAMES = {
    "max_results",
    "max_items",
    "limit",
    "page",
    "page_size",
    "per_page",
    "offset",
    "cursor",
    "sort",
    "sort_by",
    "sort_order",
    "order",
    "order_by",
    "direction",
    "lang",
    "language",
    "locale",
    "format",
    "unit",
    "units",
    "currency",
    "timezone",
}

_SCOPE_SENSITIVE_FLAG_TOKENS = (
    "inactive",
    "archived",
    "deleted",
    "hidden",
    "disabled",
    "unavailable",
    "out_of_stock",
    "out-of-stock",
    "private",
    "restricted",
)


_DEFAULT_EXTRACT_RE = re.compile(r"(?:Defaults to|Default:?)\s+([^.;\n]+)", re.IGNORECASE)
_DEFAULT_PHRASE_RE = re.compile(r"\s*(?:Defaults to|Default:?)\s+[^.]*\.?", re.IGNORECASE)

_NEUTRAL_ENUM_TOKENS = {"", "all", "any", "none", "*"}


def _is_neutral_enum_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _NEUTRAL_ENUM_TOKENS


def _strip_default_phrase(description: Any) -> str:
    if description is None:
        return ""
    if not isinstance(description, str):
        description = str(description)
    cleaned = _DEFAULT_PHRASE_RE.sub("", description).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _extract_default_from_description(description: Any) -> Optional[Any]:
    if not isinstance(description, str) or not description:
        return None

    match = _DEFAULT_EXTRACT_RE.search(description)
    if not match:
        return None

    raw = match.group(1).strip().strip(",")
    raw = re.split(r"\s+(?:if|when|unless)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip().strip(",")

    # Strip wrapping quotes/backticks.
    if (raw.startswith(("'", '"', "`")) and raw.endswith(("'", '"', "`"))) and len(raw) >= 2:
        raw = raw[1:-1]
    lowered = raw.strip().lower()

    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    if "empty array" in lowered or lowered in ("[]",):
        return []
    if "empty object" in lowered or lowered in ("{}",):
        return {}

    if re.fullmatch(r"[+-]?\d+", raw):
        try:
            return int(raw)
        except Exception:
            pass
    if re.fullmatch(r"[+-]?\d+\.\d+", raw):
        try:
            return float(raw)
        except Exception:
            pass

    return raw


def _default_is_empty(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _looks_like_projection_param(name: str, schema: Dict[str, Any]) -> bool:
    lowered = name.strip().lower()
    if lowered in _PROJECTION_PARAM_NAMES:
        return True
    if any(lowered.endswith(suffix) for suffix in _PROJECTION_PARAM_SUFFIXES):
        return True

    # Common domain-specific projection-like names.
    # Example: a "characteristics" array meaning "which fields to return".
    if lowered in ("characteristics", "attributes"):
        schema_type = schema.get("type")
        if schema_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                if items.get("type") == "string" or "enum" in items:
                    return True

    desc = schema.get("description")
    if isinstance(desc, str) and desc:
        d = desc.lower()
        if any(k in d for k in ("field", "fields", "column", "columns", "projection", "select")) and any(
            k in d for k in ("return", "include", "only", "select", "projection")
        ):
            return True
    return False


def _is_presentation_only_param(name: str, schema: Dict[str, Any]) -> bool:
    """
    Presentation-only defaults are allowed to be non-neutral, as long as they do not
    change which records are selected/updated (e.g., language/max_results/include_details).
    Everything else is treated as scope-affecting by default.
    """
    lowered = name.strip().lower()
    if lowered in _CONTROL_PARAM_NAMES or lowered.endswith(("_limit", "_page", "_page_size")):
        return True

    schema_type = schema.get("type")
    if schema_type == "boolean" and lowered.startswith(("include_", "with_", "show_", "return_")):
        if any(tok in lowered for tok in _SCOPE_SENSITIVE_FLAG_TOKENS):
            return False
        return True

    if lowered in ("verbose", "detailed", "debug", "dry_run"):
        return True

    return False


def _control_default_for(name: str, schema: Dict[str, Any]) -> Optional[Any]:
    """
    Best-effort defaults for common presentation-only control params.
    """
    lowered = name.strip().lower()
    enum = schema.get("enum")
    if isinstance(enum, list):
        for v in enum:
            if v is not None and not _default_is_empty(v):
                return v

    schema_type = schema.get("type")
    if lowered in ("max_results", "max_items", "limit", "page_size", "per_page"):
        if schema_type in ("integer", "number"):
            return 10
    if lowered == "page" and schema_type == "integer":
        return 1
    if lowered == "offset" and schema_type == "integer":
        return 0
    if lowered in ("sort_order", "order", "direction") and schema_type == "string":
        return "asc"
    if lowered in ("language", "lang", "locale") and schema_type == "string":
        return "English"
    return None


def _coerce_default_type(schema: Dict[str, Any], default_value: Any) -> Any:
    """
    Best-effort coercion to satisfy "default JSON type MUST match schema.type exactly".
    Returns the coerced value or the original value if coercion is not safe.
    """
    schema_type = schema.get("type")
    if not isinstance(schema_type, str):
        return default_value

    if schema_type == "number":
        if isinstance(default_value, bool):
            return default_value
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            return float(default_value)
        return default_value

    if schema_type == "integer":
        if isinstance(default_value, bool):
            return default_value
        if isinstance(default_value, float) and default_value.is_integer():
            return int(default_value)
        return default_value

    if schema_type == "string":
        if default_value is None:
            return default_value
        if not isinstance(default_value, str):
            return str(default_value)
        return default_value

    return default_value


def _default_type_matches(schema: Dict[str, Any], default_value: Any) -> bool:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str):
        return True
    if schema_type == "string":
        return isinstance(default_value, str)
    if schema_type == "integer":
        return isinstance(default_value, int) and not isinstance(default_value, bool)
    if schema_type == "number":
        return isinstance(default_value, float) and not isinstance(default_value, bool)
    if schema_type == "boolean":
        return isinstance(default_value, bool)
    if schema_type == "array":
        return isinstance(default_value, list)
    if schema_type == "object":
        return isinstance(default_value, dict)
    return True


def _choose_enum_default(enum: List[Any], schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str):
        schema_type = "string"

    if schema_type == "string":
        strings = [v for v in enum if isinstance(v, str)]
        non_neutral = [v for v in strings if not _is_neutral_enum_token(v)]
        if non_neutral:
            return non_neutral[0]
        if strings:
            return strings[0]
        return "example"

    if schema_type == "integer":
        ints = [v for v in enum if isinstance(v, int) and not isinstance(v, bool)]
        if ints:
            return ints[0]
        return 0

    if schema_type == "number":
        nums = [v for v in enum if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums:
            v = nums[0]
            return float(v) if isinstance(v, int) else v
        return 0.0

    if schema_type == "boolean":
        bools = [v for v in enum if isinstance(v, bool)]
        if bools:
            return bools[0]
        return False

    return enum[0] if enum else _infer_default(schema)


def _infer_default(schema: Dict[str, Any]) -> Any:
    # Composition: pick the first usable alternative.
    for key in ("anyOf", "oneOf", "allOf"):
        alts = schema.get(key)
        if isinstance(alts, list) and alts:
            for alt in alts:
                if not isinstance(alt, dict):
                    continue
                candidate = _infer_default(alt)
                if candidate is not None and not _default_is_empty(candidate):
                    return candidate

    if "const" in schema and schema["const"] is not None:
        return schema["const"]

    enum = schema.get("enum")
    if isinstance(enum, list):
        for v in enum:
            if v is not None and not _default_is_empty(v):
                return v

    schema_type: Any = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t not in (None, "null")]
        schema_type = non_null[0] if non_null else "string"
    if not isinstance(schema_type, str):
        schema_type = "string"

    type_aliases = {
        "float": "number",
        "double": "number",
        "decimal": "number",
        "int": "integer",
        "long": "integer",
        "bool": "boolean",
        "dict": "object",
        "map": "object",
        "list": "array",
        "str": "string",
    }
    schema_type = type_aliases.get(schema_type, schema_type)

    if schema_type == "string":
        fmt = schema.get("format")
        if isinstance(fmt, str) and fmt in DEFAULT_STRING_BY_FORMAT:
            return DEFAULT_STRING_BY_FORMAT[fmt]
        return "example"

    if schema_type == "integer":
        default_int = 0
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)):
            default_int = int(minimum)
        if isinstance(maximum, (int, float)) and default_int > int(maximum):
            default_int = int(maximum)
        return default_int

    if schema_type == "number":
        default_num = 0.0
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)):
            default_num = float(minimum)
        if isinstance(maximum, (int, float)) and default_num > float(maximum):
            default_num = float(maximum)
        return default_num

    if schema_type == "boolean":
        return False

    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            items = {"type": "string"}
        item_default = _infer_default(items)
        if item_default is None or _default_is_empty(item_default):
            item_default = "example"

        min_items = schema.get("minItems")
        length = 1
        if isinstance(min_items, int) and min_items > 1:
            length = min(min_items, 3)

        if length == 1:
            return [item_default]
        if isinstance(item_default, (int, float)):
            return [item_default + i for i in range(length)]
        if isinstance(item_default, str):
            return [f"{item_default}_{i+1}" for i in range(length)]
        return [item_default for _ in range(length)]

    if schema_type == "object":
        props = schema.get("properties")
        if not isinstance(props, dict) or not props:
            return {"value": "example"}

        required = schema.get("required")
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            required = []

        keys: List[str] = [k for k in required if k in props]
        if not keys:
            keys = [sorted(props.keys())[0]]

        obj: Dict[str, Any] = {}
        for key in keys:
            subschema = props.get(key)
            if isinstance(subschema, dict):
                v = _infer_default(subschema)
            else:
                v = "example"
            if v is None or _default_is_empty(v):
                v = "example"
            obj[key] = v
        return obj

    return "example"


def normalize_json_schema(schema: Any, *, is_root_parameters: bool = False) -> Any:
    """
    Normalize a JSON Schema fragment in-place (returns a new object).
    """
    if not isinstance(schema, dict):
        return schema

    normalized: Dict[str, Any] = copy.deepcopy(schema)

    # Normalize schema composition keywords.
    for key in ("anyOf", "oneOf", "allOf"):
        alts = normalized.get(key)
        if isinstance(alts, list):
            normalized[key] = [normalize_json_schema(x) for x in alts if x is not None]
    for key in ("not", "if", "then", "else"):
        subschema = normalized.get(key)
        if isinstance(subschema, dict):
            normalized[key] = normalize_json_schema(subschema)

    # Drop null-valued keys early.
    for k in list(normalized.keys()):
        if normalized[k] is None:
            normalized.pop(k, None)

    # Project-wide forbidden JSON Schema keywords (no regex/dynamic-key objects).
    for k in ("pattern", "patternProperties", "propertyNames", "unevaluatedProperties"):
        normalized.pop(k, None)

    # Reduce schema complexity: avoid length constraints that frequently over-constrain generation.
    for k in ("minItems", "maxItems", "uniqueItems", "minLength", "maxLength"):
        normalized.pop(k, None)

    # Normalize description and strip legacy default phrases.
    if "description" in normalized:
        normalized["description"] = _strip_default_phrase(normalized.get("description"))

    # Remove nulls from enum.
    if "enum" in normalized and isinstance(normalized["enum"], list):
        normalized["enum"] = [v for v in normalized["enum"] if v is not None]
        if not normalized["enum"]:
            normalized.pop("enum", None)

    # Remove null from type unions.
    schema_type = normalized.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t not in (None, "null")]
        if len(non_null) == 1:
            normalized["type"] = non_null[0]
        else:
            normalized["type"] = non_null
    elif schema_type == "null":
        normalized["type"] = "string"

    # Normalize common type aliases emitted by LLMs.
    if isinstance(normalized.get("type"), str):
        type_aliases = {
            "float": "number",
            "double": "number",
            "decimal": "number",
            "int": "integer",
            "long": "integer",
            "bool": "boolean",
            "dict": "object",
            "map": "object",
            "list": "array",
            "str": "string",
        }
        normalized["type"] = type_aliases.get(normalized["type"], normalized["type"])

    # Union wrapper canonicalization:
    # If this node is a composition wrapper (anyOf/oneOf/allOf), it MUST NOT also behave like a concrete
    # object/array/string schema at the same level (those constraints would apply to *all* branches and
    # frequently make the schema unsatisfiable, e.g., type=object + anyOf includes string "any").
    if any(isinstance(normalized.get(k), list) and normalized.get(k) for k in ("anyOf", "oneOf")):
        for k in (
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "format",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
        ):
            normalized.pop(k, None)

    # Enforce project-wide format allow-list (and make format imply string type).
    fmt = normalized.get("format")
    if fmt is not None:
        if isinstance(fmt, str) and fmt in ALLOWED_STRING_FORMATS:
            if "type" not in normalized:
                normalized["type"] = "string"
        else:
            normalized.pop("format", None)

    # Normalize object schemas.
    if normalized.get("type") == "object" or "properties" in normalized or "additionalProperties" in normalized:
        normalized["type"] = "object"

        props = normalized.get("properties")
        if not isinstance(props, dict):
            props = {}
        required = normalized.get("required")
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            required = []

        if is_root_parameters and props:
            # Project rule: ban projection/field-selection parameters.
            for k in list(props.keys()):
                subschema = props.get(k)
                if not isinstance(k, str) or not isinstance(subschema, dict):
                    continue
                if _looks_like_projection_param(k, subschema):
                    props.pop(k, None)
            required = [k for k in required if k in props]

        if not props and not is_root_parameters:
            description = normalized.get("description")
            if not isinstance(description, str):
                description = ""
            lowered = description.lower()
            looks_like_mapping = any(
                hint in lowered
                for hint in (
                    "key-value",
                    "key value",
                    "key/value",
                    "mapping",
                    "map",
                    "dictionary",
                    "dict",
                    "pairs",
                    "where keys",
                    "键值",
                    "映射",
                    "字典",
                )
            )
            if looks_like_mapping:
                props = {
                    "entries": {
                        "type": "array",
                        "description": "List of key-value entries.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "description": "Entry key."},
                                "value": {"type": "string", "description": "Entry value (string-encoded)."},
                            },
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    }
                }
                required = ["entries"]
            else:
                props = {
                    "value": {"type": "string", "description": "Value payload."},
                }
                required = ["value"]

        # Keep required keys that exist in properties.
        prop_keys = [k for k in props.keys() if isinstance(k, str)]
        required = [k for k in required if k in prop_keys]
        required_set = {k for k in required if isinstance(k, str)}

        # Nested object schemas: optional keys are allowed (required ⊆ properties.keys()).
        # Root tool parameters: optional keys are allowed at the top-level, and MUST have defaults.

        normalized["properties"] = props
        normalized["required"] = required
        normalized["additionalProperties"] = False

        for name, subschema in list(props.items()):
            if not isinstance(subschema, dict):
                subschema = {"type": "string"}
            extracted_default = None
            if is_root_parameters and name not in required_set:
                extracted_default = _extract_default_from_description(subschema.get("description"))
            subschema = normalize_json_schema(subschema)

            # Enforce *_id parameter rules (top-level only).
            if is_root_parameters and name.endswith("_id"):
                desc = subschema.get("description")
                subschema = {"type": "string"}
                if isinstance(desc, str) and desc.strip():
                    subschema["description"] = desc
                subschema.pop("default", None)
                if name not in required_set:
                    required.append(name)
                    required_set.add(name)
                props[name] = subschema
                continue

            # Enforce: object-typed properties MUST be REQUIRED at ANY object level.
            is_object_prop = (
                subschema.get("type") == "object"
                or "properties" in subschema
                or "additionalProperties" in subschema
            )
            if is_object_prop and name not in required_set:
                required.append(name)
                required_set.add(name)

            is_required = name in required_set
            is_optional = not is_required

            # Enforce defaults:
            # - OPTIONAL keys MUST have non-null defaults (at all object levels).
            # - REQUIRED keys MUST NOT have defaults (at all object levels).
            if is_optional:
                if "default" not in subschema or subschema.get("default") is None:
                    control_default = _control_default_for(name, subschema) if is_root_parameters else None
                    if isinstance(extracted_default, str) and subschema.get("type") == "string":
                        subschema["default"] = extracted_default
                    elif control_default is not None:
                        subschema["default"] = control_default
                    else:
                        enum = subschema.get("enum")
                        if isinstance(enum, list) and enum:
                            subschema["default"] = _choose_enum_default(enum, subschema)
                        else:
                            subschema["default"] = _infer_default(subschema)
                else:
                    # Enforce: enum-like params MUST NOT default to neutral/no-op tokens.
                    enum = subschema.get("enum")
                    if subschema.get("type") == "string" and isinstance(enum, list) and enum:
                        if subschema.get("default") not in enum or _is_neutral_enum_token(subschema.get("default")):
                            subschema["default"] = _choose_enum_default(enum, subschema)
            else:
                subschema.pop("default", None)

            if "default" in subschema:
                subschema["default"] = _coerce_default_type(subschema, subschema.get("default"))
                if not _default_type_matches(subschema, subschema.get("default")):
                    enum = subschema.get("enum")
                    if isinstance(enum, list) and enum:
                        subschema["default"] = _choose_enum_default(enum, subschema)
                    else:
                        subschema["default"] = _infer_default(subschema)

            fmt = subschema.get("format")
            if subschema.get("type") == "string" and isinstance(fmt, str) and "default" in subschema:
                if not string_default_conforms_format(subschema.get("default"), fmt):
                    subschema["default"] = DEFAULT_STRING_BY_FORMAT.get(fmt, "example")

            props[name] = subschema

        normalized["properties"] = props
        normalized["required"] = required

    # Normalize array schemas.
    if normalized.get("type") == "array" or "items" in normalized:
        normalized["type"] = "array"
        items = normalized.get("items")
        if isinstance(items, dict):
            normalized["items"] = normalize_json_schema(items)
        elif isinstance(items, list):
            normalized["items"] = [normalize_json_schema(x) for x in items if x is not None]
        elif "items" not in normalized:
            # Best-effort strictness: require items schema if absent.
            normalized["items"] = {"type": "string"}

    # Ensure default is non-null if present (empty defaults are allowed by protocol).
    if "default" in normalized and normalized.get("default") is None:
        normalized["default"] = _infer_default(normalized)

    # Final pass: remove any lingering nulls.
    for k in list(normalized.keys()):
        if normalized[k] is None:
            normalized.pop(k, None)
    return normalized


def normalize_call_doc(call_doc: Any) -> Dict[str, Any]:
    """
    Normalize an OpenAI function CallDoc (type=function + function + parameters schema).
    """
    if not isinstance(call_doc, dict):
        call_doc = {}

    doc: Dict[str, Any] = copy.deepcopy(call_doc)
    doc["type"] = "function"

    func = doc.get("function")
    if not isinstance(func, dict):
        func = {}

    name = func.get("name")
    if not isinstance(name, str) or not name:
        name = "unknown_tool"

    description = func.get("description")
    if description is None:
        description = ""
    if not isinstance(description, str):
        description = str(description)

    parameters = func.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = normalize_json_schema(parameters, is_root_parameters=True)

    # Ensure parameters root is strict object schema.
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    parameters["type"] = "object"
    parameters["$schema"] = JSON_SCHEMA_DRAFT_URI_2020_12
    if not isinstance(parameters.get("properties"), dict):
        parameters["properties"] = {}
    if not isinstance(parameters.get("required"), list):
        parameters["required"] = []
    parameters["required"] = [k for k in parameters["required"] if isinstance(k, str) and k in parameters["properties"]]
    # Ensure any top-level `*_id` parameters are required (no defaults).
    prop_keys = [k for k in parameters["properties"].keys() if isinstance(k, str)]
    req_set = set(parameters["required"])
    for k in prop_keys:
        if k.endswith("_id") and k not in req_set:
            parameters["required"].append(k)
            req_set.add(k)
    # Ensure any top-level object parameters are required (no defaults).
    for k in prop_keys:
        subschema = parameters["properties"].get(k)
        if not isinstance(subschema, dict):
            continue
        is_object_param = subschema.get("type") == "object" or "properties" in subschema or "additionalProperties" in subschema
        if not is_object_param:
            continue
        subschema.pop("default", None)
        parameters["properties"][k] = subschema
        if k not in req_set:
            parameters["required"].append(k)
            req_set.add(k)
    parameters["additionalProperties"] = False

    doc["function"] = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }

    # Keep docs clean: strip common metadata keys if present.
    doc.pop("node_type", None)
    doc.pop("merged_from", None)
    return doc
