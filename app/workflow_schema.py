"""Data contracts: supported schema dialect, value validation and compatibility."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import ipaddress
import json
import math
import re
from typing import Any, Dict, List, Optional


# A deliberately bounded JSON Schema dialect. Unsupported keywords fail at
# registration rather than silently weakening a published contract.
SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
               "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
               "minProperties", "maxProperties", "pattern", "format", "default", "title", "description",
               "x-semantic-type", "x-sensitivity"}


def check_schema(schema, path="$"):
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema must be an object")
    unknown = set(schema) - SCHEMA_KEYS
    if unknown:
        raise ValueError(f"{path}: unsupported schema keywords: {sorted(unknown)}")
    types = schema.get("type", [])
    types = types if isinstance(types, list) else [types]
    if any(t not in {"object", "array", "string", "integer", "number", "boolean", "null"} for t in types):
        raise ValueError(f"{path}: unknown JSON type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not isinstance(schema.get("required", []), list):
        raise ValueError(f"{path}: invalid properties or required")
    if any(not isinstance(v, str) for v in schema.get("required", [])):
        raise ValueError(f"{path}: required names must be strings")
    for key, child in properties.items():
        check_schema(child, f"{path}.{key}")
    if "items" in schema:
        check_schema(schema["items"], path + "[]")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError(f"{path}: additionalProperties must be boolean")
    if "format" in schema and schema["format"] not in {"ipv4", "ipv6", "date-time"}:
        raise ValueError(f"{path}: unsupported format")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error) as exc:
            raise ValueError(f"{path}: invalid pattern") from exc
    for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], (int, float)) or not math.isfinite(schema[key])):
            raise ValueError(f"{path}: invalid {key}")
        if key in schema and key not in {"minimum", "maximum"} and (schema[key] < 0 or int(schema[key]) != schema[key]):
            raise ValueError(f"{path}: invalid {key}")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}: enum must be a non-empty array")
    if schema.get("x-sensitivity", "internal") not in {"public", "internal", "confidential", "restricted"}:
        raise ValueError(f"{path}: invalid sensitivity")
    if "default" in schema and validate_json_schema(schema["default"], schema):
        raise ValueError(f"{path}: default violates contract")


def version_key(version):
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError("version must be an exact major.minor.patch version")
    return tuple(int(part) for part in version.split("."))


def manifest_digest(manifest):
    payload = manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.dict()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_json_schema(value: Any, schema: Dict[str, Any], *, path: str = "$", max_errors: int = 20) -> List[str]:
    """Validate the common JSON Schema vocabulary used by component contracts."""

    errors: List[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value does not match const")
    if "enum" in schema and value not in schema.get("enum", []):
        errors.append(f"{path}: value is not one of enum values")

    alternatives = schema.get("oneOf") or schema.get("anyOf")
    if alternatives:
        matches = sum(
            not validate_json_schema(value, option, path=path, max_errors=1)
            for option in alternatives
            if isinstance(option, dict)
        )
        if ("oneOf" in schema and matches != 1) or ("anyOf" in schema and matches < 1):
            errors.append(f"{path}: value does not match the schema alternatives")
        return errors[:max_errors]

    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(_json_type_matches(value, item) for item in expected_types):
            errors.append(f"{path}: expected type {expected!r}")
            return errors[:max_errors]

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}: properties must be an object")
            return errors[:max_errors]
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"{path}.{field}: required property is missing")
        if schema.get("additionalProperties") is False:
            for field in value:
                if field not in properties:
                    errors.append(f"{path}.{field}: additional property is not allowed")
        for field, field_schema in properties.items():
            if field in value and isinstance(field_schema, dict):
                errors.extend(validate_json_schema(value[field], field_schema, path=f"{path}.{field}", max_errors=max_errors))
                if len(errors) >= max_errors:
                    return errors[:max_errors]
        if "minProperties" in schema and len(value) < int(schema["minProperties"]):
            errors.append(f"{path}: has fewer than minProperties")
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            errors.append(f"{path}: exceeds maxProperties")
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: has fewer than minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: exceeds maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_json_schema(item, item_schema, path=f"{path}[{index}]", max_errors=max_errors))
                if len(errors) >= max_errors:
                    return errors[:max_errors]
    elif isinstance(value, str):
        if schema.get("format") in {"ipv4", "ipv6"}:
            try:
                parsed = ipaddress.ip_address(value)
                if parsed.version != (4 if schema["format"] == "ipv4" else 6):
                    raise ValueError()
            except ValueError:
                errors.append(f"{path}: invalid {schema['format']}")
        if schema.get("format") == "date-time":
            from datetime import datetime
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError()
            except ValueError:
                errors.append(f"{path}: invalid date-time")
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: exceeds maxLength")
        if "pattern" in schema:
            try:
                matched = re.search(str(schema["pattern"]), value)
            except re.error:
                matched = None
            if matched is None:
                errors.append(f"{path}: does not match pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite number")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: exceeds maximum")
    return errors[:max_errors]


def _json_type_matches(value: Any, expected: Any) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(str(expected), False)


def _schema_for_port(schema: Dict[str, Any], port: str, *, output: bool) -> Optional[Dict[str, Any]]:
    if not isinstance(schema, dict) or not schema:
        return None
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(properties.get(port), dict):
        return properties[port]
    if port == "output" and output:
        return schema
    return None


def _schemas_compatible(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """Conservative assignability: every source value must satisfy the target."""
    if "const" in source:
        return not validate_json_schema(source["const"], target) and _labels_compatible(source, target)
    if "enum" in source:
        return all(not validate_json_schema(value, target) for value in source["enum"]) and _labels_compatible(source, target)
    if "const" in target or "enum" in target:
        return False
    if not _labels_compatible(source, target):
        return False
    source_type = source.get("type")
    target_type = target.get("type")
    if target_type:
        source_types = source_type if isinstance(source_type, list) else [source_type]
        target_types = target_type if isinstance(target_type, list) else [target_type]
        if any(t not in target_types and not (t == "integer" and "number" in target_types) for t in source_types):
            return False
    for keyword in ("format", "pattern"):
        if keyword in target and source.get(keyword) != target[keyword]:
            return False
    for keyword in ("minimum", "minLength", "minItems", "minProperties"):
        if keyword in target and (keyword not in source or source[keyword] < target[keyword]):
            return False
    for keyword in ("maximum", "maxLength", "maxItems", "maxProperties"):
        if keyword in target and (keyword not in source or source[keyword] > target[keyword]):
            return False
    if "items" in target and not _schemas_compatible(source.get("items", {}), target["items"]):
        return False
    if target_type == "object" or "properties" in target or "required" in target:
        source_props = source.get("properties", {}) if isinstance(source.get("properties", {}), dict) else {}
        target_props = target.get("properties", {}) if isinstance(target.get("properties", {}), dict) else {}
        for field in target.get("required", []):
            if field not in source.get("required", []):
                return False
        for field, target_prop in target_props.items():
            if field in source_props and not _schemas_compatible(source_props[field], target_prop):
                return False
            if field not in source_props and source.get("additionalProperties", True) and target_prop:
                return False
        if target.get("additionalProperties") is False and (source.get("additionalProperties", True) or set(source_props) - set(target_props)):
            return False
    return True


def _labels_compatible(source, target):
    levels = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    return (not target.get("x-semantic-type") or source.get("x-semantic-type") == target["x-semantic-type"]) and levels.get(source.get("x-sensitivity", "internal"), 3) <= levels.get(target.get("x-sensitivity", "internal"), 1)


def with_defaults(value, schema):
    result = deepcopy(value)
    for key, prop in schema.get("properties", {}).items():
        if key not in result and "default" in prop:
            result[key] = deepcopy(prop["default"])
    return result
