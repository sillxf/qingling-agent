"""Compatibility exports. Implementations live in the modules named below.

New code imports component_registry, workflow_schema or workflow_validation
according to its responsibility; existing integrations may keep this module.
"""
from .component_registry import (
    ComponentContext, ComponentDefinition, ComponentHandler,
    ComponentNotFound, ComponentRegistry,
)
from .workflow_schema import (
    SCHEMA_KEYS, check_schema, manifest_digest, validate_json_schema,
    version_key, with_defaults,
    _schemas_compatible, _schema_for_port, _labels_compatible, _json_type_matches,
)
from .workflow_validation import (
    execution_order, validate_workflow, validate_workflow_contracts,
)
