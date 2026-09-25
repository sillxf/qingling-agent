"""Installed component definitions and the single validated invocation boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .models import ComponentManifest, ComponentResult
from .workflow_schema import check_schema, validate_json_schema, version_key


@dataclass
class ComponentContext:
    """The data and execution metadata passed to one Component."""

    node_id: str
    component: str
    run: Any
    profile: Any
    inputs: Dict[str, Any]
    outputs: Dict[str, Dict[str, Any]]
    config: Dict[str, Any]
    version: str = "1.0.0"
    deadline_monotonic: Optional[float] = None
    cancel_event: Any = None

    @property
    def run_id(self):
        return self.run.id

    @property
    def tenant_id(self):
        return self.run.tenant_id

    @property
    def user_id(self):
        return self.run.user_id

    @property
    def roles(self):
        return list(self.run.roles)

    @property
    def correlation_id(self):
        return self.run.correlation_id

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TimeoutError("component cancelled")
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise TimeoutError("component deadline exceeded")


ComponentHandler = Callable[[ComponentContext], Dict[str, Any] | ComponentResult]


@dataclass(frozen=True)
class ComponentDefinition:
    name: str
    handler: ComponentHandler
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    manifest: ComponentManifest


class ComponentNotFound(ValueError):
    pass


class ComponentRegistry:
    """Registry used by the generic Workflow executor."""

    def __init__(self) -> None:
        self._definitions = {}
        self._lock = threading.RLock()
        self.disabled = set()

    def register(
        self,
        name: str,
        handler: ComponentHandler,
        *,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        manifest: Optional[ComponentManifest] = None,
    ) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("component name cannot be empty")
        if not callable(handler):
            raise TypeError("component handler must be callable")
        manifest = deepcopy(manifest) if manifest else ComponentManifest(id=normalized, version=version, input_schema=input_schema or {}, output_schema=output_schema or {})
        if manifest.id != normalized:
            raise ValueError("component manifest id does not match registration")
        version_key(manifest.version)
        for schema in (manifest.input_schema, manifest.output_schema, manifest.config_schema):
            check_schema(schema)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", normalized):
            raise ValueError("invalid component id")
        key = (normalized, manifest.version)
        with self._lock:
            if key in self._definitions:
                raise ValueError(f"component is already registered: {normalized}@{manifest.version}")
            self._definitions[key] = ComponentDefinition(normalized, handler, deepcopy(manifest.input_schema), deepcopy(manifest.output_schema), manifest)

    def has(self, name: str, version: str = "1.0.0") -> bool:
        return (str(name).strip(), version) in self._definitions

    def definition(self, name: str, version: str = "1.0.0") -> ComponentDefinition:
        try:
            value = self._definitions[(str(name).strip(), version)]
            return ComponentDefinition(value.name, value.handler, deepcopy(value.input_schema), deepcopy(value.output_schema), deepcopy(value.manifest))
        except KeyError as exc:
            raise ComponentNotFound(f"component is not registered: {name}@{version}") from exc

    def list_manifests(self):
        with self._lock:
            return [deepcopy(item.manifest) for item in self._definitions.values()]

    def validate_inputs(self, component: str, inputs: Dict[str, Any], override: Optional[Dict[str, Any]] = None, version: str = "1.0.0") -> None:
        schema = self.definition(component, version).input_schema
        errors = validate_json_schema(inputs, schema) if schema else []
        if override:
            errors.extend(validate_json_schema(inputs, override))
        if errors:
            raise ValueError(f"component {component!r} input contract failed: {'; '.join(errors[:3])}")

    def validate_outputs(self, component: str, outputs: Dict[str, Any], version: str = "1.0.0") -> None:
        schema = self.definition(component, version).output_schema
        errors = validate_json_schema(outputs, schema) if schema else []
        if errors:
            raise ValueError(f"component {component!r} output contract failed: {'; '.join(errors[:3])}")

    def invoke(self, context: ComponentContext) -> Dict[str, Any]:
        result = self.execute(context)
        if result.status == "failed":
            raise ValueError(result.error or "component failed")
        return result.outputs

    def execute(self, context: ComponentContext) -> ComponentResult:
        definition = self.definition(context.component, context.version)
        if (context.component, context.version) in self.disabled:
            raise ValueError("component version is disabled")
        self.validate_inputs(context.component, context.inputs, version=context.version)
        errors = validate_json_schema(context.config, definition.manifest.config_schema)
        if errors:
            raise ValueError("component config contract failed: " + "; ".join(errors))
        context.check_cancelled()
        value = definition.handler(context)
        if not isinstance(value, (dict, ComponentResult)):
            raise ValueError(f"component {context.component!r} must return an object")
        result = value if isinstance(value, ComponentResult) else ComponentResult(outputs=value)
        if result.status != "failed":
            self.validate_outputs(context.component, result.outputs, context.version)
        context.check_cancelled()
        return deepcopy(result)
