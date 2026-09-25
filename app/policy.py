from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable

from .models import AgentProfile, PolicyDecision, ToolContract


class PolicyGate:
    """Authorizes candidate tool calls outside the model."""

    _high_risk = {"high", "critical"}

    def evaluate(
        self,
        profile: AgentProfile,
        tool: ToolContract,
        args: Dict[str, Any],
    ) -> PolicyDecision:
        mode = profile.permissions.get(tool.permission, "deny")
        errors = self._validate_args(tool.input_schema, args)
        if errors:
            return PolicyDecision(
                id=uuid.uuid4().hex,
                tool_name=tool.name,
                mode=mode,
                allowed=False,
                reason="invalid tool arguments: " + "; ".join(errors),
            )
        requires_approval = tool.requires_approval or tool.risk_level in self._high_risk
        if mode == "deny":
            return PolicyDecision(
                id=uuid.uuid4().hex,
                tool_name=tool.name,
                mode=mode,
                allowed=False,
                reason=f"permission {tool.permission!r} is denied by the agent profile",
            )
        if mode == "ask" or requires_approval:
            return PolicyDecision(
                id=uuid.uuid4().hex,
                tool_name=tool.name,
                mode="ask",
                allowed=False,
                requires_approval=True,
                reason="operator approval is required before this tool can run",
            )
        return PolicyDecision(
            id=uuid.uuid4().hex,
            tool_name=tool.name,
            mode="allow",
            allowed=True,
            reason="allowed by the agent profile and tool contract",
        )

    @staticmethod
    def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Iterable[str]:
        if not schema:
            return []
        errors = []
        for field in schema.get("required", []):
            if field not in args:
                errors.append(f"missing required field {field!r}")
        properties = schema.get("properties", {})
        for field, value in args.items():
            expected = properties.get(field, {}).get("type")
            if expected == "string" and not isinstance(value, str):
                errors.append(f"field {field!r} must be a string")
            elif expected == "object" and not isinstance(value, dict):
                errors.append(f"field {field!r} must be an object")
            elif expected == "array" and not isinstance(value, list):
                errors.append(f"field {field!r} must be an array")
        return errors
