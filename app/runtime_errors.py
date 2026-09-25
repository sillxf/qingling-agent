"""Shared run-control exceptions; executors never import RuntimeService."""
from .models import Approval


class RuntimeErrorBase(RuntimeError):
    pass


class ApprovalRequired(RuntimeErrorBase):
    def __init__(self, approval: Approval) -> None:
        super().__init__("operator approval is required")
        self.approval = approval


class PolicyDenied(RuntimeErrorBase):
    pass


class RunTimeout(RuntimeErrorBase):
    def __init__(self, scope: str = "run") -> None:
        self.scope = scope
        super().__init__(f"run timed out ({scope})")


class RunCancelled(RuntimeErrorBase):
    pass


class TenantAccessDenied(RuntimeErrorBase):
    pass
