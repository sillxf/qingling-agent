from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .observability import new_correlation_id


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for persisted domain objects."""

    return datetime.now(timezone.utc)


ExecuteEngine = Literal["workflow", "opencode", "promptchain"]
RoutePolicy = Literal["explicit", "hybrid", "auto"]
PermissionMode = Literal["allow", "ask", "deny"]
RiskLevel = Literal["low", "medium", "high", "critical"]
PlanStepKind = Literal["component", "tool", "model", "control"]
PlanStepStatus = Literal["pending", "running", "succeeded", "failed", "waiting_approval"]
RunStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "retrying",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
]


class ModelConfig(BaseModel):
    chat: str = "sec-llm"
    embedding: str = "bge-m3"
    temperature: float = 0.1
    max_tokens: int = 4096


class FewShotExample(BaseModel):
    question: str
    answer: str


class AgentProfile(BaseModel):
    id: str
    name: str
    version: str = "1.0.0"
    execute_engine: ExecuteEngine = "workflow"
    system_prompt: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)
    max_react_steps: int = 30
    max_retry: int = 3
    skills: List[str] = Field(default_factory=list)
    mcp_servers: List[str] = Field(default_factory=list)
    subagents: List[str] = Field(default_factory=list)
    permissions: Dict[str, PermissionMode] = Field(default_factory=dict)
    few_shots: List[FewShotExample] = Field(default_factory=list)
    risk_policy: str = "approval_required"
    workflow_ref: Optional[str] = None
    # ``explicit`` preserves the original profile-driven execution behavior.
    route_policy: RoutePolicy = "explicit"
    enabled: bool = True


class StructuredIntent(BaseModel):
    """Normalized intent used by planning and routing."""

    task: str
    params: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    expected_output: str = "answer"
    confidence: float = 0.0
    source: str = "rule"
    needs_clarification: bool = False


class PlanStep(BaseModel):
    step_id: str
    objective: str
    kind: PlanStepKind = "component"
    component: str = ""
    tool_name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    success_criteria: str = "步骤输出满足预期"
    status: PlanStepStatus = "pending"


class TaskPlan(BaseModel):
    plan_id: str
    goal: str
    strategy: str = "deterministic_template"
    steps: List[PlanStep] = Field(default_factory=list)
    max_steps: int = 12
    source: str = "template"
    status: str = "draft"


class PlanValidation(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    approval_required: List[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    engine: ExecuteEngine
    profile_id: str
    profile_version: str
    workflow_ref: Optional[str] = None
    workflow_version: Optional[str] = None
    reason: str = ""
    source: str = "profile"


class DecisionSnapshot(BaseModel):
    decision_id: str
    created_at: datetime = Field(default_factory=utc_now)
    intent: StructuredIntent
    plan: TaskPlan
    validation: PlanValidation
    route: RouteDecision
    plan_hash: str = ""


class PortRef(BaseModel):
    node: str
    port: str


class WorkflowNode(BaseModel):
    id: str
    component: str
    version: str = "1.0.0"
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    bindings: Dict[str, Any] = Field(default_factory=dict)
    join: Literal["all", "any"] = "all"
    on_error: Literal["fail", "skip", "route"] = "fail"
    position: Dict[str, float] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    from_ref: PortRef = Field(..., alias="from")
    to_ref: PortRef = Field(..., alias="to")

    if hasattr(BaseModel, "model_validate"):
        model_config = {"populate_by_name": True}
    else:
        class Config:
            allow_population_by_field_name = True


class WorkflowManifest(BaseModel):
    id: str
    version: str = "1.0.0"
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge] = Field(default_factory=list)
    entry: str
    exit: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: Optional[str] = None
    description: str = ""
    max_parallel: int = Field(default=4, ge=1, le=16)
    max_nodes: int = Field(default=100, ge=1, le=500)
    max_output_bytes: int = Field(default=1_000_000, ge=1024, le=10_000_000)


class ComponentExecution(BaseModel):
    timeout_seconds: float = Field(default=30, gt=0, le=600)
    max_retries: int = Field(default=0, ge=0, le=10)
    idempotent: bool = False


class ComponentSecurity(BaseModel):
    permission: Literal["read", "analyze", "mutate", "bash", "edit"] = "read"
    risk_level: RiskLevel = "low"
    tenant_scoped: bool = True
    roles: List[str] = Field(default_factory=list)
    requires_approval: bool = False
    compensation: Optional[str] = None


class ComponentManifest(BaseModel):
    id: str
    version: str = "1.0.0"
    display_name: str = ""
    description: str = ""
    kind: Literal["function", "tool", "model", "agent", "control"] = "function"
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    execution: ComponentExecution = Field(default_factory=ComponentExecution)
    security: ComponentSecurity = Field(default_factory=ComponentSecurity)
    maintainer: str = "local"
    source: str = "installed"
    dependencies: Dict[str, str] = Field(default_factory=dict)
    # References an installed executable, never an import path or uploaded code.
    implementation: Optional[str] = None


class ComponentResult(BaseModel):
    status: Literal["succeeded", "partial", "abstain", "failed"] = "succeeded"
    outputs: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Any] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    active_ports: Optional[List[str]] = None


class WorkflowValidation(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ToolContract(BaseModel):
    name: str
    description: str = ""
    permission: Literal["read", "analyze", "mutate", "bash", "edit"] = "read"
    risk_level: RiskLevel = "low"
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    reversible: bool = True
    timeout_seconds: int = 30
    max_retries: int = 0
    retry_policy: Literal["never", "transient", "idempotent", "manual"] = "never"
    requires_approval: bool = False


class PolicyDecision(BaseModel):
    id: str
    tool_name: str
    mode: PermissionMode
    allowed: bool
    requires_approval: bool = False
    reason: str
    created_at: datetime = Field(default_factory=utc_now)


class RunCreateRequest(BaseModel):
    tenant_id: str
    agent_id: str
    user_id: str = "anonymous"
    # Populated from the trusted authentication principal at the API
    # boundary.  Development mode may supply it explicitly for local tests.
    roles: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None
    mode: Literal["chat", "api", "queue"] = "api"


class ChatMessageRequest(BaseModel):
    tenant_id: str
    agent_id: str = "event-investigation"
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    message: str
    stream: bool = True
    idempotency_key: Optional[str] = None
    correlation_id: Optional[str] = None


class Run(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    roles: List[str] = Field(default_factory=list)
    agent_id: str
    agent_version: str
    status: RunStatus = "queued"
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    artifact_ids: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_category: Optional[str] = None
    error_retryable: Optional[bool] = None
    timeout_scope: Optional[str] = None
    correlation_id: str = Field(default_factory=new_correlation_id)
    deadline_at: Optional[datetime] = None
    # Decision data is server-generated and kept separate from client input.
    decision_snapshot: Optional[Dict[str, Any]] = None
    execution_engine: Optional[ExecuteEngine] = None
    workflow_ref: Optional[str] = None
    workflow_version: Optional[str] = None
    execution_state: Dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    id: str
    run_id: str
    node_id: str
    status: str
    attempt: int = 0
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class DeadLetter(BaseModel):
    id: str
    run_id: str
    reason: str
    error_code: Optional[str] = None
    error_category: Optional[str] = None
    attempts: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Compensation(BaseModel):
    id: str
    run_id: str
    status: Literal["pending", "resolved", "rejected"] = "pending"
    action: str = "inspect"
    note: Optional[str] = None
    decided_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: Optional[datetime] = None


class RunEvent(BaseModel):
    seq: int
    run_id: str
    event_type: str
    schema_version: str = "1.0"
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Approval(BaseModel):
    id: str
    run_id: str
    tenant_id: str
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    policy_decision_id: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    requested_by: str = "system"
    decided_by: Optional[str] = None
    comment: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: Optional[datetime] = None
    # Approved actions are one-time capabilities.  These fields make replay
    # state explicit and survive a SQLite restart.
    consumed_at: Optional[datetime] = None
    consumed_by_run_id: Optional[str] = None


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    decided_by: str = "operator"
    comment: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: Any
    schema_version: str = "1.0"
    code: Optional[str] = None
    category: Optional[str] = None
    retryable: Optional[bool] = None
    correlation_id: Optional[str] = None
    run_id: Optional[str] = None


class ErrorInfo(BaseModel):
    code: str
    category: str
    message: str
    retryable: bool = False
    retry_after_ms: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    schema_version: str = "1.0"
    error: ErrorInfo
    correlation_id: Optional[str] = None
    run_id: Optional[str] = None
    node_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
