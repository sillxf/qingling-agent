from __future__ import annotations

import json
import time
from threading import Lock
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .auth import (
    Authenticator,
    AuthenticationError,
    AuthorizationError,
    build_authenticator,
    enforce_roles,
    enforce_tenant,
)
from .bootstrap import load_demo_configuration
from .models import (
    AgentProfile,
    ApprovalDecisionRequest,
    ChatMessageRequest,
    Run,
    RunCreateRequest,
    WorkflowManifest,
    Compensation,
    Checkpoint,
    DeadLetter,
)
from .observability import normalize_correlation_id
from .rag import (
    KnowledgeAccessContext,
    KnowledgeAccessDenied,
    KnowledgeDocument,
    KnowledgeEmbeddingError,
    KnowledgeError,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeNotFoundError,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeValidationError,
)
from .runtime import RunTimeout, RuntimeService, TenantAccessDenied
from .state import TERMINAL_STATES
from .store import NotFoundError
from .workbench import install_workbench
from .api_dependencies import (
    api_error, authenticate_principal, bind_request_identity, workflow_access,
)


# 422 retains its wire value; old Starlette releases lack the renamed constant.
HTTP_422_STATUS = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


def create_app(
    runtime: Optional[RuntimeService] = None,
    authenticator: Optional[Authenticator] = None,
) -> FastAPI:
    runtime_service = runtime if runtime is not None else RuntimeService()
    app_settings = runtime_service.settings
    request_authenticator = authenticator if authenticator is not None else build_authenticator(app_settings)
    if not runtime_service.store.list_profiles():
        config_dir = Path(app_settings.config_dir).expanduser().resolve() if app_settings.config_dir else None
        load_demo_configuration(runtime_service, config_dir)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            # An injected service is owned by its caller, not the HTTP adapter.
            if runtime is None:
                runtime_service.close()

    app = FastAPI(title=app_settings.app_name, version=__version__, lifespan=lifespan)
    app.state.runtime = runtime_service
    app.state.authenticator = request_authenticator

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_, exc: HTTPException):
        if isinstance(exc.detail, dict) and exc.detail.get("schema_version") == "1.0":
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(TenantAccessDenied)
    @app.exception_handler(AuthorizationError)
    async def handle_access_denied(request: Request, _exc):
        correlation_id = normalize_correlation_id(request.headers.get("X-Correlation-ID"))
        error = api_error(403, code="FORBIDDEN", category="authorization",
                          message="resource access is not authorized", correlation_id=correlation_id)
        return JSONResponse(status_code=403, content=error.detail, headers=error.headers)

    @app.exception_handler(NotFoundError)
    async def handle_resource_not_found(request: Request, _exc):
        correlation_id = normalize_correlation_id(request.headers.get("X-Correlation-ID"))
        error = api_error(404, code="NOT_FOUND", category="validation",
                          message="resource was not found", correlation_id=correlation_id)
        return JSONResponse(status_code=404, content=error.detail, headers=error.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError):
        correlation_id = normalize_correlation_id(request.headers.get("X-Correlation-ID"))
        fields = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", []))
            fields.append({"field": location, "type": error.get("type", "invalid")})
        body = {
            "schema_version": "1.0",
            "error": {
                "code": "INVALID_ARGUMENT",
                "category": "validation",
                "message": "request validation failed",
                "retryable": False,
                "details": {"fields": fields},
            },
            "correlation_id": correlation_id,
        }
        return JSONResponse(
            status_code=HTTP_422_STATUS,
            content=body,
            headers={"X-Correlation-ID": correlation_id},
        )

    def get_runtime() -> RuntimeService:
        return app.state.runtime

    def require_principal(authorization: Optional[str]) -> object:
        return authenticate_principal(app.state.authenticator, authorization)

    def require_management_role(authorization: Optional[str]) -> object:
        return authenticate_principal(app.state.authenticator, authorization, {"admin", "operator"})

    def submit_request(
        request: RunCreateRequest,
        correlation_header: Optional[str] = None,
        authorization: Optional[str] = None,
    ) -> Run:
        principal = require_principal(authorization)
        request.correlation_id = normalize_correlation_id(request.correlation_id or correlation_header)
        bind_request_identity(request, principal)
        try:
            return get_runtime().submit(request)
        except NotFoundError as exc:
            raise api_error(
                status.HTTP_404_NOT_FOUND,
                code="NOT_FOUND",
                category="validation",
                message="requested agent was not found",
                correlation_id=request.correlation_id,
            ) from exc
        except ValueError as exc:
            raise api_error(
                HTTP_422_STATUS,
                code="VALIDATION_ERROR",
                category="validation",
                message=str(exc),
                correlation_id=request.correlation_id,
            ) from exc

    def knowledge_context(
        principal: object,
        *,
        requested_tenant: Optional[str] = None,
        requested_user: Optional[str] = None,
        requested_roles: Optional[list[str]] = None,
        document_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> KnowledgeAccessContext:
        """Build the trusted RAG scope at the API boundary.

        Authenticated principals are authoritative for tenant, user and roles.
        Development authentication intentionally keeps request identity fields
        available for local ACL integration tests; document routes can infer a
        tenant from the local store when no tenant query parameter is supplied.
        """

        authenticated = bool(getattr(principal, "authenticated", False))
        principal_tenant = getattr(principal, "tenant_id", None)
        if authenticated:
            if not isinstance(principal_tenant, str) or not principal_tenant.strip():
                raise api_error(
                    status.HTTP_403_FORBIDDEN,
                    code="FORBIDDEN",
                    category="authorization",
                    message="the authenticated principal has no tenant",
                    correlation_id=correlation_id,
                )
            if requested_tenant is not None:
                try:
                    enforce_tenant(principal, requested_tenant)
                except AuthorizationError as exc:
                    raise api_error(
                        status.HTTP_403_FORBIDDEN,
                        code="FORBIDDEN",
                        category="authorization",
                        message="tenant access is not authorized",
                        correlation_id=correlation_id,
                    ) from exc
            tenant_id = principal_tenant
            user_id = getattr(principal, "subject", "anonymous")
            roles = list(getattr(principal, "roles", ()) or ())
        else:
            tenant_id = requested_tenant
            if not tenant_id and document_id:
                # Development mode has no tenant claim.  Looking up the local
                # document lets GET/DELETE remain convenient without weakening
                # the authenticated path above.
                try:
                    tenant_id = get_runtime().rag.store.get_document(document_id).tenant_id
                except KnowledgeNotFoundError:
                    tenant_id = None
            tenant_id = tenant_id or "development"
            principal_subject = getattr(principal, "subject", "development") or "development"
            # Pydantic request models materialize optional identity fields with
            # development-friendly defaults (``anonymous``/``[]``). Treat
            # those defaults as omitted so the local development principal's
            # built-in admin role remains effective; non-default values are
            # still useful for exercising ACLs locally.
            user_id = requested_user if requested_user and requested_user != "anonymous" else principal_subject
            roles = list(requested_roles) if requested_roles else list(getattr(principal, "roles", ()) or ())

        try:
            return KnowledgeAccessContext(tenant_id=tenant_id, user_id=user_id, roles=roles)
        except (TypeError, ValueError) as exc:
            raise api_error(
                HTTP_422_STATUS,
                code="VALIDATION_ERROR",
                category="validation",
                message="invalid knowledge access context",
                correlation_id=correlation_id,
            ) from exc

    def raise_knowledge_error(
        exc: Exception,
        *,
        correlation_id: Optional[str],
        document_id: Optional[str] = None,
    ) -> None:
        """Translate domain errors to the versioned REST error envelope."""

        if isinstance(exc, KnowledgeAccessDenied):
            raise api_error(
                status.HTTP_403_FORBIDDEN,
                code="FORBIDDEN",
                category="authorization",
                message="knowledge access is not authorized",
                correlation_id=correlation_id,
            ) from exc
        if isinstance(exc, KnowledgeNotFoundError):
            raise api_error(
                status.HTTP_404_NOT_FOUND,
                code="NOT_FOUND",
                category="validation",
                message="knowledge document was not found",
                correlation_id=correlation_id,
            ) from exc
        if isinstance(exc, KnowledgeEmbeddingError):
            raise api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                code="KNOWLEDGE_EMBEDDING_ERROR",
                category="knowledge",
                message="knowledge embedding service is unavailable",
                correlation_id=correlation_id,
                retryable=True,
            ) from exc
        if isinstance(exc, KnowledgeValidationError):
            raise api_error(
                HTTP_422_STATUS,
                code="VALIDATION_ERROR",
                category="validation",
                message=str(exc),
                correlation_id=correlation_id,
            ) from exc
        if isinstance(exc, KnowledgeError):
            raise api_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                code="KNOWLEDGE_ERROR",
                category="knowledge",
                message="knowledge service is temporarily unavailable",
                correlation_id=correlation_id,
                retryable=True,
            ) from exc
        raise exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": app_settings.app_name, "version": __version__, "environment": app_settings.environment}

    @app.get("/v1/agents")
    def list_agents(authorization: Optional[str] = Header(default=None)) -> list[AgentProfile]:
        require_principal(authorization)
        return get_runtime().store.list_profiles()

    @app.post("/v1/agents/profile", response_model=AgentProfile)
    def publish_profile(
        profile: AgentProfile,
        authorization: Optional[str] = Header(default=None),
    ) -> AgentProfile:
        require_management_role(authorization)
        try:
            return get_runtime().publish_profile(profile)
        except ValueError as exc:
            raise api_error(HTTP_422_STATUS, code="VALIDATION_ERROR", category="validation", message=str(exc)) from exc

    @app.post("/v1/workflows/validate")
    def validate_workflow(
        workflow: WorkflowManifest,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = require_management_role(authorization)
        workflow_access(principal, workflow)
        return get_runtime().validate_workflow(workflow)

    @app.post("/v1/workflows/publish", response_model=WorkflowManifest)
    def publish_workflow(
        workflow: WorkflowManifest,
        authorization: Optional[str] = Header(default=None),
    ) -> WorkflowManifest:
        principal = require_management_role(authorization)
        workflow_access(principal, workflow, write=True)
        try:
            published = get_runtime().publish_workflow(workflow)
            get_runtime().store.append_audit({"action": "workflow.publish", "user_id": principal.subject,
                "tenant_id": workflow.tenant_id, "workflow": workflow.id, "version": workflow.version})
            return published
        except ValueError as exc:
            raise api_error(HTTP_422_STATUS, code="VALIDATION_ERROR", category="validation", message=str(exc)) from exc

    @app.get("/v1/tools")
    def list_tools(authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        return get_runtime().list_tools()

    @app.get("/v1/skills")
    def list_skills(authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        return get_runtime().list_skills()

    @app.get("/v1/skills/{skill_id}")
    def get_skill(skill_id: str, authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        try:
            return get_runtime().get_skill(skill_id)
        except Exception as exc:
            if getattr(exc, "code", "") == "SKILL_NOT_FOUND":
                raise api_error(status.HTTP_404_NOT_FOUND, code="NOT_FOUND", category="resource", message="skill was not found") from exc
            raise api_error(HTTP_422_STATUS, code="SKILL_INVALID", category="validation", message="skill is invalid") from exc

    @app.get("/v1/mcp/servers")
    def list_mcp_servers(authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        return get_runtime().list_mcp_servers()

    @app.get("/v1/mcp/tools")
    def list_mcp_tools(server_id: Optional[str] = Query(default=None), authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        return get_runtime().list_mcp_tools([server_id] if server_id else None)

    @app.get("/v1/subagents")
    def list_subagents(authorization: Optional[str] = Header(default=None)):
        require_principal(authorization)
        return get_runtime().list_subagents()

    @app.post(
        "/v1/knowledge/documents",
        response_model=KnowledgeIngestResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_knowledge_document(
        request: KnowledgeIngestRequest,
        response: Response,
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> KnowledgeIngestResponse:
        principal = require_principal(authorization)
        correlation_id = normalize_correlation_id(correlation_header)
        # An authenticated non-admin caller may only create documents owned by
        # its verified subject.  Do not pass a caller-controlled owner through
        # to the RAG layer; administrators may intentionally assign ownership
        # for delegated imports.
        ingest_request = request
        if getattr(principal, "authenticated", False) and not bool(
            set(getattr(principal, "roles", ()) or ()).intersection({"admin", "knowledge_admin"})
        ):
            if request.owner_user_id not in (None, "", principal.subject):
                raise api_error(
                    status.HTTP_403_FORBIDDEN,
                    code="FORBIDDEN",
                    category="authorization",
                    message="document owner must match the authenticated principal",
                    correlation_id=correlation_id,
                )
            if hasattr(request, "model_copy"):
                ingest_request = request.model_copy(deep=True)
            else:
                ingest_request = request.copy(deep=True)
            ingest_request.owner_user_id = principal.subject
        context = knowledge_context(
            principal,
            requested_tenant=ingest_request.tenant_id,
            requested_user=ingest_request.owner_user_id,
            correlation_id=correlation_id,
        )
        try:
            result = get_runtime().rag.ingest_document(ingest_request, context=context)
        except KnowledgeError as exc:
            raise_knowledge_error(exc, correlation_id=correlation_id)
        response.headers["X-Correlation-ID"] = correlation_id
        return result

    @app.post("/v1/knowledge/search", response_model=KnowledgeSearchResponse)
    def search_knowledge(
        request: KnowledgeSearchRequest,
        response: Response,
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> KnowledgeSearchResponse:
        principal = require_principal(authorization)
        correlation_id = normalize_correlation_id(correlation_header)
        context = knowledge_context(
            principal,
            requested_tenant=request.tenant_id,
            requested_user=request.user_id,
            requested_roles=request.roles,
            correlation_id=correlation_id,
        )
        try:
            result = get_runtime().rag.search(request, context=context)
        except KnowledgeError as exc:
            raise_knowledge_error(exc, correlation_id=correlation_id)
        response.headers["X-Correlation-ID"] = correlation_id
        return result

    @app.get("/v1/knowledge/documents/{document_id}", response_model=KnowledgeDocument)
    def get_knowledge_document(
        document_id: str,
        response: Response,
        tenant_id: Optional[str] = Query(default=None),
        user_id: Optional[str] = Query(default=None),
        roles: Optional[list[str]] = Query(default=None),
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> KnowledgeDocument:
        principal = require_principal(authorization)
        correlation_id = normalize_correlation_id(correlation_header)
        context = knowledge_context(
            principal,
            requested_tenant=tenant_id,
            requested_user=user_id,
            requested_roles=roles,
            document_id=document_id,
            correlation_id=correlation_id,
        )
        try:
            result = get_runtime().rag.get_document(document_id, context=context)
        except KnowledgeError as exc:
            raise_knowledge_error(exc, correlation_id=correlation_id, document_id=document_id)
        response.headers["X-Correlation-ID"] = correlation_id
        return result

    @app.delete("/v1/knowledge/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_knowledge_document(
        document_id: str,
        response: Response,
        tenant_id: Optional[str] = Query(default=None),
        user_id: Optional[str] = Query(default=None),
        roles: Optional[list[str]] = Query(default=None),
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> Response:
        principal = require_principal(authorization)
        correlation_id = normalize_correlation_id(correlation_header)
        context = knowledge_context(
            principal,
            requested_tenant=tenant_id,
            requested_user=user_id,
            requested_roles=roles,
            document_id=document_id,
            correlation_id=correlation_id,
        )
        try:
            get_runtime().rag.delete_document(document_id, context=context)
        except KnowledgeError as exc:
            raise_knowledge_error(exc, correlation_id=correlation_id, document_id=document_id)
        response.status_code = status.HTTP_204_NO_CONTENT
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @app.post("/v1/runs", response_model=Run, status_code=status.HTTP_202_ACCEPTED)
    def create_run(
        request: RunCreateRequest,
        response: Response,
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> Run:
        run = submit_request(request, correlation_header, authorization)
        response.headers["X-Correlation-ID"] = run.correlation_id
        return run

    @app.post("/v1/chat/messages", response_model=Run, status_code=status.HTTP_202_ACCEPTED)
    def chat_message(
        request: ChatMessageRequest,
        response: Response,
        correlation_header: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> Run:
        run_request = RunCreateRequest(
                tenant_id=request.tenant_id,
                agent_id=request.agent_id,
                user_id=request.user_id,
                session_id=request.session_id,
                input={"message": request.message},
                idempotency_key=request.idempotency_key,
                correlation_id=request.correlation_id,
                mode="chat",
            )
        run = submit_request(run_request, correlation_header, authorization)
        response.headers["X-Correlation-ID"] = run.correlation_id
        return run

    @app.get("/v1/runs/{run_id}", response_model=Run)
    def get_run(run_id: str, authorization: Optional[str] = Header(default=None)) -> Run:
        principal = require_principal(authorization)
        try:
            return get_runtime().get_run(
                run_id,
                tenant_id=getattr(principal, "tenant_id", None),
                user_id=getattr(principal, "subject", None),
                roles=getattr(principal, "roles", None),
            )
        except NotFoundError as exc:
            raise api_error(
                status.HTTP_404_NOT_FOUND,
                code="NOT_FOUND",
                category="validation",
                message="run was not found",
                run_id=run_id,
            ) from exc
        except TenantAccessDenied as exc:
            raise api_error(
                status.HTTP_403_FORBIDDEN,
                code="FORBIDDEN",
                category="authorization",
                message="tenant access is not authorized",
                run_id=run_id,
            ) from exc

    @app.get("/v1/runs/{run_id}/events")
    def stream_events(
        run_id: str,
        last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
        authorization: Optional[str] = Header(default=None),
    ) -> StreamingResponse:
        principal = require_principal(authorization)
        try:
            run = get_runtime().get_run(
                run_id,
                tenant_id=getattr(principal, "tenant_id", None),
                user_id=getattr(principal, "subject", None),
                roles=getattr(principal, "roles", None),
            )
        except NotFoundError as exc:
            raise api_error(
                status.HTTP_404_NOT_FOUND,
                code="NOT_FOUND",
                category="validation",
                message="run was not found",
                run_id=run_id,
            ) from exc
        except TenantAccessDenied as exc:
            raise api_error(
                status.HTTP_403_FORBIDDEN,
                code="FORBIDDEN",
                category="authorization",
                message="tenant access is not authorized",
                run_id=run_id,
            ) from exc

        try:
            initial_seq = max(0, int(last_event_id or "0"))
        except ValueError:
            initial_seq = 0

        def event_stream() -> Generator[str, None, None]:
            after_seq = initial_seq
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                run, events = get_runtime().event_snapshot(
                    run_id,
                    after_seq,
                    tenant_id=getattr(principal, "tenant_id", None),
                    user_id=getattr(principal, "subject", None),
                    roles=getattr(principal, "roles", None),
                )
                for event in events:
                    after_seq = event.seq
                    yield f"id: {event.seq}\nevent: {event.event_type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                if run.status in TERMINAL_STATES or run.status == "waiting_approval":
                    break
                time.sleep(0.1)

        stream = StreamingResponse(event_stream(), media_type="text/event-stream")
        stream.headers["X-Correlation-ID"] = run.correlation_id
        return stream

    @app.post("/v1/runs/{run_id}/cancel", response_model=Run)
    def cancel_run(run_id: str, authorization: Optional[str] = Header(default=None)) -> Run:
        principal = require_principal(authorization)
        try:
            return get_runtime().cancel_for_tenant(
                run_id,
                tenant_id=getattr(principal, "tenant_id", None),
                user_id=getattr(principal, "subject", None),
                roles=getattr(principal, "roles", None),
            )
        except NotFoundError as exc:
            raise api_error(status.HTTP_404_NOT_FOUND, code="NOT_FOUND", category="validation", message="run was not found", run_id=run_id) from exc
        except ValueError as exc:
            raise api_error(status.HTTP_409_CONFLICT, code="CONFLICT", category="execution", message=str(exc), run_id=run_id) from exc
        except TenantAccessDenied as exc:
            raise api_error(status.HTTP_403_FORBIDDEN, code="FORBIDDEN", category="authorization", message="tenant access is not authorized", run_id=run_id) from exc

    @app.post("/v1/approvals/{approval_id}")
    def decide_approval(
        approval_id: str,
        request: ApprovalDecisionRequest,
        authorization: Optional[str] = Header(default=None),
    ):
        principal = require_principal(authorization)
        try:
            enforce_roles(principal, {"admin", "operator"})
        except AuthorizationError as exc:
            raise api_error(status.HTTP_403_FORBIDDEN, code="FORBIDDEN", category="authorization", message="the caller lacks the required role") from exc
        try:
            approval = get_runtime().decide_approval(
                approval_id,
                request.approved,
                getattr(principal, "subject", request.decided_by)
                if getattr(principal, "authenticated", False)
                else request.decided_by,
                request.comment,
                tenant_id=getattr(principal, "tenant_id", None),
            )
            return approval
        except NotFoundError as exc:
            raise api_error(status.HTTP_404_NOT_FOUND, code="NOT_FOUND", category="validation", message="approval was not found") from exc
        except (ValueError, RunTimeout) as exc:
            raise api_error(status.HTTP_409_CONFLICT, code="CONFLICT", category="policy", message=str(exc)) from exc
        except TenantAccessDenied as exc:
            raise api_error(status.HTTP_403_FORBIDDEN, code="FORBIDDEN", category="authorization", message="tenant access is not authorized") from exc

    @app.get("/v1/dead-letters", response_model=list[DeadLetter])
    def list_dead_letters(authorization: Optional[str] = Header(default=None)):
        principal = require_principal(authorization)
        enforce_roles(principal, {"admin", "operator"})
        return get_runtime().list_dead_letters(tenant_id=getattr(principal, "tenant_id", None))

    @app.get("/v1/runs/{run_id}/checkpoints", response_model=list[Checkpoint])
    def list_checkpoints(run_id: str, authorization: Optional[str] = Header(default=None)):
        principal = require_principal(authorization)
        get_runtime().get_run(run_id, tenant_id=getattr(principal, "tenant_id", None), user_id=getattr(principal, "subject", None), roles=getattr(principal, "roles", None))
        return get_runtime().list_checkpoints(run_id)

    @app.get("/v1/compensations", response_model=list[Compensation])
    def list_compensations(run_id: Optional[str] = Query(default=None), authorization: Optional[str] = Header(default=None)):
        principal = require_principal(authorization)
        enforce_roles(principal, {"admin", "operator"})
        return get_runtime().list_compensations(run_id, tenant_id=getattr(principal, "tenant_id", None))

    @app.post("/v1/compensations/{compensation_id}", response_model=Compensation)
    def resolve_compensation(compensation_id: str, action: str = "inspect", note: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
        principal = require_principal(authorization)
        enforce_roles(principal, {"admin", "operator"})
        return get_runtime().resolve_compensation(
            compensation_id, action=action, note=note,
            decided_by=getattr(principal, "subject", "operator"),
            tenant_id=getattr(principal, "tenant_id", None),
        )

    install_workbench(app)
    return app


# Keep the legacy ASGI target `app.main:app` without creating services merely
# because tests, demos or library callers import the factory. Only an explicit
# lookup of `app` starts the default application; normal CLI startup uses factory.
_default_app_lock = Lock()


def __getattr__(name: str):
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _default_app_lock:
        if "app" not in globals():
            globals()["app"] = create_app()
        return globals()["app"]
