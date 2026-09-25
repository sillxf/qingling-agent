"""Authenticated workbench endpoints; the static shell contains no secrets."""
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .auth import AuthorizationError, enforce_tenant
from .api_dependencies import principal_for, workflow_access, bind_request_identity
from .models import ComponentManifest, RunCreateRequest, WorkflowManifest
from .store import NotFoundError


class DraftRequest(BaseModel):
    workflow: WorkflowManifest
    revision: int = Field(default=0, ge=0)


class StateRequest(BaseModel):
    state: str


class ActivationRequest(BaseModel):
    id: str
    version: str
    enabled: bool = True


def install_workbench(app):
    router = APIRouter()

    def checked(call):
        try:
            return call()
        except NotFoundError as exc:
            raise HTTPException(404, "resource not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def audit(runtime, principal, action, **data):
        runtime.store.append_audit({"action": action, "user_id": principal.subject, "tenant_id": principal.tenant_id, **data})

    @router.get("/workbench", include_in_schema=False)
    def shell():
        return FileResponse(Path(__file__).parent / "static" / "workbench.html", media_type="text/html")

    @router.get("/v1/components")
    def components(request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization)
        items = request.app.state.runtime.workflow_catalog.components()
        for item in items:
            item["references"] = [w for w in item["references"] if not w["tenant_id"] or principal.is_development or w["tenant_id"] == principal.tenant_id]
        return items

    @router.post("/v1/components/register")
    def register(manifest: ComponentManifest, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin"})
        runtime = request.app.state.runtime
        result = checked(lambda: runtime.workflow_catalog.register(manifest))
        audit(runtime, principal, "component.register", component=manifest.id, version=manifest.version)
        return result

    @router.put("/v1/components/{component}/{version}/state")
    def component_state(component: str, version: str, body: StateRequest, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin"})
        runtime = request.app.state.runtime
        result = checked(lambda: runtime.workflow_catalog.component_state(component, version, body.state))
        audit(runtime, principal, "component.state", component=component, version=version, state=body.state)
        return result

    @router.get("/v1/workbench/workflows")
    def workflows(request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization)
        runtime = request.app.state.runtime
        return runtime.workflow_catalog.list_workflows(
            tenant_id=principal.tenant_id, include_all=principal.is_development,
        )

    @router.put("/v1/workbench/draft")
    def draft(body: DraftRequest, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin", "operator"})
        workflow_access(principal, body.workflow, write=True)
        runtime = request.app.state.runtime
        result = checked(lambda: runtime.workflow_catalog.save_draft(body.workflow, body.revision))
        audit(runtime, principal, "workflow.draft", workflow=body.workflow.id, version=body.workflow.version)
        return result

    @router.post("/v1/workbench/activate")
    def activate(body: ActivationRequest, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin", "operator"})
        runtime = request.app.state.runtime
        workflow = checked(lambda: runtime.store.get_workflow(body.id, body.version))
        workflow_access(principal, workflow, write=True)
        result = checked(lambda: runtime.workflow_catalog.activate(body.id, body.version, body.enabled))
        audit(runtime, principal, "workflow.activate", workflow=body.id, version=body.version, enabled=body.enabled)
        return result

    @router.post("/v1/workbench/execute/{workflow_id}/{version}", status_code=202)
    def execute(workflow_id: str, version: str, body: RunCreateRequest, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin", "operator"})
        runtime = request.app.state.runtime
        workflow = checked(lambda: runtime.workflow_catalog.resolve(workflow_id, version))
        workflow_access(principal, workflow)
        try:
            bind_request_identity(body, principal)
        except HTTPException as exc:
            raise HTTPException(403, "tenant access is not authorized") from exc
        return checked(lambda: runtime.submit(body, workflow=workflow))

    @router.get("/v1/workbench/runs/{run_id}/trace")
    def trace(run_id: str, request: Request, authorization: Optional[str] = Header(default=None)):
        principal = principal_for(request, authorization, {"admin", "operator"})
        runtime = request.app.state.runtime
        run = checked(lambda: runtime.store.get_run(run_id))
        try:
            enforce_tenant(principal, run.tenant_id)
        except AuthorizationError as exc:
            raise HTTPException(403, "tenant access is not authorized") from exc
        # Full payloads stay in the access-controlled checkpoint store. The
        # trace view exposes state, schema keys and metrics, not raw secrets.
        records = run.execution_state.get("workflow", {}).get("nodes", {})
        return {"run_id": run.id, "status": run.status, "workflow": run.workflow_ref, "version": run.workflow_version,
            "nodes": {key: {"status": value["status"], "metrics": value.get("metrics", {}), "output_keys": list(value.get("outputs", {}))} for key, value in records.items()}}

    app.include_router(router)
