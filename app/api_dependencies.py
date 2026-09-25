"""Shared HTTP identity binding and error translation.

Domain modules use auth.py and runtime_errors.py; only API routes depend on this
module. Legacy workbench detail strings remain available during API migration.
"""
from typing import Optional

from fastapi import HTTPException, status
from .auth import AuthenticationError, AuthorizationError, enforce_roles, enforce_tenant
from .models import RunCreateRequest


def api_error(
    status_code: int,
    *,
    code: str,
    category: str,
    message: str,
    correlation_id: Optional[str] = None,
    run_id: Optional[str] = None,
    retryable: bool = False,
) -> HTTPException:
    headers = {"X-Correlation-ID": correlation_id} if correlation_id else None
    return HTTPException(
        status_code=status_code,
        detail={
            "schema_version": "1.0",
            "error": {
                "code": code,
                "category": category,
                "message": message,
                "retryable": retryable,
            },
            "correlation_id": correlation_id,
            "run_id": run_id,
        },
        headers=headers,
    )


def authenticate_principal(authenticator, authorization, roles=None, *, structured=True):
    try:
        principal = authenticator.authenticate(authorization)
        if roles:
            enforce_roles(principal, set(roles))
        return principal
    except AuthenticationError as exc:
        if not structured:
            raise HTTPException(401, "authentication required") from exc
        raise api_error(401, code="UNAUTHENTICATED", category="authentication",
                        message="authentication is required") from exc
    except AuthorizationError as exc:
        if not structured:
            raise HTTPException(403, "role is not authorized") from exc
        raise api_error(403, code="FORBIDDEN", category="authorization",
                        message="the caller lacks the required role") from exc


def principal_for(request, authorization, roles=None):
    return authenticate_principal(
        request.app.state.authenticator, authorization, roles, structured=False,
    )


def bind_request_identity(request: RunCreateRequest, principal: object) -> None:
    try:
        enforce_tenant(principal, request.tenant_id)
    except AuthorizationError as exc:
        raise api_error(
            status.HTTP_403_FORBIDDEN,
            code="FORBIDDEN",
            category="authorization",
            message="tenant access is not authorized",
            correlation_id=request.correlation_id,
        ) from exc
    if getattr(principal, "authenticated", False):
        request.user_id = principal.subject
        # Roles are security-sensitive just like the subject.  Persist the
        # roles from the verified principal so later Runtime/tool calls do
        # not depend on mutable request data.
        request.roles = list(getattr(principal, "roles", ()) or ())

def workflow_access(principal, workflow, *, write=False):
    try:
        if workflow.tenant_id:
            enforce_tenant(principal, workflow.tenant_id)
        elif write:
            enforce_roles(principal, {"admin"})
    except AuthorizationError as exc:
        raise HTTPException(403, "workflow access is not authorized") from exc
