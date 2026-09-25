"""Authentication and authorization primitives for the API boundary.

The first implementation intentionally uses static bearer tokens so the
runtime can exercise tenant and role boundaries without adding a JWT or IAM
dependency. Production deployments should replace the authenticator through
the same small protocol and keep token verification outside the agent runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, FrozenSet, List, Optional, Protocol


class AuthError(RuntimeError):
    """Base class for authentication configuration and request failures."""


class AuthConfigurationError(AuthError):
    pass


class AuthenticationError(AuthError):
    pass


class AuthorizationError(AuthError):
    pass


@dataclass(frozen=True)
class Principal:
    """The trusted identity attached to one API request."""

    subject: str
    tenant_id: Optional[str]
    roles: FrozenSet[str]
    authenticated: bool = True
    token_id: Optional[str] = None

    @property
    def is_development(self) -> bool:
        return not self.authenticated

    def has_any_role(self, required: set[str]) -> bool:
        return bool(self.roles.intersection(required))


class Authenticator(Protocol):
    def authenticate(self, authorization: Optional[str]) -> Principal: ...


class DevelopmentAuthenticator:
    """Explicitly non-production identity used when auth is disabled."""

    def authenticate(self, authorization: Optional[str]) -> Principal:
        return Principal(
            subject="development",
            tenant_id=None,
            roles=frozenset({"development", "admin", "operator"}),
            authenticated=False,
            token_id="development",
        )


@dataclass(frozen=True)
class _TokenRecord:
    digest: bytes
    principal: Principal


class StaticTokenAuthenticator:
    """Constant-time bearer token verifier backed by configured token hashes."""

    def __init__(self, records: List[_TokenRecord]) -> None:
        if not records:
            raise AuthConfigurationError("at least one authentication token is required")
        self._records = tuple(records)

    @classmethod
    def from_json(cls, raw: str) -> "StaticTokenAuthenticator":
        if not isinstance(raw, str) or not raw.strip():
            raise AuthConfigurationError("QINGLING_AUTH_TOKENS_JSON is required when authentication is enabled")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthConfigurationError("QINGLING_AUTH_TOKENS_JSON must be valid JSON") from exc
        if not isinstance(payload, list) or not payload:
            raise AuthConfigurationError("authentication token configuration must be a non-empty JSON array")

        records: List[_TokenRecord] = []
        seen: set[bytes] = set()
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise AuthConfigurationError(f"authentication token entry {index} must be an object")
            token = item.get("token")
            subject = item.get("subject")
            tenant_id = item.get("tenant_id")
            roles = item.get("roles", [])
            token_id = item.get("token_id")
            if not isinstance(token, str) or len(token) < 8:
                raise AuthConfigurationError(f"authentication token entry {index} has an invalid token")
            if not isinstance(subject, str) or not subject.strip():
                raise AuthConfigurationError(f"authentication token entry {index} has an invalid subject")
            if not isinstance(tenant_id, str) or not tenant_id.strip():
                raise AuthConfigurationError(f"authentication token entry {index} has an invalid tenant_id")
            if not isinstance(roles, list) or any(not isinstance(role, str) or not role.strip() for role in roles):
                raise AuthConfigurationError(f"authentication token entry {index} has invalid roles")
            if token_id is not None and (not isinstance(token_id, str) or not token_id.strip()):
                raise AuthConfigurationError(f"authentication token entry {index} has an invalid token_id")
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            if digest in seen:
                raise AuthConfigurationError("authentication token values must be unique")
            seen.add(digest)
            records.append(
                _TokenRecord(
                    digest=digest,
                    principal=Principal(
                        subject=subject.strip(),
                        tenant_id=tenant_id.strip(),
                        roles=frozenset(role.strip() for role in roles),
                        authenticated=True,
                        token_id=token_id.strip() if isinstance(token_id, str) else None,
                    ),
                )
            )
        return cls(records)

    def authenticate(self, authorization: Optional[str]) -> Principal:
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            raise AuthenticationError("bearer authentication is required")
        token = authorization[7:].strip()
        if not token:
            raise AuthenticationError("bearer authentication is required")
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for record in self._records:
            if hmac.compare_digest(digest, record.digest):
                return record.principal
        raise AuthenticationError("bearer token is invalid")


def build_authenticator(app_settings: Any) -> Authenticator:
    if not bool(getattr(app_settings, "auth_enabled", False)):
        return DevelopmentAuthenticator()
    return StaticTokenAuthenticator.from_json(getattr(app_settings, "auth_tokens_json", ""))


def enforce_tenant(principal: Principal, tenant_id: str) -> None:
    """Reject a request that attempts to act outside its trusted tenant."""

    if principal.is_development:
        return
    if not principal.tenant_id or not hmac.compare_digest(principal.tenant_id, tenant_id):
        raise AuthorizationError("tenant access is not authorized")


def enforce_roles(principal: Principal, required: set[str]) -> None:
    """Require one of the supplied roles when authentication is enabled."""

    if principal.is_development:
        return
    if not principal.has_any_role(required):
        raise AuthorizationError("the caller lacks the required role")
