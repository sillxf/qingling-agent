from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("QINGLING_APP_NAME", "青灵智能体")
    app_version: str = os.getenv("QINGLING_APP_VERSION", "0.1.0")
    environment: str = os.getenv("QINGLING_ENV", "development")
    max_react_steps: int = _env_int("QINGLING_MAX_REACT_STEPS", 30)
    max_retry: int = _env_int("QINGLING_MAX_RETRY", 3)
    run_workers: int = _env_int("QINGLING_RUN_WORKERS", 4)
    run_timeout_seconds: int = _env_int("QINGLING_RUN_TIMEOUT_SECONDS", 300)
    decision_model_planning: bool = _env_bool("QINGLING_DECISION_MODEL_PLANNING", False)
    decision_max_steps: int = _env_int("QINGLING_DECISION_MAX_STEPS", 12)
    store_backend: str = os.getenv("QINGLING_STORE_BACKEND", "memory")
    sqlite_path: str = os.getenv("QINGLING_SQLITE_PATH", "data/qingling.db")
    # The HTTP gateway is opt-in; RuntimeService still defaults to the
    # deterministic offline gateway unless an adapter is explicitly injected.
    model_gateway_base_url: str = os.getenv(
        "QINGLING_MODEL_GATEWAY_BASE_URL",
        os.getenv("QINGLING_MODEL_BASE_URL", ""),
    )
    model_gateway_api_key: str = os.getenv(
        "QINGLING_MODEL_GATEWAY_API_KEY",
        os.getenv("QINGLING_MODEL_API_KEY", ""),
    )
    model_gateway_timeout_seconds: float = _env_float(
        "QINGLING_MODEL_GATEWAY_TIMEOUT_SECONDS",
        _env_float("QINGLING_MODEL_TIMEOUT_SECONDS", 30.0),
    )
    model_gateway_max_retries: int = _env_int(
        "QINGLING_MODEL_GATEWAY_MAX_RETRIES",
        _env_int("QINGLING_MODEL_MAX_RETRIES", 2),
    )
    model_gateway_backoff_seconds: float = _env_float(
        "QINGLING_MODEL_GATEWAY_BACKOFF_SECONDS",
        _env_float("QINGLING_MODEL_BACKOFF_SECONDS", 0.25),
    )
    # Authentication is opt-in for local development. When enabled, the
    # static token JSON is parsed and validated during application startup.
    auth_enabled: bool = _env_bool("QINGLING_AUTH_ENABLED", False)
    auth_tokens_json: str = os.getenv("QINGLING_AUTH_TOKENS_JSON", "")
    event_api_base_url: str = os.getenv("QINGLING_EVENT_API_BASE_URL", "")
    event_api_key: str = os.getenv("QINGLING_EVENT_API_KEY", "")
    event_api_timeout_seconds: float = _env_float("QINGLING_EVENT_API_TIMEOUT_SECONDS", 10.0)
    event_api_max_retries: int = _env_int("QINGLING_EVENT_API_MAX_RETRIES", 2)
    event_api_backoff_seconds: float = _env_float("QINGLING_EVENT_API_BACKOFF_SECONDS", 0.25)
    # Knowledge/RAG storage is configurable independently so local knowledge
    # data can be persisted without changing the Run store migration plan.
    # An omitted backend means "derive from the primary store setting". This keeps
    # ``Settings(store_backend="sqlite")`` intuitive while still allowing an
    # explicit knowledge backend to remain independent.
    knowledge_store_backend: Optional[str] = os.getenv("QINGLING_KNOWLEDGE_STORE_BACKEND")
    # ``None`` means use the primary SQLite path when the effective knowledge
    # backend is SQLite; callers may explicitly point it at a dedicated file.
    knowledge_sqlite_path: Optional[str] = os.getenv("QINGLING_KNOWLEDGE_SQLITE_PATH")
    knowledge_embedding_model: str = os.getenv("QINGLING_KNOWLEDGE_EMBEDDING_MODEL", "bge-m3")
    # OpenCode resource roots are local, versioned directories.  MCP servers
    # are explicitly allow-listed as JSON definitions; an empty value keeps
    # the offline runtime unchanged.
    config_dir: Optional[str] = os.getenv("QINGLING_CONFIG_DIR")
    skills_root: str = os.getenv("QINGLING_SKILLS_ROOT", "")
    skill_max_chars: int = _env_int("QINGLING_SKILL_MAX_CHARS", 120_000)
    mcp_servers_json: str = os.getenv("QINGLING_MCP_SERVERS_JSON", "")
    opencode_context_max_chars: int = _env_int("QINGLING_OPENCODE_CONTEXT_MAX_CHARS", 120_000)
    opencode_sandbox_root: str = os.getenv("QINGLING_OPENCODE_SANDBOX_ROOT", "data/opencode-sandbox")
    opencode_max_file_bytes: int = _env_int("QINGLING_OPENCODE_MAX_FILE_BYTES", 1_000_000)
    opencode_max_output_chars: int = _env_int("QINGLING_OPENCODE_MAX_OUTPUT_CHARS", 30_000)
    opencode_shell_timeout_seconds: float = _env_float("QINGLING_OPENCODE_SHELL_TIMEOUT_SECONDS", 10.0)

    def __post_init__(self) -> None:
        if self.knowledge_store_backend is None:
            object.__setattr__(
                self,
                "knowledge_store_backend",
                self.store_backend,
            )
        if self.knowledge_sqlite_path is None:
            effective_backend = str(self.knowledge_store_backend or "memory").strip().lower()
            default_path = (
                self.sqlite_path if effective_backend == "sqlite" else "data/qingling-knowledge.db"
            )
            object.__setattr__(self, "knowledge_sqlite_path", default_path)


settings = Settings()
