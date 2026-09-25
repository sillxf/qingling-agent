from __future__ import annotations

"""Small, safe OpenCode resources layer.

This module deliberately keeps Skills and MCP behind explicit registries.  A
profile can only load resources that were registered by the service, and MCP
tools are exposed to the normal ToolRegistry so the existing policy and audit
boundaries remain in force.
"""

import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import ToolContract
from .observability import redact_text


class OpenCodeResourceError(RuntimeError):
    code = "OPENCODE_RESOURCE_ERROR"


class SkillNotFound(OpenCodeResourceError):
    code = "SKILL_NOT_FOUND"


class SkillValidationError(OpenCodeResourceError):
    code = "SKILL_INVALID"


class MCPConfigurationError(OpenCodeResourceError):
    code = "MCP_CONFIGURATION_ERROR"


class MCPCallError(OpenCodeResourceError):
    code = "MCP_CALL_ERROR"


class SandboxViolation(OpenCodeResourceError):
    code = "SANDBOX_VIOLATION"


class HookRejected(OpenCodeResourceError):
    code = "HOOK_REJECTED"


def _safe_id(value: Any, label: str = "id") -> str:
    text = str(value or "").strip()
    if not text or len(text) > 128 or any(ord(char) < 32 for char in text):
        raise SkillValidationError(f"invalid {label}")
    return text


def _inside(root: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    root: str
    content: str
    sha256: str
    resources: List[str] = field(default_factory=list)


class SkillRegistry:
    """Registry for versioned local SKILL.md resources."""

    def __init__(self, root: str | Path, *, max_chars: int = 120_000) -> None:
        self.root = Path(root).resolve()
        self.max_chars = max(1_000, int(max_chars))
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.RLock()
        self.discover()

    def discover(self) -> None:
        if not self.root.exists():
            return
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            try:
                self.register_directory(skill_file.parent)
            except SkillValidationError:
                # A malformed optional skill must not make the service
                # unusable.  It remains absent and cannot be selected.
                continue

    def register_directory(self, directory: str | Path) -> Skill:
        path = Path(directory).resolve()
        if not _inside(self.root, path):
            raise SkillValidationError("skill directory is outside configured root")
        skill_file = path / "SKILL.md"
        if not skill_file.is_file():
            raise SkillValidationError("skill must contain SKILL.md")
        content = skill_file.read_text(encoding="utf-8")
        if not content.strip() or len(content) > self.max_chars:
            raise SkillValidationError("SKILL.md is empty or too large")
        skill_id = _safe_id(path.name, "skill id")
        name, description, body = self._parse_frontmatter(content)
        resources = [
            str(item.relative_to(path)).replace("\\", "/")
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.name != "SKILL.md" and _inside(path, item)
        ]
        item = Skill(
            id=skill_id,
            name=name or skill_id,
            description=description,
            root=str(path),
            content=body,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            resources=resources,
        )
        with self._lock:
            self._skills[skill_id] = item
        return item

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[str, str, str]:
        if not content.startswith("---"):
            return "", "", content
        lines = content.splitlines()
        try:
            end = lines.index("---", 1)
        except ValueError:
            return "", "", content
        metadata: Dict[str, str] = {}
        for line in lines[1:end]:
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip().strip('"\'')
        return metadata.get("name", ""), metadata.get("description", ""), "\n".join(lines[end + 1 :]).strip()

    def list(self) -> List[Skill]:
        with self._lock:
            return sorted(self._skills.values(), key=lambda item: item.id)

    def get(self, skill_id: str) -> Skill:
        with self._lock:
            try:
                return self._skills[_safe_id(skill_id, "skill id")]
            except KeyError as exc:
                raise SkillNotFound(f"skill not found: {skill_id}") from exc

    def load_many(self, skill_ids: Iterable[str]) -> List[Skill]:
        return [self.get(item) for item in skill_ids]

    def prompt_context(self, skill_ids: Iterable[str]) -> str:
        chunks: List[str] = []
        for skill in self.load_many(skill_ids):
            chunks.append(
                f"<skill id=\"{skill.id}\" sha256=\"{skill.sha256}\">\n"
                f"{skill.content}\n</skill>"
            )
        return "\n\n".join(chunks)


class FileSystemSandbox:
    """Run-scoped filesystem and shell boundary for OpenCode tools."""

    def __init__(self, root: str | Path, *, max_file_bytes: int = 1_000_000, max_output_chars: int = 30_000) -> None:
        self.root = Path(root).resolve()
        self.max_file_bytes = max(1_024, int(max_file_bytes))
        self.max_output_chars = max(1_000, int(max_output_chars))
        self.root.mkdir(parents=True, exist_ok=True)

    def run_root(self, run_id: str, tenant_id: str) -> Path:
        safe_run = hashlib.sha256(str(run_id).encode()).hexdigest()[:32]
        safe_tenant = hashlib.sha256(str(tenant_id).encode()).hexdigest()[:32]
        path = (self.root / safe_tenant / safe_run).resolve()
        if not _inside(self.root, path):
            raise SandboxViolation("sandbox path escaped root")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve(self, run_id: str, tenant_id: str, relative: str) -> Path:
        if not isinstance(relative, str) or not relative.strip() or len(relative) > 512:
            raise SandboxViolation("invalid sandbox path")
        base = self.run_root(run_id, tenant_id)
        candidate = (base / relative).resolve()
        if not _inside(base, candidate):
            raise SandboxViolation("sandbox path traversal denied")
        return candidate

    def read(self, run_id: str, tenant_id: str, relative: str) -> Dict[str, Any]:
        path = self.resolve(run_id, tenant_id, relative)
        if not path.is_file():
            raise SandboxViolation("sandbox file not found")
        if path.stat().st_size > self.max_file_bytes:
            raise SandboxViolation("sandbox file is too large")
        return {"path": relative.replace("\\", "/"), "content": path.read_text(encoding="utf-8"), "size": path.stat().st_size}

    def list(self, run_id: str, tenant_id: str, relative: str = ".") -> Dict[str, Any]:
        path = self.resolve(run_id, tenant_id, relative)
        if not path.exists() or not path.is_dir():
            raise SandboxViolation("sandbox directory not found")
        items = []
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            items.append({"name": child.name, "type": "directory" if child.is_dir() else "file", "size": child.stat().st_size if child.is_file() else None})
        return {"path": relative.replace("\\", "/"), "items": items}

    def search(self, run_id: str, tenant_id: str, query: str, relative: str = ".") -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise SandboxViolation("invalid search query")
        base = self.resolve(run_id, tenant_id, relative)
        if not base.is_dir():
            raise SandboxViolation("sandbox search root is not a directory")
        hits: List[Dict[str, Any]] = []
        for path in base.rglob("*"):
            if not path.is_file() or path.stat().st_size > self.max_file_bytes:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    hits.append({"path": str(path.relative_to(self.run_root(run_id, tenant_id))).replace("\\", "/"), "line": number, "text": line[:2_000]})
                    if len(hits) >= 200:
                        return {"query": query, "hits": hits, "truncated": True}
        return {"query": query, "hits": hits, "truncated": False}

    def write(self, run_id: str, tenant_id: str, relative: str, content: str) -> Dict[str, Any]:
        if not isinstance(content, str) or len(content.encode("utf-8")) > self.max_file_bytes:
            raise SandboxViolation("sandbox content is too large")
        path = self.resolve(run_id, tenant_id, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": relative.replace("\\", "/"), "size": len(content.encode("utf-8"))}

    def exec(self, run_id: str, tenant_id: str, command: str, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        if not isinstance(command, str) or not command.strip() or len(command) > 4_000:
            raise SandboxViolation("invalid sandbox command")
        timeout = max(0.1, min(float(timeout_seconds), 30.0))
        base = self.run_root(run_id, tenant_id)
        try:
            completed = subprocess.run(
                command,
                cwd=str(base),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxViolation("sandbox command timed out") from exc
        return {
            "returncode": int(completed.returncode),
            "stdout": (completed.stdout or "")[: self.max_output_chars],
            "stderr": (completed.stderr or "")[: self.max_output_chars],
            "truncated": len(completed.stdout or "") > self.max_output_chars or len(completed.stderr or "") > self.max_output_chars,
        }


Hook = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]


class HookRegistry:
    """In-process lifecycle hooks; hooks may enrich or reject an operation."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Hook]] = {}

    def register(self, point: str, hook: Hook) -> None:
        if not point.strip():
            raise HookRejected("hook point cannot be empty")
        self._hooks.setdefault(point, []).append(hook)

    def run(self, point: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(payload)
        for hook in self._hooks.get(point, []):
            try:
                result = hook(point, dict(current))
            except Exception as exc:
                raise HookRejected(f"hook failed at {point}") from exc
            if result is None:
                continue
            if not isinstance(result, dict):
                raise HookRejected(f"hook returned invalid payload at {point}")
            if result.get("reject"):
                raise HookRejected(str(result.get("reason") or f"hook rejected at {point}"))
            current.update(result)
        return current


@dataclass(frozen=True)
class MCPToolDefinition:
    server_id: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    permission: str = "read"
    risk_level: str = "low"
    reversible: bool = True
    requires_approval: bool = False

    @property
    def qualified_name(self) -> str:
        return f"mcp.{self.server_id}.{self.name}"


@dataclass
class MCPServerDefinition:
    id: str
    transport: str = "http"
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    tools: List[MCPToolDefinition] = field(default_factory=list)
    handlers: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = field(default_factory=dict)
    _process: Any = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)


class MCPRegistry:
    """MCP server catalog with HTTP JSON-RPC and optional stdio transport."""

    def __init__(self) -> None:
        self._servers: Dict[str, MCPServerDefinition] = {}
        self._tool_index: Dict[str, MCPToolDefinition] = {}

    def register(self, server: MCPServerDefinition) -> None:
        server.id = _safe_id(server.id, "MCP server id")
        if server.transport not in {"http", "stdio", "inproc"}:
            raise MCPConfigurationError("unsupported MCP transport")
        if server.transport == "http" and not server.url:
            raise MCPConfigurationError("HTTP MCP server requires url")
        if server.transport == "stdio" and not server.command:
            raise MCPConfigurationError("stdio MCP server requires command")
        if server.id in self._servers:
            raise MCPConfigurationError(f"MCP server already registered: {server.id}")
        self._servers[server.id] = server
        for tool in server.tools:
            if tool.server_id != server.id:
                raise MCPConfigurationError("MCP tool server id mismatch")
            if tool.qualified_name in self._tool_index:
                raise MCPConfigurationError(f"MCP tool already registered: {tool.qualified_name}")
            self._tool_index[tool.qualified_name] = tool

    def list_servers(self) -> List[MCPServerDefinition]:
        return sorted(self._servers.values(), key=lambda item: item.id)

    def list_tools(self, server_ids: Optional[Iterable[str]] = None) -> List[MCPToolDefinition]:
        allowed = set(server_ids) if server_ids is not None else set(self._servers)
        return sorted(
            [tool for tool in self._tool_index.values() if tool.server_id in allowed],
            key=lambda item: item.qualified_name,
        )

    def get_tool(self, qualified_name: str) -> MCPToolDefinition:
        try:
            return self._tool_index[qualified_name]
        except KeyError as exc:
            raise MCPCallError(f"MCP tool not found: {qualified_name}") from exc

    def invoke(self, qualified_name: str, args: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        tool = self.get_tool(qualified_name)
        server = self._servers[tool.server_id]
        if tool.name in server.handlers:
            return server.handlers[tool.name](args, context)
        if server.transport == "http":
            return self._http_call(server, tool.name, args, context)
        if server.transport == "stdio":
            return self._stdio_call(server, tool.name, args)
        raise MCPCallError(f"no handler for MCP tool: {qualified_name}")

    @staticmethod
    def _http_call(server: MCPServerDefinition, tool_name: str, args: Dict[str, Any], context: Any) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if getattr(context, "tenant_id", None):
            headers["X-Tenant-ID"] = str(context.tenant_id)
        if getattr(context, "user_id", None):
            headers["X-User-ID"] = str(context.user_id)
        if getattr(context, "run_id", None):
            headers["X-Run-ID"] = str(context.run_id)
        if getattr(context, "correlation_id", None):
            headers["X-Correlation-ID"] = str(context.correlation_id)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        }).encode("utf-8")
        try:
            response = urlopen(Request(server.url or "", data=body, headers=headers, method="POST"), timeout=server.timeout_seconds)
            payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise MCPCallError(redact_text(f"MCP HTTP call failed: {exc}")) from exc
        if not isinstance(payload, Mapping) or payload.get("error"):
            raise MCPCallError("MCP server returned an error")
        result = payload.get("result", payload)
        if not isinstance(result, Mapping):
            raise MCPCallError("MCP tool result must be an object")
        return dict(result)

    @staticmethod
    def _stdio_call(server: MCPServerDefinition, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        with server._lock:
            if server._process is None or server._process.poll() is not None:
                env = os.environ.copy()
                env.update(server.env)
                server._process = subprocess.Popen(
                    [server.command or "", *server.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": uuid.uuid4().hex,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            })
            try:
                assert server._process.stdin is not None and server._process.stdout is not None
                server._process.stdin.write(request + "\n")
                server._process.stdin.flush()
                line = server._process.stdout.readline()
                payload = json.loads(line)
            except (OSError, ValueError, AssertionError) as exc:
                raise MCPCallError("MCP stdio call failed") from exc
            if not isinstance(payload, Mapping) or payload.get("error"):
                raise MCPCallError("MCP server returned an error")
            result = payload.get("result", payload)
            if not isinstance(result, Mapping):
                raise MCPCallError("MCP tool result must be an object")
            return dict(result)


def _tool_from_config(server_id: str, item: Mapping[str, Any]) -> MCPToolDefinition:
    name = _safe_id(item.get("name"), "MCP tool name")
    return MCPToolDefinition(
        server_id=server_id,
        name=name,
        description=str(item.get("description", ""))[:1_000],
        input_schema=dict(item.get("input_schema") or item.get("inputSchema") or {}),
        permission=str(item.get("permission", "read")),
        risk_level=str(item.get("risk_level", "low")),
        reversible=bool(item.get("reversible", True)),
        requires_approval=bool(item.get("requires_approval", False)),
    )


def build_mcp_registry(app_settings: Any) -> MCPRegistry:
    registry = MCPRegistry()
    raw = str(getattr(app_settings, "mcp_servers_json", "") or "").strip()
    if not raw:
        return registry
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPConfigurationError("QINGLING_MCP_SERVERS_JSON must be valid JSON") from exc
    if not isinstance(values, list):
        raise MCPConfigurationError("QINGLING_MCP_SERVERS_JSON must be an array")
    for item in values:
        if not isinstance(item, Mapping):
            raise MCPConfigurationError("MCP server definitions must be objects")
        server_id = _safe_id(item.get("id"), "MCP server id")
        tools = [_tool_from_config(server_id, tool) for tool in (item.get("tools") or []) if isinstance(tool, Mapping)]
        registry.register(MCPServerDefinition(
            id=server_id,
            transport=str(item.get("transport", "http")),
            url=str(item.get("url")) if item.get("url") else None,
            command=str(item.get("command")) if item.get("command") else None,
            args=[str(value) for value in item.get("args", [])],
            env={str(key): str(value) for key, value in dict(item.get("env") or {}).items()},
            timeout_seconds=max(0.1, float(item.get("timeout_seconds", 30.0))),
            tools=tools,
        ))
    return registry


def mcp_contract(tool: MCPToolDefinition) -> ToolContract:
    return ToolContract(
        name=tool.qualified_name,
        description=tool.description or f"MCP tool {tool.name}",
        permission=tool.permission if tool.permission in {"read", "analyze", "mutate", "bash", "edit"} else "read",
        risk_level=tool.risk_level if tool.risk_level in {"low", "medium", "high", "critical"} else "low",
        input_schema=tool.input_schema,
        reversible=tool.reversible,
        requires_approval=tool.requires_approval,
    )


@dataclass(frozen=True)
class SubAgentDefinition:
    id: str
    system_prompt: str
    skills: List[str] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    max_steps: int = 8


class SubAgentRegistry:
    """Declarative sub-agent catalog; execution remains in the parent Runtime."""

    def __init__(self) -> None:
        self._items: Dict[str, SubAgentDefinition] = {}

    def register(self, item: SubAgentDefinition) -> None:
        item_id = _safe_id(item.id, "subagent id")
        if item_id in self._items:
            raise OpenCodeResourceError(f"subagent already registered: {item_id}")
        self._items[item_id] = SubAgentDefinition(
            id=item_id,
            system_prompt=str(item.system_prompt or "")[:20_000],
            skills=list(item.skills),
            mcp_servers=list(item.mcp_servers),
            max_steps=max(1, min(int(item.max_steps), 30)),
        )

    def get(self, item_id: str) -> SubAgentDefinition:
        try:
            return self._items[_safe_id(item_id, "subagent id")]
        except KeyError as exc:
            raise OpenCodeResourceError(f"subagent not found: {item_id}") from exc

    def list(self) -> List[SubAgentDefinition]:
        return sorted(self._items.values(), key=lambda item: item.id)
