from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .catalog_storage import MemoryCatalogStorage, SQLiteCatalogStorage
from .models import AgentProfile, Approval, Checkpoint, Compensation, DeadLetter, Run, RunEvent, RunCreateRequest, WorkflowManifest, utc_now
from .observability import new_correlation_id, sanitize_audit_event


class NotFoundError(KeyError):
    pass


def _model_json(model: Any) -> str:
    """Serialize a Pydantic model on both v1 and v2 without leaking internals."""

    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def _model_payload(model: Any) -> Dict[str, Any]:
    return json.loads(_model_json(model))


def _parse_model(model_type: Any, payload: Any) -> Any:
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(payload)
    return model_type.parse_obj(payload)


class StoreProtocol(Protocol):
    """Persistence contract consumed by RuntimeService."""

    def register_profile(self, profile: AgentProfile) -> AgentProfile: ...
    def get_profile(self, agent_id: str, version: Optional[str] = None) -> AgentProfile: ...
    def list_profiles(self) -> List[AgentProfile]: ...
    def register_workflow(self, workflow: WorkflowManifest) -> WorkflowManifest: ...
    def list_workflows(self) -> List[WorkflowManifest]: ...
    def catalog_list(self, kind: str) -> List[Dict[str, Any]]: ...
    def catalog_put(self, kind: str, item_id: str, version: str, payload: Dict[str, Any], *, expected_revision: Optional[int] = None, immutable: bool = False) -> Dict[str, Any]: ...
    def get_workflow(self, workflow_id: str, version: Optional[str] = None) -> WorkflowManifest: ...
    def find_idempotent_run(self, tenant_id: str, key: Optional[str]) -> Optional[Run]: ...
    def create_run(
        self,
        request: RunCreateRequest,
        profile: AgentProfile,
        *,
        correlation_id: Optional[str] = None,
        deadline_at: Optional[datetime] = None,
    ) -> Run: ...
    def get_run(self, run_id: str) -> Run: ...
    def list_runs(self) -> List[Run]: ...
    def update_run(self, run: Run) -> Run: ...
    def append_event(self, run_id: str, event_type: str, data: Optional[Dict[str, object]] = None) -> RunEvent: ...
    def list_events(self, run_id: str, after_seq: int = 0) -> List[RunEvent]: ...
    def create_approval(
        self,
        run_id: str,
        tenant_id: str,
        tool_name: str,
        args: Dict[str, object],
        policy_decision_id: str,
        requested_by: str,
        correlation_id: Optional[str] = None,
    ) -> Approval: ...
    def get_approval(self, approval_id: str) -> Approval: ...
    def update_approval(self, approval: Approval) -> Approval: ...
    def consume_approval(
        self,
        approval_id: str,
        *,
        run_id: str,
        tool_name: str,
        args: Dict[str, object],
    ) -> Approval: ...
    def append_audit(self, event: Dict[str, object]) -> None: ...
    def list_audit(self) -> List[Dict[str, object]]: ...
    def enqueue_run(self, run_id: str) -> None: ...
    def claim_run(self, worker_id: str, run_id: Optional[str] = None) -> Optional[str]: ...
    def ack_run(self, run_id: str) -> None: ...
    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint: ...
    def list_checkpoints(self, run_id: str) -> List[Checkpoint]: ...
    def enqueue_dead_letter(self, item: DeadLetter) -> DeadLetter: ...
    def list_dead_letters(self) -> List[DeadLetter]: ...
    def create_compensation(self, item: Compensation) -> Compensation: ...
    def update_compensation(self, item: Compensation) -> Compensation: ...
    def list_compensations(self, run_id: Optional[str] = None) -> List[Compensation]: ...


def build_store(app_settings: Any) -> StoreProtocol:
    """Build the configured store without making persistence implicit."""

    backend = str(getattr(app_settings, "store_backend", "memory")).strip().lower()
    if backend == "memory":
        return InMemoryStore()
    if backend == "sqlite":
        return SQLiteStore(getattr(app_settings, "sqlite_path", "data/qingling.db"))
    raise ValueError(f"unsupported store backend: {backend}")


class _JsonStoreMixin:
    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)

    @staticmethod
    def _load(value: str) -> Any:
        return json.loads(value)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


class InMemoryStore(MemoryCatalogStorage):
    """A replaceable store for the first runnable vertical slice."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.profiles: Dict[Tuple[str, str], AgentProfile] = {}
        self.active_profile_versions: Dict[str, str] = {}
        self.workflows: Dict[Tuple[str, str], WorkflowManifest] = {}
        self.runs: Dict[str, Run] = {}
        self.events: Dict[str, List[RunEvent]] = {}
        self.approvals: Dict[str, Approval] = {}
        self.idempotency: Dict[Tuple[str, str], str] = {}
        self.audit_events: List[Dict[str, object]] = []
        self.queue: List[Dict[str, str]] = []
        self.checkpoints: Dict[str, List[Checkpoint]] = {}
        self.dead_letters: Dict[str, DeadLetter] = {}
        self.compensations: Dict[str, Compensation] = {}

    def register_profile(self, profile: AgentProfile) -> AgentProfile:
        with self._lock:
            self.profiles[(profile.id, profile.version)] = deepcopy(profile)
            self.active_profile_versions[profile.id] = profile.version
            return deepcopy(profile)

    def get_profile(self, agent_id: str, version: Optional[str] = None) -> AgentProfile:
        with self._lock:
            selected_version = version or self.active_profile_versions.get(agent_id)
            if selected_version is None:
                raise NotFoundError(f"agent profile not found: {agent_id}")
            profile = self.profiles.get((agent_id, selected_version))
            if profile is None:
                raise NotFoundError(f"agent profile not found: {agent_id}@{selected_version}")
            return deepcopy(profile)

    def list_profiles(self) -> List[AgentProfile]:
        with self._lock:
            return [deepcopy(p) for p in self.profiles.values()]

    def register_workflow(self, workflow: WorkflowManifest) -> WorkflowManifest:
        with self._lock:
            current = self.workflows.get((workflow.id, workflow.version))
            if current and _model_payload(current) != _model_payload(workflow):
                raise ValueError("published workflow version is immutable")
            self.workflows[(workflow.id, workflow.version)] = deepcopy(workflow)
            return deepcopy(workflow)

    def list_workflows(self):
        with self._lock:
            return deepcopy(list(self.workflows.values()))

    def get_workflow(self, workflow_id: str, version: Optional[str] = None) -> WorkflowManifest:
        with self._lock:
            if version is None:
                versions = [v for (wid, v) in self.workflows if wid == workflow_id]
                version = max(versions, key=lambda v: tuple(int(p) for p in v.split("."))) if versions else None
            workflow = self.workflows.get((workflow_id, version)) if version else None
            if workflow is None:
                raise NotFoundError(f"workflow not found: {workflow_id}@{version or 'latest'}")
            return deepcopy(workflow)

    def find_idempotent_run(self, tenant_id: str, key: Optional[str]) -> Optional[Run]:
        if not key:
            return None
        with self._lock:
            run_id = self.idempotency.get((tenant_id, key))
            return deepcopy(self.runs[run_id]) if run_id and run_id in self.runs else None

    def create_run(
        self,
        request: RunCreateRequest,
        profile: AgentProfile,
        *,
        correlation_id: Optional[str] = None,
        deadline_at: Optional[datetime] = None,
    ) -> Run:
        with self._lock:
            if request.idempotency_key:
                existing = self.find_idempotent_run(request.tenant_id, request.idempotency_key)
                if existing:
                    return existing
            run = Run(
                id=uuid.uuid4().hex,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                roles=list(request.roles),
                agent_id=profile.id,
                agent_version=profile.version,
                input=deepcopy(request.input),
                session_id=request.session_id,
                correlation_id=correlation_id or request.correlation_id or new_correlation_id(),
                deadline_at=deadline_at,
            )
            self.runs[run.id] = run
            self.events[run.id] = []
            if request.idempotency_key:
                self.idempotency[(request.tenant_id, request.idempotency_key)] = run.id
            return deepcopy(run)

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            if run_id not in self.runs:
                raise NotFoundError(f"run not found: {run_id}")
            return deepcopy(self.runs[run_id])

    def list_runs(self) -> List[Run]:
        with self._lock:
            return [deepcopy(run) for run in self.runs.values()]

    def update_run(self, run: Run) -> Run:
        with self._lock:
            self.runs[run.id] = deepcopy(run)
            return deepcopy(run)

    def append_event(self, run_id: str, event_type: str, data: Optional[Dict[str, object]] = None) -> RunEvent:
        with self._lock:
            if run_id not in self.runs:
                raise NotFoundError(f"run not found: {run_id}")
            event = RunEvent(
                seq=len(self.events[run_id]) + 1,
                run_id=run_id,
                event_type=event_type,
                schema_version=str((data or {}).get("schema_version", "1.0")),
                tenant_id=(data or {}).get("tenant_id"),
                correlation_id=(data or {}).get("correlation_id"),
                data=data or {},
            )
            self.events[run_id].append(event)
            return deepcopy(event)

    def list_events(self, run_id: str, after_seq: int = 0) -> List[RunEvent]:
        with self._lock:
            if run_id not in self.events:
                raise NotFoundError(f"run not found: {run_id}")
            return [deepcopy(e) for e in self.events[run_id] if e.seq > after_seq]

    def create_approval(
        self,
        run_id: str,
        tenant_id: str,
        tool_name: str,
        args: Dict[str, object],
        policy_decision_id: str,
        requested_by: str,
        correlation_id: Optional[str] = None,
    ) -> Approval:
        with self._lock:
            approval = Approval(
                id=uuid.uuid4().hex,
                run_id=run_id,
                tenant_id=tenant_id,
                tool_name=tool_name,
                args=deepcopy(args),
                policy_decision_id=policy_decision_id,
                requested_by=requested_by,
                correlation_id=correlation_id,
            )
            self.approvals[approval.id] = approval
            return deepcopy(approval)

    def get_approval(self, approval_id: str) -> Approval:
        with self._lock:
            if approval_id not in self.approvals:
                raise NotFoundError(f"approval not found: {approval_id}")
            return deepcopy(self.approvals[approval_id])

    def update_approval(self, approval: Approval) -> Approval:
        with self._lock:
            self.approvals[approval.id] = deepcopy(approval)
            return deepcopy(approval)

    def consume_approval(
        self,
        approval_id: str,
        *,
        run_id: str,
        tool_name: str,
        args: Dict[str, object],
    ) -> Approval:
        with self._lock:
            approval = self.approvals.get(approval_id)
            if approval is None:
                raise NotFoundError(f"approval not found: {approval_id}")
            if (
                approval.status != "approved"
                or approval.consumed_at is not None
                or approval.run_id != run_id
                or approval.tool_name != tool_name
                or approval.args != args
            ):
                raise ValueError("approval is invalid, mismatched, or already consumed")
            approval.consumed_at = utc_now()
            approval.consumed_by_run_id = run_id
            self.approvals[approval.id] = deepcopy(approval)
            return deepcopy(approval)

    def append_audit(self, event: Dict[str, object]) -> None:
        with self._lock:
            record = sanitize_audit_event(dict(event))
            record.setdefault("audit_id", uuid.uuid4().hex)
            record.setdefault("created_at", utc_now().isoformat())
            self.audit_events.append(deepcopy(record))

    def list_audit(self) -> List[Dict[str, object]]:
        with self._lock:
            return deepcopy(self.audit_events)

    def enqueue_run(self, run_id: str) -> None:
        with self._lock:
            if not any(item["run_id"] == run_id and item["status"] == "queued" for item in self.queue):
                self.queue.append({"run_id": run_id, "status": "queued"})

    def claim_run(self, worker_id: str, run_id: Optional[str] = None) -> Optional[str]:
        with self._lock:
            for item in self.queue:
                if item["status"] == "queued" and (run_id is None or item["run_id"] == run_id):
                    item.update(status="running", worker_id=worker_id)
                    return item["run_id"]
        return None

    def ack_run(self, run_id: str) -> None:
        with self._lock:
            self.queue = [item for item in self.queue if item["run_id"] != run_id]

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        with self._lock:
            items = self.checkpoints.setdefault(checkpoint.run_id, [])
            items.append(deepcopy(checkpoint))
            return deepcopy(checkpoint)

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        with self._lock:
            return deepcopy(self.checkpoints.get(run_id, []))

    def enqueue_dead_letter(self, item: DeadLetter) -> DeadLetter:
        with self._lock:
            self.dead_letters[item.id] = deepcopy(item)
            return deepcopy(item)

    def list_dead_letters(self) -> List[DeadLetter]:
        with self._lock:
            return deepcopy(list(self.dead_letters.values()))

    def create_compensation(self, item: Compensation) -> Compensation:
        with self._lock:
            self.compensations[item.id] = deepcopy(item)
            return deepcopy(item)

    def update_compensation(self, item: Compensation) -> Compensation:
        with self._lock:
            if item.id not in self.compensations:
                raise NotFoundError(f"compensation not found: {item.id}")
            self.compensations[item.id] = deepcopy(item)
            return deepcopy(item)

    def list_compensations(self, run_id: Optional[str] = None) -> List[Compensation]:
        with self._lock:
            values = list(self.compensations.values())
            return deepcopy([item for item in values if run_id is None or item.run_id == run_id])


class SQLiteStore(SQLiteCatalogStorage, _JsonStoreMixin):
    """Small durable store for local deployments and integration tests.

    It deliberately implements the same contract as ``InMemoryStore``. The
    production PostgreSQL adapter can replace it without changing RuntimeService.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog (
                    kind TEXT, id TEXT, version TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(kind, id, version)
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                );
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (id, version)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    tenant_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, key)
                );
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_queue (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, worker_id TEXT, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS dead_letters (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS compensations (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events (run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def register_profile(self, profile: AgentProfile) -> AgentProfile:
        payload = _model_json(profile)
        with self._lock, self._conn:
            self._conn.execute("UPDATE profiles SET active = 0 WHERE id = ?", (profile.id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO profiles (id, version, active, payload) VALUES (?, ?, 1, ?)",
                (profile.id, profile.version, payload),
            )
        return deepcopy(profile)

    def get_profile(self, agent_id: str, version: Optional[str] = None) -> AgentProfile:
        with self._lock:
            if version is None:
                row = self._conn.execute(
                    "SELECT version FROM profiles WHERE id = ? AND active = 1 ORDER BY version DESC LIMIT 1",
                    (agent_id,),
                ).fetchone()
                version = row["version"] if row else None
            row = self._conn.execute(
                "SELECT payload FROM profiles WHERE id = ? AND version = ?",
                (agent_id, version),
            ).fetchone() if version else None
            if row is None:
                raise NotFoundError(f"agent profile not found: {agent_id}@{version or 'latest'}")
            return _parse_model(AgentProfile, self._load(row["payload"]))

    def list_profiles(self) -> List[AgentProfile]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM profiles ORDER BY id, version").fetchall()
            return [_parse_model(AgentProfile, self._load(row["payload"])) for row in rows]

    def register_workflow(self, workflow: WorkflowManifest) -> WorkflowManifest:
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute("SELECT payload FROM workflows WHERE id = ? AND version = ?", (workflow.id, workflow.version)).fetchone()
            if row and _model_payload(_parse_model(WorkflowManifest, json.loads(row[0]))) != _model_payload(workflow):
                raise ValueError("published workflow version is immutable")
            self._conn.execute(
                "INSERT OR IGNORE INTO workflows (id, version, payload) VALUES (?, ?, ?)",
                (workflow.id, workflow.version, _model_json(workflow)),
            )
        return deepcopy(workflow)

    def list_workflows(self):
        with self._lock:
            return [_parse_model(WorkflowManifest, json.loads(row[0])) for row in self._conn.execute("SELECT payload FROM workflows").fetchall()]

    def get_workflow(self, workflow_id: str, version: Optional[str] = None) -> WorkflowManifest:
        with self._lock:
            if version is None:
                versions = self._conn.execute("SELECT version, payload FROM workflows WHERE id = ?", (workflow_id,)).fetchall()
                selected = max(versions, key=lambda r: tuple(int(p) for p in r["version"].split("."))) if versions else None
                row = selected
            else:
                row = self._conn.execute(
                    "SELECT payload FROM workflows WHERE id = ? AND version = ?",
                    (workflow_id, version),
                ).fetchone()
            if row is None:
                raise NotFoundError(f"workflow not found: {workflow_id}@{version or 'latest'}")
            return _parse_model(WorkflowManifest, self._load(row["payload"]))

    def find_idempotent_run(self, tenant_id: str, key: Optional[str]) -> Optional[Run]:
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT r.payload FROM idempotency i JOIN runs r ON r.id = i.run_id WHERE i.tenant_id = ? AND i.key = ?",
                (tenant_id, key),
            ).fetchone()
            return _parse_model(Run, self._load(row["payload"])) if row else None

    def create_run(
        self,
        request: RunCreateRequest,
        profile: AgentProfile,
        *,
        correlation_id: Optional[str] = None,
        deadline_at: Optional[datetime] = None,
    ) -> Run:
        with self._lock, self._conn:
            if request.idempotency_key:
                row = self._conn.execute(
                    "SELECT r.payload FROM idempotency i JOIN runs r ON r.id = i.run_id WHERE i.tenant_id = ? AND i.key = ?",
                    (request.tenant_id, request.idempotency_key),
                ).fetchone()
                if row:
                    return _parse_model(Run, self._load(row["payload"]))
            run = Run(
                id=uuid.uuid4().hex,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                roles=list(request.roles),
                agent_id=profile.id,
                agent_version=profile.version,
                input=deepcopy(request.input),
                session_id=request.session_id,
                correlation_id=correlation_id or request.correlation_id or new_correlation_id(),
                deadline_at=deadline_at,
            )
            if request.idempotency_key:
                inserted = self._conn.execute(
                    "INSERT OR IGNORE INTO idempotency (tenant_id, key, run_id) VALUES (?, ?, ?)",
                    (request.tenant_id, request.idempotency_key, run.id),
                ).rowcount
                if inserted == 0:
                    row = self._conn.execute(
                        "SELECT r.payload FROM idempotency i JOIN runs r ON r.id = i.run_id WHERE i.tenant_id = ? AND i.key = ?",
                        (request.tenant_id, request.idempotency_key),
                    ).fetchone()
                    if row:
                        return _parse_model(Run, self._load(row["payload"]))
                    raise RuntimeError("idempotency record exists without its run")
            self._conn.execute(
                "INSERT INTO runs (id, tenant_id, payload) VALUES (?, ?, ?)",
                (run.id, run.tenant_id, _model_json(run)),
            )
            return deepcopy(run)

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"run not found: {run_id}")
            return _parse_model(Run, self._load(row["payload"]))

    def list_runs(self) -> List[Run]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM runs ORDER BY rowid").fetchall()
            return [_parse_model(Run, self._load(row["payload"])) for row in rows]

    def update_run(self, run: Run) -> Run:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE runs SET tenant_id = ?, payload = ? WHERE id = ?",
                (run.tenant_id, _model_json(run), run.id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"run not found: {run.id}")
            return deepcopy(run)

    def append_event(self, run_id: str, event_type: str, data: Optional[Dict[str, object]] = None) -> RunEvent:
        with self._lock, self._conn:
            if self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run not found: {run_id}")
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM events WHERE run_id = ?", (run_id,)).fetchone()
            payload = data or {}
            event = RunEvent(
                seq=int(row["next_seq"]),
                run_id=run_id,
                event_type=event_type,
                schema_version=str(payload.get("schema_version", "1.0")),
                tenant_id=payload.get("tenant_id"),
                correlation_id=payload.get("correlation_id"),
                data=payload,
            )
            self._conn.execute(
                "INSERT INTO events (run_id, seq, payload) VALUES (?, ?, ?)",
                (run_id, event.seq, _model_json(event)),
            )
            return deepcopy(event)

    def list_events(self, run_id: str, after_seq: int = 0) -> List[RunEvent]:
        with self._lock:
            if self._conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"run not found: {run_id}")
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, after_seq),
            ).fetchall()
            return [_parse_model(RunEvent, self._load(row["payload"])) for row in rows]

    def create_approval(
        self,
        run_id: str,
        tenant_id: str,
        tool_name: str,
        args: Dict[str, object],
        policy_decision_id: str,
        requested_by: str,
        correlation_id: Optional[str] = None,
    ) -> Approval:
        approval = Approval(
            id=uuid.uuid4().hex,
            run_id=run_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            args=deepcopy(args),
            policy_decision_id=policy_decision_id,
            requested_by=requested_by,
            correlation_id=correlation_id,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO approvals (id, run_id, tenant_id, payload) VALUES (?, ?, ?, ?)",
                (approval.id, run_id, tenant_id, _model_json(approval)),
            )
        return deepcopy(approval)

    def get_approval(self, approval_id: str) -> Approval:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"approval not found: {approval_id}")
            return _parse_model(Approval, self._load(row["payload"]))

    def update_approval(self, approval: Approval) -> Approval:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE approvals SET run_id = ?, tenant_id = ?, payload = ? WHERE id = ?",
                (approval.run_id, approval.tenant_id, _model_json(approval), approval.id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"approval not found: {approval.id}")
            return deepcopy(approval)

    def consume_approval(
        self,
        approval_id: str,
        *,
        run_id: str,
        tool_name: str,
        args: Dict[str, object],
    ) -> Approval:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT payload FROM approvals WHERE id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"approval not found: {approval_id}")
            approval = _parse_model(Approval, self._load(row["payload"]))
            if (
                approval.status != "approved"
                or approval.consumed_at is not None
                or approval.run_id != run_id
                or approval.tool_name != tool_name
                or approval.args != args
            ):
                raise ValueError("approval is invalid, mismatched, or already consumed")
            approval.consumed_at = utc_now()
            approval.consumed_by_run_id = run_id
            cursor = self._conn.execute(
                "UPDATE approvals SET payload = ? WHERE id = ?",
                (_model_json(approval), approval_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"approval not found: {approval_id}")
            return deepcopy(approval)

    def append_audit(self, event: Dict[str, object]) -> None:
        record = sanitize_audit_event(dict(event))
        record.setdefault("audit_id", uuid.uuid4().hex)
        record.setdefault("created_at", utc_now().isoformat())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audits (audit_id, payload) VALUES (?, ?)",
                (str(record["audit_id"]), self._dump(record)),
            )

    def list_audit(self) -> List[Dict[str, object]]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM audits ORDER BY seq").fetchall()
            return [self._load(row["payload"]) for row in rows]

    def enqueue_run(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO run_queue(run_id,status,updated_at) VALUES (?, 'queued', ?)", (run_id, utc_now().isoformat()))

    def claim_run(self, worker_id: str, run_id: Optional[str] = None) -> Optional[str]:
        with self._lock, self._conn:
            stale = (utc_now() - timedelta(seconds=60)).isoformat()
            query = "SELECT run_id FROM run_queue WHERE (status='queued' OR (status='running' AND updated_at < ?))"
            params: tuple[Any, ...] = (stale,)
            if run_id:
                query += " AND run_id=?"
                params += (run_id,)
            query += " ORDER BY updated_at LIMIT 1"
            row = self._conn.execute(query, params).fetchone()
            if not row:
                return None
            run_id = row["run_id"]
            self._conn.execute("UPDATE run_queue SET status='running', worker_id=?, updated_at=? WHERE run_id=? AND (status='queued' OR (status='running' AND updated_at < ?))", (worker_id, utc_now().isoformat(), run_id, stale))
            return run_id

    def ack_run(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM run_queue WHERE run_id=?", (run_id,))

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO checkpoints(id,run_id,payload) VALUES (?,?,?)", (checkpoint.id, checkpoint.run_id, _model_json(checkpoint)))
            return deepcopy(checkpoint)

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM checkpoints WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()
            return [_parse_model(Checkpoint, self._load(row["payload"])) for row in rows]

    def enqueue_dead_letter(self, item: DeadLetter) -> DeadLetter:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO dead_letters(id,run_id,payload) VALUES (?,?,?)", (item.id, item.run_id, _model_json(item)))
            return deepcopy(item)

    def list_dead_letters(self) -> List[DeadLetter]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM dead_letters ORDER BY rowid").fetchall()
            return [_parse_model(DeadLetter, self._load(row["payload"])) for row in rows]

    def create_compensation(self, item: Compensation) -> Compensation:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO compensations(id,run_id,payload) VALUES (?,?,?)", (item.id, item.run_id, _model_json(item)))
            return deepcopy(item)

    def update_compensation(self, item: Compensation) -> Compensation:
        with self._lock, self._conn:
            self._conn.execute("UPDATE compensations SET payload=? WHERE id=?", (_model_json(item), item.id))
            return deepcopy(item)

    def list_compensations(self, run_id: Optional[str] = None) -> List[Compensation]:
        with self._lock:
            query = "SELECT payload FROM compensations" + (" WHERE run_id=?" if run_id else "") + " ORDER BY rowid"
            rows = self._conn.execute(query, (run_id,) if run_id else ()).fetchall()
            return [_parse_model(Compensation, self._load(row["payload"])) for row in rows]
