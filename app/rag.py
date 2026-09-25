"""Bounded, tenant-aware retrieval augmented generation primitives.

The module intentionally keeps ingestion and retrieval independent from the
runtime Store.  It is suitable for local use now and exposes a small store
protocol that can later be replaced by PostgreSQL/vector infrastructure.
Retrieved text is always marked as untrusted data; it must never be appended to
a system prompt without an explicit data boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import quote

from pydantic import BaseModel, Field

from .model_gateway import DeterministicModelGateway, ModelCallContext, ModelGateway
from .models import utc_now


MAX_DOCUMENT_CHARS = 1_000_000
MAX_TITLE_CHARS = 512
MAX_SOURCE_URI_CHARS = 2_048
MAX_QUERY_CHARS = 4_096
MAX_METADATA_CHARS = 32_768
MAX_CHUNK_SIZE = 8_192
MAX_CHUNKS_PER_DOCUMENT = 10_000
MAX_EMBEDDING_DIMENSIONS = 4_096
MAX_RESULTS = 50


class KnowledgeError(RuntimeError):
    """Base class for knowledge-base failures."""


class KnowledgeValidationError(ValueError, KnowledgeError):
    """The request cannot be safely accepted."""


class KnowledgeNotFoundError(KeyError, KnowledgeError):
    """The requested object does not exist in the caller's tenant scope."""


class KnowledgeAccessDenied(PermissionError, KnowledgeError):
    """The caller is not authorized to read or mutate an object."""


class KnowledgeEmbeddingError(KnowledgeError):
    """Embedding provider returned an unusable result."""


class DocumentACL(BaseModel):
    """Read ACL attached to a document.

    ``tenant`` is the default least-surprising development policy: every
    principal in the same tenant can read the document.  ``private`` limits
    access to ``owner_user_id`` and ``restricted`` requires an explicit user
    or role match.  The service still enforces the tenant boundary first.
    """

    visibility: str = "tenant"
    allowed_users: List[str] = Field(default_factory=list)
    allowed_roles: List[str] = Field(default_factory=list)

    @property
    def users(self) -> List[str]:
        """Compatibility alias used by integrations that call them users."""

        return self.allowed_users

    @property
    def roles(self) -> List[str]:
        """Compatibility alias used by integrations that call them roles."""

        return self.allowed_roles


class KnowledgeAccessContext(BaseModel):
    """Trusted identity supplied by the API/authentication boundary."""

    tenant_id: str
    user_id: str = "anonymous"
    roles: List[str] = Field(default_factory=list)

    @property
    def role_set(self) -> set[str]:
        return {role for role in self.roles if isinstance(role, str)}


# Short alias used by callers that describe the object as a RAG principal.
RAGAccessContext = KnowledgeAccessContext


class KnowledgeDocument(BaseModel):
    """Versioned source document retained for citation back-links."""

    id: str
    tenant_id: str
    title: str
    content: str
    source_uri: str = ""
    version: str = "1.0.0"
    owner_user_id: str = "system"
    acl: DocumentACL = Field(default_factory=DocumentACL)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    injection_flags: List[str] = Field(default_factory=list)
    poisoned: bool = False
    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def document_id(self) -> str:
        return self.id


class KnowledgeChunk(BaseModel):
    """A bounded, independently retrievable section of a document."""

    id: str
    document_id: str
    tenant_id: str
    ordinal: int
    text: str
    start_offset: int = 0
    end_offset: int = 0
    embedding: List[float] = Field(default_factory=list)
    content_hash: str = ""
    injection_flags: List[str] = Field(default_factory=list)
    poisoned: bool = False


class Citation(BaseModel):
    """Stable source reference returned with every retrieval hit."""

    citation_id: str
    document_id: str
    chunk_id: str
    title: str
    source_uri: str = ""
    version: str = "1.0.0"
    excerpt: str
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    content_hash: str = ""
    poisoned: bool = False
    injection_flags: List[str] = Field(default_factory=list)
    backlink: str = ""
    # Explicit data-boundary marker for prompt construction code.
    untrusted: bool = True


class KnowledgeIngestRequest(BaseModel):
    """Input for one idempotent document version."""

    tenant_id: str
    title: str
    content: str
    source_uri: str = ""
    version: str = "1.0.0"
    document_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    acl: DocumentACL = Field(default_factory=DocumentACL)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 800
    chunk_overlap: int = 120
    reject_poisoned: bool = False


class KnowledgeIngestResponse(BaseModel):
    document: KnowledgeDocument
    chunks: List[KnowledgeChunk] = Field(default_factory=list)
    chunk_count: int = 0
    injection_flags: List[str] = Field(default_factory=list)
    poisoned: bool = False


class KnowledgeSearchRequest(BaseModel):
    tenant_id: str
    query: str
    user_id: str = "anonymous"
    roles: List[str] = Field(default_factory=list)
    top_k: int = 5
    min_score: float = 0.0
    include_flagged: bool = False


class KnowledgeSearchResult(BaseModel):
    citation: Citation
    text: str
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    # Consumers should render this as data, never as an instruction.
    untrusted: bool = True


class KnowledgeSearchResponse(BaseModel):
    query: str
    tenant_id: str
    results: List[KnowledgeSearchResult] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    total_candidates: int = 0
    filtered_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    abstain: bool = False


class KnowledgeStoreProtocol(Protocol):
    def put_document(self, document: KnowledgeDocument, chunks: Sequence[KnowledgeChunk]) -> None: ...

    def get_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> KnowledgeDocument: ...

    def list_documents(self, *, tenant_id: Optional[str] = None) -> List[KnowledgeDocument]: ...

    def delete_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> None: ...

    def list_chunks(
        self,
        *,
        tenant_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]: ...


# Public spelling used by callers that expect a store interface.
KnowledgeStore = KnowledgeStoreProtocol


def _json_model(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def _parse_model(model_type: Any, payload: Any) -> Any:
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(payload)
    return model_type.parse_obj(payload)


def _safe_text(value: Any, field: str, maximum: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{field} must be a string")
    if required and not value.strip():
        raise KnowledgeValidationError(f"{field} cannot be empty")
    if len(value) > maximum:
        raise KnowledgeValidationError(f"{field} exceeds the maximum length")
    if any(ord(char) < 0x20 and char not in "\t\n\r" for char in value):
        raise KnowledgeValidationError(f"{field} contains a control character")
    return value


def _safe_identifier(value: Any, field: str) -> str:
    text = _safe_text(value, field, 128)
    # IDs are used in event/citation references and logs.  Keep an explicit,
    # bounded alphabet so control/path-like values cannot cross boundaries.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", text):
        raise KnowledgeValidationError(f"{field} has an invalid format")
    return text


def _validate_acl(acl: DocumentACL) -> DocumentACL:
    if acl.visibility not in {"tenant", "private", "restricted"}:
        raise KnowledgeValidationError("acl.visibility must be tenant, private, or restricted")
    for field_name, values in (("acl.allowed_users", acl.allowed_users), ("acl.allowed_roles", acl.allowed_roles)):
        if len(values) > 256:
            raise KnowledgeValidationError(f"{field_name} contains too many entries")
        for value in values:
            _safe_identifier(value, field_name)
    # Return a detached normalized copy so caller mutation cannot alter policy.
    return deepcopy(acl)


def _metadata_copy(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        raise KnowledgeValidationError("metadata must be an object")
    try:
        encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise KnowledgeValidationError("metadata must contain JSON-compatible values") from exc
    if len(encoded) > MAX_METADATA_CHARS:
        raise KnowledgeValidationError("metadata exceeds the maximum size")
    return deepcopy(metadata)


_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|override|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|system|developer)\s+(?:instructions?|messages?)"
            r"|(?:忽略|无视|忘记|覆盖).{0,12}(?:之前|先前|上面|系统|开发者).{0,12}(?:指令|提示|消息)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(?:reveal|print|show|泄露|输出).{0,24}(?:system\s+prompt|developer\s+message|系统提示|开发者消息|密钥|token|api[_ -]?key)",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_execution_request",
        re.compile(
            r"(?:call|invoke|execute|run|使用|调用|执行).{0,24}(?:tool|function|command|bash|curl|工具|函数|命令)",
            re.IGNORECASE,
        ),
    ),
    (
        "safety_bypass",
        re.compile(
            r"(?:disable|bypass|绕过|关闭|禁用).{0,20}(?:safety|security|guardrail|安全|防护|审批)",
            re.IGNORECASE,
        ),
    ),
)


def detect_prompt_injection(value: str) -> List[str]:
    """Return stable flags without modifying or executing the source text."""

    flags: List[str] = []
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            flags.append(name)
    # Zero-width and directional formatting characters can hide instructions.
    if any(unicodedata.category(char) == "Cf" for char in value):
        flags.append("hidden_formatting")
    return flags


class InMemoryKnowledgeStore:
    """Thread-safe bounded store used by local deployments and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._documents: Dict[str, KnowledgeDocument] = {}
        self._chunks: Dict[str, List[KnowledgeChunk]] = {}

    def put_document(self, document: KnowledgeDocument, chunks: Sequence[KnowledgeChunk]) -> None:
        _validate_stored_document(document, chunks)
        with self._lock:
            existing = self._documents.get(document.id)
            if existing is not None and existing.tenant_id != document.tenant_id:
                raise KnowledgeAccessDenied("document belongs to another tenant")
            self._documents[document.id] = deepcopy(document)
            self._chunks[document.id] = [deepcopy(chunk) for chunk in chunks]

    def get_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> KnowledgeDocument:
        with self._lock:
            document = self._documents.get(document_id)
            if document is None or (tenant_id is not None and document.tenant_id != tenant_id):
                raise KnowledgeNotFoundError(f"knowledge document not found: {document_id}")
            return deepcopy(document)

    def list_documents(self, *, tenant_id: Optional[str] = None) -> List[KnowledgeDocument]:
        with self._lock:
            values = [document for document in self._documents.values() if tenant_id is None or document.tenant_id == tenant_id]
            return deepcopy(sorted(values, key=lambda item: (item.id, item.version)))

    def delete_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> None:
        with self._lock:
            document = self._documents.get(document_id)
            if document is None or (tenant_id is not None and document.tenant_id != tenant_id):
                raise KnowledgeNotFoundError(f"knowledge document not found: {document_id}")
            del self._documents[document_id]
            self._chunks.pop(document_id, None)

    def list_chunks(
        self,
        *,
        tenant_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        with self._lock:
            selected: Iterable[Tuple[str, List[KnowledgeChunk]]] = self._chunks.items()
            if document_id is not None:
                selected = ((document_id, self._chunks.get(document_id, [])),)
            values: List[KnowledgeChunk] = []
            for _, chunks in selected:
                values.extend(chunk for chunk in chunks if tenant_id is None or chunk.tenant_id == tenant_id)
            return deepcopy(sorted(values, key=lambda item: (item.document_id, item.ordinal)))


class SQLiteKnowledgeStore:
    """Small durable adapter with the same contract as the memory store."""

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
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES knowledge_documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_tenant ON knowledge_documents(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant_doc ON knowledge_chunks(tenant_id, document_id, ordinal);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put_document(self, document: KnowledgeDocument, chunks: Sequence[KnowledgeChunk]) -> None:
        _validate_stored_document(document, chunks)
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT tenant_id FROM knowledge_documents WHERE id = ?", (document.id,)).fetchone()
            if existing is not None and existing["tenant_id"] != document.tenant_id:
                raise KnowledgeAccessDenied("document belongs to another tenant")
            self._conn.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document.id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO knowledge_documents(id, tenant_id, version, payload) VALUES (?, ?, ?, ?)",
                (document.id, document.tenant_id, document.version, _json_model(document)),
            )
            self._conn.executemany(
                "INSERT INTO knowledge_chunks(id, document_id, tenant_id, ordinal, payload) VALUES (?, ?, ?, ?, ?)",
                [(chunk.id, chunk.document_id, chunk.tenant_id, chunk.ordinal, _json_model(chunk)) for chunk in chunks],
            )

    def get_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> KnowledgeDocument:
        with self._lock:
            row = self._conn.execute("SELECT tenant_id, payload FROM knowledge_documents WHERE id = ?", (document_id,)).fetchone()
            if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
                raise KnowledgeNotFoundError(f"knowledge document not found: {document_id}")
            return _parse_model(KnowledgeDocument, json.loads(row["payload"]))

    def list_documents(self, *, tenant_id: Optional[str] = None) -> List[KnowledgeDocument]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute("SELECT payload FROM knowledge_documents ORDER BY id, version").fetchall()
            else:
                rows = self._conn.execute("SELECT payload FROM knowledge_documents WHERE tenant_id = ? ORDER BY id, version", (tenant_id,)).fetchall()
            return [_parse_model(KnowledgeDocument, json.loads(row["payload"])) for row in rows]

    def delete_document(self, document_id: str, *, tenant_id: Optional[str] = None) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT tenant_id FROM knowledge_documents WHERE id = ?", (document_id,)).fetchone()
            if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
                raise KnowledgeNotFoundError(f"knowledge document not found: {document_id}")
            self._conn.execute("DELETE FROM knowledge_documents WHERE id = ?", (document_id,))

    def list_chunks(
        self,
        *,
        tenant_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        with self._lock:
            clauses: List[str] = []
            params: List[str] = []
            if tenant_id is not None:
                clauses.append("tenant_id = ?")
                params.append(tenant_id)
            if document_id is not None:
                clauses.append("document_id = ?")
                params.append(document_id)
            query = "SELECT payload FROM knowledge_chunks"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY document_id, ordinal"
            rows = self._conn.execute(query, params).fetchall()
            return [_parse_model(KnowledgeChunk, json.loads(row["payload"])) for row in rows]


def _validate_stored_document(document: KnowledgeDocument, chunks: Sequence[KnowledgeChunk]) -> None:
    _safe_identifier(document.id, "document.id")
    _safe_identifier(document.tenant_id, "document.tenant_id")
    _safe_text(document.title, "document.title", MAX_TITLE_CHARS)
    _safe_text(document.content, "document.content", MAX_DOCUMENT_CHARS)
    _safe_text(document.source_uri, "document.source_uri", MAX_SOURCE_URI_CHARS, required=False)
    _validate_acl(document.acl)
    _metadata_copy(document.metadata)
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise KnowledgeValidationError("document contains too many chunks")
    for expected, chunk in enumerate(chunks):
        if chunk.document_id != document.id or chunk.tenant_id != document.tenant_id or chunk.ordinal != expected:
            raise KnowledgeValidationError("chunk document, tenant, or ordinal does not match")
        _safe_identifier(chunk.id, "chunk.id")
        _safe_text(chunk.text, "chunk.text", MAX_CHUNK_SIZE, required=False)
        if chunk.start_offset < 0 or chunk.end_offset < chunk.start_offset or chunk.end_offset > len(document.content):
            raise KnowledgeValidationError("chunk offsets are outside the document")
        if document.content[chunk.start_offset : chunk.end_offset] != chunk.text:
            raise KnowledgeValidationError("chunk text does not match document content")
        expected_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if chunk.content_hash and chunk.content_hash != expected_hash:
            raise KnowledgeValidationError("chunk content hash does not match text")
        if len(chunk.embedding) > MAX_EMBEDDING_DIMENSIONS:
            raise KnowledgeValidationError("chunk embedding is too large")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in chunk.embedding):
            raise KnowledgeValidationError("chunk embedding contains an invalid value")
    expected_document_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
    if document.content_hash and document.content_hash != expected_document_hash:
        raise KnowledgeValidationError("document content hash does not match content")


def _tokenize(value: str) -> List[str]:
    # Keeping CJK characters as individual tokens makes the offline matcher
    # useful without a heavyweight language tokenizer.
    return re.findall(r"[a-z0-9_:.@/+-]+|[\u4e00-\u9fff]", value.lower())


def _lexical_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    query_counts: Dict[str, int] = {}
    text_counts: Dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    for token in text_tokens:
        text_counts[token] = text_counts.get(token, 0) + 1
    overlap = sum(min(count, text_counts.get(token, 0)) for token, count in query_counts.items())
    score = overlap / max(1, len(query_tokens))
    if query.strip().lower() in text.lower():
        score = max(score, 1.0)
    return min(1.0, score)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    # Map cosine [-1, 1] into a bounded relevance score [0, 1].
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


class RAGService:
    """Ingest and retrieve knowledge while enforcing a trusted scope."""

    def __init__(
        self,
        *,
        store: Optional[KnowledgeStoreProtocol] = None,
        model_gateway: Optional[ModelGateway] = None,
        embedding_model: str = "bge-m3",
        max_document_chars: int = MAX_DOCUMENT_CHARS,
        max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    ) -> None:
        self.store = store or InMemoryKnowledgeStore()
        self.model_gateway = model_gateway or DeterministicModelGateway()
        self.embedding_model = _safe_text(embedding_model, "embedding_model", 128)
        self.max_document_chars = max(1, min(int(max_document_chars), MAX_DOCUMENT_CHARS))
        self.max_chunks_per_document = max(1, min(int(max_chunks_per_document), MAX_CHUNKS_PER_DOCUMENT))

    def ingest_document(
        self,
        request: Optional[KnowledgeIngestRequest] = None,
        *,
        context: Optional[KnowledgeAccessContext] = None,
        embedding_context: Optional[ModelCallContext] = None,
        **kwargs: Any,
    ) -> KnowledgeIngestResponse:
        """Validate, flag, chunk, embed and persist one document version."""

        if request is None:
            request = KnowledgeIngestRequest(**kwargs)
        elif kwargs:
            raise KnowledgeValidationError("request and keyword fields cannot be combined")
        if not isinstance(request, KnowledgeIngestRequest):
            request = _parse_model(KnowledgeIngestRequest, request)
        if context is None:
            raise KnowledgeValidationError("trusted knowledge context is required for ingest")
        tenant_id = _safe_identifier(request.tenant_id, "tenant_id")
        self._assert_context_tenant(context, tenant_id)
        title = _safe_text(request.title, "title", MAX_TITLE_CHARS)
        content = _safe_text(request.content, "content", self.max_document_chars)
        source_uri = _safe_text(request.source_uri, "source_uri", MAX_SOURCE_URI_CHARS, required=False)
        version = _safe_text(request.version, "version", 128)
        document_id = request.document_id or uuid.uuid4().hex
        document_id = _safe_identifier(document_id, "document_id")
        owner = request.owner_user_id or context.user_id
        owner = _safe_identifier(owner, "owner_user_id")
        if owner != context.user_id and not self._is_admin(context):
            raise KnowledgeAccessDenied("document owner must match the trusted caller")
        acl = _validate_acl(request.acl)
        metadata = _metadata_copy(request.metadata)
        try:
            chunk_size = int(request.chunk_size)
            chunk_overlap = int(request.chunk_overlap)
        except (TypeError, ValueError) as exc:
            raise KnowledgeValidationError("chunk_size and chunk_overlap must be integers") from exc
        if chunk_size < 16 or chunk_size > MAX_CHUNK_SIZE:
            raise KnowledgeValidationError("chunk_size must be between 16 and 8192")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise KnowledgeValidationError("chunk_overlap must be non-negative and smaller than chunk_size")

        document_flags = detect_prompt_injection(content)
        if document_flags and request.reject_poisoned:
            raise KnowledgeValidationError("document was rejected because prompt-injection indicators were detected")
        pieces = self._chunk(content, chunk_size, chunk_overlap)
        if len(pieces) > self.max_chunks_per_document:
            raise KnowledgeValidationError("document produces too many chunks")
        texts = [piece[0] for piece in pieces]
        embeddings = self._embed(texts, context=embedding_context)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = utc_now()
        document = KnowledgeDocument(
            id=document_id,
            tenant_id=tenant_id,
            title=title,
            content=content,
            source_uri=source_uri,
            version=version,
            owner_user_id=owner,
            acl=acl,
            metadata=metadata,
            content_hash=content_hash,
            injection_flags=document_flags,
            poisoned=bool(document_flags),
            created_at=now,
            updated_at=now,
        )
        chunks: List[KnowledgeChunk] = []
        for ordinal, ((text, start, end), embedding) in enumerate(zip(pieces, embeddings)):
            flags = detect_prompt_injection(text)
            chunks.append(
                KnowledgeChunk(
                    id=f"{document_id}:{ordinal}",
                    document_id=document_id,
                    tenant_id=tenant_id,
                    ordinal=ordinal,
                    text=text,
                    start_offset=start,
                    end_offset=end,
                    embedding=embedding,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    injection_flags=flags,
                    poisoned=bool(flags),
                )
            )
        self.store.put_document(document, chunks)
        return KnowledgeIngestResponse(
            document=deepcopy(document),
            chunks=deepcopy(chunks),
            chunk_count=len(chunks),
            injection_flags=list(document_flags),
            poisoned=bool(document_flags),
        )

    # ``ingest`` is a concise compatibility alias for API adapters.
    ingest = ingest_document

    def get_document(self, document_id: str, *, context: KnowledgeAccessContext) -> KnowledgeDocument:
        document_id = _safe_identifier(document_id, "document_id")
        self._validate_context(context)
        document = self.store.get_document(document_id, tenant_id=context.tenant_id)
        if not document.enabled or not self._can_read(document, context):
            raise KnowledgeNotFoundError(f"knowledge document not found: {document_id}")
        return document

    def delete_document(self, document_id: str, *, context: KnowledgeAccessContext) -> None:
        self._validate_context(context)
        document = self.store.get_document(document_id, tenant_id=context.tenant_id)
        if not self._is_admin(context) and document.owner_user_id != context.user_id:
            raise KnowledgeAccessDenied("only the document owner or an administrator may delete it")
        self.store.delete_document(document_id, tenant_id=context.tenant_id)

    def search(
        self,
        request: Optional[KnowledgeSearchRequest | str] = None,
        *,
        context: Optional[KnowledgeAccessContext] = None,
        embedding_context: Optional[ModelCallContext] = None,
        **kwargs: Any,
    ) -> KnowledgeSearchResponse:
        """Return ACL-filtered citations ranked by lexical and vector scores."""

        if context is None:
            raise KnowledgeValidationError("tenant context is required for search")
        if isinstance(request, str):
            # Support the concise ``search(query, context=ctx, top_k=...)``
            # adapter form while keeping tenant and identity authoritative
            # from the trusted context.
            fields = self._context_fields(context)
            fields.update(kwargs)
            fields["tenant_id"] = context.tenant_id
            fields["user_id"] = context.user_id
            fields["roles"] = list(context.roles)
            fields["query"] = request
            request = KnowledgeSearchRequest(**fields)
        elif request is None:
            # ``search(query="...", context=ctx)`` is the common service
            # adapter form; bind the tenant and identity from the trusted
            # context rather than accepting them from an untrusted payload.
            if context is not None:
                fields = self._context_fields(context)
                fields.update(kwargs)
                # Identity fields are always authoritative from the trusted
                # context; caller-supplied kwargs may only control query
                # options such as top_k/min_score.
                fields["tenant_id"] = context.tenant_id
                fields["user_id"] = context.user_id
                fields["roles"] = list(context.roles)
                request = KnowledgeSearchRequest(**fields)
            else:
                request = KnowledgeSearchRequest(**kwargs)
        elif kwargs:
            raise KnowledgeValidationError("request and keyword fields cannot be combined")
        if not isinstance(request, KnowledgeSearchRequest):
            request = _parse_model(KnowledgeSearchRequest, request)
        tenant_id = _safe_identifier(request.tenant_id, "tenant_id")
        query = _safe_text(request.query, "query", MAX_QUERY_CHARS)
        self._validate_context(context)
        effective_context = context
        self._assert_context_tenant(effective_context, tenant_id)
        if not isinstance(request.top_k, int) or isinstance(request.top_k, bool) or request.top_k < 1 or request.top_k > MAX_RESULTS:
            raise KnowledgeValidationError("top_k must be between 1 and 50")
        try:
            min_score = float(request.min_score)
        except (TypeError, ValueError) as exc:
            raise KnowledgeValidationError("min_score must be a number") from exc
        if not math.isfinite(min_score) or min_score < 0 or min_score > 1:
            raise KnowledgeValidationError("min_score must be between 0 and 1")

        documents = {document.id: document for document in self.store.list_documents(tenant_id=tenant_id)}
        all_chunks = self.store.list_chunks(tenant_id=tenant_id)
        candidates: List[Tuple[KnowledgeChunk, KnowledgeDocument, float, float, bool]] = []
        filtered = 0
        for chunk in all_chunks:
            document = documents.get(chunk.document_id)
            if document is None or not document.enabled or not self._can_read(document, effective_context):
                filtered += 1
                continue
            # A document-level flag is inherited by every chunk.  This avoids
            # a boundary-spanning injection being hidden in an otherwise clean
            # chunk and keeps the default retrieval policy fail-closed.
            flagged = bool(document.poisoned or document.injection_flags or chunk.poisoned or chunk.injection_flags)
            if flagged and not request.include_flagged:
                filtered += 1
                continue
            lexical = _lexical_score(query, chunk.text)
            candidates.append((chunk, document, lexical, 0.0, flagged))
        if not candidates:
            return KnowledgeSearchResponse(
                query=query,
                tenant_id=tenant_id,
                total_candidates=len(all_chunks),
                filtered_count=filtered,
                warnings=["NO_SAFE_EVIDENCE"],
                abstain=True,
            )
        query_embedding = self._embed([query], context=embedding_context)[0]
        ranked: List[KnowledgeSearchResult] = []
        for chunk, document, lexical, _, flagged in candidates:
            vector = _cosine(query_embedding, chunk.embedding)
            # The deterministic offline gateway intentionally returns stable
            # placeholder vectors, not semantic representations.  Do not let
            # their near-collinearity turn every unrelated chunk into evidence;
            # a real gateway may still provide vector-only matches.
            if isinstance(self.model_gateway, DeterministicModelGateway) and lexical <= 0:
                vector = 0.0
            score = round(0.55 * lexical + 0.45 * vector, 6)
            if score <= 0 or score < min_score:
                filtered += 1
                continue
            citation_id = f"{document.id}:{chunk.ordinal}"
            backlink = self._backlink(document, chunk)
            citation_flags = list(dict.fromkeys([*document.injection_flags, *chunk.injection_flags]))
            citation = Citation(
                citation_id=citation_id,
                document_id=document.id,
                chunk_id=chunk.id,
                title=document.title,
                source_uri=document.source_uri,
                version=document.version,
                excerpt=chunk.text[:500],
                score=score,
                lexical_score=round(lexical, 6),
                vector_score=round(vector, 6),
                content_hash=chunk.content_hash or document.content_hash,
                poisoned=flagged,
                injection_flags=citation_flags,
                backlink=backlink,
            )
            ranked.append(
                KnowledgeSearchResult(
                    citation=citation,
                    text=chunk.text,
                    score=score,
                    lexical_score=round(lexical, 6),
                    vector_score=round(vector, 6),
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.citation.document_id, item.citation.chunk_id))
        ranked = ranked[: request.top_k]
        warnings: List[str] = []
        if request.include_flagged and any(result.citation.poisoned for result in ranked):
            warnings.append("FLAGGED_CONTENT_INCLUDED_AS_UNTRUSTED_DATA")
        if not ranked:
            warnings.append("NO_SAFE_EVIDENCE")
        return KnowledgeSearchResponse(
            query=query,
            tenant_id=tenant_id,
            results=ranked,
            citations=[result.citation for result in ranked],
            total_candidates=len(all_chunks),
            filtered_count=filtered,
            warnings=warnings,
            abstain=not bool(ranked),
        )

    def _embed(
        self,
        texts: Sequence[str],
        *,
        context: Optional[ModelCallContext] = None,
    ) -> List[List[float]]:
        if not texts:
            return []
        self._check_embedding_context(context)
        try:
            embed_with_context = getattr(self.model_gateway, "embed_with_context", None)
            if callable(embed_with_context):
                vectors = embed_with_context(
                    model=self.embedding_model,
                    texts=list(texts),
                    context=context,
                )
            else:
                # Keep compatibility with lightweight third-party adapters
                # that only implement the original ``embed`` method.
                vectors = self.model_gateway.embed(model=self.embedding_model, texts=list(texts))
        except Exception as exc:
            raise KnowledgeEmbeddingError("embedding provider failed") from exc
        self._check_embedding_context(context)
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise KnowledgeEmbeddingError("embedding provider returned an invalid vector count")
        normalized: List[List[float]] = []
        expected_dimension: Optional[int] = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise KnowledgeEmbeddingError("embedding provider returned an empty vector")
            if len(vector) > MAX_EMBEDDING_DIMENSIONS:
                raise KnowledgeEmbeddingError("embedding vector exceeds the maximum dimension")
            if expected_dimension is None:
                expected_dimension = len(vector)
            if len(vector) != expected_dimension:
                raise KnowledgeEmbeddingError("embedding vectors have inconsistent dimensions")
            clean: List[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise KnowledgeEmbeddingError("embedding vector contains a non-finite value")
                clean.append(float(value))
            normalized.append(clean)
        return normalized

    @staticmethod
    def _check_embedding_context(context: Optional[ModelCallContext]) -> None:
        """Fail before/after an embedding call when its execution budget is gone."""

        if context is None:
            return
        if context.is_cancelled():
            raise KnowledgeEmbeddingError("embedding request cancelled")
        # ``inf`` keeps an unconstrained context unconstrained while still
        # detecting an expired timeout/deadline when one was supplied.
        if context.remaining_seconds(float("inf")) <= 0:
            raise KnowledgeEmbeddingError("embedding request timed out")

    @staticmethod
    def _chunk(content: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        pieces: List[Tuple[str, int, int]] = []
        start = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            text = content[start:end]
            if text:
                pieces.append((text, start, end))
            if end >= len(content):
                break
            start = end - overlap
        return pieces

    @staticmethod
    def _backlink(document: KnowledgeDocument, chunk: KnowledgeChunk) -> str:
        suffix = "#chunk=" + quote(chunk.id, safe="")
        return (document.source_uri.rstrip("#") + suffix) if document.source_uri else f"knowledge://{document.id}/{quote(chunk.id, safe='')}"

    @staticmethod
    def _context_fields(context: Optional[KnowledgeAccessContext]) -> Dict[str, Any]:
        if context is None:
            raise KnowledgeValidationError("tenant context is required for search")
        return {"tenant_id": context.tenant_id, "user_id": context.user_id, "roles": list(context.roles)}

    @staticmethod
    def _validate_context(context: KnowledgeAccessContext) -> None:
        if not isinstance(context, KnowledgeAccessContext):
            raise KnowledgeValidationError("trusted knowledge context is required")
        _safe_identifier(context.tenant_id, "context.tenant_id")
        _safe_identifier(context.user_id, "context.user_id")
        if len(context.roles) > 128:
            raise KnowledgeValidationError("context contains too many roles")
        for role in context.roles:
            _safe_identifier(role, "context.role")

    @classmethod
    def _assert_context_tenant(cls, context: KnowledgeAccessContext, tenant_id: str) -> None:
        cls._validate_context(context)
        if context.tenant_id != tenant_id:
            raise KnowledgeAccessDenied("tenant access is not authorized")

    @staticmethod
    def _is_admin(context: KnowledgeAccessContext) -> bool:
        return bool(context.role_set.intersection({"admin", "knowledge_admin"}))

    @classmethod
    def _can_read(cls, document: KnowledgeDocument, context: KnowledgeAccessContext) -> bool:
        if document.tenant_id != context.tenant_id:
            return False
        if cls._is_admin(context):
            return True
        acl = document.acl
        if acl.visibility == "tenant":
            return True
        if acl.visibility == "private":
            return document.owner_user_id == context.user_id
        return document.owner_user_id == context.user_id or context.user_id in acl.allowed_users or bool(context.role_set.intersection(acl.allowed_roles))


# Friendly factory aliases for application bootstrap code.
KnowledgeBaseService = RAGService
InMemoryRAGStore = InMemoryKnowledgeStore
SQLiteRAGStore = SQLiteKnowledgeStore
RAGStore = KnowledgeStoreProtocol
ACL = DocumentACL
Document = KnowledgeDocument
Chunk = KnowledgeChunk
IngestRequest = KnowledgeIngestRequest
IngestResponse = KnowledgeIngestResponse
SearchRequest = KnowledgeSearchRequest
SearchResponse = KnowledgeSearchResponse
SearchResult = KnowledgeSearchResult

# ``retrieve`` is conventional wording for a RAG service and keeps adapters
# thin without duplicating logic or security checks.
RAGService.retrieve = RAGService.search


def build_knowledge_store(*, backend: str = "memory", path: str | Path = "data/qingling-knowledge.db") -> KnowledgeStoreProtocol:
    normalized = str(backend).strip().lower()
    if normalized == "memory":
        return InMemoryKnowledgeStore()
    if normalized == "sqlite":
        return SQLiteKnowledgeStore(path)
    raise KnowledgeValidationError(f"unsupported knowledge store backend: {backend}")


__all__ = [
    "ACL",
    "Citation",
    "Chunk",
    "Document",
    "DocumentACL",
    "InMemoryKnowledgeStore",
    "InMemoryRAGStore",
    "KnowledgeAccessContext",
    "KnowledgeAccessDenied",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeEmbeddingError",
    "KnowledgeError",
    "KnowledgeIngestRequest",
    "KnowledgeIngestResponse",
    "KnowledgeNotFoundError",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
    "KnowledgeSearchResult",
    "KnowledgeStore",
    "KnowledgeStoreProtocol",
    "KnowledgeValidationError",
    "IngestRequest",
    "IngestResponse",
    "RAGAccessContext",
    "RAGStore",
    "RAGService",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "KnowledgeBaseService",
    "SQLiteKnowledgeStore",
    "SQLiteRAGStore",
    "build_knowledge_store",
    "detect_prompt_injection",
]
