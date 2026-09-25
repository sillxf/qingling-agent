"""Domain contracts for the read-only asset security assessment use case."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


Severity = Literal["info", "low", "medium", "high", "critical"]
AssessmentRisk = Literal["unknown", "low", "medium", "high", "critical"]
EvidenceStatus = Literal["confirmed", "unknown", "conflict", "stale"]


class AssetRecord(BaseModel):
    """A normalized asset inventory record. It is data, never an instruction."""

    asset_id: str
    tenant_id: str
    ip: Optional[str] = None
    hostname: Optional[str] = None
    criticality: Severity = "medium"
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)


class VulnerabilityRecord(BaseModel):
    vulnerability_id: str
    tenant_id: str
    asset_id: Optional[str] = None
    ip: Optional[str] = None
    title: str = ""
    severity: Severity = "medium"
    cvss: Optional[float] = Field(default=None, ge=0, le=10)
    status: Literal["open", "in_progress", "remediated", "accepted"] = "open"
    cve: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)


class ExposureRecord(BaseModel):
    exposure_id: str
    tenant_id: str
    asset_id: Optional[str] = None
    ip: Optional[str] = None
    exposure_type: str = "network"
    severity: Severity = "medium"
    internet_exposed: bool = False
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    protocol: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)


class EvidenceRef(BaseModel):
    evidence_id: str
    subject_type: Literal["asset", "vulnerability", "exposure"]
    subject_id: str
    source_id: str
    observed_at: datetime = Field(default_factory=utc_now)
    status: EvidenceStatus = "confirmed"
    confidence: float = Field(default=1.0, ge=0, le=1)
    citation: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class AssessmentFinding(BaseModel):
    finding_id: str
    kind: Literal["asset", "vulnerability", "exposure", "data_quality"]
    severity: Severity
    title: str
    description: str
    evidence_ids: List[str] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    assessment_id: str
    target: Dict[str, str]
    risk_level: AssessmentRisk
    confidence: float = Field(ge=0, le=1)
    findings: List[AssessmentFinding] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    partial: bool = False
    assessed_at: datetime = Field(default_factory=utc_now)
