"""Deterministic, read-only asset security assessment use case."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional

from .security_models import (
    AssessmentFinding,
    AssessmentResult,
    AssetRecord,
    EvidenceRef,
    ExposureRecord,
    VulnerabilityRecord,
)
from .security_sources import (
    AssetSource,
    ExposureSource,
    SourceContext,
    SourceResult,
    VulnerabilitySource,
)
from .models import ComponentExecution, ComponentManifest, ComponentResult, ComponentSecurity


_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


class SecurityAssessmentService:
    """Combines three read-only sources without invoking an LLM or mutating data."""

    def __init__(self, asset_source: AssetSource, vulnerability_source: VulnerabilitySource, exposure_source: ExposureSource) -> None:
        self.asset_source = asset_source
        self.vulnerability_source = vulnerability_source
        self.exposure_source = exposure_source

    def assess(self, target: str | Mapping[str, Any], context: SourceContext) -> AssessmentResult:
        query = self._normalize_target(target)
        context.check_active()
        asset_result = self._search(self.asset_source, query, context)
        vulnerability_result = self._search(self.vulnerability_source, query, context)
        exposure_result = self._search(self.exposure_source, query, context)
        return self._build_result(query, asset_result, vulnerability_result, exposure_result)

    @staticmethod
    def _normalize_target(target: str | Mapping[str, Any]) -> Dict[str, str]:
        if isinstance(target, str):
            value = target.strip()
            if not value:
                raise ValueError("assessment target cannot be empty")
            return {"ip": value}
        if not isinstance(target, Mapping):
            raise ValueError("assessment target must be a string or object")
        normalized = {key: str(target[key]).strip() for key in ("asset_id", "ip", "hostname") if target.get(key) not in (None, "")}
        if not normalized:
            raise ValueError("assessment target requires asset_id, ip, or hostname")
        return normalized

    @staticmethod
    def _search(source: Any, query: Mapping[str, Any], context: SourceContext) -> SourceResult[Any]:
        try:
            result = source.search(query, context)
            if not isinstance(result, SourceResult):
                raise TypeError("security source must return SourceResult")
            return result
        except TimeoutError:
            raise
        except Exception as exc:
            # Source failure is evidence of incompleteness, not proof of safety.
            source_id = str(getattr(source, "source_id", source.__class__.__name__))
            return SourceResult(items=[], source_id=source_id, partial=True, errors=[str(exc)[:256]])

    def _build_result(self, target: Dict[str, str], asset: SourceResult[Any], vulnerabilities: SourceResult[Any], exposure: SourceResult[Any]) -> AssessmentResult:
        findings: list[AssessmentFinding] = []
        evidence: list[EvidenceRef] = []
        missing: list[str] = []
        result_sets = (("asset", asset), ("vulnerability", vulnerabilities), ("exposure", exposure))
        active_vulnerabilities = [record for record in vulnerabilities.items if record.status not in {"remediated", "accepted"}]
        for kind, source_result in result_sets:
            if not source_result.items:
                missing.append(f"{kind}_inventory")
            if source_result.partial or source_result.errors:
                missing.append(f"{kind}_source")
            for record in source_result.items:
                record_id = self._record_id(record, kind)
                evidence_id = f"{source_result.source_id}:{record_id}"
                evidence.append(EvidenceRef(
                    evidence_id=evidence_id,
                    subject_type=kind,
                    subject_id=record_id,
                    source_id=source_result.source_id,
                    observed_at=record.observed_at,
                    citation=f"{source_result.source_id}/{record_id}",
                    attributes=_dump(record),
                ))
                if kind == "asset":
                    severity = record.criticality
                    if severity in {"high", "critical"}:
                        findings.append(AssessmentFinding(finding_id=evidence_id, kind="asset", severity=severity,
                            title="关键资产", description=f"资产关键性为 {severity}。", evidence_ids=[evidence_id]))
                elif kind == "vulnerability" and record in active_vulnerabilities:
                    findings.append(AssessmentFinding(finding_id=evidence_id, kind="vulnerability", severity=record.severity,
                        title=record.title or (record.cve or "未命名漏洞"), description=f"存在 {record.severity} 级未闭环漏洞。", evidence_ids=[evidence_id]))
                elif kind == "exposure":
                    findings.append(AssessmentFinding(finding_id=evidence_id, kind="exposure", severity=record.severity,
                        title=f"{record.exposure_type} 暴露面", description=f"发现 {record.severity} 级暴露面。", evidence_ids=[evidence_id]))
        rank = max((_SEVERITY_RANK.get(finding.severity, 0) for finding in findings), default=-1)
        risk_level = ("low", "low", "medium", "high", "critical")[rank] if rank >= 0 else "unknown"
        source_count = sum(bool(result.items) and not result.errors for _, result in result_sets)
        confidence = 0.0 if not source_count else min(0.95, 0.35 + 0.2 * source_count)
        if missing:
            confidence = max(0.0, confidence - 0.1 * len(set(missing)))
        partial = bool(missing)
        recommendations = self._recommendations(risk_level, active_vulnerabilities, exposure.items, missing)
        return AssessmentResult(
            assessment_id=str(uuid.uuid4()), target=target, risk_level=risk_level, confidence=round(confidence, 3),
            findings=findings, evidence_refs=evidence, missing_evidence=sorted(set(missing)), recommendations=recommendations, partial=partial,
        )

    @staticmethod
    def _record_id(record: Any, kind: str) -> str:
        return str(getattr(record, {"asset": "asset_id", "vulnerability": "vulnerability_id", "exposure": "exposure_id"}[kind]))

    @staticmethod
    def _recommendations(risk_level: str, vulnerabilities: list[VulnerabilityRecord], exposures: list[ExposureRecord], missing: list[str]) -> list[str]:
        recommendations: list[str] = []
        if vulnerabilities:
            recommendations.append("优先核实并修复未闭环漏洞，确认补丁或风险接受状态。")
        if exposures:
            recommendations.append("核对暴露面是否符合业务预期，收敛不必要的公网端口和服务。")
        if missing:
            recommendations.append("补齐缺失的资产、漏洞和暴露面数据后重新评估。")
        if not recommendations:
            recommendations.append("保持只读复核并按数据新鲜度定期重新评估。")
        return recommendations


def assess_component(context: Any, *, service: SecurityAssessmentService) -> ComponentResult:
    """Workflow adapter; all identity comes from the trusted ComponentContext."""
    value = context.inputs.get("target", context.inputs.get("input"))
    source_context = SourceContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        run_id=context.run_id,
        correlation_id=context.correlation_id,
        roles=tuple(context.roles),
        deadline_monotonic=context.deadline_monotonic,
        cancel_event=context.cancel_event,
    )
    result = service.assess(value, source_context)
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    else:
        payload = result.dict()
    return ComponentResult(
        status="partial" if result.partial else ("abstain" if result.risk_level == "unknown" else "succeeded"),
        outputs=payload,
        evidence=[item.model_dump(mode="json") if hasattr(item, "model_dump") else item.dict() for item in result.evidence_refs],
        warnings=list(result.missing_evidence),
    )


def register_security_assessment_component(registry: Any, service: SecurityAssessmentService) -> None:
    """Install the versioned low-risk read-only assessment component."""
    name = "security.asset.assess"
    if registry.has(name):
        return
    registry.register(
        name,
        lambda context: assess_component(context, service=service),
        input_schema={"type": "object", "properties": {"target": {"type": "object"}, "input": {"type": "object"}}, "additionalProperties": True},
        output_schema={"type": "object", "required": ["assessment_id", "target", "risk_level", "confidence", "findings", "evidence_refs", "missing_evidence"]},
        manifest=ComponentManifest(
            id=name,
            display_name="Asset security assessment",
            description="Deterministic read-only assessment over registered asset, vulnerability and exposure sources.",
            kind="function",
            input_schema={"type": "object", "properties": {"target": {"type": "object"}, "input": {"type": "object"}}, "additionalProperties": True},
            output_schema={"type": "object", "required": ["assessment_id", "target", "risk_level", "confidence", "findings", "evidence_refs", "missing_evidence"]},
            security=ComponentSecurity(permission="read", risk_level="low", tenant_scoped=True),
            execution=ComponentExecution(idempotent=True, max_retries=1),
        ),
    )
