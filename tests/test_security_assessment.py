from app.security_assessment import SecurityAssessmentService, register_security_assessment_component
from app.security_models import AssetRecord, ExposureRecord, VulnerabilityRecord
from app.security_sources import (
    InMemoryAssetSource,
    InMemoryExposureSource,
    InMemoryVulnerabilitySource,
    SourceContext,
)
from app.component_registry import ComponentContext, ComponentRegistry
from app.models import AgentProfile, Run


def context(tenant="tenant-a"):
    return SourceContext(tenant_id=tenant, user_id="analyst", run_id="run-1", correlation_id="corr-1")


def service_with_records():
    assets = InMemoryAssetSource([
        AssetRecord(asset_id="asset-1", tenant_id="tenant-a", ip="10.0.0.1", criticality="high"),
        AssetRecord(asset_id="other", tenant_id="tenant-b", ip="10.0.0.1", criticality="critical"),
    ])
    vulnerabilities = InMemoryVulnerabilitySource([
        VulnerabilityRecord(vulnerability_id="cve-1", tenant_id="tenant-a", asset_id="asset-1", ip="10.0.0.1", severity="critical", cve="CVE-2026-1"),
        VulnerabilityRecord(vulnerability_id="closed", tenant_id="tenant-a", asset_id="asset-1", ip="10.0.0.1", severity="critical", status="remediated"),
    ])
    exposures = InMemoryExposureSource([
        ExposureRecord(exposure_id="exp-1", tenant_id="tenant-a", asset_id="asset-1", ip="10.0.0.1", severity="high", internet_exposed=True),
    ])
    return SecurityAssessmentService(assets, vulnerabilities, exposures)


def test_memory_sources_are_tenant_scoped():
    result = service_with_records().asset_source.search({"ip": "10.0.0.1"}, context())
    assert [record.asset_id for record in result.items] == ["asset-1"]


def test_assessment_is_deterministic_and_ignores_closed_vulnerabilities():
    result = service_with_records().assess({"ip": "10.0.0.1"}, context())
    assert result.risk_level == "critical"
    assert result.confidence > 0.5
    assert {finding.kind for finding in result.findings} == {"asset", "vulnerability", "exposure"}
    assert [ref.subject_id for ref in result.evidence_refs].count("closed") == 1
    assert "vulnerability_inventory" not in result.missing_evidence


def test_missing_sources_abstain_and_never_claim_safety():
    service = SecurityAssessmentService(InMemoryAssetSource(), InMemoryVulnerabilitySource(), InMemoryExposureSource())
    result = service.assess("10.0.0.2", context())
    assert result.risk_level == "unknown"
    assert result.confidence == 0.0
    assert result.partial is True
    assert set(result.missing_evidence) == {"asset_inventory", "vulnerability_inventory", "exposure_inventory"}
    assert any("补齐" in item for item in result.recommendations)


def test_assessment_component_is_registered_as_read_only():
    registry = ComponentRegistry()
    service = service_with_records()
    register_security_assessment_component(registry, service)
    manifest = registry.definition("security.asset.assess").manifest
    assert manifest.security.permission == "read"
    assert manifest.security.risk_level == "low"
