"""Public evidence contracts, including adversarial and incomplete inputs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.demo import run_demo
from app.evidence_report import build_report


def groups(**records):
    return {key: {"ip": "192.0.2.10", "source": "request_supplied_evidence", "records": records.get(key, [])}
            for key in ("asset", "vulnerabilities", "events")}


def test_risk_findings_have_resolvable_citations_and_no_fake_confidence():
    inputs = groups(
        asset=[{"id": "asset-1", "severity": "critical"}],
        vulnerabilities=[{"id": "vuln-1", "severity": "critical", "status": "open"}],
        events=[{"id": "event-1", "severity": "high", "status": "open"}],
    )
    before = deepcopy(inputs)
    result = build_report(inputs)
    assert inputs == before
    assert result == build_report(inputs)
    assert result["assessment"] == "risk_detected"
    assert result["observed_risk_level"] == "critical"
    assert result["evidence_count"] == 3
    assert len(result["findings"]) == 2
    assert "confidence" not in result
    refs = {item["citation_id"] for item in result["evidence"]}
    assert len(refs) == result["evidence_count"]
    assert all(set(finding["evidence_ids"]) <= refs for finding in result["findings"])
    assert all(item["verification_status"] == "unverified" for item in result["evidence"])


@pytest.mark.parametrize("state", ["remediated", "accepted", "resolved", "closed", "false_positive", "benign"])
def test_closed_evidence_is_retained_but_not_an_active_risk(state):
    result = build_report(groups(vulnerabilities=[{"id": "vuln-1", "severity": "critical", "status": state}]))
    assert result["assessment"] == "insufficient_evidence"
    assert result["evidence_count"] == 1
    assert result["findings"] == []


def test_no_evidence_or_asset_importance_never_proves_safety_or_attack():
    for inputs in (groups(), groups(asset=[{"severity": "critical"}])):
        result = build_report(inputs)
        assert result["assessment"] == "insufficient_evidence"
        assert result["observed_risk_level"] == "unknown"
        assert result["missing_evidence"]
        assert result["limitations"]


def test_unknown_severity_and_duplicate_source_ids_do_not_fabricate_findings():
    result = build_report(groups(events=[{"id": "same", "severity": "unknown"}, {"id": "same", "severity": "high"}]))
    assert len(result["findings"]) == 1
    assert len({item["citation_id"] for item in result["evidence"]}) == 2


def test_markdown_does_not_render_untrusted_evidence_as_html_or_links():
    result = build_report(groups(events=[{"title": '<img src=x> [click](https://invalid.test)\n# forged section', "severity": "high"}]))
    assert "<img" not in result["report"]
    assert "[click](" not in result["report"]
    assert "\n# forged" not in result["report"]
    assert "&lt;img" in result["report"]


def test_mismatched_targets_fail_closed():
    inputs = groups()
    inputs["events"]["ip"] = "192.0.2.99"
    with pytest.raises(ValueError, match="one IP"):
        build_report(inputs)


def test_bundled_dataset_is_explicitly_synthetic():
    import app
    dataset = json.loads((Path(app.__file__).parent / "resources/examples/security_cases.json").read_text(encoding="utf-8"))
    assert dataset["provenance"] == "synthetic_fixtures_not_real_security_events"
    assert len(dataset["cases"]) == 6


def test_end_to_end_api_acceptance_demo():
    result = run_demo()
    assert result["passed"] == result["total"] == 9
    assert result["not_a_model_accuracy_benchmark"] is True
    assert result["sample_report"]["findings"]


def test_acceptance_demo_with_phone_like_generated_ids(phone_like_uuids):
    result = run_demo()
    assert result["passed"] == result["total"] == 9
