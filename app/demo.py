"""Repeatable, credential-free API acceptance demo. Run: python -m app.demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import tempfile
import time
from typing import Any

from .auth import StaticTokenAuthenticator
from .config import Settings
from .model_gateway import DeterministicModelGateway
from .runtime import RuntimeService


def _require(condition: bool, message: str) -> None:
    # Deliberately not `assert`: the demo must fail even under python -O.
    if not condition:
        raise RuntimeError(message)


def _wait(client, run_id: str, headers: dict[str, str], timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/runs/{run_id}", headers=headers)
        _require(response.status_code == 200, f"run lookup failed: {response.status_code}")
        run = response.json()
        if run["status"] in {"succeeded", "failed", "timed_out", "cancelled", "waiting_approval"}:
            return run
        time.sleep(0.01)
    raise RuntimeError(f"run {run_id} did not settle within {timeout}s")


def run_demo() -> dict[str, Any]:
    # TestClient is a development dependency; no listener, network, API key or
    # user database is used. Import lazily so normal CLI installation stays lean.
    from fastapi.testclient import TestClient
    from .main import create_app

    dataset = json.loads((Path(__file__).parent / "resources/examples/security_cases.json").read_text(encoding="utf-8"))
    token, other_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    authenticator = StaticTokenAuthenticator.from_json(json.dumps([
        {"token": token, "subject": "demo-operator", "tenant_id": "demo-tenant", "roles": ["operator"]},
        {"token": other_token, "subject": "other-operator", "tenant_id": "other-tenant", "roles": ["operator"]},
    ]))
    headers = {"Authorization": f"Bearer {token}"}
    results = []
    sample_report = None
    with tempfile.TemporaryDirectory(prefix="qingling-demo-") as temporary:
        runtime = RuntimeService(model_gateway=DeterministicModelGateway(), app_settings=Settings(
            store_backend="memory", knowledge_store_backend="memory", config_dir=None,
            model_gateway_base_url="", event_api_base_url="", mcp_servers_json="", skills_root="",
            opencode_sandbox_root=temporary, run_timeout_seconds=10, model_gateway_backoff_seconds=0,
        ))
        try:
            with TestClient(create_app(runtime, authenticator)) as client:
                for case in dataset["cases"]:
                    response = client.post("/v1/workbench/execute/asset-safety/1.1.0", headers=headers, json={
                        "tenant_id": "demo-tenant", "agent_id": "asset-investigation", "input": case["input"],
                    })
                    _require(response.status_code == 202, f"{case['id']}: submission failed: {response.text}")
                    run = _wait(client, response.json()["id"], headers)
                    output = run.get("output") or {}
                    actual = {"status": run["status"], "error_code": run.get("error_code"), **output,
                              "finding_count": len(output.get("findings", []))}
                    for key, expected in case["expected"].items():
                        _require(actual.get(key) == expected, f"{case['id']}: {key} expected {expected!r}, got {actual.get(key)!r}")
                    evidence_ids = {item["citation_id"] for item in output.get("evidence", [])}
                    for finding in output.get("findings", []):
                        _require(bool(finding["evidence_ids"]) and set(finding["evidence_ids"]) <= evidence_ids,
                                 "finding has missing or fabricated citations")
                    if case["id"] == "risk-with-citations":
                        sample_report = output
                    results.append({"case": case["id"], "passed": True, "status": run["status"],
                                    "assessment": output.get("assessment")})

                forbidden = client.get(f"/v1/runs/{run['id']}", headers={"Authorization": f"Bearer {other_token}"})
                _require(forbidden.status_code == 403, "cross-tenant Run access was not rejected")
                results.append({"case": "cross-tenant-run-read-rejected", "passed": True})

                for approved in (True, False):
                    response = client.post("/v1/runs", headers=headers, json={
                        "tenant_id": "demo-tenant", "agent_id": "security-assistant-react",
                        "input": {"tool_call": {"tool": "response.block_ip", "args": {
                            "ip": "192.0.2.10", "reason": "synthetic acceptance test; dry-run only"}}},
                    })
                    _require(response.status_code == 202, "approval scenario submission failed")
                    run = _wait(client, response.json()["id"], headers)
                    _require(run["status"] == "waiting_approval", "high-risk tool did not stop for approval")
                    events = runtime.store.list_events(run["id"])
                    approval_id = next(event.data["approval_id"] for event in events if event.event_type == "approval.required")
                    response = client.post(f"/v1/approvals/{approval_id}", headers=headers, json={"approved": approved})
                    _require(response.status_code == 200,
                             f"approval decision failed: HTTP {response.status_code}: {response.text[:1000]}")
                    final = _wait(client, run["id"], headers)
                    if approved:
                        _require(final["status"] == "succeeded", "approved run did not complete")
                        _require(final["output"]["tool_observation"]["dry_run"] is True, "demo must never cause a real block")
                    else:
                        _require(final["status"] == "failed" and final["error_code"] == "POLICY_DENIED", "rejected tool was not denied")
                    results.append({"case": "approval-then-dry-run" if approved else "rejection-prevents-action", "passed": True})
        finally:
            runtime.close()
    return {"schema_version": "1.0", "mode": "offline_acceptance", "dataset_provenance": dataset["provenance"],
            "not_a_model_accuracy_benchmark": True, "passed": len(results), "total": len(results),
            "cases": results, "sample_report": sample_report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON results and a sibling Markdown evidence report")
    args = parser.parse_args()
    try:
        result = run_demo()
    except (RuntimeError, ImportError) as exc:
        parser.exit(1, f"Demo failed: {exc}\nInstall development dependencies with pip install -e '.[dev]'.\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(result["sample_report"]["report"], encoding="utf-8")
    print(f"{result['passed']}/{result['total']} acceptance cases passed (synthetic data, offline, no real actions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
