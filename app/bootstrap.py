from __future__ import annotations

import json
from pathlib import Path

from .models import AgentProfile, WorkflowManifest
from .runtime import RuntimeService


def _parse(model, payload):
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


def load_demo_configuration(runtime: RuntimeService, config_dir: Path | None = None) -> None:
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "resources" / "config"
    profile_path = config_dir / "agent_profile.json"
    react_profile_path = config_dir / "react_profile.json"
    workflow_path = config_dir / "workflow_event_investigation.json"
    profile = _parse(AgentProfile, json.loads(profile_path.read_text(encoding="utf-8")))
    workflow = _parse(WorkflowManifest, json.loads(workflow_path.read_text(encoding="utf-8")))
    runtime.publish_workflow(workflow)
    runtime.publish_profile(profile)
    asset_path = config_dir / "workflow_asset_safety.json"
    if asset_path.exists():
        runtime.publish_workflow(_parse(WorkflowManifest, json.loads(asset_path.read_text(encoding="utf-8"))))
    if react_profile_path.exists():
        react_profile = _parse(AgentProfile, json.loads(react_profile_path.read_text(encoding="utf-8")))
        runtime.publish_profile(react_profile)

    report_path = config_dir / "workflow_asset_safety_report.json"
    if report_path.exists():
        runtime.publish_workflow(_parse(WorkflowManifest, json.loads(report_path.read_text(encoding="utf-8"))))
    asset_profile = config_dir / "asset_profile.json"
    if asset_profile.exists():
        runtime.publish_profile(_parse(AgentProfile, json.loads(asset_profile.read_text(encoding="utf-8"))))
