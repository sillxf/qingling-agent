"""Publication and management for installed components and frozen workflows."""
from copy import deepcopy

from .models import ComponentManifest, WorkflowManifest
from .store import NotFoundError, _model_payload, _parse_model
from .workflow_schema import manifest_digest
from .workflow_validation import validate_workflow


class WorkflowCatalog:
    def __init__(self, store, registry):
        self.store, self.registry = store, registry
        for manifest in registry.list_manifests():
            store.catalog_put("component", manifest.id, manifest.version, _model_payload(manifest), immutable=True)
        for record in store.catalog_list("component"):
            manifest = _parse_model(ComponentManifest, record["payload"])
            if not registry.has(manifest.id, manifest.version) and manifest.implementation:
                self._install_alias(manifest)
        for record in store.catalog_list("component_state"):
            if record["payload"]["state"] == "disabled":
                registry.disabled.add((record["id"], record["version"]))

    def _install_alias(self, manifest):
        if not manifest.implementation or "@" not in manifest.implementation:
            raise ValueError("implementation must reference an installed component@version")
        component, version = manifest.implementation.rsplit("@", 1)
        source = self.registry.definition(component, version)
        # An alias can add documentation, but cannot weaken the executable's
        # input/output contract, permission requirements or execution limits.
        for field in ("input_schema", "output_schema", "config_schema", "security", "execution", "kind", "dependencies"):
            if getattr(manifest, field) != getattr(source.manifest, field):
                raise ValueError(f"installed implementation contract mismatch: {field}")
        self.registry.register(manifest.id, source.handler, manifest=manifest)

    def register(self, manifest):
        # Validate before persisting; publication cannot import arbitrary code.
        if self.registry.has(manifest.id, manifest.version):
            if self.registry.definition(manifest.id, manifest.version).manifest != manifest:
                raise ValueError("component version already exists")
        else:
            self._install_alias(manifest)
        return self.store.catalog_put("component", manifest.id, manifest.version, _model_payload(manifest), immutable=True)

    def component_state(self, component, version, state):
        if state not in {"published", "deprecated", "disabled"}:
            raise ValueError("invalid component state")
        self.registry.definition(component, version)
        record = self.store.catalog_put("component_state", component, version, {"state": state})
        if state == "disabled":
            self.registry.disabled.add((component, version))
        else:
            self.registry.disabled.discard((component, version))
        return record

    def components(self):
        states = {(r["id"], r["version"]): r["payload"]["state"] for r in self.store.catalog_list("component_state")}
        result = []
        workflows = self.store.list_workflows()
        for record in self.store.catalog_list("component"):
            item = deepcopy(record["payload"])
            item["state"] = states.get((item["id"], item["version"]), "published")
            item["available"] = self.registry.has(item["id"], item["version"]) and item["state"] != "disabled"
            item["references"] = [{"id": w.id, "version": w.version, "tenant_id": w.tenant_id} for w in workflows if any(n.component == item["id"] and n.version == item["version"] for n in w.nodes)]
            result.append(item)
        return result

    def assert_owner(self, workflow):
        existing = [w for w in self.store.list_workflows() if w.id == workflow.id]
        drafts = self.store.catalog_list("workflow_draft")
        owners = [w.tenant_id for w in existing] + [r["payload"].get("tenant_id") for r in drafts if r["id"] == workflow.id]
        if any(owner != workflow.tenant_id for owner in owners):
            raise ValueError("workflow id is reserved by another owner")

    def save_draft(self, workflow, revision):
        self.assert_owner(workflow)
        return self.store.catalog_put("workflow_draft", workflow.id, workflow.version, _model_payload(workflow), expected_revision=revision)

    def publish(self, workflow):
        self.assert_owner(workflow)
        validation = validate_workflow(workflow, self.registry)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        states = {(r["id"], r["version"]): r["payload"]["state"] for r in self.store.catalog_list("component_state")}
        for node in workflow.nodes:
            if states.get((node.component, node.version)) == "deprecated":
                raise ValueError(f"new publication cannot reference deprecated component: {node.component}")
        digests = {node.id: manifest_digest(self.registry.definition(node.component, node.version).manifest) for node in workflow.nodes}
        for node in workflow.nodes:
            if node.component == "control.loop":
                body = self.registry.definition(node.config["component"], node.config["version"]).manifest
                digests[node.id + ":body"] = manifest_digest(body)
        published = self.store.register_workflow(workflow)
        self.store.catalog_put("workflow_release", workflow.id, workflow.version, {"components": digests}, immutable=True)
        return published

    def activate(self, workflow_id, version, enabled=True):
        workflow = self.store.get_workflow(workflow_id, version)
        if enabled:
            self.verify(workflow)
        return self.store.catalog_put("workflow_active", workflow_id, "active", {"version": version, "enabled": enabled})

    def resolve(self, workflow_id, version=None):
        active = next((r for r in self.store.catalog_list("workflow_active") if r["id"] == workflow_id), None)
        if version is None and active:
            if not active["payload"]["enabled"]:
                raise ValueError("workflow is disabled")
            version = active["payload"]["version"]
        workflow = self.store.get_workflow(workflow_id, version)
        self.verify(workflow)
        return workflow

    def ensure_enabled(self, workflow_id):
        active = next((r for r in self.store.catalog_list("workflow_active") if r["id"] == workflow_id), None)
        if active and not active["payload"]["enabled"]:
            raise ValueError("workflow is disabled for new runs")

    def verify(self, workflow):
        validation = validate_workflow(workflow, self.registry)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        release = next((r for r in self.store.catalog_list("workflow_release") if r["id"] == workflow.id and r["version"] == workflow.version), None)
        if release:
            for node in workflow.nodes:
                actual = manifest_digest(self.registry.definition(node.component, node.version).manifest)
                if release["payload"]["components"].get(node.id) != actual:
                    raise ValueError("published component manifest changed; execution blocked")
                if node.component == "control.loop":
                    body = self.registry.definition(node.config["component"], node.config["version"]).manifest
                    if (body.id, body.version) in self.registry.disabled or release["payload"]["components"].get(node.id + ":body") != manifest_digest(body):
                        raise ValueError("loop body changed or disabled; execution blocked")
        return workflow

    def list_workflows(self, *, tenant_id=None, include_all=False):
        """Return the management view without exposing catalog storage to routes."""
        def visible(workflow):
            return not workflow.tenant_id or include_all or workflow.tenant_id == tenant_id

        published = [w for w in self.store.list_workflows() if visible(w)]
        drafts = [
            record for record in self.store.catalog_list("workflow_draft")
            if visible(_parse_model(WorkflowManifest, record["payload"]))
        ]
        published_ids = {workflow.id for workflow in published}
        active = [
            record for record in self.store.catalog_list("workflow_active")
            if record["id"] in published_ids
        ]
        return {"published": published, "drafts": drafts, "active": active}
