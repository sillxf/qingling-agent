"""Release audit regressions: authorization on alternate paths and import isolation."""
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.models import Compensation
from app.runtime import RuntimeService
from app.store import InMemoryStore, SQLiteStore
from tests.test_auth import auth_settings, headers


@pytest.fixture(params=["memory", "sqlite"])
def secured_api(request, tmp_path):
    store = InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "audit.db")
    runtime = RuntimeService(store=store, app_settings=auth_settings())
    with TestClient(create_app(runtime), raise_server_exceptions=False) as client:
        try:
            yield client, runtime
        finally:
            runtime.close()


def submit(client):
    payload = {"tenant_id": "tenant-a", "agent_id": "event-investigation",
               "idempotency_key": "release-case-001", "input": {"message": "private operator evidence"}}
    response = client.post("/v1/runs", headers=headers("operator-token-456"), json=payload)
    assert response.status_code == 202
    return payload, response.json()["id"]


def test_idempotency_cannot_bypass_run_ownership(secured_api):
    client, _ = secured_api
    payload, run_id = submit(client)
    assert client.get(f"/v1/runs/{run_id}", headers=headers("analyst-token-123")).status_code == 403
    denied = client.post("/v1/runs", headers=headers("analyst-token-123"), json=payload)
    assert denied.status_code == 403
    assert "private operator evidence" not in denied.text
    repeated = client.post("/v1/runs", headers=headers("operator-token-456"), json=payload)
    assert repeated.status_code == 202 and repeated.json()["id"] == run_id


def test_compensation_update_is_tenant_scoped_and_denial_is_non_mutating(secured_api):
    client, runtime = secured_api
    _, run_id = submit(client)
    runtime.store.create_compensation(Compensation(id="compensation-a", run_id=run_id))
    before = len(runtime.store.list_events(run_id))
    denied = client.post("/v1/compensations/compensation-a", headers=headers("operator-token-789"))
    assert denied.status_code == 403
    assert runtime.store.list_compensations(run_id)[0].status == "pending"
    assert not any(event.event_type == "compensation.resolved" for event in runtime.store.list_events(run_id)[before:])
    allowed = client.post("/v1/compensations/compensation-a", headers=headers("operator-token-456"))
    assert allowed.status_code == 200
    assert allowed.json()["decided_by"] == "operator-a"


def test_resource_errors_are_http_errors_not_internal_failures(secured_api):
    client, _ = secured_api
    _, run_id = submit(client)
    assert client.get(f"/v1/runs/{run_id}/checkpoints", headers=headers("operator-token-789")).status_code == 403
    assert client.get("/v1/runs/missing/checkpoints", headers=headers("operator-token-456")).status_code == 404
    assert client.post("/v1/compensations/missing", headers=headers("operator-token-456")).status_code == 404
    assert client.get("/v1/dead-letters", headers=headers("analyst-token-123")).status_code == 403


@pytest.mark.parametrize("arguments", [["-c", "import app.main"], ["-m", "app.demo"]])
def test_import_and_offline_demo_do_not_initialize_user_database(tmp_path, arguments):
    db = tmp_path / "configured-user.db"
    sandbox = tmp_path / "configured-sandbox"
    env = {key: value for key, value in os.environ.items() if not key.startswith("QINGLING_")}
    env.update(QINGLING_STORE_BACKEND="sqlite", QINGLING_SQLITE_PATH=str(db),
               QINGLING_OPENCODE_SANDBOX_ROOT=str(sandbox))
    result = subprocess.run([sys.executable, *arguments], cwd=Path(__file__).parents[1],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not db.exists(), "import/demo must not open or recover the user's configured database"
    assert not sandbox.exists(), "import/demo must not initialize the user's configured sandbox"


def test_legacy_asgi_target_is_created_lazily_and_cached(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("QINGLING_")}
    env["QINGLING_OPENCODE_SANDBOX_ROOT"] = str(tmp_path / "legacy-sandbox")
    code = '''
import app.main as main
from fastapi.testclient import TestClient
assert "app" not in vars(main)
application = main.app
assert main.app is application
with TestClient(application) as client:
    assert client.get("/health").status_code == 200
assert application.state.runtime._closed
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
