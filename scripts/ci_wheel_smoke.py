"""Install the built wheel in a fresh venv outside the checkout and smoke-test it.

Run after ``python -m build`` from the project root. No editable install or
project PYTHONPATH is used. The temporary directory is left in the OS temp
area for diagnostics; ephemeral CI runners remove it after the job.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


SMOKE = r'''
import pathlib
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.request

from fastapi.testclient import TestClient
import app
from app.main import create_app

installed = pathlib.Path(app.__file__).resolve()
venv = pathlib.Path(sys.prefix).resolve()
assert installed.is_relative_to(venv), f"imported source, not wheel: {installed}"
with TestClient(create_app()) as client:
    assert client.get("/health").status_code == 200
    page = client.get("/workbench")
    assert page.status_code == 200 and "<html" in page.text.lower()
    assert client.app.state.runtime.store.get_profile("asset-investigation").id == "asset-investigation"
    assert client.app.state.runtime.skills.get("event-investigation") is not None
print(f"health/workbench/profile/skill OK from {installed}")

scripts = pathlib.Path(sysconfig.get_path("scripts"))
cli = scripts / ("qingling.exe" if sys.platform == "win32" else "qingling")
subprocess.run([str(cli), "--help"], check=True, stdout=subprocess.DEVNULL)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
proc = subprocess.Popen([str(cli), "--host", "127.0.0.1", "--port", str(port)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(150):
        if proc.poll() is not None:
            raise AssertionError(f"qingling CLI exited early: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                assert response.status == 200
            break
        except OSError:
            time.sleep(0.2)
    else:
        raise AssertionError("qingling CLI health did not respond")
finally:
    proc.terminate()
    proc.wait(timeout=10)
print("qingling CLI HTTP health OK")
'''


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    wheels = list((project / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in dist/, got {len(wheels)}")
    outside = Path(tempfile.mkdtemp(prefix="qingling-wheel-smoke-")).resolve()
    if outside.is_relative_to(project):
        raise SystemExit("temporary wheel test directory must be outside the checkout")
    venv = outside / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    demo = scripts / ("qingling-demo.exe" if os.name == "nt" else "qingling-demo")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # CI must not pick up development settings or credentials from a caller.
    for key in list(env):
        if key.startswith("QINGLING_"):
            env.pop(key)
    constraints = project / "requirements" / "constraints.txt"
    subprocess.run([str(python), "-m", "pip", "install", "-c", str(constraints), f"{wheels[0]}[dev]"],
                   cwd=outside, env=env, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], cwd=outside, env=env, check=True)
    subprocess.run([str(python), "-c", SMOKE], cwd=outside, env=env, check=True)
    subprocess.run([str(demo), "--output", "demo-cli.json"], cwd=outside, env=env, check=True)
    subprocess.run([str(python), "-m", "app.demo", "--output", "demo-module.json"],
                   cwd=outside, env=env, check=True)
    print(f"wheel smoke passed; isolated directory: {outside}")


if __name__ == "__main__":
    main()
