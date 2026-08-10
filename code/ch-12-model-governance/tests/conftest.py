"""Shared fixtures: a stub flag client, and a real AppConfig agent in local dev mode."""

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENT_IMAGE = "public.ecr.aws/aws-appconfig/aws-appconfig-agent:2.x"
ION_SUFFIX = ".application%ion%type=AWS.AppConfig.FeatureFlags"


class StubFlags:
    """A stand-in flag client: a fixed, deterministic 20% challenger split."""

    def configuration(self, context: dict) -> dict:
        """Return the evaluated-flag shape the agent would, variant by loanId."""
        challenger = int(context["loanId"][-2:]) % 5 == 0
        return {
            "challenger_rollout": {
                "enabled": True,
                "_variant": "challenger" if challenger else "champion",
                "model": "credit-challenger" if challenger else "credit-scorecard",
            }
        }


@pytest.fixture
def stub_flags() -> StubFlags:
    """Return the stub flag client for router and gateway tests (no agent needed)."""
    return StubFlags()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def agent_url(tmp_path_factory):
    """Run the real AppConfig agent, serving the chapter's rollout plus a widened profile.

    The agent's local development mode reads its Ion files once at startup, so the
    widened (pct 50) rollout is a second configuration profile rather than an edit.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker is not available")
    sys.path.insert(0, str(ROOT / "etl"))
    from to_ion import to_ion

    flags = json.loads((ROOT / "local/flags/feature-flags.json").read_text())
    wide = json.loads(json.dumps(flags))
    wide["challenger_rollout"]["_variants"][0]["rule"] = (
        '(split by::$loanId pct::50 seed::"rollout-2026")'
    )
    cfg = tmp_path_factory.mktemp("agent-configs")
    (cfg / f"credit-governance:local:rollout{ION_SUFFIX}").write_text(to_ion(flags))
    (cfg / f"credit-governance:local:rollout50{ION_SUFFIX}").write_text(to_ion(wide))

    port = _free_port()
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:2772",
            "-v",
            f"{cfg}:/configs",
            "-e",
            "LOCAL_DEVELOPMENT_DIRECTORY=/configs",
            AGENT_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start the AppConfig agent: {started.stderr.strip()}")
    container = started.stdout.strip()
    url = f"http://127.0.0.1:{port}"
    probe = f"{url}/applications/credit-governance/environments/local/configurations/rollout"
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(probe, timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.skip("the AppConfig agent did not become ready")
        yield url
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False
        )
