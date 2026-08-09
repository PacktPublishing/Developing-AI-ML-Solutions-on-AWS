"""The feature-flag seam: the AWS AppConfig agent, or the from-source evaluator.

get_appconfig() returns a client whose configuration(context) yields the evaluated
feature flags for a request. The authoritative path is the real AWS AppConfig agent
(AgentAppConfig): the same published agent image runs as a sidecar locally and on ECS,
so the split buckets every loan identically in both worlds. AppConfig (with a flag
store) is the from-source reference: it evaluates the variant rules in process with
rule_evaluator.py, useful for the non-split logic and for running without Docker. Note
that the reference evaluator's split is not bit-identical to the agent's, which is why
the rollout decision runs through the agent.

Env: APPCONFIG_AGENT_URL -> query the agent (with APPCONFIG_APP / APPCONFIG_ENV /
     APPCONFIG_PROFILE); APPCONFIG_LOCAL=1 -> the from-source evaluator over a local
     file (APPCONFIG_FLAGS); otherwise the AppConfig management API.
"""

import json
import os
import urllib.request
from pathlib import Path

from .rule_evaluator import evaluate_config


class LocalFlagStore:
    """Read the feature-flag document from a local file (the flag store)."""

    def __init__(self, path: str | Path) -> None:
        """Point at the local feature-flag JSON."""
        self._path = Path(path)

    def raw(self) -> dict:
        """Return the raw (unevaluated) feature-flag document."""
        return json.loads(self._path.read_text())


class AppConfigStore:
    """Fetch the raw configuration through the AppConfig management API."""

    def __init__(
        self, application: str, profile: str, region: str | None = None
    ) -> None:
        """Resolve against a named AppConfig application and configuration profile."""
        import boto3

        self._app, self._profile = application, profile
        self._client = boto3.client(
            "appconfig", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )

    def _resolve(self, method: str, name: str, **kwargs) -> str:
        """Resolve an AppConfig resource name to its id."""
        for item in getattr(self._client, method)(**kwargs).get("Items", []):
            if item.get("Name") == name:
                return item["Id"]
        raise RuntimeError(f"AppConfig resource '{name}' not found")

    def raw(self) -> dict:
        """Return the latest hosted configuration version, with its _variants intact."""
        app = self._resolve("list_applications", self._app)
        profile = self._resolve(
            "list_configuration_profiles", self._profile, ApplicationId=app
        )
        versions = self._client.list_hosted_configuration_versions(
            ApplicationId=app, ConfigurationProfileId=profile
        )["Items"]
        latest = max(v["VersionNumber"] for v in versions)
        body = self._client.get_hosted_configuration_version(
            ApplicationId=app, ConfigurationProfileId=profile, VersionNumber=latest
        )["Content"].read()
        return json.loads(body)


class AppConfig:
    """The from-source reference: a flag store plus in-process rule evaluation."""

    def __init__(self, store) -> None:
        """Wrap a flag store (local file or AppConfig management API)."""
        self._store = store

    def configuration(self, context: dict) -> dict:
        """Evaluate every flag's variant rules against the request context."""
        return evaluate_config(self._store.raw(), context)


class AgentAppConfig:
    """The authoritative path: query the AWS AppConfig agent, which evaluates the rules."""

    def __init__(
        self, base_url: str, application: str, environment: str, profile: str
    ) -> None:
        """Build the agent's configuration URL for one application/environment/profile."""
        self._url = (
            f"{base_url.rstrip('/')}/applications/{application}"
            f"/environments/{environment}/configurations/{profile}"
        )

    def configuration(self, context: dict) -> dict:
        """Ask the agent for the evaluated flags, passing the context it buckets on."""
        header = "&".join(f"{k}={v}" for k, v in context.items())
        req = urllib.request.Request(self._url, headers={"Context": header})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())


def get_appconfig():
    """Return the feature-flag client: the agent, the local evaluator, or management API."""
    agent = os.environ.get("APPCONFIG_AGENT_URL")
    if agent:
        return AgentAppConfig(
            agent,
            os.environ.get("APPCONFIG_APP", "credit-governance"),
            os.environ.get("APPCONFIG_ENV", "local"),
            os.environ.get("APPCONFIG_PROFILE", "rollout"),
        )
    if os.environ.get("APPCONFIG_LOCAL") == "1":
        path = os.environ.get("APPCONFIG_FLAGS", "local/flags/feature-flags.json")
        return AppConfig(LocalFlagStore(path))
    return AppConfig(
        AppConfigStore(
            application=os.environ.get("APPCONFIG_APP", "credit-governance"),
            profile=os.environ.get("APPCONFIG_PROFILE", "rollout"),
        )
    )
